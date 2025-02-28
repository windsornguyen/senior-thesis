import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from dataclasses import dataclass, field
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

# --- TODO: These two are always compiled even when kernel.compile is disabled. We should fix this. ---
from thesis.models.mamba.causal_conv1d_compilable import causal_conv1d_fn, causal_conv1d_update
from thesis.models.mamba.ssm_compilable import mamba_chunk_scan_combined
# -----------------------------------------------------------------------------------------------------

from thesis.models.norms import build_norm

from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*num_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class InitConfig:
    dt_max: float = 0.1
    dt_min: float = 0.001

    dt_init_floor: float = 1e-4

    A_init_min: float = 1
    A_init_max: float = 16


DEFAULT_INIT_CONFIG = InitConfig()


@dataclass
class BaseMambaConfig:
    """
    Configuration for the Mamba family of models.
    """

    dim: int = 512
    num_layers: int = 8
    num_heads: int = 8

    state_dim: int = 128
    num_groups: int = 1
    conv_size: int | None = 4

    bias: bool = False  # Linear bias
    conv_bias: bool = True  # Convolutional bias
    dt_bias: bool = False
    D_has_head_dim: bool = False
    learnable_init_states: bool = False

    ffn_dim_multiplier: float = 2.0
    multiple_of: int = 256  # Enforce that MLP hidden layer size is multiple of a large power of 2

    norm_eps: float = 1e-6
    norm_type: str = "rmsnorm"

    # CUDA-related items
    chunk_size: int = 256
    use_mem_eff_path: bool = False

    # Initialization-related items
    init_use_depth: bool = False
    init_base_std: float | None = None
    init_std_factor: str = "disabled"  # e.g. "global_depth"
    init_config: InitConfig = field(default_factory=InitConfig)


class SSM(nn.Module):
    """
    State Space Model (SSM) implementation with selective state updates and convolution.

    Implements the core SSM computation with support for both training and inference modes.
    During inference, uses cached states for efficient token-by-token generation.
    """

    def __init__(self, config: BaseMambaConfig) -> None:
        """Initialize SSM parameters and layers.
        Args:
            config: Configuration containing model hyperparameters
        """
        super().__init__()
        self.config = config
        vars(self).update(vars(config))

        assert self.dim > 0, "Model dimension (config.dim) must be positive"
        assert self.num_heads > 0, "Number of heads (config.num_heads) must be positive"
        assert self.state_dim > 0, "State dimension (config.state_dim) must be positive"

        if self.ffn_dim_multiplier is None:
            raise ValueError(
                "ffn_dim_multiplier must be set to a valid float (e.g. 2.0) " "to determine hidden_dim in SSM."
            )
        assert self.ffn_dim_multiplier > 0, "ffn_dim_multiplier must be > 0"

        self.hidden_dim = int(self.ffn_dim_multiplier * self.dim)
        self.hidden_dim = config.multiple_of * (  # Round up to multiple_of
            (self.hidden_dim + self.multiple_of - 1) // self.multiple_of
        )

        assert (
            self.hidden_dim % self.num_heads == 0
        ), f"Hidden dim {self.hidden_dim} not divisible by num_heads={self.num_heads}."

        self.head_dim = self.hidden_dim // self.num_heads

        self.dt_limit_kwargs = {}
        dt_limit = (self.init_config.dt_min, self.init_config.dt_max)
        if dt_limit != (0.0, float("inf")):
            self.dt_limit_kwargs = dict(dt_limit=dt_limit)

        # Order: [z, x, B, C, dt]
        d_input = 2 * self.hidden_dim + 2 * self.num_groups * self.state_dim + self.num_heads

        self.input = nn.Linear(self.dim, d_input, bias=self.bias)

        # Only create Conv1d if self.conv_size is specified
        if self.conv_size is not None:
            conv_dim = self.hidden_dim + 2 * self.num_groups * self.state_dim

            # Depthwise-ish conv (groups = out_channels)
            # TODO: Check that this is used if causal_conv1d_fn and causal_conv1d_update cannot be imported
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=self.conv_size,
                groups=conv_dim,
                bias=self.conv_bias,  # <- This is a boolean in your config, so pass that or True/False
                padding=self.conv_size - 1,  # for "causal" style
            )

        if config.dt_bias:
            self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        else:
            self.dt_bias = nn.Parameter(torch.zeros(self.num_heads), requires_grad=False)

        self.A_log = nn.Parameter(torch.empty(self.num_heads))

        if config.D_has_head_dim:
            self.D = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        else:
            self.D = nn.Parameter(torch.ones(self.num_heads))

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.state_dim))

        self.norm = build_norm(config.norm_type, dim=self.hidden_dim, eps=self.norm_eps)
        self.output = nn.Linear(self.hidden_dim, self.dim, bias=self.bias)

    def _causal_conv(
        self,
        zxbcdt: torch.Tensor,
        tok_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        ssm_impl: str = "ssm",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Make slightly less verbose
        """Processes input through causal convolution path, handling both full sequence and incremental cases.

        This function implements two processing modes:
        1. Full sequence ("ssm"): Used during training and initial prompt processing.
        2. Incremental ("ssm_update"): Used during token-by-token generation.

        Args:
            zxbcdt: Input tensor containing concatenated [z, x, B, C, dt] components
            tok_idx: Token indices for sequence processing. Required for "ssm" mode.
                Defaults to None.
            cu_seqlens: Cumulative sequence lengths for variable length processing.
                Used only in "ssm" mode with caching. Defaults to None.
            ssm_impl: Implementation mode, either "ssm" for full sequence processing
                or "ssm_update" for incremental generation. Defaults to "ssm".

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple containing separated components (z, x, B, C, dt), where:
                - z: Gating branch
                - x: Main branch
                - B, C: SSM state matrices (analogous to K, Q in attention)
                - dt: Time delta values

        Notes:
            - When using "ssm" mode during inference, a cache should be pre-initialized
            externally. This design allows for flexible caching strategies without
            modifying model code.
            - The "ssm_update" mode requires a cache to exist and will use it for
            incremental state updates during generation.
            - B, C components correspond to Key, Query in the SSM/attention duality.
        """
        # Split input into components
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.hidden_dim,
                self.hidden_dim + 2 * self.num_groups * self.state_dim,
                self.num_heads,
            ],
            dim=-1,
        )

        if ssm_impl == "ssm":
            if hasattr(self, "cache"):
                conv_varlen_states = causal_conv1d_varlen_states(
                    xBC.squeeze(0),
                    cu_seqlens,
                    state_len=self.cache.conv_cache.shape[-1],
                )
                self.cache.conv_cache.copy_(conv_varlen_states)

            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                seq_idx=tok_idx,
            ).transpose(1, 2)
        elif ssm_impl == "ssm_update":
            xBC = causal_conv1d_update(
                x=xBC.squeeze(0),
                conv_state=self.cache.conv_cache,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            ).unsqueeze(0)
        else:
            raise NotImplementedError(f"SSM implementation {ssm_impl} not supported")

        # Split processed tensor into components
        x, B, C = torch.split(
            xBC,
            [
                self.hidden_dim,
                self.num_groups * self.state_dim,
                self.num_groups * self.state_dim,
            ],
            dim=-1,
        )

        return z, x, B, C, dt

    def _non_causal_conv(self, zxbcdt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, x, B, C, dt = torch.split(
            zxbcdt,
            [
                self.hidden_dim,
                self.hidden_dim,
                self.num_groups * self.state_dim,
                self.num_groups * self.state_dim,
                self.num_heads,
            ],
            dim=-1,
        )
        return z, x, B, C, dt

    def _fwd(self, x, dt, A, B, C, tok_idx, cu_seqlens, initial_states):
        """
        For training

        Returns:
            (bsz, seq_len, num_heads, head_dim)
        """
        y = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=tok_idx,
            cu_seqlens=cu_seqlens,
            initial_states=initial_states,
            **self.dt_limit_kwargs,
        )

        if hasattr(self, "cache"):
            y, varlen_states = y
            self.cache.state_cache.copy_(varlen_states)

        return y

    def _step(self, x, seq_len, dt, A, B, C):
        """
        For inference / generation.
        """
        x = x.squeeze(0)
        A = A[..., None, None].expand(self.num_heads, self.head_dim, self.state_dim)
        dt = dt.permute(1, 2, 0).expand(seq_len, self.num_heads, self.head_dim)
        D = self.D
        if D is not None and D.dim() == 1:
            D = D.unsqueeze(1).expand(self.num_heads, self.head_dim)
        B, C = B.squeeze(0), C.squeeze(0)
        y = selective_state_update(
            self.cache.state_cache,
            x,
            dt,
            A,
            B,
            C,
            D,
            z=None,
            dt_bias=(
                torch.zeros(self.num_heads, self.head_dim).to(x)
                if self.dt_bias is None
                else self.dt_bias.unsqueeze(1).expand(self.num_heads, self.head_dim)
            ),
            dt_softplus=True,
        ).unsqueeze(0)

        return y

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        zxbcdt = self.input(x)

        A = -torch.exp(self.A_log.float())
        initial_states = self.init_states.expand(bsz, -1, -1, -1) if self.learnable_init_states else None

        # Causal conv path
        if self.conv_size is not None:
            # Memory-efficient Triton kernel path
            if self.use_mem_eff_path:
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=tok_idx,
                    activation="silu",
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.output.weight,
                    outproj_bias=self.output.bias,
                    headdim=self.head_dim,
                    ngroups=self.num_groups,
                    norm_before_gate=False,  # Post-norm, y = self.norm(y * F.silu(z))
                    initial_states=initial_states,
                    **self.dt_limit_kwargs,
                )
                return out
            else:
                # CUDA kernel path
                z, x, B, C, dt = self._causal_conv(zxbcdt)
        else:
            # Non-causal conv path
            z, x, B, C, dt = self._non_causal_conv(zxbcdt)

        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        B = B.view(bsz, seq_len, self.num_groups, self.state_dim)
        C = C.view(bsz, seq_len, self.num_groups, self.state_dim)

        # Chunked SSM scan
        if ssm_impl == "ssm":
            # (bsz, seq_len, num_heads, head_dim)
            y = self._fwd(x, dt, A, B, C, tok_idx, cu_seqlens, initial_states)
        elif ssm_impl == "ssm_update":
            y = self._step(x, seq_len, dt, A, B, C)
        else:
            raise NotImplementedError(f"SSM implementation {ssm_impl} not supported")

        y = y.view(bsz, seq_len, self.hidden_dim)

        # Could be different activation function, including None.
        # Mamba people post_norm here also (sometimes norm(z)*y or norm(z*y))
        # y = self.norm(y) * F.silu(z)
        y = self.norm(y * F.silu(z))
        out = self.output(y)

        return out

    @torch.inference_mode()
    def reset_parameters(self, init_std, factor) -> None:
        config = self.config
        init_config = config.init_config
        if init_config is None:
            init_config = DEFAULT_INIT_CONFIG

        # Linear layers
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor

        nn.init.trunc_normal_(
            self.input.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )

        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )

        # SSM
        if self.dt_bias is not None and self.dt_bias.requires_grad:
            log_dt_min = math.log(init_config.dt_min)
            log_dt_max = math.log(init_config.dt_max)

            # Sample log_dt ~ Uniform[log_dt_min, log_dt_max]
            log_dt = torch.rand(self.num_heads, device=self.dt_bias.device) * (log_dt_max - log_dt_min) + log_dt_min
            dt = torch.exp(log_dt)
            dt = torch.clamp(dt, min=init_config.dt_init_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_dt)

        elif self.dt_bias is not None:
            # If dt_bias is not trainable, we can just keep it zero or set to any constant
            self.dt_bias.fill_(0.0)

        # Convolution
        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(
                self.conv1d.weight,
                mean=0.0,
                std=conv_std,
                a=-3 * conv_std,
                b=3 * conv_std,
            )
            if self.conv1d.bias is not None:
                nn.init.zeros_(self.conv1d.bias)

        # Learnable init states
        if self.learnable_init_states:
            self.init_states.zero_()

        # Initialize A_log ~ log( Uniform(A_init_min, A_init_max) )
        self.A_log.uniform_(init_config.A_init_min, init_config.A_init_max)
        self.A_log.log_()

        if self.D is not None:
            self.D.data.fill_(1.0)

        # Reset norm parameters
        self.norm.reset_parameters()


class MambaBlock(nn.Module):
    def __init__(self, config: BaseMambaConfig):
        super().__init__()
        self.norm = build_norm(config.norm_type, dim=config.dim, eps=config.norm_eps)
        self.ssm = SSM(config)

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        x = x + self.ssm(self.norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        return x

    @torch.inference_mode()
    def init_weights(self, init_std=None, factor=1.0):
        self.norm.reset_parameters()
        self.ssm.reset_parameters(init_std, factor)


class BaseMamba(nn.Module):
    def __init__(self, config: BaseMambaConfig):
        super().__init__()
        self.model_dim = config.dim
        self.init_base_std = config.init_base_std

        self.init_config = config.init_config
        self.init_std_factor = InitStdFactor(config.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(MambaBlock(config))

    def forward(
        self,
        h: torch.Tensor,
        tok_idx: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        return h

    @torch.inference_mode()
    def reset_parameters(self):
        pass

    @torch.inference_mode()
    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.model_dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)


@dataclass
class MambaConfig(BaseMambaConfig):

    # Language modeling specific parameters
    vocab_size: int = -1  # Will error if unchanged, makes you double check!
    weight_tying: bool = False

    # Training parameters
    loss_reduction: str = "mean"
    torch_dtype: torch.dtype = torch.bfloat16
    seed: int = 1746

    # SSM specific parameters
    ssm_chunk_size: int = field(default=256)  # Alias for chunk_size in BaseMambaConfig
    use_attn: bool = False
    softcap: float | None = None

    def __post_init__(self):
        # Ensure chunk_size is set from ssm_chunk_size for backward compatibility
        self.chunk_size = self.ssm_chunk_size


class Mamba(BaseMamba):
    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)
        self.weight_tying = config.weight_tying
        self.loss_reduction = config.loss_reduction

        assert config.vocab_size > 0, "vocab_size must be set and > 0"

        self.tok_emb = torch.nn.Embedding(config.vocab_size, config.dim)

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        if config.weight_tying:
            self.output.weight = self.tok_emb.weight

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        tok_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        h = self.tok_emb(x)
        h = super().forward(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        logits = self.output(self.norm(h))
        return logits

    @torch.inference_mode()
    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.model_dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_emb.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    @torch.inference_mode()
    def init_weights(self, buffer_device: torch.device = None):
        """
        Initialize model parameters and optionally compute buffers on a specific device.

        Args:
            buffer_device (torch.device, optional): If provided, any large or precomputed
                buffers (like RoPE frequency tensors) will be allocated or re-created on
                this device during initialization. This can avoid overhead from transferring
                buffers between CPU and GPU after creation. If None, buffers default to the
                device of the first parameter or CPU.

        Usage:
            - Pass a GPU device (e.g., ``torch.device('cuda')``) when you want to ensure
            buffers are created directly on GPU, preventing extra transfers.
            - Pass a CPU device (e.g., ``torch.device('cpu')``) if you want to keep
            large buffers in CPU memory (common in CPU-offload or pipeline-parallel setups).
            - Leave it as ``None`` to rely on the model's existing parameter device or
            the default PyTorch device context.

        When / Why:
            - Useful in distributed or pipeline-parallel training where parameters may
            initially live on CPU, but you still need certain buffers on GPU to avoid
            overhead during forward passes.
            - Prevents large re-allocations or re-copies when big buffers (like RoPE
            frequency tables) are needed per rank.
        """
        super().init_weights()

    @classmethod
    def from_model_args(cls, config: MambaConfig) -> "Mamba":
        """
        Initialize a Mamba model from a MambaConfig object.

        Args:
            config (MambaConfig): Mamba configuration arguments.

        Returns:
            Mamba: Mamba-2 model.
        """
        return cls(config)

    def _flops_per_tok(self, num_layers, seq_len, dim, causal):
        # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
        return 3.5 * (4 * num_layers * seq_len * dim // (2 if causal else 1))

    def get_num_flops_per_token(self, num_non_embed_params: int, num_layers: int, dim: int, seq_len: int) -> int:
        return 6 * num_non_embed_params + self._flops_per_tok(num_layers, seq_len, dim, True)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm.default,
        torch.ops.c10d_functional.reduce_scatter_tensor.default,
        torch.ops.mamba_ssm.ssm_chunk_scan_combined_fwd.default,
        # For low-precision training, it's useful to always save the result of max(abs(tensor))
        torch.ops.aten.abs.default,
        torch.ops.aten.max.default,
    }


def main():
    from mamba_ssm import Mamba2 as MambaRef

    x = torch.randn(2, 64, 192).cuda()

    # Create and run the first model
    model = MambaRef(
        d_model=192,
        expand=2,
        d_conv=4,
        d_state=64,
        headdim=48,
    ).cuda()
    y = model(x)
    print("Mamba reference output: ", y)
    print("Mean of MambaRef output: ", y.mean().item())
    print("Stddev of MambaRef output: ", y.std().item())

    # Create and run the second model
    config = MambaConfig(vocab_size=200064, use_mem_eff_path=True)
    model2 = Mamba(
        config=config,
    ).cuda()

    # Fix: Convert x to torch.LongTensor
    x_indices = torch.randint(0, config.vocab_size, (2, 64), dtype=torch.long).cuda()

    y2 = model2(x_indices)
    print("Mamba output: ", y2)
    print("Mean of Mamba output: ", y2.mean().item())
    print("Stddev of Mamba output: ", y2.std().item())


if __name__ == "__main__":
    main()
