# -*- Author: Windsor Nguyen '25 -*-

import inspect
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import convolve, fftconvolve

from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from thesis.experiments.utils.assoc_scan.kernel import associative_scan
from thesis.utils.pytorch import get_device_info
from thesis.utils.systems import get_peak_flops

from thesis.utils.logger import logger

loss_fn = nn.CrossEntropyLoss()

os.environ["TIKTOKEN_CACHE_DIR"] = (
    "/scratch/gpfs/mn4560/hazan-lab/tensorized_filters/tensorized_filters/tiktoken_cache"
)
import tiktoken

# -----------------------------------------------------------------------------


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def tensordot_conv(
    f: torch.Tensor,  # [L, h]
    u: torch.Tensor,  # [B, H, L, h]
) -> torch.Tensor:
    """
    Performs a causal FFT convolution for projected filters.

    Args:
        f (Tensor): [L, h] Spectral filters, one length-L kernel per head-channel, 
            where h = head_dim.
        u (Tensor): [B, H, L, h] Batched input sequences.

    Returns:
        Tensor: [B, H, L, h] Causal convolution of each (B, H, h) stream with its 
        corresponding filter f[:, h].

    Notes:
        Vectorization order is channels → heads → batch (fastest-changing 
        dimension first), allowing the operation to fuse into a single CUDA 
        kernel when using `torch.compile` or `vmap`.
    """
    causal_conv = lambda sig, ker: fftconvolve(sig, ker, mode="full")[..., :sig.shape[-1]]
    cmap = torch.vmap(causal_conv, in_dims=(1, 1), out_dims=0)   # over head dim
    hmap = torch.vmap(cmap, in_dims=(0, None), out_dims=0)  # over heads
    bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)  # over batches
    return bmap(u, f).permute(0, 1, 3, 2)                   # [B, H, L, h]


@torch.compile
def stu_conv(filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    r"""Applies K shared filters causally to all channels using nested vmap.

    Performs 1D causal convolution, applying *each* of the ``K_num`` filters
    provided in `filters` independently to *every* channel (`h`) of the input,
    across all batches (`B`) and heads (`H`). This implementation uses nested
    `torch.vmap` calls and results in an expanded output dimension ``K_num``.

    Args:
        filters (torch.Tensor): Bank of shared filters. Shape: ``[K_len, K_num]``,
            where ``K_len`` is the kernel length and ``K_num`` is the number
            of distinct filters.
        inputs (torch.Tensor): Input sequences (typically pre-projected). Shape:
            ``[B, H, L, h]``, where ``L`` is sequence length.

    Returns:
        torch.Tensor: Output tensor after convolution. Shape: ``[B, H, L, K_num, h]``.

    Example::

        >>> B, H, L, h, K_len, K_num = 2, 3, 16, 4, 5, 7
        >>> filters = torch.randn(K_len, K_num)
        >>> inputs = torch.randn(B, H, L, h)
        >>> output = stu_conv(filters, inputs)
        >>> output.shape
        torch.Size([2, 3, 16, 7, 4])
    """
    if filters.dim() != 2:
        raise ValueError("filters must be 2D tensor of shape [K_len, K_num]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, h]")

    inputs = inputs.transpose(-1, -2)  # [B, H, h, L]
    causal_conv = lambda sig, ker: fftconvolve(sig, ker, mode="full")[..., : sig.shape[-1]]

    kmap = torch.vmap(causal_conv, in_dims=(None, 1), out_dims=0)  # Map over K_num filters
    cmap = torch.vmap(kmap, in_dims=(0, None), out_dims=1)  # Map over h signal, add K dim after h
    hmap = torch.vmap(cmap, in_dims=(0, None), out_dims=1)  # Map over H signal, add K dim after H
    bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)  # Map over B signal, add K dim before H

    y = bmap(inputs, filters)  # [B, K_num, H, h, L]
    return y.permute(0, 2, 4, 1, 3)  # [B, H, L, K_num, h]


@torch.compile
def leaky_relu(x: torch.Tensor, squared: bool = True, alpha: float = 1e-2) -> torch.Tensor:
    r"""Apply the leaky ReLU activation, with optional squaring of the output.

    Args:
        x (Tensor): Input tensor.
        squared (bool, optional): If True, returns the square of the leaky ReLU output. 
            Defaults to True.
        alpha (float, optional): Slope for the negative part of the input. 
            Defaults to 1e-2.

    Returns:
        Tensor: The activated tensor, optionally squared.
    """
    out = torch.where(
        condition=x > 0,
        input=x,
        other=alpha * x,
    )
    if squared:
        return out * out
    return out


@torch.compile
def combine_fn(self, x: Tuple[Any], y: Tuple[Any]) -> Tuple[Any]:
        """Combine two leaves of the associative scan tree for AA.

        Args:
            x: First leaf of the scan tree, containing the current state.
            y: Second leaf of the scan tree, containing the new input.

        Returns:
            A tuple representing the combined state of the two leaves.

        NOTE:

            Each leaf is a tuple (m, s, n, Z, g) where:
                - m: Running maximum (for stable online softmax)
                - s: Running sum of exp(score - m)
                - n: Running sum of exp(score - m) * value
                - Z: Running sum of gated outer products
                - g: Running sum of gates

        """
        m_x, s_x, n_x, Z_x, g_x = x
        m_y, s_y, n_y, Z_y, g_y = y

        # Compute the new maximum
        m_new = torch.max(m_x, m_y)

        # Scaling factors
        exp_x = torch.exp(m_x - m_new)
        exp_y = torch.exp(m_y - m_new)

        # Update softmax aggregator components
        s_new = s_x * exp_x + s_y * exp_y
        n_new = n_x * exp_x.view(-1, 1) + n_y * exp_y.view(-1, 1)

        # Update gated Z and gate accumulation
        Z_new = Z_x + Z_y
        g_new = g_x + g_y

        return (m_new, s_new, n_new, Z_new, g_new)

@torch.compile
def scan_fn(
    qk_slice: torch.Tensor,
    v_slice: torch.Tensor,
    Z_slice: torch.Tensor,
    g_slice: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Associatively scan over a (B, H) stream.

    Args:
        qk_slice (Tensor): [L] Similarity logits for the stream.
        v_slice (Tensor): [L, h] L2-normalized V (first-order numerator).
        Z_slice (Tensor): [L, h, h] Gated outer-product accumulator.
        g_slice (Tensor): [L] Scalar gate sequence.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            - max_cumul (Tensor): [L] Maximum cumulative similarity.
            - norm_cumul (Tensor): [L] Cumulative normalization factors.
            - v_cumul (Tensor): [L, h] Cumulative first-order vectors.
            - Z_cumul (Tensor): [L, h, h] Cumulative second-order matrices.
            - gate_cumul (Tensor): [L] Cumulative gate values.
    """
    leaves = (
        qk_slice,                   # m (initialized to qk cosine similarity)
        torch.ones_like(qk_slice),  # s (denominator starts at 1)
        v_slice,                    # n (V numerator)
        Z_slice,                    # Z (second-order accumulator)
        g_slice,                    # g (gate accumulator)
    )
    return associative_scan(
        combine_fn=combine_fn,
        xs=leaves,
        dim=0,
        combine_mode="generic",
    )

@torch.compile
def spectron_scan(sim, v, gated_Z, gates):
    B, H = sim.shape[:2]
    
    sim_flat, v_flat, gated_Z_flat, gates_flat = (t.flatten(0, 1) for t in (sim, v, gated_Z, gates))
    hmap = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    bmap = hmap(sim_flat, v_flat, gated_Z_flat, gates_flat)

    return tuple(result.unflatten(0, (B, H)) for result in bmap)


class BaseConfigForCausalLM(PretrainedConfig):
    """Base PretrainedConfig class to be decorated with dataclass"""

    model_type = "base_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class SpectronConfig(BaseConfigForCausalLM):
    model_type = "Spectron"
    spectral_filters: torch.Tensor
    spectral_filters_fft: Optional[torch.Tensor] = None

    bsz: int = 1
    dim: int = 1024
    num_heads: int = 12
    num_local_heads: Optional[int] = -1
    num_layers: int = 12
    seq_len: int = 4096
    vocab_size: int = 200064
    inter_dim: Optional[int] = 3072
    mlp_scale: Optional[float] = 12.0
    use_tensordot: Optional[bool] = True
    weight_tying: Optional[bool] = True
    bias: Optional[bool] = False
    eps: float = 1e-5
    torch_dtype: torch.dtype = torch.bfloat16
    device: torch.device = None

    def __post_init__(self):
        if self.num_local_heads == -1:
            self.num_local_heads = self.num_heads
        if self.inter_dim is None:
            hidden_dim = self.mlp_scale * self.dim
            num_hidden = int(2 * hidden_dim / 3)
            self.inter_dim = find_multiple(num_hidden, 256)
        self.head_dim = self.dim // self.num_heads

    @classmethod
    def from_name(cls, name: str):
        # presets = {
        #     "tiny": dict(dim=128, num_heads=4, num_layers=2, vocab_size=10000),
        #     "small": dict(dim=256, num_heads=8, num_layers=4, vocab_size=20000),
        #     "gpt2-small": dict(dim=768, num_heads=12, num_layers=12, vocab_size=50257),
        #     # add more as needed
        # }
        # if name not in presets:
        #     raise ValueError(f"Unknown model config name: {name}")

        # return cls(**presets[name])
        print("Not yet implemented")
        pass


class MLP(nn.Module):
    def __init__(self, config: SpectronConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.inter_dim)
        self.w2 = nn.Linear(config.inter_dim, config.dim)
        self.w2.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class AssociativeAttention(nn.Module):
    """Online Associative Attention module."""

    def __init__(
        self,
        config: SpectronConfig,
        spectral_filters: torch.Tensor,
        spectral_filters_fft: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.register_buffer("spectral_filters", None, persistent=False)
        self.register_buffer("spectral_filters_fft", None, persistent=False)
        self.spectral_filters = spectral_filters
        self.spectral_filters_fft = spectral_filters_fft
        self.use_tensordot = config.use_tensordot

        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.wo.SCALE_INIT = 1

        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_local_heads = config.num_local_heads

        self.wg = nn.Linear(config.head_dim**2, 1)
        self.eps = config.eps

        self.register_parameter("qk_norm_scale", nn.Parameter(torch.empty(1, self.num_heads, 1)))
        self.register_parameter("kv_norm_scale", nn.Parameter(torch.empty(1, self.num_heads, 1, 1, 1)))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim

        bhld = x.view(B, L, H, h).transpose(1, 2)
        if self.use_tensordot:
            approx = self.lora_lo(self.spectral_basis)
            filters = self.lora_hi(approx)
            x_tilde = tensordot_conv(filters, bhld)
        else:
            x_tilde = self.stu_conv(self.spectral_filters, bhld)
            x_tilde = x_tilde.permute(0, 2, 3, 1, 4).reshape(B, L, self.k * H * h)
            x_tilde = self.M_phi(x_tilde)
        
        q = self.wq(x_tilde).view(B, L, H, h).transpose(1, 2)
        k = self.wk(x_tilde).view(B, L, H, h).transpose(1, 2)
        v = self.wv(x_tilde).view(B, L, H, h).transpose(1, 2)
        q, k, v = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        # Inner and outer products (cosine similarity and state covariance matrix)
        sim = torch.einsum("bhlq,bhlk->bhl", q, k) * self.qk_norm_scale
        Z   = torch.einsum("bhsn,bhsp->bhspn", k, v) * self.kv_norm_scale

        gate_inputs = Z.view(B, H, L, -1)
        gates_logits = self.wg(gate_inputs)

        # [B, H, L, 1]
        gates = leaky_relu(gates_logits, squared=True) + self.eps

        # [B, H, L, 1, 1]
        gates = gates.squeeze(-1)

        # Apply gating to Z
        gated_Z = Z * gates.unsqueeze(-1).unsqueeze(-1)

        # Scan over all dims simultaneously
        m_scan, s_scan, n_scan, Z_scan, g_scan = spectron_scan(sim, v, gated_Z, gates)

        # -*- Compute final attention outputs -*-
        linear_attn = n_scan / (s_scan.unsqueeze(-1) + self.eps)

        # Compute online softmax in safe manner
        softmax_weights = torch.exp(sim - m_scan) / (s_scan + self.eps)

        # Compute gated accumulation normalization
        H = Z_scan / (g_scan.unsqueeze(-1).unsqueeze(-1) + self.eps)
        H = torch.einsum("bhtp,bhtpn->bhtn", q, H)

        # Query from the accumulated state history
        ctx  = torch.einsum("bhld,bhlde->bhle", q, H)
        qla  = torch.einsum("bhtp,bhtpn->bhtn", q, linear_attn)
        lerp = ctx + (qla - ctx) * softmax_weights.unsqueeze(-1)

        # Concatenate the heads back together
        y = lerp.transpose(1, 2).reshape(B, L, -1)
        out = self.wo(y)

        return out


class SpectronBlock(nn.Module):
    def __init__(
        self,
        config: SpectronConfig,
        spectral_filters: torch.Tensor,
        spectral_filters_fft: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.aa_norm = nn.LayerNorm(config.dim)
        self.aa = AssociativeAttention(config, spectral_filters, spectral_filters_fft)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.aa(self.aa_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Spectron(nn.Module):
    def __init__(
        self,
        config: SpectronConfig,
        spectral_filters: torch.Tensor,
        spectral_filters_fft: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList()

        for _ in range(config.num_layers):
            self.layers.append(SpectronBlock(config, spectral_filters, spectral_filters_fft))

        self.norm_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if self.config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = self.config.dim**-0.5

    def init_weights(self, module):
        std = self.std
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs) -> CausalLMOutput:
        x = self.tok_emb(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        return_logits = kwargs.pop("return_logits", False)
        if labels is not None or return_logits:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
        else:
            # TODO: Not sure if applicable for Spectron!
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim

        loss = None
        if labels is not None:
            loss = loss_fn(logits.flatten(0, 1), labels.flatten(0, 1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def estimate_mfu(self, obs_toks_per_sec: float, world_size: int) -> float:
        """
        Estimate model FLOPs utilization (MFU) as a fraction of peak theoretical FLOPs used.

        This estimator breaks down the per-token FLOPs into three main components:

        1. Parameter cost: 6 * N, where N = total number of parameters.

        2. ScanAttention cost: For each layer (L) and head (H) with head-dim h:
            - q·k dot product: ~2 * h FLOPs
            - Outer product(s) and gating:
                ~6 * h^2 FLOPs
            Total per layer-head: (2h + 6h^2), hence overall cost is: L * H * (2h + 6h^2).

        3. STU cost (if enabled):
                L * [8 * d * (c_rfft * log2(2T)) + 12 * d],
            where d = model dim, T = sequence length, and c_rfft is an optional constant (default 1).

        The total per-token FLOPs is then:

            flops_per_token = 6 * N + L * H * (2h + 6h^2) + (STU cost, if applicable)

        With total_peak_flops = peak_flops_per_gpu * world_size, the theoretical peak tokens/sec is:

            theoretical_peak_toks_per_sec = total_peak_flops / flops_per_token

        Finally, MFU is given by:

            MFU = obs_toks_per_sec / theoretical_peak_toks_per_sec

        Args:
            obs_toks_per_sec (float): Observed tokens processed per second (across all GPUs).
            world_size (int): Number of GPUs.

        Returns:
            float: Estimated MFU (a fraction between 0 and 1).
        """
        # Get per-GPU peak FLOPs and scale by world_size.
        device_type = get_device_info(return_type="type")
        peak_flops_per_gpu = get_peak_flops(device_type)
        total_peak_flops = peak_flops_per_gpu * world_size

        # Extract configuration values.
        L = self.config.num_layers
        H = self.config.num_heads
        d = self.config.dim
        h = d // H  # Head dimension
        T = self.config.seq_len
        N = self.get_num_params()

        # (i) Parameter multiplication cost.
        param_cost = 6 * N

        # (ii) Cost from ScanAttention operations per token.
        # For each (layer, head): 2*h (sim) + 6*h^2 (outer & gating costs)
        scan_attention_cost = L * H * (2 * h + 6 * (h**2))

        # Baseline FLOPs per token.
        flops_per_token = param_cost + scan_attention_cost

        # (iii) If STU modules are enabled, add FFT-based convolution cost.
        c_rfft = 1  # Constant for FFT-based convolution (defaults to 1)
        stu_cost = L * (8 * d * (c_rfft * math.log2(2 * T)) + 12 * d)
        flops_per_token += stu_cost

        # Compute the theoretical maximum tokens per second.
        theoretical_peak_tokens_per_sec = total_peak_flops / flops_per_token

        # MFU is the ratio of observed tokens/sec to the theoretical peak.
        mfu = obs_toks_per_sec / theoretical_peak_tokens_per_sec

        return mfu

    def get_num_params(self):
        """Return the number of parameters in the model."""
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def configure_optimizers(self, weight_decay, max_lr, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        logger.info(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, capturable=torch.cuda.is_available(), fused=use_fused
        )
        return optimizer
