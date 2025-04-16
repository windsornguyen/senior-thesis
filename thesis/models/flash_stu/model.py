# ===----------------------------------------------------------------------===##
# File: flash_stu/model.py
# Author: Windsor Nguyen '25
# Date: April 14th, 2025
# ===----------------------------------------------------------------------===##

import inspect
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torchtune.modules import RotaryPositionalEmbeddings as RoPE

from fla.modules import FusedCrossEntropyLoss

try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to torch.rfft."
    )
    flash_fft_available = False

try:
    from flash_attn_interface import flash_attn_func
except ImportError as e:
    print(
        f"Unable to import FlashAttention-3: {e}. Attempting to install FlashAttention-2...\n"
    )

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(
        f"Unable to import FlashAttention-2: {e}. No alternative currently available.\n"
    )


# ===----------------------------------------------------------------------===##
#                              Utility Functions
# ===----------------------------------------------------------------------===##

IGNORE_IDX = -100

loss_fn = FusedCrossEntropyLoss(ignore_index=IGNORE_IDX)

def nearest_power_of_two(n: int, round_up: bool = False) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length() if round_up else (n).bit_length() - 1)

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries.reshape(-1, 1) + entries.reshape(1, -1)

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z

def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)

# ===----------------------------------------------------------------------===##
#                              Model Definitions
# ===----------------------------------------------------------------------===##

class BaseConfigForCausalLM(PretrainedConfig):
    """Base PretrainedConfig class to be decorated with dataclass"""

    model_type = "base_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@dataclass
class FlashSTUConfig(BaseConfigForCausalLM):

    model_type = "FlashSTU"
    spectral_filters: torch.Tensor

    bsz: int = 1
    dim: int = 1024
    r: int = 1024
    num_heads: int = 12
    num_local_heads: Optional[int] = -1
    num_layers: int = 12
    seq_len: int = 4096
    n: int = 8191
    window_size: int = 2048
    vocab_size: int = 200064
    inter_dim: Optional[int] = 3072
    mlp_scale: Optional[float] = 12.0
    weight_tying: Optional[bool] = True
    bias: Optional[bool] = False
    rope_theta: Optional[float] = 10000.0
    softcap: Optional[float] = 50.0
    num_eigh: Optional[int] = 24
    use_hankel_L: Optional[bool] = False
    use_flash_fft: Optional[bool] = True
    use_tensordot: Optional[bool] = True
    use_attn: Optional[bool] = True
    use_alibi: Optional[bool] = False
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
    def __init__(self, config: FlashSTUConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.inter_dim)
        self.w2 = nn.Linear(config.inter_dim, config.dim)
        self.w2.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))

class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.wo.SCALE_INIT = 1

        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_local_heads = config.num_local_heads
        self.window_size = config.window_size
        self.softcap = config.softcap

        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if config.use_alibi else None
        self.rotary_emb = RoPE(
            dim=self.head_dim,
            max_seq_len=config.seq_len,
            base=config.rope_theta,
        )

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        
        if self.alibi_slopes is None:
            q, k = self.rotary_emb(q), self.rotary_emb(k)

        y = flash_attn_func(
            q=q.to(torch.bfloat16),
            k=k.to(torch.bfloat16),
            v=v.to(torch.bfloat16),
            causal=True,
            window_size=(self.window_size, 0),
            # softcap=self.softcap,
            alibi_slopes=self.alibi_slopes,
        ).to(torch.float32)

        out = y.reshape(bsz, seq_len, -1)
        out = self.wo(out)

        return out
    
    def _generate_slopes(self, n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, num_heads: int, interpolation_factor: float = 0.25):
        # If n_heads is a power of 2, generate slopes directly
        if math.log2(num_heads).is_integer():
            slopes = self._generate_slopes(num_heads)
        else:
            # Get slopes for the nearest power of two
            n = nearest_power_of_two(num_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)

            # Generate extra slopes
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: num_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=torch.device("cuda"))  # FA ALiBi must be on CUDA
        slopes = slopes * interpolation_factor  # https://arxiv.org/pdf/2310.13017
        return slopes

class STU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.spectral_filters = config.spectral_filters
        self.spectral_filters_fft = None

        self.n = config.n
        self.num_eigh = config.num_eigh
        self.d_in = config.dim
        self.d_out = config.dim
        self.r = config.r
        self.use_hankel_L = config.use_hankel_L
        self.use_tensordot = config.use_tensordot
        self.flash_fft = FlashFFTConv(self.n, dtype=torch.bfloat16) \
            if config.use_flash_fft and flash_fft_available else None
        
        # TODO: Add dimensionality reduction `r` here.
        if self.use_tensordot:
            self.M_inputs = nn.Parameter(torch.zeros(self.d_in, self.d_out))
            self.M_filters = nn.Parameter(torch.zeros(self.num_eigh, self.d_in))
        else:
            self.M_phi_plus = nn.Parameter(torch.zeros(self.num_eigh, self.d_in, self.d_out))
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.zeros(self.num_eigh, self.d_in, self.d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        if self.use_tensordot:
            
            # Contract inputs and filters over (K, D) dims first, then convolve
            x_proj = x @ self.M_inputs
            phi_proj = self.spectral_filters @ self.M_filters
            if self.flash_fft:
                spectral_plus, spectral_minus = self.flash_conv(x_proj, phi_proj, self.flash_fft, self.use_tensordot)
            else:
                spectral_plus, spectral_minus = self.conv(x_proj, phi_proj, self.n, self.use_tensordot)

        else:
            # Convolve inputs and filters first, then contract over (K, D) dims
            if self.flash_fft:
                U_plus, U_minus = self.flash_conv(x, self.spectral_filters, self.flash_fft, self.use_tensordot)
            else:
                U_plus, U_minus = self.conv(x, self.spectral_filters, self.n, self.use_tensordot)

            B, L, K, D = U_plus.shape
            spectral_plus = U_plus.reshape(B, L, K * D) @ self.M_phi_plus.reshape(K * D, self.d_out)
            if not self.use_hankel_L:
                spectral_minus = U_minus.reshape(B, L, K * D) @ self.M_phi_minus.reshape(K * D, self.d_out)

        out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
        return out

    def conv(self, u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs convolution via FFT with causal alignment using a negative featurization.

        The input tensor u is modulated by an alternating sign tensor (sgn) that multiplies every other
        time step by -1. This "negative featurization" modulates the phase so that in this implementation
        the correct causal output is obtained by simply slicing the first L elements (i.e. [:seq_len]).
        Note: Using a conventional slice [seq_len-1:2*seq_len-1] would yield a flipped alignment, resulting in leakage.

        Args:
            u: Input tensor of shape (bsz, seq_len, d_in).
            v: Kernel tensor; expected shape is (seq_len, d_out) if use_tensordot is True.
            n: FFT length (typically set to 2*seq_len - 1 for linear convolution with implicit right zero-padding).
            use_tensordot: Boolean flag to control kernel reshaping.

        Returns:
            A tuple (U_plus, U_minus) where:
            - U_plus is the primary convolution output.
            - U_minus is the secondary output, corrected by the sign tensor.
        """
        bsz, seq_len, d_in = u.shape

        sgn = torch.full((1, seq_len, 1), 1, device=u.device)
        sgn[:, 1::2] *= -1  # Apply negative featurization: multiply every other element by -1.

        if use_tensordot:
            _, d_out = v.shape
            v = v.view(1, -1, d_out, 1).to(torch.float32).contiguous()
        else:
            _, K = v.shape
            sgn = sgn.unsqueeze(-1)
            v = v.view(1, -1, K, 1, 1).to(torch.float32).contiguous()  # (bsz, seq_len, K, d_in, stack)
            u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

        v = torch.fft.rfft(v, n=n, dim=1)

        U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
        U = torch.fft.rfft(U, n=n, dim=1)
        
        # Slicing the first seq_len outputs yields the proper causal convolution given the negative modulation.
        U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
        U_plus, U_minus = torch.unbind(U_conv, dim=-1)
        U_minus = U_minus * sgn

        return U_plus, U_minus

    def flash_conv(
        self, u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_tensordot: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flash FFT convolution.

        Args:
            u (torch.Tensor): Input tensor of shape `(B, L, d_in)`, where:
                - `B` is the batch size,
                - `L` is the sequence length,
                - `d_in` is the input dimension.
            v (torch.Tensor): Filter tensor of shape `(K, d_in)`, where:
                - `K` is the number of filters,
                - `d_in` is the input dimension.
            flash_fft (FlashFFTConv): An instance of the FlashFFTConv module, used to perform the convolution.
            use_tensordot (bool, optional): If `True`, performs the tensordot approximation (default is `True`).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple `(U_plus, U_minus)`:
                - `U_plus`: Convolved output tensor with positive eigenvalues.
                - Shape depends on `use_tensordot`:
                    - If `use_tensordot=True`: `(B, L, d_in)`
                    - If `use_tensordot=False`: `(B, L, K, d_in)`
                - `U_minus`: Convolved output tensor with negative eigenvalues.
                - Shape depends on `use_tensordot`:
                    - If `use_tensordot=True`: `(B, L, d_in)`
                    - If `use_tensordot=False`: `(B, L, K, d_in)`

        Raises:
            ValueError: If the input tensor shapes do not conform to the expected dimensions.

        Example:
            >>> u = torch.randn(4, 16, 32)  # (B, L, d_in)
            >>> v = torch.randn(8, 32)      # (K, d_in)
            >>> flash_fft = FlashFFTConv(n=16, dtype=torch.float32)
            >>> U_plus, U_minus = flash_convolve(u, v, flash_fft, use_tensordot=True)
            >>> print(U_plus.shape, U_minus.shape)
            torch.Size([4, 16, 32]) torch.Size([4, 16, 32])

            """
        bsz, seq_len, d_in = u.shape
        _, K = v.shape

        padded_len = nearest_power_of_two(seq_len, round_up=True)
        pad_len = padded_len - seq_len

        sgn = torch.full((1, 1, padded_len), 1, device=u.device)
        sgn[:, :, 1::2] = -1

        if use_tensordot:
            u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16)
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32)
            u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
        else:
            u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1)
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1)
            u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

        U_conv = flash_fft(u_conv, v_padded)

        # Trim the output back to the original sequence length
        U_conv = U_conv[..., :seq_len]
        u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

        if use_tensordot:
            u_minus = u_minus * sgn[:, :, :seq_len]
            U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
        else:
            sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
            U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
            U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

        return U_plus, U_minus

class SlidingWindowAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.swa_norm = nn.LayerNorm(config.dim)
        self.swa = SlidingWindowAttention(config)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.swa(self.swa_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class STULayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.stu_norm = nn.LayerNorm(config.dim)
        self.stu = STU(config)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.stu(self.stu_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class FlashSTU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList()

        for layer_idx in range(config.num_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config))
            else:
                self.layers.append(SlidingWindowAttentionLayer(config)) \
                    if config.use_attn else self.layers.append(STULayer(config))

        self.norm_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if self.config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight
        
        self.std = self.config.dim ** -0.5

    def _init_weights(self, module):
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
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        loss = None
        if labels is not None:
            loss = loss_fn(logits.flatten(0, 1), labels.flatten(0, 1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def estimate_mfu(self, obs_toks_per_sec: float, world_size: int) -> float:
        """Estimate model FLOPs utilization (MFU) as a fraction of peak theoretical FLOPs used.

        The estimated FLOPs per token are given by the sum of:
        (i)  a parameter-multiplication cost: 6 * N, where N is the number of parameters, and
        (ii)  the attention cost (per Chinchilla): 12 * L * H * Q * T,
            with L = num_layers, H = num_heads, Q = (model dim) / H, T = seq_len,
        and, if STU modules are used,
        (iii) the additional cost from FFT-based convolution:
            L * [8 · d · (c_rfft · log₂(2T)) + 12 · d],
        where d is self.config.dim.

        Thus, if STU modules are used, the FLOPs per token is:

            flops_per_token = 6 * N + 12 * L * H * Q * T + L * [8 * d * (c_rfft * log2(2T)) + 12 * d]

        Given total peak FLOPs (across all GPUs), we compute the theoretical peak tokens/sec
        as: total_peak_flops / flops_per_token.

        Then:

            MFU = obs_toks_per_sec / (total_peak_flops / flops_per_token)

        Args:
            obs_toks_per_sec (float): Observed tokens processed per second (total over all GPUs).
            world_size (int): Number of GPUs.

        Returns:
            float: Estimated MFU (a fraction between 0 and 1, roughly).

        """
        # Get device-specific peak FLOPs per GPU (external helper)
        device_type = get_device_info(return_type="type")
        peak_flops_per_gpu = get_peak_flops(device_type)
        total_peak_flops = peak_flops_per_gpu * world_size

        L = self.config.num_layers
        H = self.config.num_heads
        Q = self.config.dim // H
        T = self.config.seq_len
        N = self.get_num_params()

        # Baseline attention-based cost (from Chinchilla):
        base_flops_per_token = 6 * N + 12 * L * H * Q * T

        # If STU modules are used, add extra FFT-based convolution FLOPs:
        if getattr(self.config, "use_stu", False):
            stu_flops = self._calc_stu_flops()
            flops_per_token = base_flops_per_token + stu_flops
        else:
            flops_per_token = base_flops_per_token

        theoretical_peak_tokens_per_sec = total_peak_flops / flops_per_token
        mfu = obs_toks_per_sec / theoretical_peak_tokens_per_sec

        return mfu

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            logger.info(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, capturable=torch.cuda.is_available(), fused=use_fused)
        return optimizer


# ===----------------------------------------------------------------------===##
#                              Testing
# ===----------------------------------------------------------------------===##

if __name__ == "__main__":
    config_path = "config.json"
    
    with open(config_path, "r") as f:
        config_data = json.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    torch_dtype = getattr(torch, config_data["torch_dtype"])

    configs = FlashSTUConfig(
        bsz=config_data["bsz"],
        dim=config_data["dim"],
        num_heads=config_data["num_heads"],
        num_layers=config_data["num_layers"],
        seq_len=config_data["seq_len"],
        weight_tying=config_data["weight_tying"],
        window_size=config_data["window_size"],
        vocab_size=config_data["vocab_size"],
        mlp_scale=config_data["mlp_scale"],
        bias=config_data["bias"],
        num_eigh=config_data["num_eigh"],
        r=config_data["r"],
        use_hankel_L=config_data["use_hankel_L"],
        use_flash_fft=config_data["use_flash_fft"],
        use_tensordot=config_data["use_tensordot"],
        use_attn=config_data["use_attn"],
        softcap=config_data["softcap"],
        rope_theta=config_data["rope_theta"],
        use_alibi=config_data["use_alibi"],
        dilation=config_data["dilation"],
        torch_dtype=torch_dtype,
    )

    filters = get_spectral_filters(
        seq_len=config_data["seq_len"], 
        K=config_data["num_eigh"],
        use_hankel_L=config_data["use_hankel_L"],
        device=device,
    )

    print("Configs:")
    for key, value in vars(configs).items():
        print(f"  {key}: {value}")

    model = FlashSTU(configs, filters).to(device=device, dtype=torch_dtype)

    x = torch.randint(
        0, configs.vocab_size, 
        (config_data["bsz"], config_data["seq_len"]), 
        dtype=torch.long
    ).to(device)

    outputs = model(x)

    print("Output shape:", outputs.shape)
    print("Sample output:", outputs[0, 0, :10])
