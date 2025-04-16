import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import torchaudio

from typing import Tuple

from tqdm import tqdm
from torchtune.modules import RotaryPositionalEmbeddings as RoPE
from thesis.experiments.synthetics.registry import registry

from thesis.experiments.synthetics.args import args
from thesis.utils.logger import logger
from flash_attn import flash_attn_func
from thesis.experiments.utils.assoc_scan.kernel import associative_scan

# from transformers.modeling_outputs import CausalLMOutput
from fla.modules import FusedCrossEntropyLoss

IGNORE_IDX = -100
loss_fn = FusedCrossEntropyLoss(ignore_index=IGNORE_IDX)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")


def apply_compile(model: nn.Module, **kwargs) -> None:
    """
    Apply torch.compile to each Flash STU block. This makes compilation efficient
    due to repeated structure. Alternatively, one can compile the whole model
    (after applying DP).
    """
    logger.info(f"Compiling each {model.__class__.__name__} block with torch.compile...")
    start = time.perf_counter()
    for layer_id, flash_stu_block in model.layers.named_children():
        flash_stu_block = torch.compile(flash_stu_block, **kwargs)
        model.layers.register_module(layer_id, flash_stu_block)
    end = time.perf_counter()
    logger.info(f"Finished compiling each {model.__class__.__name__} block in {end - start:.4f} seconds.")


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
        self.w2.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh") * self.w3(x))


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.dim, self.num_heads = args.dim, args.num_heads
        assert args.dim % args.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = args.dim // args.num_heads

        self.wq = nn.Linear(args.dim, args.dim)
        self.wk = nn.Linear(args.dim, args.dim)
        self.wv = nn.Linear(args.dim, args.dim)

        self.c_proj = nn.Linear(args.dim, args.dim, bias=args.bias)
        self.c_proj.SCALE_INIT = 1

        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if args.use_alibi else None
        self.window_size = args.window_size
        self.rope = RoPE(self.head_dim, args.seq_len, args.rope_theta)
        self.softcap = args.softcap

        self.dropout = args.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

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

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
    ) -> torch.Tensor:
        if x is not None:
            q = k = v = x
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")

        bsz, q_len, dim = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape

        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # FlashAttention expects bsz, seq_len, num_heads, head_dim
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim)
        v = v.view(bsz, v_len, self.num_heads, self.head_dim)

        # if self.alibi_slopes is None:  # Either RoPE or ALiBi for positional embedding
        q, k = self.rope(q), self.rope(k)

        y = flash_attn_func(  # https://arxiv.org/pdf/2307.08691
            q=q.to(torch.bfloat16),
            k=k.to(torch.bfloat16),
            v=v.to(torch.bfloat16),
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            # window_size=(self.window_size, 0), # Set to seq_len if full attention
            # alibi_slopes=self.alibi_slopes, # https://arxiv.org/pdf/2108.12409
            # NOTE: Softcapping cannot be used simultaneously with dropout
            # softcap=self.softcap,  # https://arxiv.org/pdf/2408.00118
        ).to(torch.float32)

        out = y.contiguous().view(bsz, q_len, -1)
        out = self.resid_dropout(self.c_proj(out))
        return out


class AttentionLayer(nn.Module):
    def __init__(self, args) -> None:
        super(AttentionLayer, self).__init__()
        self.attn_norm = nn.LayerNorm(args.dim)
        self.attn = Attention(args=args)
        self.mlp_norm = nn.LayerNorm(args.dim)
        self.mlp = MLP(args.dim, args.mlp_scale * args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x=self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        assert args.dim % args.num_heads == 0, f"dim ({args.dim}) must be divisible num_heads ({args.num_heads})"
        self.head_dim = args.dim // args.num_heads

        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList()
        for _ in range(args.num_layers):
            self.layers.append(AttentionLayer(args))

        self.out_norm = nn.LayerNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=args.bias)
        self.lm_head.SCALE_INIT = 1

        if args.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = args.dim**-0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x)

        out = self.lm_head(self.out_norm(x))
        return out

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        std = self.std
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * len(self.layers)) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


def get_hankel(seq_len: int, use_hankel_L: bool = False, device=None) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
    i_plus_j = entries.view(-1, 1) + entries.view(1, -1)
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False, device: torch.device = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k = phi_k * sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


def convfft(u: torch.Tensor, v: torch.Tensor, mode: str = "full", causal: bool = False) -> torch.Tensor:
    """Perform generic convolution using FFT. Supports various modes and filter shapes.

    Args:
        u: Input tensor of shape (B, L, d).
        v: Filter tensor of shape (F, d) or (F, d, k) for k filters.
        mode: Convolution mode ('full', 'same', 'valid').
        causal: Whether to apply causal convolution (default: False).

    Returns:
        Convolved tensor of shape depending on mode:
            - 'full': (B, L + F - 1, d[, k])
            - 'same': (B, L, d[, k])
            - 'valid': (B, L - F + 1, d[, k])

    """
    assert mode in {"full", "same", "valid"}, f"Invalid mode '{mode}'"
    B, L, d = u.shape
    dtype = u.dtype  # Store original dtype

    # Ensure v has shape (F, d, k)
    if v.ndim == 2:
        F_len, d_v = v.shape
        assert d == d_v, "Filter and input dimensions must match."
        v = v.unsqueeze(-1)  # shape (F, d, 1)
    elif v.ndim == 3:
        F_len, d_v, _ = v.shape
        assert d == d_v, "Filter and input dimensions must match."
    else:
        raise ValueError("Filter tensor must be either (F, d) or (F, d, k)")

    conv_len = L + F_len - 1
    fft_len = nearest_power_of_two(conv_len, round_up=True)

    # Convert to float32 for FFT computation
    u_float = u.to(torch.float32)
    v_float = v.to(torch.float32)

    # Pad tensors
    u_padded = F.pad(u_float, (0, 0, 0, fft_len - L))
    v_padded = F.pad(v_float, (0, 0, 0, 0, 0, fft_len - F_len))

    # Perform FFT operations in float32
    U_fft = torch.fft.rfft(u_padded, n=fft_len, dim=1)
    V_fft = torch.fft.rfft(v_padded, n=fft_len, dim=0)

    U_fft = U_fft.unsqueeze(-1)
    V_fft = V_fft.unsqueeze(0).expand(B, -1, -1, -1)

    conv_result = torch.fft.irfft(U_fft * V_fft, n=fft_len, dim=1)

    if causal:
        start_idx = F_len - 1
    else:
        start_idx = 0

    if mode == "full":
        end_idx = start_idx + conv_len
    elif mode == "same":
        end_idx = start_idx + L
    elif mode == "valid":
        end_idx = start_idx + L - F_len + 1

    result = conv_result[:, start_idx:end_idx]

    if result.shape[-1] == 1:
        result = result.squeeze(-1)

    # Convert back to original dtype
    return result.to(dtype=dtype)


def bld_convfft(
    u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs FFT-based convolution with causal alignment using a negative featurization.

    The input tensor u is modulated by an alternating sign tensor (sgn) that multiplies every other
    time step by -1. This "negative featurization" modulates the phase so that in this implementation
    the correct causal output is obtained by simply slicing the first seq_len elements (i.e. [:seq_len]).
    Note: Using a conventional slice [seq_len-1:2*seq_len-1] would yield a flipped alignment, resulting in leakage.

    Args:
        u: Input tensor of shape (B, seq_len, d_in).
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


class SimpleSpectralAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        num_heads: int = 4,
        use_hankel_L: bool = False,
        use_tensordot: bool = True,
        r: int = 64,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_tensordot = use_tensordot
        self.use_hankel_L = use_hankel_L
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
        self.K = k  # num_eigh
        self.r = r
        self.eps = 1e-8

        # STU filters
        self.stu_filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        if self.use_tensordot:
            self.tensordot_proj = nn.Linear(d_model, d_model)

        # STU projection matrices
        if use_tensordot:
            self.M_inputs = nn.Parameter(torch.empty(d_model, r, dtype=dtype, device=device))
            self.M_filters = nn.Parameter(torch.empty(k, r, dtype=dtype, device=device))
            self.out_proj_stu = nn.Linear(r, d_model, bias=True, device=device, dtype=dtype)
        else:
            self.M_phi_plus = nn.Parameter(torch.empty(k, d_model, d_model, dtype=dtype, device=device))
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.empty(k, d_model, d_model, dtype=dtype, device=device))

        self.qk_norm_scale = nn.Parameter(torch.ones((1, self.num_heads, 1))) / torch.sqrt(torch.tensor(float(self.head_dim)))
        self.kv_norm_scale = nn.Parameter(torch.ones((1, 1, 1, self.head_dim, self.head_dim)))

        # Attention components
        self.wq = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        # Gate for combining attention and spectral features
        self.wg = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.wg_z = nn.Linear(self.head_dim ** 2, 1, device=device, dtype=dtype)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def compute_stu_features(self, u: torch.Tensor) -> torch.Tensor:
        """Compute STU features"""
        B, T, d = u.shape

        if self.use_tensordot:
            # Project first
            u_proj = u @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            # p_coeffs_conv = p_coeffs_conv @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            phi_proj = self.stu_filters @ self.M_filters  # (L, K) x (K, r) -> (L, r)

            # Then, convolve: (B, L, r) ⊗ (L, r) -> (B, L, r)
            spectral_plus, spectral_minus = bld_convfft(u_proj, phi_proj, self.n, self.use_tensordot)

            # Final output
            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            # out = self.out_proj_stu(out + p_coeffs_conv)
        else:
            # Convolve first to get featurized inputs: (B, L, d_in) x (L, K) -> (B, L, K, d_in)
            U_plus, U_minus = bld_convfft(u, self.stu_filters, self.n, self.use_tensordot)

            # Compute sum-product of featurized inputs and M matrices over the K filters
            B, L, K, d_in = U_plus.shape

            # Spectral output: (B, L, K * d_in) x (K * d_in, d_out) -> (B, L, d_out)
            spectral_plus = U_plus.view(B, L, K * d_in) @ self.M_phi_plus.view(K * d_in, self.d_model)

            if not self.use_hankel_L:
                spectral_minus = U_minus.view(B, L, K * d_in) @ self.M_phi_minus.view(K * d_in, self.d_model)

            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

        return out

    def td_convfft(self, f: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Perform causal 1D (tensordot approx.) convolution via FFT for multi-headed inputs.

        Args:
            f: Spectral filters of shape [L, D].
            u: Input sequences of shape [B, H, L, h].

        Returns:
            Convolved sequences of shape [B, H, L, h].
        """
        reshaped = lambda x: x.reshape(u.shape[2], u.shape[1], u.shape[3]).transpose(1, 0, 2)
        tr_conv = lambda x, y: torchaudio.functional.fftconvolve(x, y)[: x.shape[0]]
        cconv = torch.vmap(tr_conv, in_dims=(0, 0), out_dims=0)
        hconv = lambda u1, f1: cconv(u1.permute(2, 0, 1), f1.permute(2, 0, 1)).T
        hmap = torch.vmap(hconv, in_dims=(0, 0), out_dims=0)
        bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)
        return bmap(u, reshaped(f))

    def full_convfft(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute FFT-based convolution for multi-headed inputs.

        Args:
            v: Spectral filters of shape [L, K].
            u: Inputs of shape [B, H, L, h].

        Returns:
            Convolution output of shape [B, H, L, K, h].
        """
        L = u.shape[2]
        tr_conv = lambda filter_k, channel_h: torchaudio.functional.fftconvolve(channel_h, filter_k)[:L]
        conv_filter_with_channels = torch.vmap(tr_conv, in_dims=(None, 1), out_dims=1)
        conv_all_filters_channels = torch.vmap(conv_filter_with_channels, in_dims=(1, None), out_dims=1)
        conv_one_head = lambda u1, v_filters: conv_all_filters_channels(v_filters, u1)
        conv_heads = torch.vmap(conv_one_head, in_dims=(0, None), out_dims=0)
        conv_batch = torch.vmap(conv_heads, in_dims=(0, None), out_dims=0)
        return conv_batch(u, v)

    def forward(self, x: torch.Tensor, chunk_len: int = 128) -> torch.Tensor:
        # B, T, d = x.shape

        # # Branch 1: Compute STU features
        # x_tilde = self.compute_stu_features(x)

        # # Branch 2: Compute multihead linear attention
        # q = self.wq(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d_head)
        # k = self.wk(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d_head)
        # v = self.wv(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d_head)
        # # k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        # # if self.use_tensordot:
        # #     filters = self.tensordot_proj(self.stu_filters)
        # #     k = self.td_convfft(filters, k)
        # #     v = self.td_convfft(filters, v)
        # # else:
        # #     k = self.full_convfft(self.stu_filters, k)
        # #     v = self.full_convfft(self.stu_filters, v)

        # # Z = (
        # #     torch.einsum("bhlkd,bhlke->bhlde", v, k)
        # #     if not self.use_tensordot
        # #     else torch.einsum("bhld,bhle->bhlde", v, k)
        # # ) * self.kv_norm_scale

        # # Linear attention computation in spectral basis
        # Z = torch.einsum("bhtp,bhtn->bhtpn", v, k)  # (B, H, T, d_head, d_head)

        # # gate_input_z = Z.reshape(*Z.shape[:3], -1)
        # # gates_logits_z = self.wg_z(gate_input_z)
        # # gates_z = F.relu(gates_logits_z) ** 2 + self.eps
        # # gates_z = gates_z[..., None]

        # # gated_Z = gates_z * Z

        # H = torch.cumsum(Z, dim=2)

        # Y = torch.einsum("bhtp,bhtpn->bhtn", q, H)  # (B, H, T, d_head)

        # # Compute gate values
        # gate_values = torch.sigmoid(self.wg(x))  # (B, T, d)
        
        # # Merge heads
        # y_attn = Y.permute(0, 2, 1, 3).contiguous().view(B, T, d)  # (B, T, d)
        
        # # Combine branches using element-wise gating
        # # -> x_tilde * (1 - gate_values) + y_attn * gate_values
        # y_combined = x_tilde + (y_attn - x_tilde) * gate_values

        # # Final projection and normalization
        # return self.wo(y_combined)
        B, L, D = x.shape

        # Branch 1: Compute STU features
        x_tilde = self.compute_stu_features(x)  # (B, T, d_model)

        # Branch 2: Compute multihead linear attention
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_head)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_head)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_head)

        # Linear attention computation
        Z = torch.einsum("bhtp,bhtn->bhtpn", v, k)  # (B, H, T, d_head, d_head)
        H = torch.cumsum(Z, dim=2)
        Y = torch.einsum("bhtp,bhtpn->bhtn", q, H)  # (B, H, T, d_head)

        # Merge heads
        y_attn = Y.transpose(1, 2).reshape(B, L, D)  # (B, L, D)

        # Compute gate values
        gate_values = torch.sigmoid(self.wg(x))  # (B, L, D)

        # Combine branches using element-wise gating
        # -> x_tilde * (1 - gate_values) + y_attn * gate_values
        y_combined = x_tilde + (y_attn - x_tilde) * gate_values

        # Final projection and normalization
        return self.wo(y_combined)


class SpectralAttention(nn.Module):
    """Associative attention mechanism with spectral filtering and gating."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        spectral_basis: torch.Tensor,
        use_tensordot: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.spectral_basis = spectral_basis
        self.use_tensordot = use_tensordot
        self.eps = eps

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.wo.SCALE_INIT = 1

        if self.use_tensordot:
            self.tensordot_proj = nn.Linear(dim, dim)
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)

        # Adjust wg_z input dimension to h*h
        self.wg_z = nn.Linear(self.head_dim * self.head_dim, 1, bias=True)
        self.gate_proj = nn.Linear(dim, dim, bias=True)

        kv_initial_value = torch.ones((1, 1, 1, self.head_dim, self.head_dim))
        qk_initial_value = torch.ones((1, num_heads, 1)) / torch.sqrt(torch.tensor(float(self.head_dim)))

        self.kv_norm_scale = nn.Parameter(kv_initial_value)
        self.qk_norm_scale = nn.Parameter(qk_initial_value)

    def td_convfft(self, f: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Perform causal 1D (tensordot approx.) convolution via FFT for multi-headed inputs.

        Args:
            f: Spectral filters of shape [L, D].
            u: Input sequences of shape [B, H, L, h].

        Returns:
            Convolved sequences of shape [B, H, L, h].
        """
        reshaped = lambda x: x.reshape(u.shape[2], u.shape[1], u.shape[3]).transpose(1, 0, 2)
        tr_conv = lambda x, y: torchaudio.functional.fftconvolve(x, y)[: x.shape[0]]
        cconv = torch.vmap(tr_conv, in_dims=(0, 0), out_dims=0)
        hconv = lambda u1, f1: cconv(u1.permute(2, 0, 1), f1.permute(2, 0, 1)).T
        hmap = torch.vmap(hconv, in_dims=(0, 0), out_dims=0)
        bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)
        return bmap(u, reshaped(f))

    def bhld_convfft(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute FFT-based convolution for multi-headed inputs.

        Args:
            v: Spectral filters of shape [L, K].
            u: Inputs of shape [B, H, L, h].

        Returns:
            Convolution output of shape [B, H, L, K, h].
        """
        L = u.shape[2]
        tr_conv = lambda filter_k, channel_h: torchaudio.functional.fftconvolve(channel_h, filter_k)[:L]
        conv_filter_with_channels = torch.vmap(tr_conv, in_dims=(None, 1), out_dims=1)
        conv_all_filters_channels = torch.vmap(conv_filter_with_channels, in_dims=(1, None), out_dims=1)
        conv_one_head = lambda u1, v_filters: conv_all_filters_channels(v_filters, u1)
        conv_heads = torch.vmap(conv_one_head, in_dims=(0, None), out_dims=0)
        conv_batch = torch.vmap(conv_heads, in_dims=(0, None), out_dims=0)
        return conv_batch(u, v)

    def combine_fn(self, x: Tuple, y: Tuple) -> Tuple:
        """Combine function for scan: (m, s, n, Z, g)."""
        m_x, s_x, n_x, Z_x, g_x = x
        m_y, s_y, n_y, Z_y, g_y = y

        m_new = torch.maximum(m_x, m_y)
        exp_x = torch.exp(m_x - m_new)
        exp_y = torch.exp(m_y - m_new)
        s_new = s_x * exp_x + s_y * exp_y

        # n_x is always [L, h], exp_x is [L]. Broadcast exp_x to [L, 1].
        n_new = n_x * exp_x[..., None] + n_y * exp_y[..., None]

        Z_new = Z_x + Z_y
        g_new = g_x + g_y
        return m_new, s_new, n_new, Z_new, g_new

    def scan_fn(
        self, qk_slice: torch.Tensor, v_norm_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor
    ):
        """Process a single slice for scanning using v_norm for n state."""
        # Initial leaves: (scores, initial_s, original_v_norm, Z_slice, g_slice)
        leaves = (qk_slice, torch.ones_like(qk_slice), v_norm_slice, Z_slice, g_slice)
        return associative_scan(combine_fn=self.combine_fn, xs=leaves, dim=0, combine_mode="generic")

    def batched_scan_fn(self, sim: torch.Tensor, v_norm: torch.Tensor, gated_Z: torch.Tensor, gates: torch.Tensor):
        """Applies the full scan across batch and head dimensions."""
        # v_norm has shape [B, H, L, h]
        # Apply vmap over batch (0) and head (1) dimensions
        return torch.vmap(
            torch.vmap(self.scan_fn, in_dims=(0, 0, 0, 0), out_dims=0), in_dims=(0, 0, 0, 0), out_dims=0
        )(sim, v_norm, gated_Z, gates)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim

        q = self.wq(x).reshape(B, L, H, h).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, H, h).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, H, h).permute(0, 2, 1, 3)

        sim = torch.einsum("bhld,bhld->bhl", q, k) * self.qk_norm_scale

        k_norm = F.normalize(k, p=2.0, dim=-1, eps=self.eps)
        v_norm = F.normalize(v, p=2.0, dim=-1, eps=self.eps)

        if self.use_tensordot:
            filters = self.tensordot_proj(self.spectral_basis)
            k_filtered = self.tensordot_conv(filters, k_norm)
            v_filtered = self.tensordot_conv(filters, v_norm)
        else:
            k_filtered = self.full_conv(self.spectral_basis, k_norm)
            v_filtered = self.full_conv(self.spectral_basis, v_norm)

        Z = (
            torch.einsum("bhlkd,bhlke->bhlde", v_filtered, k_filtered)
            if not self.use_tensordot
            else torch.einsum("bhld,bhle->bhlde", v_filtered, k_filtered)
        ) * self.kv_norm_scale

        gate_input_z = Z.reshape(*Z.shape[:3], -1)
        gates_logits_z = self.wg_z(gate_input_z)
        gates_z = F.relu(gates_logits_z) ** 2 + self.eps
        gates_z = gates_z[..., None]

        gated_Z = gates_z * Z

        m_scan, s_scan, n_scan, Z_scan, g_scan = self.batched_scan_fn(sim, v_norm, gated_Z, gates_z)

        # Ensure eps is a tensor for torch.maximum
        eps_tensor = torch.tensor(self.eps, device=g_scan.device, dtype=g_scan.dtype)
        state_matrix = Z_scan / torch.maximum(g_scan, eps_tensor)
        state_path = torch.einsum("bhld, bhlde -> bhle", q, state_matrix)

        state_path = F.normalize(state_path, dim=-1, eps=self.eps)
        state_path = state_path.permute(0, 2, 1, 3).reshape(B, L, D)
        output = self.wo(state_path)

        return output


class SpectralAttentionLayer(nn.Module):
    """
    A single layer that applies SpectralAttention, followed by an MLP,
    each of which is added (residual) to the input, then normalized.

    Args:
        seq_len (int): Sequence length (T).
        d_model (int): Model dimension.
        k (int): Projection dimension for the spectral filters.
        use_hankel_L (bool): Whether to use a Hankel matrix.
        device: Torch device.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        num_heads: int = 4,
        use_hankel_L: bool = False,
        use_tensordot: bool = True,
        r: int = 64,
        device=None,
    ):
        super().__init__()
        self.spectral_attention = SimpleSpectralAttention(
            seq_len, d_model, k, num_heads, use_hankel_L, use_tensordot=use_tensordot, r=r, device=device
        )
        # spectral_basis = get_spectral_filters(seq_len, k, use_hankel_L, device=torch.device("cuda"))
        # self.spectral_attention = SpectralAttention(
        #     dim=d_model,
        #     num_heads=num_heads,
        #     seq_len=seq_len,
        #     spectral_basis=spectral_basis,
        #     use_tensordot=use_tensordot,
        # )
        self.spec_attn_norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpectralAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        x = x + self.spectral_attention(self.spec_attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Spectron(nn.Module):
    """
    A stacked spectral-transformer-like model. Uses an embedding, a dropout,
    multiple SpectralAttention layers in sequence, and an output projection.

    Args:
        seq_len (int): Sequence length.
        d_model (int): Model dimension.
        k (int): Projection dimension for the spectral filters.
        vocab_size (int): Vocabulary size.
        num_layers (int): Number of SpectralAttention layers.
        dropout (float): Dropout probability.
        use_hankel_L (bool): Whether to use a Hankel matrix in the attention.
        device: Torch device.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        num_heads: int,
        vocab_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_hankel_L: bool = False,
        use_tensordot: bool = True,
        r: int = 64,
        device=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.k = k
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.use_hankel_L = use_hankel_L
        self.use_tensordot = use_tensordot
        self.r = r
        self.device = device

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                SpectralAttentionLayer(
                    seq_len=seq_len,
                    d_model=d_model,
                    k=k,
                    num_heads=num_heads,
                    use_hankel_L=use_hankel_L,
                    use_tensordot=use_tensordot,
                    r=r,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional: Weight tying
        # self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.std = 1 / math.sqrt(d_model)
        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the Spectron model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, S).

        Returns:
            torch.Tensor: Logits tensor of shape (B, S, vocab_size).
        """
        _b, s = input_ids.shape
        h = self.tok_emb(input_ids)  # (B, S, D)

        for layer in self.layers:
            h = layer(h)  # Each layer expects/returns (B, S, D)

        h = self.norm(h)  # (B, S, D)
        logits = self.lm_head(h)  # (B, S, vocab_size)

        return logits

    def _init_weights(self, m):
        std = self.std
        if isinstance(m, nn.Linear):
            if hasattr(m, "SCALE_INIT"):
                std *= (2 * len(self.layers)) ** -0.5
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)


# -----------------------------------------------------------------------------
# Training and Evaluation Functions
# -----------------------------------------------------------------------------
def compute_metrics(model, loader, criterion, device=device):
    """
    Compute average loss and token-level accuracy for a given dataloader.
    Ignores target_ignore_idx (-100) in calculations.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            # Calculate loss for the batch
            # Note: Ensure criterion ignores the index correctly
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.detach().item()

            # Calculate accuracy for the batch
            predictions = logits.argmax(dim=-1)
            valid_mask = targets != IGNORE_IDX  # Use the global IGNORE_IDX
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().detach().item()
            total_tokens += valid_mask.sum().detach().item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
    model.train()  # Set model back to training mode
    return avg_loss, accuracy


def compute_recall_accuracy(model, loader, device=device):
    """
    Compute accuracy specifically on positions where we test key-value recall.
    These are positions where targets != -100 and we're testing if the model
    correctly recalls the value associated with a previously seen key.
    """
    model.eval()
    correct_recalls = 0
    total_recalls = 0
    printed_batch = False  # Flag to print only the first batch
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)

            # Find positions where we're testing recall (targets != -100)
            recall_positions = targets != -100

            # Calculate accuracy only on recall positions
            match = (predictions == targets) & recall_positions
            correct_recalls += match.sum().detach().item()
            total_recalls += recall_positions.sum().detach().item()

            # --- Debug Print for the first batch ---
            if i == 0 and not printed_batch:
                print("\n--- Debug Recall Eval (First Batch Example) ---")
                idx_to_print = 0
                inp_print = inputs[idx_to_print].cpu().tolist()
                tgt_print = targets[idx_to_print].cpu().tolist()
                pred_print = predictions[idx_to_print].cpu().tolist()

                recall_idx = -1
                for k, t_val in enumerate(tgt_print):
                    if t_val != -100:
                        recall_idx = k
                        break

                print(f"Input:     {inp_print}")
                print(f"Target:    {tgt_print}")
                print(f"Predicted: {pred_print}")
                if recall_idx != -1:
                    print(
                        f"Recall Step @ index {recall_idx}: Target={tgt_print[recall_idx]}, Predicted={pred_print[recall_idx]}"
                    )
                else:
                    print("No recall target found in this example.")
                print("-----------------------------------------------")
                printed_batch = True  # Ensure we only print once per eval call

    recall_acc = 100.0 * correct_recalls / (total_recalls if total_recalls > 0 else 1)
    print(f"Key-Value Recall Accuracy: {recall_acc:.2f}% (Correct: {correct_recalls}, Total: {total_recalls})")
    return recall_acc


def train_model(model, train_loader, val_loader, max_steps: int = 10000, eval_interval: int = 50):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    model.train()  # Ensure model is in training mode

    # History trackers
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    eval_steps = []

    step = 0
    # Use a single progress bar for steps
    desc = f"Training {model.__class__.__name__}"
    pbar = tqdm(total=max_steps, desc=desc)

    # --- Variables for accumulating metrics between evaluations ---
    current_train_loss = 0.0
    current_correct_train_tokens = 0
    current_total_train_tokens = 0
    batches_since_last_eval = 0
    # Initialize validation metrics for postfix display
    last_val_loss = float("inf")
    last_val_acc = 0.0
    # ---

    while step < max_steps:
        # No explicit epoch loop needed if max_steps is the limit
        for inputs, targets in train_loader:  # targets here are inputs[1:] for training
            if step >= max_steps:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)  # These are the next-token targets

            # --- Training Step ---
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # --- End Training Step ---

            # --- Accumulate Training Metrics for Interval ---
            current_train_loss += loss.detach().item()
            with torch.no_grad():  # Accuracy calculation doesn't need gradients
                predictions = logits.argmax(dim=-1)
                valid_mask = targets != IGNORE_IDX
                match = (predictions == targets) & valid_mask
                current_correct_train_tokens += match.sum().detach().item()
                current_total_train_tokens += valid_mask.sum().detach().item()
            batches_since_last_eval += 1
            # ---

            step += 1
            pbar.update(1)

            # --- Update Postfix Every Step ---
            running_avg_train_loss = (
                current_train_loss / batches_since_last_eval if batches_since_last_eval > 0 else float("nan")
            )
            running_avg_train_acc = (
                100.0 * current_correct_train_tokens / current_total_train_tokens
                if current_total_train_tokens > 0
                else float("nan")
            )
            pbar.set_postfix(
                step=step,
                trn_loss=f"{running_avg_train_loss:.3f}",
                trn_acc=f"{running_avg_train_acc:.2f}%",
                val_loss=f"{last_val_loss:.3f}",  # Show last known val metrics
                val_acc=f"{last_val_acc:.2f}%",
                refresh=False,  # Avoid potential flickering, update handled by interval
            )
            # ---

            # --- Periodic Evaluation ---
            if step % eval_interval == 0 or step == max_steps:
                # Calculate final average training metrics for the interval
                avg_train_loss = current_train_loss / batches_since_last_eval if batches_since_last_eval > 0 else 0.0
                avg_train_acc = (
                    100.0 * current_correct_train_tokens / current_total_train_tokens
                    if current_total_train_tokens > 0
                    else 0.0
                )

                # Calculate validation metrics
                val_loss, val_acc = compute_metrics(model, val_loader, loss_fn, device)
                last_val_loss = val_loss  # Update last known val metrics
                last_val_acc = val_acc

                # Store metrics for history plots
                train_loss_history.append(avg_train_loss)
                train_acc_history.append(avg_train_acc)
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                eval_steps.append(step)

                # Update progress bar postfix with final interval metrics
                pbar.set_postfix(
                    step=step,
                    trn_loss=f"{avg_train_loss:.3f}",
                    trn_acc=f"{avg_train_acc:.2f}%",
                    val_loss=f"{val_loss:.3f}",
                    val_acc=f"{val_acc:.2f}%",
                    refresh=True,  # Force refresh now
                )

                # Reset accumulators for the next interval
                current_train_loss = 0.0
                current_correct_train_tokens = 0
                current_total_train_tokens = 0
                batches_since_last_eval = 0
            # --- End Periodic Evaluation ---

    pbar.close()
    # Return all histories
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history, eval_steps


def show_example_predictions(model, model_name: str, dataset):
    model.eval()
    with torch.no_grad():
        first_input, first_target = dataset[0]
        last_input, last_target = dataset[-1]
        first_input = first_input.unsqueeze(0).to(device)
        last_input = last_input.unsqueeze(0).to(device)

        first_logits = model(first_input)
        last_logits = model(last_input)

        first_pred = first_logits.argmax(dim=-1).squeeze(0).cpu()
        last_pred = last_logits.argmax(dim=-1).squeeze(0).cpu()

        print(f"\n{model_name} - First Example")
        print("Input:     ", dataset[0][0].cpu().tolist())
        print("Target:    ", first_target.cpu().tolist())
        print("Predicted: ", first_pred.tolist())
        print(f"\n{model_name} - Last Example")
        print("Input:     ", dataset[-1][0].cpu().tolist())
        print("Target:    ", last_target.cpu().tolist())
        print("Predicted: ", last_pred.tolist())


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(1746)
    torch.cuda.manual_seed_all(1746)

    # Dataset parameters
    vocab_size = args.vocab_size
    seq_len = args.seq_len
    num_train = 12800
    num_test = num_train // 10

    # Create data loaders using the registry
    print("Creating data loaders via registry...")
    loader, val_loader = registry.create_data_loaders(
        task_name="mqar",
        batch_size=args.bsz,
        num_train=num_train,
        num_test=num_test,
        backend="torch",  # Ensure torch backend
        device=device,
        in_memory=True,  # Use in-memory dataset
        # Task-specific arguments for single-query ICR:
        vocab_size=vocab_size,
        seq_len=seq_len,
        # multi_query=True,  # Explicitly set to single-query
        # Add other potential kwargs if needed, e.g., noise params if applicable
    )
    print("Data loaders created.")

    models_to_train = []
    if args.model == "all" or args.model == "transformer":
        trans_model = Transformer(args).to(device=device, dtype=getattr(torch, args.dtype))
        models_to_train.append(("Transformer", trans_model))

    if args.model == "all" or args.model == "spectron":
        spectron = Spectron(
            seq_len=seq_len,
            d_model=128,
            k=24,
            num_heads=1,
            vocab_size=vocab_size,
            num_layers=2,
            dropout=0.0,
            use_hankel_L=False,
            use_tensordot=False,
            r=128,
            device=device,
        ).to(device)
        models_to_train.append(("Spectron", spectron))

    # Train models and collect results
    results = {}
    for model_name, model in models_to_train:
        print(f"\nTraining {model_name}...")
        apply_compile(model)
        tr_loss, tr_acc, v_loss, v_acc, steps = train_model(
            model, loader, val_loader, max_steps=args.steps, eval_interval=args.eval
        )
        # Store results only if needed, otherwise just run training
        # results[model_name] = (loss_history, acc_history, eval_steps)

        # Show example predictions after training
        print(f"\n--- Example Predictions for {model_name} ---")
        show_example_predictions(model, model_name, loader.dataset)

    # Example adaptation if you uncomment plotting
    if len(models_to_train) > 0:  # Check if models were trained
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)  # 2 rows: Loss, Accuracy

        # Assuming only one model for simplicity, adapt if multiple
        model_name = models_to_train[0][0]

        # Plot Loss
        axes[0].plot(steps, tr_loss, label=f"{model_name} Train Loss", color="tab:blue", alpha=0.8)
        axes[0].plot(steps, v_loss, label=f"{model_name} Val Loss", color="tab:orange", linestyle="--")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Training & Validation Loss ({model_name})")
        axes[0].legend()
        axes[0].grid(True)

        # Plot Accuracy
        axes[1].plot(steps, tr_acc, label=f"{model_name} Train Acc", color="tab:blue", alpha=0.8)
        axes[1].plot(steps, v_acc, label=f"{model_name} Val Acc", color="tab:orange", linestyle="--")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title(f"Training & Validation Accuracy ({model_name})")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
