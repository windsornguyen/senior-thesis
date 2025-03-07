import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from thesis.experiments.synthetics.mqar import generate_mqar
from thesis.models.flash_stu.model import FlashSTUModel
from thesis.models.mamba.model import MambaModel


# -----------------------------------------------------------------------------
# Helper: Build Causal Mask (for transformer models)
# -----------------------------------------------------------------------------
def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


# -----------------------------------------------------------------------------
# Custom Transformer Components (unchanged)
# -----------------------------------------------------------------------------
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        B, T, _ = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        output = self.out_proj(attn_output)
        return output, attn_weights


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dim_feedforward: int = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.self_attn = CustomMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        attn_out, _ = self.self_attn(src, src, src, mask=mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CustomTransformerEncoderLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            src = layer(src, mask=mask)
        return src


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def get_monic_chebyshev_coeffs(n: int) -> torch.Tensor:
    def chebyshev_t_int(n: int) -> list[int]:
        if n == 0:
            return [1]
        elif n == 1:
            return [1, 0]
        T0 = [1]
        T1 = [1, 0]
        for _ in range(2, n + 1):
            T2 = [2 * c for c in T1] + [0]
            d = len(T2) - len(T0)
            padded_T0 = [0] * d + T0
            T2 = [a - b for a, b in zip(T2, padded_T0)]
            T0, T1 = T1, T2
        return T2

    coeffs = torch.tensor(chebyshev_t_int(n), dtype=torch.complex128)
    if n > 0:
        coeffs = coeffs / (2.0 ** (n - 1))
    return coeffs


def get_hankel(seq_len: int, use_hankel_L: bool = False, device=None) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False, device: torch.device = None, dtype=torch.float32
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L, device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k = phi_k * sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


class LearnableSpectralFilters(nn.Module):
    def __init__(self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


# class SpectralAttention(nn.Module):
#     """
#     SpectralAttention implements a spectral-based attention mechanism.
#     It projects inputs into a (B, T, seq_len) space, applies learned spectral filters
#     to obtain queries (Q) and keys (K), and applies a linear projection to obtain values (V).
#     The attention output is produced via a sequence of einsum operations:
#       - Outer products between V and K are scaled by a decay parameter.
#       - A lower-triangular Hankel mask aggregates the results.
#       - Finally, the query (Q) is applied to the aggregated tensor.
#     The output is then projected back to the original model dimension.

#     Args:
#         seq_len (int): Sequence length (T).
#         d_model (int): Input model dimension.
#         k (int): Projection dimension for the spectral filters and linear layer.
#         use_hankel_L (bool): Whether to use a Hankel matrix in computing attention.
#         device: Torch device.
#     """

#     def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
#         super().__init__()
#         self.seq_len = seq_len
#         dtype = torch.bfloat16

#         # Optionally project input from d_model -> seq_len if needed.
#         self.pre_proj = (
#             nn.Linear(d_model, seq_len, device=device, dtype=dtype)
#             if d_model != seq_len
#             else nn.Identity()
#         )

#         # Q and K are learned spectral filters.
#         self.Q = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype=dtype)
#         self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype=dtype)
#         # V is a linear layer.
#         self.v_proj = nn.Linear(seq_len, k, device=device, dtype=dtype)

#         # Final projection from k back to d_model.
#         self.o_proj = nn.Linear(k, d_model, device=device, dtype=dtype)

#         # Decay parameter: one per time step.
#         self.decay = nn.Parameter(torch.ones(seq_len, device=device, dtype=dtype))

#         # Precompute and store the lower-triangular Hankel matrix L (shape [T, T]).
#         L = get_hankel(seq_len, use_hankel_L, device=device).to(dtype)
#         self.L = nn.Parameter(torch.tril(L))

#     def forward(self, x: torch.Tensor, chunk_len: int = 128) -> torch.Tensor:
#         """
#         Forward pass of SpectralAttention.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T, d_model).
#             chunk_len (int): Chunk size for potential block-sum approaches (unused here).

#         Returns:
#             torch.Tensor: Output tensor of shape (B, T, d_model).
#         """
#         B, T, _ = x.shape

#         # Pre-project input to match internal sequence length.
#         x_proj = self.pre_proj(x)  # Shape: (B, T, seq_len)

#         # Compute Q and K via learned spectral filters.
#         Q_out = torch.einsum("bti,ik->btk", x_proj, self.Q())
#         K_out = torch.einsum("bti,ik->btk", x_proj, self.K())

#         # Compute V via linear projection.
#         V_out = self.v_proj(x_proj)  # Shape: (B, T, k)

#         # Compute outer product between V and K, scaling by decay.
#         # V_out: (B, T, k), K_out: (B, T, k), decay: (T,)
#         Z = torch.einsum("btp,btn,t->btpn", V_out, K_out, self.decay)

#         # Expand Hankel matrix L for the batch.
#         L_batched = self.L.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, T, T)

#         # Aggregate the outer products using the Hankel mask.
#         H = torch.einsum("bts,bspn->btpn", L_batched, Z)

#         # Apply the query to the aggregated tensor.
#         Y = torch.einsum("btk,btkn->btn", Q_out, H)

#         # Final projection to output the model dimension.
#         return self.o_proj(Y)


# class SpectralAttentionLayer(nn.Module):
#     """
#     A single layer that applies SpectralAttention, followed by an MLP,
#     each of which is added (residual) to the input, then normalized.

#     Args:
#         seq_len (int): Sequence length (T).
#         d_model (int): Model dimension.
#         k (int): Projection dimension for the spectral filters.
#         use_hankel_L (bool): Whether to use a Hankel matrix.
#         device: Torch device.
#     """

#     def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
#         super().__init__()
#         self.spectral_attention = SpectralAttention(seq_len, d_model, k, use_hankel_L, device)
#         self.mlp = MLP(d_model, 4 * d_model)
#         self.spec_attn_norm = nn.RMSNorm(d_model)
#         self.mlp_norm = nn.RMSNorm(d_model)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of SpectralAttentionLayer.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T, d_model).

#         Returns:
#             torch.Tensor: Output tensor of shape (B, T, d_model).
#         """
#         x = x + self.spectral_attention(self.spec_attn_norm(x))
#         x = x + self.mlp(self.mlp_norm(x))
#         return x


# class SpectronMQAR(nn.Module):
#     """
#     A stacked spectral-transformer-like model. Uses an embedding, a dropout,
#     multiple SpectralAttention layers in sequence, and an output projection.

#     Args:
#         seq_len (int): Sequence length.
#         d_model (int): Model dimension.
#         k (int): Projection dimension for the spectral filters.
#         vocab_size (int): Vocabulary size.
#         d_out (int): Output dimension (defaults to vocab_size).
#         num_layers (int): Number of SpectralAttention layers.
#         dropout (float): Dropout probability.
#         use_hankel_L (bool): Whether to use a Hankel matrix in the attention.
#         device: Torch device.
#     """

#     def __init__(
#         self,
#         seq_len: int,
#         d_model: int,
#         k: int,
#         vocab_size: int,
#         d_out: int | None = None,
#         num_layers: int = 1,
#         dropout: float = 0.1,
#         use_hankel_L: bool = False,
#         device=None,
#     ):
#         super().__init__()

#         if d_out is None:
#             d_out = vocab_size

#         # Embedding and dropout
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.in_dropout = nn.Dropout(dropout)
#         self.out_dropout = nn.Dropout(dropout)

#         # Stack of SpectralAttention layers
#         self.layers = nn.ModuleList(
#             [SpectralAttentionLayer(seq_len, d_model, k, use_hankel_L, device=device) for _ in range(num_layers)]
#         )

#         self.norm = nn.LayerNorm(d_model)
#         self.out_proj = nn.Linear(d_model, d_out)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the Spectron model.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T) containing token indices.

#         Returns:
#             torch.Tensor: Output logits of shape (B, T, d_out).
#         """
#         # Embed and apply dropout
#         x_emb = self.in_dropout(self.embedding(x))

#         # Pass through stacked layers
#         out = x_emb
#         for layer in self.layers:
#             out = layer(out)

#         # Normalize and project
#         out = self.norm(out)
#         out = self.out_proj(out)
#         return self.out_dropout(out)


def get_opt_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7 / 6) * math.log2(seq_len)))


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


def fft_conv(u: torch.Tensor, v: torch.Tensor, mode: str = "full", causal: bool = False) -> torch.Tensor:
    """
    Perform generic convolution using FFT. Supports various modes and filter shapes.

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

    # Pad u along its length dimension (last dimension remains d)
    u_padded = F.pad(u, (0, 0, 0, fft_len - L)).to(torch.float32)  # (B, fft_len, d)
    # Pad v along its first dimension (filter length) using a 6-tuple.
    v_padded = F.pad(v, (0, 0, 0, 0, 0, fft_len - F_len)).to(torch.float32)  # (fft_len, d, k)

    U_fft = torch.fft.rfft(u_padded, n=fft_len, dim=1)  # (B, fft_len//2+1, d)
    V_fft = torch.fft.rfft(v_padded, n=fft_len, dim=0)  # (fft_len//2+1, d, k)

    U_fft = U_fft.unsqueeze(-1)  # (B, fft_len//2+1, d, 1)
    V_fft = V_fft.unsqueeze(0).expand(B, -1, -1, -1)  # (B, fft_len//2+1, d, k)

    conv_result = torch.fft.irfft(U_fft * V_fft, n=fft_len, dim=1)  # (B, fft_len, d, k)

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

    return result.to(dtype=u.dtype)


def stu_conv(
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


class SpectralAttention(nn.Module):
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
        self.d_out = d_model  # Same for now
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_tensordot = use_tensordot
        self.use_hankel_L = use_hankel_L
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
        self.K = k  # num_eigh
        self.r = r

        # STU filters
        self.stu_filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)

        # Chebyshev coefficients
        self.p_coeffs_kernel = (
            torch.tensor(get_monic_chebyshev_coeffs(get_opt_degree(seq_len)), device=device)
            .view(-1, 1)
            .repeat(1, d_model)
        )
        # (n, d_model)

        # STU projection matrices
        if use_tensordot:
            self.M_inputs = nn.Parameter(torch.empty(d_model, r, dtype=dtype, device=device))
            self.M_filters = nn.Parameter(torch.empty(k, r, dtype=dtype, device=device))
            self.out_proj_stu = nn.Linear(r, d_model, bias=True, device=device, dtype=dtype)
        else:
            self.M_phi_plus = nn.Parameter(torch.empty(k, d_model, d_model, dtype=dtype, device=device))
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.empty(k, d_model, d_model, dtype=dtype, device=device))

        # Attention components
        self.Q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.V = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        # Gate for combining attention and spectral features
        self.gate = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        # Normalization
        self.norm = nn.RMSNorm(d_model)

    def compute_stu_features(self, u: torch.Tensor) -> torch.Tensor:
        """Compute STU features"""
        B, T, d = u.shape

        # Convolve inputs w/ Chebyshev coefficients, per https://arxiv.org/pdf/2502.06545
        p_coeffs_conv = -fft_conv(u, self.p_coeffs_kernel, mode="same", causal=True)

        if self.use_tensordot:
            # Project first
            u_proj = u @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            p_coeffs_conv = p_coeffs_conv @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            phi_proj = self.stu_filters @ self.M_filters  # (L, K) x (K, r) -> (L, r)

            # Then, convolve: (B, L, r) ⊗ (L, r) -> (B, L, r)
            spectral_plus, spectral_minus = stu_conv(u_proj, phi_proj, self.n, self.use_tensordot)

            # Final output
            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            out = self.out_proj_stu(out + p_coeffs_conv)
        else:
            # Convolve first to get featurized inputs: (B, L, d_in) x (L, K) -> (B, L, K, d_in)
            U_plus, U_minus = stu_conv(u, self.stu_filters, self.n, self.use_tensordot)

            # Compute sum-product of featurized inputs and M matrices over the K filters
            B, L, K, d_in = U_plus.shape

            # Spectral output: (B, L, K * d_in) x (K * d_in, d_out) -> (B, L, d_out)
            spectral_plus = U_plus.view(B, L, K * d_in) @ self.M_phi_plus.view(K * d_in, self.d_out)

            if not self.use_hankel_L:
                spectral_minus = U_minus.view(B, L, K * d_in) @ self.M_phi_minus.view(K * d_in, self.d_model)

            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            out = out + p_coeffs_conv

        return out

    def forward(self, x: torch.Tensor, chunk_len: int = 128) -> torch.Tensor:
        B, T, d = x.shape

        # Branch 1: Compute STU features
        x_tilde = self.compute_stu_features(x)  # (B, T, d_model)

        # Branch 2: Compute multihead linear attention
        Q = self.Q(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d_head)
        K = self.K(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d_head)
        V = self.V(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d_head)

        # Apply non-linearities
        Q = F.gelu(Q, approximate="tanh")
        K = F.gelu(K, approximate="tanh")
        V = F.gelu(V, approximate="tanh")

        # Linear attention computation
        Z = torch.einsum("bhtp,bhtn->bhtpn", V, K)  # (B, H, T, d_head, d_head)
        H = torch.cumsum(Z, dim=2)  # (B, H, T, d_head, d_head)
        Y = torch.einsum("bhtp,bhtpn->bhtn", Q, H)  # (B, H, T, d_head)

        # Merge heads
        Y_attn = Y.permute(0, 2, 1, 3).contiguous().view(B, T, d)  # (B, T, d)

        # Compute gate values
        gate_values = torch.sigmoid(self.gate(x))  # (B, T, d)

        # Combine branches using element-wise gating
        Y_combined = gate_values * Y_attn + (1 - gate_values) * x_tilde

        # Final projection and normalization
        return self.norm(self.o_proj(Y_combined))


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
        self.spectral_attention = SpectralAttention(
            seq_len, d_model, k, num_heads, use_hankel_L, use_tensordot=use_tensordot, r=r, device=device
        )
        self.spectral_attention_norm = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpectralAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        x = x + self.spectral_attention(self.spectral_attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return self.norm(x)


class SpectronMQAR(nn.Module):
    """
    A stacked spectral-transformer-like model. Uses an embedding, a dropout,
    multiple SpectralAttention layers in sequence, and an output projection.

    Args:
        seq_len (int): Sequence length.
        d_model (int): Model dimension.
        k (int): Projection dimension for the spectral filters.
        vocab_size (int): Vocabulary size.
        d_out (int): Output dimension (defaults to vocab_size).
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
        d_out: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_hankel_L: bool = False,
        use_tensordot: bool = True,
        r: int = 64,
        device=None,
    ):
        super().__init__()

        if d_out is None:
            d_out = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional embeddings
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                SpectralAttentionLayer(seq_len, d_model, k, num_heads, use_hankel_L, use_tensordot, r, device=device)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all linear layers using Xavier uniform."""

        def _init_module(module):
            if isinstance(module, nn.Linear):
                # Initialize weights with Xavier uniform
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Recursively initialize weights of all submodules
            for submodule in module.children():
                _init_module(submodule)

        _init_module(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Spectron model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T) containing token indices.

        Returns:
            torch.Tensor: Output logits of shape (B, T, d_out).
        """
        B, T = x.shape

        # Add positional embeddings explicitly
        x_emb = self.embedding(x) + self.pe.unsqueeze(0)
        x_emb = self.in_dropout(x_emb)

        out = x_emb
        for layer in self.layers:
            out = layer(out)

        out = self.norm(out)
        out = self.out_proj(out)
        return self.out_dropout(out)


class FlashSTUMQAR(nn.Module):
    """Wrapper for FlashSTU to match the interface of other models."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_hankel_L: bool = False,
        use_tensordot: bool = True,
        device=None,
    ):
        super().__init__()
        from thesis.models.flash_stu.model import FlashSTUConfig, FlashSTU

        # Get spectral filters directly in bfloat16
        k = math.ceil(math.log(seq_len))
        phi = get_spectral_filters(
            seq_len=seq_len,
            K=k,
            use_hankel_L=use_hankel_L,
            device=device,
            dtype=torch.bfloat16,
        )

        # Create config with bfloat16
        config = FlashSTUConfig(
            bsz=1,
            dim=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=seq_len,
            window_size=seq_len // 8,
            vocab_size=vocab_size,
            mlp_scale=4,
            dropout=0.0,
            num_eigh=k,
            use_hankel_L=use_hankel_L,
            use_tensordot=use_tensordot,
            use_flash_fft=False,
            use_attn=True,
            torch_dtype=torch.bfloat16,
        )

        # Create model directly in bfloat16
        self.model = FlashSTU(config, phi).to(device=device, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep input as LongTensor, model will handle conversion after embedding
        return self.model(x)  # Output stays in bfloat16


class MambaMQAR(nn.Module):
    """Wrapper for Mamba to match the interface of other models."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        device=None,
    ):
        super().__init__()
        from thesis.models.mamba.model import MambaConfig, Mamba

        # Create config
        config = MambaConfig(
            dim=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            ssm_chunk_size=seq_len // 4,
            weight_tying=False,
            use_mem_eff_path=True,
            torch_dtype=torch.bfloat16,
        )

        # Create model
        self.model = Mamba(config).to(device=device)
        self.model.init_weights(buffer_device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# -----------------------------------------------------------------------------
# MQAR Task Models
# -----------------------------------------------------------------------------
class TransformerMQAR(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        """
        The transformer-based model for MQAR.
        The input embedding is over a vocabulary of size (vocab_size + 3) to account for special tokens,
        but the output prediction is over the base vocab (of size vocab_size).
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Regular token embeddings
        self.embedding = nn.Embedding(vocab_size + 3, d_model)

        # Create sinusoidal positional embeddings
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Dropout layers
        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)

        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Create and register causal mask buffer
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Add positional embeddings and apply dropout
        x_embed = self.in_dropout(self.embedding(x) + self.pe.unsqueeze(0))

        # Use the model's causal mask if no mask is provided
        mask = mask if mask is not None else self.causal_mask

        # Pass through encoder
        enc_out = self.encoder(x_embed, mask=mask)

        # Normalize and project
        out = self.norm(enc_out)
        logits = self.out_proj(out)
        return self.out_dropout(logits)


# -----------------------------------------------------------------------------
# Token-Level Accuracy Helper
# -----------------------------------------------------------------------------
def compute_token_level_accuracy(model, loader, attn_mask=None, device=None):
    """
    Compute token-level accuracy while ignoring special tokens and target_ignore_idx.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerMQAR) else model(inputs)
            predictions = logits.argmax(dim=-1)

            # Create mask for valid tokens (not target_ignore_idx)
            valid_mask = targets != -100

            # Calculate accuracy only on valid tokens
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().item()
            total_tokens += valid_mask.sum().item()

    token_acc = 100.0 * correct_tokens / (total_tokens if total_tokens > 0 else 1)
    print(f"Token-Level Accuracy (ignoring padding): {token_acc:.2f}%")
    return token_acc


# -----------------------------------------------------------------------------
# Training Loop (step-based)
# -----------------------------------------------------------------------------
def train_model(model, loader, val_loader, attn_mask=None, max_steps: int = 10000, eval_interval: int = 50):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    loss_history = []
    accuracy_history = []
    eval_steps = []
    step = 0
    epoch = 0
    latest_acc = 0.0

    desc = f"Training {model.__class__.__name__}"
    pbar = tqdm(total=max_steps, desc=desc)

    while step < max_steps:
        epoch += 1
        for inputs, targets in loader:
            if step >= max_steps:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerMQAR) else model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss = loss.item()
            loss_history.append(total_loss)
            step += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{total_loss:.4f}", acc=f"{latest_acc:.2f}%")
            pbar.update(1)

            if step % eval_interval == 0:
                acc = compute_token_level_accuracy(model, val_loader, attn_mask=attn_mask)
                accuracy_history.append(acc)
                eval_steps.append(step)
                latest_acc = acc
                pbar.set_postfix(loss=f"{total_loss:.4f}", acc=f"{acc:.2f}%")

    pbar.close()
    return loss_history, accuracy_history, eval_steps


# -----------------------------------------------------------------------------
# Device Setup & Dataset Creation
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")

batch_size = 64
seq_len = 64
d_model = 64
num_examples = 100000
vocab_size = 8192
num_pairs = 4
num_layers = 2
num_heads = 4
dropout = 0.0
k = 16
alpha = 0.01


# Create causal mask
causal_mask = build_causal_mask(seq_len).to(device)

# Create training dataset
train_dataset = generate_mqar(
    num_examples=num_examples,
    sequence_len=seq_len,
    vocab_size=vocab_size,
    num_pairs=num_pairs,
    alpha=alpha,
    seed=1746,
)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create validation dataset
val_dataset = generate_mqar(
    num_examples=num_examples // 20,
    sequence_len=seq_len,
    vocab_size=vocab_size,
    num_pairs=num_pairs,
    alpha=alpha,
    seed=1747,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Train Models
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on MQAR task")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["transformer", "spectron", "flashstu", "mamba", "all"],
        help="Model to train (transformer, spectron, flashstu, mamba, or all)",
    )
    parser.add_argument("--max-steps", type=int, default=160000, help="Maximum number of training steps")
    parser.add_argument("--eval-interval", type=int, default=250, help="Steps between evaluations")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    # Dataset parameters
    vocab_size = 16
    seq_len = 128
    num_examples = 10000

    # Generate training dataset
    train_dataset = generate_mqar(
        num_examples=num_examples,
        vocab_size=vocab_size,
        seq_len=seq_len,
        is_training=True,
        device=device,
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Generate validation dataset
    val_dataset = generate_mqar(
        num_examples=num_examples // 20,
        vocab_size=vocab_size,
        seq_len=seq_len,
        is_training=False,
        device=device,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    models_to_train = []
    if args.model == "all" or args.model == "transformer":
        trans_model = TransformerMQAR(
            seq_len=seq_len,
            d_model=64,
            vocab_size=vocab_size,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
        ).to(device)
        models_to_train.append(("Transformer", trans_model))

    if args.model == "all" or args.model == "spectron":
        spectron = SpectronMQAR(
            seq_len=seq_len,
            d_model=64,
            k=math.ceil(math.log(seq_len)),
            num_heads=8,
            vocab_size=vocab_size,
            d_out=vocab_size,
            num_layers=2,
            dropout=0.1,
            use_hankel_L=False,
            use_tensordot=False,
            r=64,
            device=device,
        ).to(device)
        models_to_train.append(("Spectron", spectron))

    if args.model == "all" or args.model == "flashstu":
        flash_stu = FlashSTUModel(
            seq_len=seq_len,
            d_model=64,
            vocab_size=vocab_size,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            use_hankel_L=False,
            use_tensordot=False,
            device=device,
        ).to(device)
        models_to_train.append(("FlashSTU", flash_stu))

    if args.model == "all" or args.model == "mamba":
        mamba = MambaModel(
            seq_len=seq_len,
            d_model=64,
            vocab_size=vocab_size,
            num_layers=2,
            device=device,
        ).to(device)
        models_to_train.append(("Mamba", mamba))

    # Train models and collect results
    results = {}
    for model_name, model in models_to_train:
        print(f"\nTraining {model_name}...")
        loss_history, acc_history, eval_steps = train_model(
            model, loader, val_loader, max_steps=args.max_steps, eval_interval=args.eval_interval
        )
        results[model_name] = (loss_history, acc_history, eval_steps)

    # Only plot if we trained more than one model
    if len(models_to_train) > 1:
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = {"Transformer": "blue", "Spectron": "green", "FlashSTU": "purple", "Mamba": "orange"}
        markers = {"Transformer": "o", "Spectron": "s", "FlashSTU": "d", "Mamba": "x"}

        # Plot training loss
        for model_name, (loss_history, _, _) in results.items():
            ax1.plot(loss_history, label=model_name, color=colors[model_name], alpha=0.7)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot validation accuracy
        for model_name, (_, acc_history, eval_steps) in results.items():
            ax2.plot(eval_steps, acc_history, label=model_name, color=colors[model_name], marker=markers[model_name])
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Token-Level Accuracy (%)")
        ax2.set_title("Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        # Add final accuracy text box
        final_acc_text = "Final Accuracy:\n" + "\n".join(
            f"{name}: {results[name][1][-1]:.2f}%" for name in results.keys()
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax2.text(1.05, 0.95, final_acc_text, transform=ax2.transAxes, fontsize=10, verticalalignment="top", bbox=props)

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Example Predictions
# -----------------------------------------------------------------------------
def show_example_predictions(model, model_name: str, attn_mask=None):
    model.eval()
    with torch.no_grad():
        first_input, first_target = train_dataset[0]
        last_input, last_target = train_dataset[-1]
        first_input = first_input.unsqueeze(0).to(device)
        last_input = last_input.unsqueeze(0).to(device)

        if isinstance(model, TransformerMQAR):
            first_logits = model(first_input, mask=attn_mask)
            last_logits = model(last_input, mask=attn_mask)
        else:
            first_logits = model(first_input)
            last_logits = model(last_input)

        first_pred = first_logits.argmax(dim=-1).squeeze(0).cpu()
        last_pred = last_logits.argmax(dim=-1).squeeze(0).cpu()

        print(f"\n{model_name} - First Example")
        print("Input:     ", first_input.squeeze().cpu().tolist())
        print("Target:    ", first_target.cpu().tolist())
        print("Predicted: ", first_pred.tolist())
        print(f"\n{model_name} - Last Example")
        print("Input:     ", last_input.squeeze().cpu().tolist())
        print("Target:    ", last_target.cpu().tolist())
        print("Predicted: ", last_pred.tolist())
