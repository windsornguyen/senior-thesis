import math

import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from flashfftconv import FlashFFTConv
except ImportError as e:
    print(f"Unable to import FlashFFTConv: {e}")
    FlashFFTConv = None

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torchtune.modules import RotaryPositionalEmbeddings as RoPE
# from flash_attn.layers.rotary import RotaryEmbedding as RoPE

from thesis.experiments.synthetics.assoc_recall import generate_assoc_recall
from thesis.experiments.synthetics.args import args

# from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from thesis.experiments.utils.assoc_scan import associative_scan


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

        self.w2.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.num_heads
        self.head_dim = args.dim // args.num_heads
        assert args.dim % args.num_heads == 0, "dim must be divisible by num_heads"
        self.rope = RoPE(self.head_dim, args.seq_len, args.rope_theta)

        self.wq = nn.Linear(args.dim, args.dim, bias=args.bias)
        self.wk = nn.Linear(args.dim, args.dim, bias=args.bias)
        self.wv = nn.Linear(args.dim, args.dim, bias=args.bias)

        self.c_proj = nn.Linear(args.dim, args.dim, bias=args.bias)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q).transpose(1, 2), self.rope(k).transpose(1, 2)

        y = sdpa(q, k, v, is_causal=True)

        out = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.c_proj(out)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm = nn.LayerNorm(args.dim)
        self.attn = Attention(args)
        self.mlp_norm = nn.LayerNorm(args.dim)
        self.mlp = MLP(args.dim, args.mlp_scale * args.dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.num_layers = args.num_layers

        self.layers = nn.ModuleList([AttentionLayer(args) for _ in range(self.num_layers)])

        self.out_norm = nn.LayerNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=args.bias)

        self.std = args.dim**-0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))
        print(self.eval())

    def forward(self, x):
        x = self.tok_emb(x)

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.out_norm(x))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params


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
            T2 = [a - b for a, b in zip(T2, padded_T0, strict=True)]
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


def poly_mul_x(poly):
    # Multiply polynomial by x: shift coefficients right by one index.
    return [0] + poly


def poly_scale(poly, factor):
    # Scale polynomial coefficients by factor.
    return [coef * factor for coef in poly]


def poly_sub(poly1, poly2):
    # Subtract poly2 from poly1; extend with zeros if necessary.
    length = max(len(poly1), len(poly2))
    result = []
    for i in range(length):
        coef1 = poly1[i] if i < len(poly1) else 0
        coef2 = poly2[i] if i < len(poly2) else 0
        result.append(coef1 - coef2)
    return result


def chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x)
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    T_nm2 = [1]  # T_0(x)
    T_nm1 = [0, 1]  # T_1(x)
    for _ in range(2, n + 1):
        # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        term = poly_mul_x(T_nm1)
        term = poly_scale(term, 2)
        T_n = poly_sub(term, T_nm2)
        T_nm2, T_nm1 = T_nm1, T_n
    return T_n


def normalized_chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x) normalized by 2**(n-1).
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    coeff = chebyshev_coeff(n)
    leading_term = coeff[-1]
    return [c / leading_term for c in coeff]


def integrate_polar_monomial(a, b, beta):
    """
    Compute the integral of z^a * z̄^b over the polar wedge:
      r ∈ [0, 1], θ ∈ [-beta, beta],
    in closed form:
      if a==b: 2*beta/(a+b+2)
      else:   2*sin((a-b)*beta)/((a-b)*(a+b+2))
    Here a and b are tensors (floats).
    """
    diff = a - b
    denom = a + b + 2
    return torch.where(
        condition=diff == 0,
        input=2 * beta / denom,
        other=2 * torch.sin(diff * beta) / (diff * denom),
    )


def get_polynomial_hankel(n, beta, t, chunk_size=2048, device="cuda", dtype=torch.float32):
    """ """
    matrix_size = t - n

    # Compute Chebyshev coefficients
    poly_coeff = normalized_chebyshev_coeff(n)
    poly = torch.tensor(poly_coeff, device=device)  # (n+1,)

    # Outer product of polynomial coefficients
    P = torch.outer(poly, poly).unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

    # Precompute the index arrays for the summation indices (ii, jj)
    ii = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    jj = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    ii, jj = torch.meshgrid(ii, jj, indexing="ij")  # (n+1, n+1)
    ii = ii.unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)
    jj = jj.unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

    # Allocate the result matrix
    Z = torch.empty((matrix_size, matrix_size), dtype=torch.complex64, device=device)

    # Process in chunks to save memory.
    for i_start in range(0, matrix_size, chunk_size):
        # Create i indices
        i_end = min(i_start + chunk_size, matrix_size)
        i_vals = torch.arange(i_start, i_end, device=device, dtype=torch.float32).view(-1, 1, 1, 1)

        for j_start in range(0, matrix_size, chunk_size):
            # Create j indices
            j_end = min(j_start + chunk_size, matrix_size)
            j_vals = torch.arange(j_start, j_end, device=device, dtype=torch.float32).view(1, -1, 1, 1)

            # Compute A and B for the chunk: shape (chunk_i, chunk_j, n+1, n+1)
            A = i_vals + ii  # Compute i + ii for each chunk element
            B = j_vals + jj

            # Compute the closed-form integral for each (i+ii, j+jj)
            int_vals = integrate_polar_monomial(A, B, beta)

            # Multiply by P and sum over the polynomial indices to yield the (i,j) entry
            chunk_Z = torch.sum(int_vals * P, dim=(2, 3))

            # Write the computed chunk to the result matrix.
            Z[i_start:i_end, j_start:j_end] = chunk_Z.to(torch.complex64)

    return Z


def get_opt_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7 / 6) * math.log2(seq_len)))


def get_polynomial_spectral_filters(
    seq_len: int,
    k: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    n = get_opt_degree(seq_len)
    beta = 1.0 / (64.0 * n**2)
    Z = get_polynomial_hankel(n, beta, seq_len + n, device=device)
    _, phi = torch.linalg.eigh(Z, UPLO="U")
    phi_k = phi[:, -k:] / math.sqrt(seq_len)

    # Validate that the eigenvectors are real since Z is Hermitian
    if torch.abs(phi_k.imag).max() > 1e-7:
        raise ValueError("Unexpectedly large imaginary components in eigenvectors")

    # Take real part only (imaginary part is due to floating point imprecision)
    return phi_k.real.to(dtype)


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


def flash_stu_conv(
    u: torch.Tensor,
    v: torch.Tensor,
    flash_fft: FlashFFTConv,
    use_tensordot: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flash FFT convolution.

    Args:
        u (torch.Tensor): Input tensor of shape `(B, L, D)`, where:
            - `B` is the batch size,
            - `L` is the sequence length,
            - `D` is the input dimension.
        v (torch.Tensor): Filter tensor of shape `(K, D)`, where:
            - `K` is the number of filters,
            - `D` is the input dimension.
        flash_fft (FlashFFTConv): An instance of the FlashFFTConv module, used to perform the convolution.
        use_tensordot (bool, optional): If `True`, performs the tensordot approximation (default is `True`).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple `(U_plus, U_minus)`:
            - `U_plus`: Convolved output tensor with positive eigenvalues.
            - Shape depends on `use_tensordot`:
                - If `use_tensordot=True`: `(B, L, D)`
                - If `use_tensordot=False`: `(B, L, K, D)`
            - `U_minus`: Convolved output tensor with negative eigenvalues.
            - Shape depends on `use_tensordot`:
                - If `use_tensordot=True`: `(B, L, D)`
                - If `use_tensordot=False`: `(B, L, K, D)`

    Raises:
        ValueError: If the input tensor shapes do not conform to the expected dimensions.

    Example:
        >>> u = torch.randn(4, 16, 32)  # (B, L, D)
        >>> v = torch.randn(8, 32)      # (K, D)
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
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
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


class HeadNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        # x: (B, H, T, d_head)
        mean = x.mean(dim=2, keepdim=True)  # (B, H, 1, d_head)
        var = x.var(dim=2, keepdim=True, unbiased=False)  # (B, H, 1, d_head)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


class MatrixHeadNorm(nn.Module):
    def __init__(self, d_head, eps=1e-6):
        super().__init__()
        self.eps = eps
        # learnable scale and bias parameters, shaped so that they broadcast
        # over (B, H, T) dimensions
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, d_head, d_head))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, d_head, d_head))

    def forward(self, x):
        """
        x: (B, H, T, d_head, d_head)
        Normalizes each d_head x d_head matrix to have zero mean and unit variance,
        where mean and variance are computed over the last two dimensions.
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


def conv(filters: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis using Torch.

    Args:
      filters: torch.Tensor of shape [seq_len, num_heads]
        Each column is the spectral filter for a head.
      keys: torch.Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        Input sequences for each head and feature.

    Returns:
      torch.Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        The result of convolving each head's filter with the corresponding input sequences.
    """

    # 1. Basic 1D convolution that truncates the output to the input length
    def conv1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.fftconvolve(x, y)[: x.shape[0]]

    # 2. Vectorize over feature dimension (head_dim)
    conv_over_features = torch.vmap(conv1d, in_dims=(None, 1), out_dims=1)

    # 3. For one head: convolve filter with key sequence across features
    def conv_head(filter_seq: torch.Tensor, key_seq: torch.Tensor) -> torch.Tensor:
        return conv_over_features(filter_seq, key_seq)

    # 4. Vectorize over heads
    conv_over_heads = torch.vmap(conv_head, in_dims=(0, 0), out_dims=0)

    # 5. Transpose filters to [num_heads, seq_len]
    filters_T = filters.T
    assert (
        keys.shape[1] == filters_T.shape[0]
    ), f"num_heads mismatch: keys {keys.shape[1]} vs filters {filters_T.shape[0]}"
    assert (
        keys.shape[2] == filters_T.shape[1]
    ), f"seq_len mismatch: keys {keys.shape[2]} vs filters {filters_T.shape[1]}"

    # 6. Vectorize over batch dimension
    conv_over_batch = torch.vmap(lambda keys_batch: conv_over_heads(filters_T, keys_batch), in_dims=0, out_dims=0)

    return conv_over_batch(keys)

def ffn_srelu(x, w1, w2):
    """https://x.com/d_haziza/status/1905310121674055906"""
    y = x @ w1  # fp8 matmul
    
    # Single fused kernel for srelu + sparsification + fp8 rowwise quantization
    y = F.relu(y) ** 2
    y = sparsify24(y)
    y = to_fp8_rowwise(y)

    # would be a rowwise fp8 2:4 GEMM call
    return y @ w2

# Full MHA variant
# class SpectralAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dim = config.dim
#         self.num_heads = config.num_heads
#         self.seq_len = config.seq_len
#         self.head_dim = config.dim // config.num_heads
#         assert self.head_dim % 2 == 0, "Head dimension must be divisible by 2 for RoPE."
#         assert config.num_eigh % config.num_heads == 0, "Number of heads must divide k."

#         filters = get_spectral_filters(config.seq_len, config.num_eigh, config.use_hankel_L, config.device, config.dtype)
#         filters = filters.view(config.seq_len, config.num_heads, config.num_eigh // config.num_heads)
#         self.register_buffer("spectral_basis", filters)

#         if config.use_tensordot:
#             self.M_inputs = nn.Parameter(torch.empty(config.dim, config.r, dtype=config.dtype, device=config.device))
#             self.M_filters = nn.Parameter(torch.empty(config.num_eigh, config.r, dtype=config.dtype, device=config.device))
#             self.out_proj_stu = nn.Linear(config.r, config.dim, bias=True, device=config.device, dtype=config.dtype)
#         else:
#             self.M_phi_plus = nn.Parameter(torch.empty(config.num_eigh, config.dim, config.dim, dtype=config.dtype, device=config.device))
#             if not config.use_hankel_L:
#                 self.M_phi_minus = nn.Parameter(torch.empty(config.num_eigh, config.dim, config.dim, dtype=config.dtype, device=config.device))

#         # Rotary positional embeddings
#         self.rope = RoPE(self.head_dim, self.seq_len, config.rope_theta)

#         self.wq = nn.Linear(config.dim, config.dim)
#         self.wk = nn.Linear(config.dim, config.dim)
#         self.wv = nn.Linear(config.dim, config.dim)
#         self.wo = nn.Linear(config.dim, config.dim)

#         self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len)))

#     def conv(self, filters: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
#         def conv1d(x, y):
#             return torch.fft.irfft(torch.fft.rfft(x, n=2*x.size(-1)) * torch.fft.rfft(y, n=2*y.size(-1)))[:x.size(-1)]

#         conv_features = torch.vmap(conv1d, in_dims=(1, None), out_dims=1)
#         conv_heads = torch.vmap(conv_features, in_dims=(None, 1), out_dims=-1)

#         return conv_heads(filters, keys)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, _ = x.shape
#         assert seq_len == self.seq_len, f"Input seq_len {seq_len} must match initialized seq_len {self.seq_len}"

#         # [B, L, H, D]
#         q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim) / math.sqrt(self.head_dim)

#         # [B, H, L, D]
#         v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # [B, H, L, D]
#         q = self.rope(q).transpose(1, 2)
#         k = self.rope(k).transpose(1, 2)

#         # [B, H, L, D]
#         # k_conv = conv(self.spectral_basis, k.transpose(1, 2))
#         # v_conv = conv(self.spectral_basis)

#         scores = torch.matmul(q, k.transpose(-2, -1))
#         scores = scores.masked_fill(self.mask == 0, float('-inf'))

#         attn_weights = F.softmax(scores, dim=-1)
#         context = torch.matmul(attn_weights, v)

#         context = context.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
#         out = self.wo(context)
#         return out


class SpectralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.seq_len = config.seq_len
        self.head_dim = config.dim // config.num_heads

        # Get spectral filters of shape [seq_len, config.num_eigh].
        filters = get_spectral_filters(
            config.seq_len, config.num_heads, config.use_hankel_L, config.device, config.dtype
        )
        self.register_buffer("spectral_basis", filters)

        # Legacy API variables (not used in this design but kept for compatibility)
        if config.use_tensordot:
            self.M_inputs = nn.Parameter(torch.empty(config.dim, config.r, dtype=config.dtype, device=config.device))
            self.M_filters = nn.Parameter(
                torch.empty(config.num_eigh, config.r, dtype=config.dtype, device=config.device)
            )
            self.out_proj_stu = nn.Linear(config.r, config.dim, bias=True, device=config.device, dtype=config.dtype)
        else:
            self.M_phi_plus = nn.Parameter(
                torch.empty(config.num_eigh, config.dim, config.dim, dtype=config.dtype, device=config.device)
            )
            if not config.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.empty(config.num_eigh, config.dim, config.dim, dtype=config.dtype, device=config.device)
                )

        # Standard Q, K, V projections.
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        # Final output projection: merge heads back to config.dim.
        self.wo = nn.Linear(config.dim, config.dim)
        # Gate projection as in the legacy design.
        self.wg = nn.Linear(self.head_dim**2, 1)
        self.eps = 1e-5

    def conv(self, filters: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute convolution to project input sequences into the spectral basis.

        Args:
            filters: torch.Tensor of shape [seq_len, num_heads]
                     Each column is the spectral filter for a head.
            keys:    torch.Tensor of shape [batch_size, num_heads, seq_len, head_dim]
                     Input sequences for each head.

        Returns:
            torch.Tensor of shape [batch_size, num_heads, seq_len, head_dim]
                     The result of convolving each head's filter with the corresponding input sequences.
        """
        conv1d = lambda f, k: torchaudio.functional.fftconvolve(k, f)[: k.shape[0]]
        fconv = torch.vmap(conv1d, in_dims=(None, 1), out_dims=1)

        # For a single head: apply fconv using that head's filter and keys.
        def conv_head(filter_vec: torch.Tensor, key_seq: torch.Tensor) -> torch.Tensor:
            return fconv(filter_vec, key_seq)

        # hconv applies conv_head over the head dimension.
        hconv = torch.vmap(conv_head, in_dims=(0, 0), out_dims=0)
        # Transpose filters so that each head's filter is along axis 0:
        # filters: [seq_len, num_heads] -> [num_heads, seq_len]
        filters_T = filters.transpose(0, 1)
        # bconv applies hconv over the batch dimension.
        bconv = torch.vmap(lambda keys_batch: hconv(filters_T, keys_batch), in_dims=0, out_dims=0)
        return bconv(keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        h = self.head_dim

        # Compute Q, K, V projections
        q = self.wq(x).reshape(B, L, H, h).transpose(1, 2)
        k = self.wk(x).reshape(B, L, H, h).transpose(1, 2) * h**-0.5
        v = self.wv(x).reshape(B, L, H, h).transpose(1, 2)

        # Spectral filter
        k_conv = self.conv(self.spectral_basis, k)  # [B, H, L, h]
        v_conv = self.conv(self.spectral_basis, v)  # [B, H, L, h]

        # Compute pairwise interactions via outer product.
        Z = torch.einsum("bhlm,bhln->bhlmn", v_conv, k_conv)  # [B, H, L, h, h]

        # Flatten only for gate projection.
        gate_input = Z.flatten(-2, -1)  # shape: [B, H, L, h*h]
        gates_logits = self.wg(gate_input)  # shape: [B, H, L, 1]
        gates = (F.relu(gates_logits) ** 2) + self.eps  # [B, H, L, 1]
        gates = gates.unsqueeze(-1)  # [B, H, L, 1, 1]

        # Multiply the original Z by the gates.
        gated_Z = gates * Z  # [B, H, L, h, h]

        # Prepare for associative scan
        gated_Z = gated_Z.permute(0, 1, 3, 4, 2)  # [B, H, h, h, L]
        gated_Z_flat = gated_Z.reshape(B, H * h * h, L)  # [B, H * h * h, L]

        # For the gates, they are scalar per head per time step.
        gates_flat = gates.squeeze(-1).squeeze(-1)  # now [B, H, L]
        # Expand each scalar to cover the h*h features.
        gates_flat = gates_flat.repeat_interleave(h * h, dim=1)  #  [B, H * h * h, L]

        # Run the associative scan along the sequence dimension.
        cumul_gated_Z_flat, cumul_gates_flat = associative_scan(gated_Z_flat, gates_flat)
        # cumul_*_flat: [B, (H*head_dim*head_dim), seq_len]

        # Unflatten the D dimension back into (num_heads, head_dim, head_dim).
        cumul_gated_Z = cumul_gated_Z_flat.reshape(B, H, h, h, L).permute(0, 1, 4, 2, 3)
        cumul_gates = cumul_gates_flat.reshape(B, H, h, h, L).permute(0, 1, 4, 2, 3)

        # Compute linear attention weights.
        attn_weights = cumul_gated_Z / (cumul_gates + self.eps)

        # Compute context by applying attention weights to query.
        # q has shape [B, H, seq_len, head_dim], attn_weights has shape [B, H, seq_len, head_dim, head_dim]
        ctxt = torch.einsum("bhli,bhldo->bhlo", q, attn_weights)
        unit_ctxt = F.normalize(ctxt, p=2, dim=3, eps=self.eps)

        # Merge heads: reshape from [B, H, seq_len, head_dim] to [B, seq_len, self.dim]
        output = unit_ctxt.transpose(1, 2).reshape(B, L, D)
        return self.wo(output)

    @staticmethod
    def chebyshev_feature_map_vector(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Computes a vector-valued Chebyshev feature map for x.
        Given x, returns [c_0, c_1 * x, c_2 * x^2, ..., c_n * x^n].
        Args:
            x (torch.Tensor): Input tensor.
            coeffs (torch.Tensor): 1D tensor of Chebyshev coefficients.
        Returns:
            torch.Tensor: Tensor with an extra last dimension of size len(coeffs).
        """
        features = [coeffs[i] * (x**i) for i in range(len(coeffs))]
        # Stack along a new last dimension.
        return torch.stack(features, dim=-1)

    def compute_stu_features(self, u: torch.Tensor) -> torch.Tensor:
        """Compute STU features"""
        B, T, d = u.shape

        # Convolve inputs w/ Chebyshev coefficients, per https://arxiv.org/pdf/2502.06545
        # p_coeffs_conv = -fft_conv(u, self.p_coeffs_kernel, mode="same", causal=True)

        if self.use_tensordot:
            # Project first
            u_proj = u @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            # p_coeffs_conv = p_coeffs_conv @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            phi_proj = self.stu_filters @ self.M_filters  # (L, K) x (K, r) -> (L, r)

            # Then, convolve: (B, L, r) ⊗ (L, r) -> (B, L, r)
            spectral_plus, spectral_minus = stu_conv(u_proj, phi_proj, self.n, self.use_tensordot)

            # Final output
            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            # out = self.out_proj_stu(out + p_coeffs_conv)
            out = self.out_proj_stu(out)
        else:
            # Convolve first to get featurized inputs: (B, L, d_in) x (L, K) -> (B, L, K, d_in)
            U_plus, U_minus = stu_conv(u, self.stu_filters, self.n, self.use_tensordot)

            # Compute sum-product of featurized inputs and M matrices over the K filters
            B, L, K, d_in = U_plus.shape

            # Spectral output: (B, L, K * d_in) x (K * d_in, d_out) -> (B, L, d_out)
            spectral_plus = U_plus.view(B, L, K * d_in) @ self.M_phi_plus.view(K * d_in, self.d_out)

            if not self.use_hankel_L:
                spectral_minus = U_minus.view(B, L, K * d_in) @ self.M_phi_minus.view(K * d_in, self.dim)

            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            # out = out + p_coeffs_conv

        return out


class SpectronConfig:
    def __init__(
        self,
        seq_len: int,
        dim: int,
        num_eigh: int,
        num_heads: int,
        use_hankel_L: bool,
        dropout: float,
        use_tensordot: bool,
        r: int,
        rope_theta: float,
        vocab_size: int,
        d_out: int,
        num_layers: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.seq_len = seq_len
        self.dim = dim
        self.num_eigh = num_eigh
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.d_out = d_out
        self.num_layers = num_layers
        self.use_hankel_L = use_hankel_L
        self.dropout = dropout
        self.use_tensordot = use_tensordot
        self.r = r
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype


class SpectralAttentionLayer(nn.Module):
    """
    A single layer that applies SpectralAttention, followed by an MLP,
    each of which is added (residual) to the input, then normalized.

    Args:
        seq_len (int): Sequence length (T).
        dim (int): Model dimension.
        k (int): Projection dimension for the spectral filters.
        use_hankel_L (bool): Whether to use a Hankel matrix.
        device: Torch device.
    """

    def __init__(
        self,
        config: SpectronConfig,
    ):
        super().__init__()
        self.spectral_attention = SpectralAttention(config)
        self.spectral_attention_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim, 4 * config.dim)
        self.mlp_norm = nn.LayerNorm(config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spectral_attention(self.spectral_attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Spectron(nn.Module):
    """
    A stacked spectral-transformer-like model. Uses an embedding, a dropout,
    multiple SpectralAttention layers in sequence, and an output projection.

    Args:
        seq_len (int): Sequence length.
        dim (int): Model dimension.
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
        config: SpectronConfig,
    ):
        super().__init__()

        if config.d_out is None:
            config.d_out = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.use_tensordot = config.use_tensordot
        self.use_hankel_L = config.use_hankel_L
        self.num_layers = config.num_layers

        self.layers = nn.ModuleList([SpectralAttentionLayer(config) for _ in range(config.num_layers)])

        self.norm = nn.LayerNorm(config.dim)
        self.out_proj = nn.Linear(config.dim, config.d_out)

        self.std = config.dim**-0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))
        print(self.eval())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Spectron model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T) containing token indices.

        Returns:
            torch.Tensor: Output logits of shape (B, T, d_out).
        """
        x = self.embedding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)
        out = self.out_proj(x)
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, SpectralAttention):
            if self.use_tensordot:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_conv)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params


class STU(nn.Module):
    def __init__(self, config, filters) -> None:
        super(STU, self).__init__()
        self.config = config
        self.stu_filters = filters
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.num_eigh = config.num_eigh
        self.d_in = config.dim
        self.d_out = config.dim
        self.r = config.r
        self.use_hankel_L = config.use_hankel_L
        self.use_tensordot = config.use_tensordot

        self.p_coeffs_kernel = nn.Parameter(
            torch.tensor(
                get_monic_chebyshev_coeffs(get_opt_degree(config.seq_len)),
                device=config.device,
            )
            .view(-1, 1)
            .repeat(1, self.d_in)
        )  # (n, d_in)

        self.flash_fft = FlashFFTConv(self.n, dtype=torch.bfloat16) if config.use_flash_fft else None

        if self.use_tensordot:
            # Projection matrices
            self.M_inputs = nn.Parameter(torch.empty(self.d_in, self.r, dtype=config.dtype))
            self.M_filters = nn.Parameter(torch.empty(self.num_eigh, self.r, dtype=config.dtype))
            self.out_proj = nn.Linear(self.r, self.d_out, bias=config.bias)
        else:
            # Full M matrix
            self.M_phi_plus = nn.Parameter(torch.empty(self.num_eigh, self.d_in, self.d_out, dtype=config.dtype))

            # If not using Hankel_L, we compute the negative featurization separately
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.empty(self.num_eigh, self.d_in, self.d_out, dtype=config.dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # Convolve inputs w/ Chebyshev coefficients, per https://arxiv.org/pdf/2502.06545
        p_coeffs_conv = -fft_conv(u, self.p_coeffs_kernel, mode="same", causal=True)

        if self.use_tensordot:
            # Project first
            u_proj = u @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            p_coeffs_conv = p_coeffs_conv @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            phi_proj = self.stu_filters @ self.M_filters  # (L, K) x (K, r) -> (L, r)

            # Then, convolve: (B, L, r) ⊗ (L, r) -> (B, L, r)
            if self.flash_fft:
                spectral_plus, spectral_minus = flash_stu_conv(u_proj, phi_proj, self.flash_fft, self.use_tensordot)
            else:
                spectral_plus, spectral_minus = stu_conv(u_proj, phi_proj, self.n, self.use_tensordot)
        else:
            # Convolve first to get featurized inputs: (B, L, d_in) x (L, K) -> (B, L, K, d_in)
            if self.flash_fft:
                U_plus, U_minus = flash_stu_conv(u, self.stu_filters, self.flash_fft, self.use_tensordot)
            else:
                U_plus, U_minus = stu_conv(u, self.stu_filters, self.n, self.use_tensordot)

            # Compute sum-product of featurized inputs and M matrices over the K filters
            B, L, K, d_in = U_plus.shape

            # Spectral output: (B, L, K * d_in) x (K * d_in, d_out) -> (B, L, d_out)
            spectral_plus = U_plus.view(B, L, K * d_in) @ self.M_phi_plus.view(K * d_in, self.d_out)

            if not self.use_hankel_L:
                spectral_minus = U_minus.view(B, L, K * d_in) @ self.M_phi_minus.view(K * d_in, self.d_out)

        out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
        out = self.out_proj(out + p_coeffs_conv) if self.use_tensordot else out + p_coeffs_conv
        return out


class STULayer(nn.Module):
    def __init__(self, config, stu_filters):
        super(STULayer, self).__init__()
        self.stu_norm = nn.LayerNorm(config.dim)
        self.stu = STU(config, stu_filters)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_scale * config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.stu_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads
        # self.rope = RoPE(self.head_dim, config.seq_len, config.rope_theta)

        # From flash-attn repo, potentially less issues with RoPE and bfloat16
        self.rope = RoPE(dim=self.head_dim, base=config.rope_theta)

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)

        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.c_proj.SCALE_INIT = 1

        self.window_size = config.window_size
        self.softcap = config.softcap

        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

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

        qkv = self.qkv(x)
        qkv = qkv.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        qkv = self.rope(qkv)

        y = flash_attn_qkvpacked_func(
            qkv=qkv.to(torch.bfloat16),
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),  # Set to seq_len if full attention
        ).to(torch.float32)

        y = y.contiguous().view(bsz, q_len, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FlashAttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm = nn.LayerNorm(args.dim)
        self.attn = FlashAttention(args)
        self.mlp_norm = nn.LayerNorm(args.dim)
        self.mlp = MLP(args.dim, args.mlp_scale * args.dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FlashSTUConfig:
    model_type = "flash_stu"

    def __init__(
        self,
        bsz: int = 8,
        dim: int = 896,
        num_heads: int = 8,
        num_layers: int = 12,
        seq_len: int = 8192,
        weight_tying: bool = True,
        window_size: int = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.1,
        num_eigh: int = 24,
        r: int = 4,
        use_hankel_L: bool = False,
        use_flash_fft: bool = True,
        use_tensordot: bool = True,
        use_attn: bool = True,
        softcap: float = 50.0,
        rope_theta: float = 10000.0,
        dilation: int = 2,  # For dilated sliding window attention mask, if used
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.weight_tying = weight_tying
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = dim
        self.mlp_scale = mlp_scale
        self.intermediate_size = self.hidden_size * self.mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.r = r
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_tensordot = use_tensordot
        self.use_attn = use_attn
        self.softcap = softcap
        self.rope_theta = rope_theta
        self.dtype = dtype
        self.device = device


class FlashSTU(nn.Module):
    config_class = FlashSTUConfig

    def __init__(self, config, filters) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        assert (
            config.dim % config.num_heads == 0
        ), f"dim ({config.dim}) must be divisible num_heads ({config.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.use_tensordot = config.use_tensordot
        self.use_hankel_L = config.use_hankel_L

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(STULayer(config, filters))

        self.out_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=config.bias)

        if config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = config.dim**-0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))
        print(self.eval())

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x)

        y_hat = self.lm_head(self.out_norm(x))
        return y_hat

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            if self.use_tensordot:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)


class Mamba(nn.Module):
    """Wrapper for Mamba to match the interface of other models."""

    def __init__(
        self,
        seq_len: int,
        dim: int,
        vocab_size: int,
        num_layers: int = 2,
        device=None,
    ):
        super().__init__()
        from thesis.models.mamba.model import MambaConfig, Mamba

        # Create config
        config = MambaConfig(
            dim=dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            ssm_chunk_size=seq_len // 4,
            weight_tying=False,
            use_mem_eff_path=True,
            dtype=torch.float32,
        )

        # Create model
        self.model = Mamba(config).to(device=device)
        self.model.init_weights(buffer_device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def row_wise_topk(sim: torch.Tensor, k: int):
    # sim: [batch, seq_len, seq_len]
    # Returns top_k values and indices along last dim (for each query)
    top_sim, top_indices = torch.topk(sim, k=k, dim=-1)
    return top_sim, top_indices


class SparseFlashSTULayer(nn.Module):
    def __init__(self, config, stu_filters):
        # TODO: Insane idea -> TopK -> MLA => insane savings
        super(SparseFlashSTULayer, self).__init__()
        self.stu_norm = nn.LayerNorm(config.dim)
        self.stu = STU(config, stu_filters)
        self.attn = Attention(config)
        self.attn_norm = nn.LayerNorm(config.dim)
        # TODO: Convert to QKV (isn't that already the case?) and Top-k this too eventually)
        self.mlp = MLP(config.dim, config.mlp_scale * config.dim)
        self.mlp_norm = nn.LayerNorm(config.dim)

        self.wq = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.wk = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.wv = nn.Linear(config.dim, config.dim, bias=config.bias)

        # TODO: Do we just want to make these an operation instead of learnable?
        self.pq = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.pk = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.pv = nn.Linear(config.dim, config.dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        z = x  # residual

        # Compute Q, K, V using standard linear projections
        q = self.wq(x)  # [b, L, dim]
        k = self.wk(x)  # [b, L, dim]
        v = self.wv(x)  # [b, L, dim]

        # Process via STU branch (spectral transform unit) with normalization
        q_stu = self.stu(self.stu_norm(q))  # [b, L, stu_k, dim]
        k_stu = self.stu(self.stu_norm(k))  # [b, L, stu_k, dim]
        v_stu = self.stu(self.stu_norm(v))  # [b, L, stu_k, dim]

        # Compute additional projections (for potential alternate attention computation)
        q_pq = self.pq(q_stu)  # [b, L, dim]
        k_pk = self.pk(k_stu)  # [b, L, dim]
        v_pv = self.pv(v_stu)  # [b, L, dim]

        # Compute similarity matrix between queries and keys (using the projected Q and K)
        # sim: [b, L, L]
        sim = q_pq @ k_pk.T

        # Get row-wise top-k similarity values and corresponding indices
        top_sim, top_indices = row_wise_topk(sim, k=config.top_k)
        # Delete full sim to simulate "zeroing out" non-topk entries (no memory savings yet)
        del sim

        # Create a new similarity matrix with zeros everywhere except the top-k entries
        # We'll reconstruct a full tensor from top_sim and top_indices.
        b, L, _ = top_sim.shape  # top_sim is [b, L, top_k]
        sim_topk = torch.zeros(b, L, L, device=x.device, dtype=x.dtype)
        sim_topk = sim_topk.scatter(dim=-1, index=top_indices, src=top_sim)
        # At this point, sim_topk has zeros for non-top-k positions.

        # TODO: Eventually try some grouped query attention ideas

        # For now, we feed the "pruned" similarity into our attention module.
        # NOTE: This is a placeholder—your Attention module might expect Q, K, V inputs,
        # not a similarity matrix. Adjust accordingly.
        attn_input = self.attn_norm(sim_topk)  # using layer norm on sim_topk
        attn_out = self.attn(attn_input)  # [b, L, dim] output from attention

        # TODO: Might we just compute the entire model in this reduced dim space and project only at the end??
        attn_out = self.o_proj(attn_out)  # project back to (b, L, dim)

        # MLP branch (not reduced for now)
        mlp_out = self.mlp(self.mlp_norm(x))

        # Combine residual connections: here we add both attention and MLP outputs to the original input
        x = x + attn_out + mlp_out
        return x


class SparseFlashSTUConfig(nn.Module):
    model_type = "sparse_flash_stu"

    def __init__(
        self,
        bsz: int = 8,
        dim: int = 896,
        num_heads: int = 8,
        num_layers: int = 12,
        seq_len: int = 8192,
        weight_tying: bool = True,
        window_size: int = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.1,
        num_eigh: int = 24,
        r: int = 4,
        use_flash_fft: bool = True,
        use_tensordot: bool = True,
        use_attn: bool = True,
        rope_theta: float = 10000.0,
        dilation: int = 2,  # For dilated sliding window attention mask, if used
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.weight_tying = weight_tying
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.mlp_scale = mlp_scale
        self.inter_dim = self.dim * self.mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.r = r
        self.use_flash_fft = use_flash_fft
        self.use_tensordot = use_tensordot
        self.use_attn = use_attn
        self.rope_theta = rope_theta
        self.dilation = dilation
        self.dtype = dtype
        self.device = device


class SparseFlashSTU(nn.Module):
    config_class = SparseFlashSTUConfig

    def __init__(self, config, filters) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        assert (
            config.dim % config.num_heads == 0
        ), f"dim ({config.dim}) must be divisible num_heads ({config.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.use_tensordot = config.use_tensordot

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(SparseFlashSTULayer(config, filters))

        self.out_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=config.bias)

        if config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = config.dim**-0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))
        print(self.eval())

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x)

        y_hat = self.lm_head(self.out_norm(x))
        return y_hat

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            if self.use_tensordot:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)


# -----------------------------------------------------------------------------
# Token-Level Accuracy Helper
# -----------------------------------------------------------------------------


def compute_token_level_accuracy(model, loader, device=None):
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
            logits = model(inputs)
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


# def compute_token_level_accuracy(model, loader, device=None):
#     """
#     Compute token-level accuracy and generate plots comparing attention weight spikiness and
#     monotonicity for a standard Softmax mechanism versus the Spectron model's attention mechanism.

#     Args:
#         model: The Spectron model.
#         loader: DataLoader providing input batches.
#         device: Device to run the model on (defaults to model's device).

#     Returns:
#         float: Token-level accuracy percentage.
#     """
#     if device is None:
#         device = next(model.parameters()).device
#     model.eval()

#     with torch.no_grad():
#         # Process the first batch
#         inputs, targets = next(iter(loader))
#         inputs, targets = inputs.to(device), targets.to(device)

#         # Get logits and debug info
#         if isinstance(model, Spectron):
#             logits, debug_info = model(inputs, debug=True)
#         else:
#             logits = model(inputs)

#         predictions = logits.argmax(dim=-1)
#         valid_mask = targets != -100
#         correct_tokens = ((predictions == targets) & valid_mask).sum().item()
#         total_tokens = valid_mask.sum().item()
#         token_acc = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0

#         # Select example and query position (e.g., approximating Query 48)
#         example_idx = 0  # First example
#         non_masked_positions = valid_mask[example_idx].nonzero(as_tuple=False)
#         chosen_pos = (
#             non_masked_positions[48 % len(non_masked_positions)].item() if non_masked_positions.numel() > 0 else 0
#         )

#         if isinstance(model, Spectron):
#             # Extract Q and K from debug_info (post-GELU)
#             Q_raw = debug_info["layer_0"]["attn_Q_raw"]  # (B, H, T, d_head)
#             K_raw = debug_info["layer_0"]["attn_K_raw"]  # (B, H, T, d_head)
#             Q_raw_ex = Q_raw[example_idx]  # (H, T, d_head)
#             K_raw_ex = K_raw[example_idx]  # (H, T, d_head)
#             d_head = Q_raw_ex.size(-1)

#             # Compute Q_t and K up to position t for classic Softmax
#             Q_chosen_raw = Q_raw_ex[:, chosen_pos, :]  # (H, d_head)
#             K_up_to_chosen_raw = K_raw_ex[:, : chosen_pos + 1, :]  # (H, t+1, d_head)

#             # Compute QK dot products for classic Softmax
#             qk_dot_raw = torch.einsum("hd,hpd->hp", Q_chosen_raw, K_up_to_chosen_raw)  # (H, t+1)

#             # Classic Softmax Attention Weights
#             attn_weights_softmax = F.softmax(qk_dot_raw / (d_head**0.5), dim=1)  # (H, t+1)
#             avg_attn_weights_softmax = attn_weights_softmax.mean(dim=0).cpu().numpy()  # (t+1,)

#             # Extract featurized Q and K for Spectron attention
#             Q_feat = debug_info["layer_0"]["attn_Q_feat"]  # (B, H, T, d_head)
#             K_feat = debug_info["layer_0"]["attn_K_feat"]  # (B, H, T, d_head)
#             Q_feat_ex = Q_feat[example_idx]  # (H, T, d_head)
#             K_feat_ex = K_feat[example_idx]  # (H, T, d_head)

#             # Compute Q_t and K up to position t for Spectron
#             Q_chosen_feat = Q_feat_ex[:, chosen_pos, :]  # (H, d_head)
#             K_up_to_chosen_feat = K_feat_ex[:, : chosen_pos + 1, :]  # (H, t+1, d_head)

#             # Compute QK dot products for Spectron
#             qk_dot_feat = torch.einsum("hd,hpd->hp", Q_chosen_feat, K_up_to_chosen_feat)  # (H, t+1)

#             # Spectron Attention Weights (approximation)
#             qk_dot_sum_feat = qk_dot_feat.sum(dim=1, keepdim=True)  # (H, 1)
#             attn_weights_spectron = qk_dot_feat / (qk_dot_sum_feat + 1e-10)  # (H, t+1)
#             avg_attn_weights_spectron = attn_weights_spectron.mean(dim=0).cpu().numpy()  # (t+1,)

#             # Compute entropy to measure spikiness
#             entropy_softmax = -(avg_attn_weights_softmax * np.log(avg_attn_weights_softmax + 1e-10)).sum()
#             entropy_spectron = -(avg_attn_weights_spectron * np.log(avg_attn_weights_spectron + 1e-10)).sum()

#             # Positions for plotting
#             positions = np.arange(chosen_pos + 1)

#             # Figure 2: Attention Weight Spikiness
#             plt.figure(figsize=(12, 3))
#             for i, (mech, weights, entropy) in enumerate(
#                 [
#                     ("Softmax", avg_attn_weights_softmax, entropy_softmax),
#                     ("Spectron", avg_attn_weights_spectron, entropy_spectron),
#                 ],
#                 1,
#             ):
#                 plt.subplot(1, 2, i)
#                 plt.scatter(positions, weights, color="green", alpha=0.7)
#                 plt.title(mech)
#                 plt.xlabel("Position")
#                 plt.ylabel(f"Query {chosen_pos} Attn. Weights")
#                 plt.xlim(0, 40)
#                 plt.ylim(0, max(0.25, weights.max() * 1.1))
#                 plt.text(0.95, 0.95, f"Entropy: {entropy:.3f}", transform=plt.gca().transAxes, ha="right", va="top")
#             plt.tight_layout()
#             plt.suptitle("Figure 2: Attention Weight Spikiness", y=1.1)
#             plt.show()

#             # Figure 3: Attention Weight Monotonicity
#             plt.figure(figsize=(12, 3))
#             for i, (mech, weights, qk_dot) in enumerate(
#                 [
#                     ("Softmax", avg_attn_weights_softmax, qk_dot_raw),
#                     ("Spectron", avg_attn_weights_spectron, qk_dot_feat),
#                 ],
#                 1,
#             ):
#                 qk = qk_dot.mean(dim=0).cpu().numpy()  # (t+1,)
#                 sorted_indices = np.argsort(qk)
#                 qk_sorted = qk[sorted_indices]
#                 attn_sorted = weights[sorted_indices]

#                 plt.subplot(1, 2, i)
#                 plt.plot(qk_sorted, attn_sorted, color="green")
#                 plt.title(mech)
#                 plt.xlabel("QK Dot Product")
#                 plt.ylabel("Attention Weight")
#             plt.tight_layout()
#             plt.suptitle("Figure 3: Attention Weight Monotonicity", y=1.1)
#             plt.show()

#     print(f"Token-Level Accuracy: {token_acc:.2f}%")
#     return token_acc


# -----------------------------------------------------------------------------
# Training Loop (step-based)
# -----------------------------------------------------------------------------
def train_model(
    model, loader, val_loader, max_steps: int = 10000, eval_interval: int = 50, dtype: torch.dtype = torch.float32
):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
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
                acc = compute_token_level_accuracy(model, val_loader, device)
                accuracy_history.append(acc)
                eval_steps.append(step)
                latest_acc = acc
                pbar.set_postfix(loss=f"{total_loss:.4f}", acc=f"{acc:.2f}%")

    pbar.close()
    return loss_history, accuracy_history, eval_steps


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.set_float32_matmul_precision("high")

    # Generate training dataset
    train_dataset = generate_assoc_recall(
        num_examples=args.num_examples,
        sequence_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_pairs=args.num_pairs,
        random_non_queries=args.random_non_queries,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)

    # Generate validation dataset
    val_dataset = generate_assoc_recall(
        num_examples=args.num_examples // 33,
        sequence_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_pairs=args.num_pairs,
        random_non_queries=args.random_non_queries,
        num_queries=args.num_queries,
        seed=args.seed + 1,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=False)

    models_to_train = []
    if args.model == "all" or args.model == "transformer":
        trans_model = Transformer(args).to(device, dtype=getattr(torch, args.dtype))
        models_to_train.append(("Transformer", trans_model))

    if args.model == "all" or args.model == "spectron":
        dtype = getattr(torch, args.dtype)
        config = SpectronConfig(
            seq_len=args.seq_len,
            dim=args.dim,
            num_eigh=args.num_eigh,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            d_out=args.vocab_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_hankel_L=False,
            use_tensordot=False,
            r=args.dim,
            rope_theta=args.rope_theta,
            device=device,
            dtype=dtype,
        )
        spectron = Spectron(config).to(device=device, dtype=dtype)
        spectron = torch.compile(spectron)
        models_to_train.append(("Spectron", spectron))

    if args.model == "all" or args.model == "flashstu":
        filters = get_spectral_filters(args.seq_len, args.num_eigh, use_hankel_L=False, device=device)
        config = FlashSTUConfig(
            seq_len=args.seq_len,
            dim=args.dim,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_scale=args.mlp_scale,
            dropout=args.dropout,
            use_hankel_L=False,
            use_tensordot=False,
            use_flash_fft=False,
            use_attn=True,
            device=device,
        )
        flash_stu = FlashSTU(config, filters).to(device=device, dtype=getattr(torch, args.dtype))
        models_to_train.append(("FlashSTU", flash_stu))

    if args.model == "all" or args.model == "mamba":
        mamba = Mamba(
            seq_len=args.seq_len,
            dim=args.dim,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            device=device,
        ).to(device=device, dtype=getattr(torch, args.dtype))
        models_to_train.append(("Mamba", mamba))

    if args.model == "all" or args.model == "sparse_flashstu":
        filters = get_spectral_filters(args.seq_len, args.num_eigh, use_hankel_L=False, device=device)
        config = SparseFlashSTUConfig(
            seq_len=args.seq_len,
            dim=args.dim,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_scale=args.mlp_scale,
            dropout=args.dropout,
            use_hankel_L=False,
            use_tensordot=False,
            use_flash_fft=False,
            use_attn=True,
            device=device,
        )
        sparse_flash_stu = SparseFlashSTU(config, filters).to(device=device, dtype=getattr(torch, args.dtype))
        models_to_train.append(("SparseFlashSTU", sparse_flash_stu))

    # Train models and collect results
    results = {}
    for model_name, model in models_to_train:
        if args.torch_compile:
            model = torch.compile(model)

        print(f"\nTraining {model_name}...")
        loss_history, acc_history, eval_steps = train_model(
            model,
            loader,
            val_loader,
            max_steps=args.steps,
            eval_interval=args.eval,
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
