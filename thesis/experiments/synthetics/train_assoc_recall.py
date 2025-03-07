import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from thesis.experiments.synthetics.assoc_recall import generate_assoc_recall
from thesis.experiments.synthetics.args import args
from flash_attn import flash_attn_func

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):    
    # For half the dimensions, build the scale factor:
    freq_seq = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (theta ** freq_seq)

    # Outer product with positions
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)

    # Build a complex exponential e^{i * theta}
    freqs_cis = torch.polar(
        torch.ones_like(angles),
        angles
    )
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    x is [B, n_heads, seq_len, head_dim_as_complex],
    so we want to broadcast freqs_cis from [max_seq_len, half_dim]
    to [1, 1, seq_len, half_dim].
    """
    seq_len = x.shape[2]
    freqs_cis = freqs_cis[:seq_len]  # slice down to current seq_len
    return freqs_cis.view(1, 1, seq_len, -1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert real -> complex by grouping last dim in pairs
    # shape => [B, n_heads, seq_len, head_dim//2, 2] => complex => [B, n_heads, seq_len, head_dim//2]
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast the frequencies to match [B, n_heads, seq_len, head_dim//2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    # Multiply => apply rotation
    xq_complex = xq_complex * freqs_cis
    xk_complex = xk_complex * freqs_cis

    # Convert back to real => shape [B, n_heads, seq_len, head_dim]
    xq_out = torch.view_as_real(xq_complex).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_complex).reshape(*xk.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)

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
        slopes = torch.tensor(slopes, device=torch.device("cuda")) # FA ALiBi must be on CUDA
        slopes = slopes * interpolation_factor  # https://arxiv.org/pdf/2310.13017
        return slopes

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
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

        if self.alibi_slopes is None: # Either RoPE or ALiBi for positional embedding
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        y = flash_attn_func(  # https://arxiv.org/pdf/2307.08691
            q=q, k=k, v=v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0), # Set toseq_len if full attention
            alibi_slopes=self.alibi_slopes, # https://arxiv.org/pdf/2108.12409

            # NOTE: Softcapping cannot be used simultaneously with dropout
            softcap=self.softcap,  # https://arxiv.org/pdf/2408.00118
        )

        out = y.contiguous().view(bsz, q_len, -1)
        out = self.resid_dropout(self.c_proj(out))
        return out

class AttentionLayer(nn.Module):
    def __init__(self, args) -> None:
        super(AttentionLayer, self).__init__()
        self.attn_norm = nn.RMSNorm(args.dim)
        self.attn = Attention(args=args)
        self.mlp_norm = nn.RMSNorm(args.dim)
        self.mlp = MLP(args.dim, args.mlp_scale * args.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor=None) -> torch.Tensor:
        x = x + self.attn(x=self.attn_norm(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        assert args.dim % args.num_heads == 0, f"dim ({args.dim}) must be divisible num_heads ({args.num_heads})"
        self.head_dim = args.dim // args.num_heads

        # From pytorch/pytorch#123411, we set persistent=True for torch.compile and PP compatibility
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                head_dim=self.head_dim,
                max_seq_len=args.seq_len,
                theta=args.theta,
            ),
            persistent=True,
        )

        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList()
        for _ in range(args.num_layers):
            self.layers.append(AttentionLayer(args))

        self.out_norm = nn.RMSNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=args.bias)
        self.lm_head.SCALE_INIT = 1

        if args.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = args.dim ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x, freqs_cis=self.freqs_cis)

        out = self.lm_head(self.out_norm(x))
        return out

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


def get_polynomial_hankel(n, beta, t, chunk_size=2048, device="cuda", dtype=torch.bfloat16):
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
    dtype: torch.dtype = torch.bfloat16,
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


class LearnableSpectralFilters(nn.Module):
    def __init__(
        self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters

class SpectralAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
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
        self.dim = dim
        self.d_out = dim  # Same for now
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
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
            .repeat(1, dim)
        )
        # (n, dim)

        # STU projection matrices
        if use_tensordot:
            self.M_inputs = nn.Parameter(torch.empty(dim, r, dtype=dtype, device=device))
            self.M_filters = nn.Parameter(torch.empty(k, r, dtype=dtype, device=device))
            self.out_proj_stu = nn.Linear(r, dim, bias=True, device=device, dtype=dtype)
        else:
            self.M_phi_plus = nn.Parameter(torch.empty(k, dim, dim, dtype=dtype, device=device))
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.empty(k, dim, dim, dtype=dtype, device=device))

        # Attention components
        self.Q = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.K = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.V = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.o_proj = nn.Linear(dim, dim, device=device, dtype=dtype)

        # Gate for combining attention and spectral features
        self.gate = nn.Linear(dim, dim, device=device, dtype=dtype)

        # Normalization
        self.norm = nn.RMSNorm(dim)

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
                spectral_minus = U_minus.view(B, L, K * d_in) @ self.M_phi_minus.view(K * d_in, self.dim)

            out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            out = out + p_coeffs_conv

        return out

    def forward(self, x: torch.Tensor, chunk_len: int = 128) -> torch.Tensor:
        B, T, d = x.shape

        # Branch 1: Compute STU features
        x_tilde = self.compute_stu_features(x)  # (B, T, dim)

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
        return self.o_proj(Y_combined)


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
        seq_len: int,
        dim: int,
        k: int,
        num_heads: int = 4,
        use_hankel_L: bool = False,
        use_tensordot: bool = True,
        r: int = 64,
        device=None,
    ):
        super().__init__()
        self.spectral_attention = SpectralAttention(
            seq_len, dim, k, num_heads, use_hankel_L, use_tensordot=use_tensordot, r=r, device=device
        )
        self.spectral_attention_norm = nn.RMSNorm(dim)
        self.mlp = MLP(dim, 4 * dim)
        self.mlp_norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpectralAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, dim).
        """
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
        seq_len: int,
        dim: int,
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

        self.embedding = nn.Embedding(vocab_size, dim)

        # Sinusoidal positional embeddings
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                SpectralAttentionLayer(seq_len, dim, k, num_heads, use_hankel_L, use_tensordot, r, device=device)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, d_out)

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


class FlashSTU(nn.Module):
    """Wrapper for FlashSTU to match the interface of other models."""

    def __init__(
        self,
        seq_len: int,
        dim: int,
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
            dim=dim,
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
            torch_dtype=torch.bfloat16,
        )

        # Create model
        self.model = Mamba(config).to(device=device)
        self.model.init_weights(buffer_device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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


# -----------------------------------------------------------------------------
# Training Loop (step-based)
# -----------------------------------------------------------------------------
def train_model(model, loader, val_loader, max_steps: int = 10000, eval_interval: int = 50, dtype: torch.dtype = torch.bfloat16):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
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
                acc = compute_token_level_accuracy(model, val_loader)
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
        spectron = Spectron(
            seq_len=args.seq_len,
            dim=args.dim,
            k=math.ceil(math.log(args.seq_len)),
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            d_out=args.vocab_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_hankel_L=False,
            use_tensordot=False,
            r=args.dim,
            device=device,
        ).to(device)
        models_to_train.append(("Spectron", spectron))

    if args.model == "all" or args.model == "flashstu":
        flash_stu = FlashSTU(
            seq_len=args.seq_len,
            dim=args.dim,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_hankel_L=False,
            use_tensordot=False,
            device=device,
        ).to(device)
        models_to_train.append(("FlashSTU", flash_stu))

    if args.model == "all" or args.model == "mamba":
        mamba = Mamba(
            seq_len=args.seq_len,
            dim=args.dim,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            device=device,
        ).to(device)
        models_to_train.append(("Mamba", mamba))

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
