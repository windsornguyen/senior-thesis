import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torchtune.modules import RotaryPositionalEmbeddings as RoPE

try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(
        f"Unable to import Triton-based flash attention: {e}. No alternative currently available."
    )


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]

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
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z, UPLO="U")
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k ** 0.25
    return phi_k.to(dtype=dtype)

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

def get_opt_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7 / 6) * math.log2(seq_len)))

def monic_chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x) normalized by 2**(n-1).
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    coeff = chebyshev_coeff(n)
    leading_term = coeff[-1]
    return [c / leading_term for c in coeff]

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

    U_fft = U_fft.unsqueeze(-1)                         # (B, fft_len//2+1, d, 1)
    V_fft = V_fft.unsqueeze(0).expand(B, -1, -1, -1)    # (B, fft_len//2+1, d, k)

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

    return result.type_as(u)

def stu_conv(u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    
    # TODO: MONKEY PATCH, REMOVE
    dtype = u.dtype

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    if use_tensordot:
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32).contiguous()
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32).contiguous() # (bsz, seq_len, K, d_in, stack)
        u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)

    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    # TODO: MONKEY PATCH, REMOVE
    U_plus = U_plus.to(dtype=dtype)
    U_minus = U_minus.to(dtype=dtype)

    return U_plus, U_minus


def flash_stu_conv(
    u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_tensordot: bool = True,
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

class STU(nn.Module):
    def __init__(self, config, filters) -> None:
        super(STU, self).__init__()
        self.config = config
        self.stu_filters = filters
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.K = config.num_eigh
        self.d_in = config.dim
        self.d_out = config.dim
        self.r = config.r
        self.use_hankel_L = config.use_hankel_L
        self.use_tensordot = config.use_tensordot

        self.p_coeffs_kernel = nn.Parameter(
            torch.tensor(
                monic_chebyshev_coeff(get_opt_degree(config.seq_len)),
                device=config.device,
            ).view(-1, 1).repeat(1, self.d_in)
        )  # (n, d_in)

        self.flash_fft = (
            FlashFFTConv(self.n, dtype=torch.bfloat16)
            if config.use_flash_fft and flash_fft_available
            else None
        )

        if self.use_tensordot:
            # Projection matrices
            self.M_inputs = nn.Parameter(torch.empty(self.d_in, self.r, dtype=config.torch_dtype))
            self.M_filters = nn.Parameter(torch.empty(self.K, self.r, dtype=config.torch_dtype))
            self.out_proj = nn.Linear(self.r, self.d_out, bias=config.bias)
        else:
            # Full M matrix
            self.M_phi_plus = nn.Parameter(torch.empty(self.K, self.d_in, self.d_out, dtype=config.torch_dtype))

            # If not using Hankel_L, we compute the negative featurization separately
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.empty(self.K, self.d_in, self.d_out, dtype=config.torch_dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # Convolve inputs w/ Chebyshev coefficients, per https://arxiv.org/pdf/2502.06545
        p_coeffs_conv = -fft_conv(u, self.p_coeffs_kernel, mode="same", causal=True)

        if self.use_tensordot:
            # Project first
            u_proj = u @ self.M_inputs                     # (B, L, d_in) x (d_in, r) -> (B, L, r)
            p_coeffs_conv = p_coeffs_conv @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
            phi_proj = self.stu_filters @ self.M_filters   # (L, K) x (K, r) -> (L, r)

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
        self.stu_norm = nn.RMSNorm(config.dim)
        self.stu = STU(config, stu_filters)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_scale * config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.stu_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads
        self.rope = RoPE(self.head_dim, config.seq_len, config.rope_theta)

        self.wq = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.wk = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.wv = nn.Linear(config.dim, config.dim, bias=config.bias)

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

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim)
        v = v.view(bsz, v_len, self.num_heads, self.head_dim)
        q, k = self.rope(q), self.rope(k)

        y = flash_attn_func(  # https://arxiv.org/pdf/2307.08691
            q=q, k=k, v=v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0), # Set to seq_len if full attention

            # NOTE: Softcapping cannot be used simultaneously with dropout
            softcap=self.softcap,  # https://arxiv.org/pdf/2408.00118
        )

        y = y.contiguous().view(bsz, q_len, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y

class AttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super(AttentionLayer, self).__init__()
        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = Attention(args=config)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_scale * config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x=self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

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

class FlashSTUConfig(PretrainedConfig):

    model_type = "FlashSTU"

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
        torch_dtype: torch.dtype = torch.float32,
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
        self.torch_dtype = torch_dtype
        self.device = device

class FlashSTU(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config, filters) -> None:
        super(FlashSTU, self).__init__(config)
        self.num_layers = config.num_layers
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.use_tensordot = config.use_tensordot
        self.use_hankel_L = config.use_hankel_L

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim, dtype=config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, filters))
            else:
                self.layers.append(AttentionLayer(config) if config.use_attn else STULayer(config, filters))

        self.norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=config.bias)

        if config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = config.dim ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x)

        y_hat = self.lm_head(self.norm(x))
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
        dropout=config_data["dropout"],
        num_eigh=config_data["num_eigh"],
        r=config_data["r"],
        use_hankel_L=config_data["use_hankel_L"],
        use_flash_fft=config_data["use_flash_fft"],
        use_tensordot=config_data["use_tensordot"],
        use_attn=config_data["use_attn"],
        softcap=config_data["softcap"],
        theta=config_data["theta"],
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
