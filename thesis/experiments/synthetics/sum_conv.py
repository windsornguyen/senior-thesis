import torch


def stu_conv(u: torch.Tensor, v: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FFT-based convolution with causal alignment and negative featurization for multiple filters.

    Args:
        u: Input tensor of shape (B, seq_len, d_in).
        v: Kernel tensor of shape (seq_len, K) (for K filters; d_in is assumed 1).
        n: FFT length (typically 2*seq_len - 1 for linear convolution).

    Returns:
        U_plus, U_minus: each of shape (B, seq_len, K, d_in)
    """
    B, seq_len, d_in = u.shape
    K = v.shape[1]

    # Build the sign modulation vector: alternate +1 and -1 along the time axis.
    sgn = torch.full((1, seq_len, 1, 1), 1, device=u.device, dtype=u.dtype)
    sgn[:, 1::2] *= -1  # Negative featurization.

    # Reshape the kernel and input to include the filter dimension.
    # For v: from (seq_len, K) to (1, seq_len, K, 1, 1).
    v_reshaped = v.view(1, seq_len, K, 1, 1).contiguous()
    # For u: expand to (B, seq_len, K, d_in) so that each filter gets the same input.
    u_expanded = u.view(B, seq_len, 1, d_in).expand(B, seq_len, K, d_in)

    # Stack u and u * sgn along a new last dim. (Negative featurization trick)
    U = torch.stack([u_expanded, u_expanded * sgn], dim=-1).to(torch.float32).contiguous()
    # Compute FFT along the time axis.
    U = torch.fft.rfft(U, n=n, dim=1)
    v_fft = torch.fft.rfft(v_reshaped, n=n, dim=1)

    # Multiply in frequency domain and then inverse FFT.
    U_conv = torch.fft.irfft(v_fft * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn  # Correct U_minus.
    return U_plus, U_minus


def stu_conv_single(u: torch.Tensor, v: torch.Tensor, n: int) -> torch.Tensor:
    """
    FFT-based convolution for a single (combined) filter using negative featurization.

    Args:
        u: Input tensor of shape (B, seq_len, d_in).
        v: Combined filter of shape (seq_len, 1).
        n: FFT length.

    Returns:
        U_plus: Convolution output of shape (B, seq_len, d_in).
    """
    B, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device, dtype=u.dtype)
    sgn[:, 1::2] *= -1  # Negative featurization.

    # Reshape v to (1, seq_len, 1, 1) so it can be FFT'd.
    v_reshaped = v.view(1, seq_len, 1, 1).contiguous()
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
    U = torch.fft.rfft(U, n=n, dim=1)
    v_fft = torch.fft.rfft(v_reshaped, n=n, dim=1)

    U_conv = torch.fft.irfft(v_fft * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn
    return U_plus


# --- Simulation Setup ---

# Dimensions
B = 2  # Batch size.
seq_len = 32  # Sequence length.
d_in = 1  # Input dimension.
K = 4  # Number of filters.
n = 2 * seq_len - 1  # FFT length for full convolution.

torch.manual_seed(0)
# Input signal: shape (B, seq_len, d_in)
u = torch.randn(B, seq_len, d_in)
print(f"u shape: {u.shape}")

# Kernel for each filter: shape (seq_len, K).
v = torch.randn(seq_len, K)
print(f"v shape: {v.shape}")

# Weight matrix for combining the filter outputs
M = torch.randn(K, 1)
print(f"M shape: {M.shape}")

# --- Approach 1: Convolve with each filter and then combine ---
U_plus, _ = stu_conv(u, v, n)
print(f"U_plus shape after stu_conv: {U_plus.shape}")

U_plus_reshaped = U_plus.view(B, seq_len, K)
print(f"U_plus_reshaped shape: {U_plus_reshaped.shape}")
spectral_plus = U_plus_reshaped @ M
print(f"spectral_plus shape: {spectral_plus.shape}")

# --- Approach 2: Sum the filters (weighted) first, then do a single convolution ---
combined_filter = (v * M.view(1, K)).sum(dim=-1, keepdim=True)
print(f"combined_filter shape: {combined_filter.shape}")
spectral_plus_combined = stu_conv_single(u, combined_filter, n)
print(f"spectral_plus_combined shape: {spectral_plus_combined.shape}")

# --- Compare the outputs ---
max_diff = (spectral_plus - spectral_plus_combined).abs().max()
print("\nMax difference between approaches:", max_diff.item())
