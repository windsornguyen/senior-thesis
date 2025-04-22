import torch
import math

from torchaudio.functional import fftconvolve

# toy dimensions
B, H, L, h, K = 2, 3, 5, 4, 2

torch.manual_seed(0)
# toy inputs
u = torch.randn(B, H, L, h)
f_proj = torch.randn(L, h)  # projected filters
f_k = torch.randn(L, K)  # K filters


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


def convfft(
    signal: torch.Tensor,
    filt: torch.Tensor,
) -> torch.Tensor:
    """Causal 1D convolution via FFT for [L,] filters.

    Given `s[t]` (length *L*) and kernel `f[t]` (same length),
    returns the *causal* convolution

        y[t] = Σ_{τ=0..t}  s[τ] · f[t‑τ]   for  t = 0..L‑1.

    Implementation: zero–pad both vectors to *2 L* FFT length, multiply
    in frequency domain, IFFT, then keep the first *L* samples.
    """
    L = signal.size(0)
    n = nearest_power_of_two(2 * L - 1, round_up=True)
    y = torch.fft.irfft(torch.fft.rfft(signal, n) * torch.fft.rfft(filt, n), n)
    return y[:L]  # causal part


def td_convfft(
    f: torch.Tensor,  # [L, h]
    u: torch.Tensor,
) -> torch.Tensor:  # [B, H, L, h]
    """Causal FFT convolution for *projected* filters.

    Args:
        f: Spectral filters, shape **[L, h]**.  One length‑*L* kernel
        per head‑channel.  (Usually `h = head_dim`.)
        u: Batched input sequences, shape **[B, H, L, h]**.

    Returns:
        Tensor of shape **[B, H, L, h]** containing the causal
        convolution of every `(B,H,h)` stream with its corresponding
        filter `f[:, h]`.

    Vectorization order: **channels → heads → batch** (fastest‑changing
    dimension first) so the operation is fused into a single CUDA
    kernel when `torch.compile`/`vmap` is used.
    """
    causal = lambda sig, ker: fftconvolve(sig, ker)[: sig.shape[0]]
    cmap = torch.vmap(causal, in_dims=(1, 1), out_dims=0)   # over h
    hmap = torch.vmap(cmap, in_dims=(0, None), out_dims=0)  # over H
    bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)  # over B
    return bmap(u, f).permute(0, 1, 3, 2)


def bhld_convfft(
    v: torch.Tensor,  # [L, K]
    u: torch.Tensor,
) -> torch.Tensor:  # [B, H, L, h]
    """Causal FFT convolution keeping **K** separate kernels.

    Args:
        v: Spectral filters of shape **[L, K]** (K kernels shared by all
        heads and channels).
        u: Inputs of shape **[B, H, L, h]**.

    Returns:
        Tensor **[B, H, L, K, h]** – causal convolution of every
        `(B,H,h)` stream with *each* of the K kernels.

    The function vmaps in the order: **kernel K → channel h → head H → batch B**.
    """
    causal = lambda sig, ker: fftconvolve(sig, ker)[: sig.shape[0]]  # (L,)×(L,)→(L,)
    kmap = torch.vmap(causal, in_dims=(1, None), out_dims=0)  # over K
    cmap = torch.vmap(kmap, in_dims=(None, 1), out_dims=1)    # over h
    hmap = torch.vmap(cmap, in_dims=(None, 0), out_dims=0)    # over H
    bmap = torch.vmap(hmap, in_dims=(None, 0), out_dims=0)    # over B
    return bmap(v, u).permute(0, 1, 4, 2, 3)


# Use imported functions
td_out = td_convfft(f_proj, u)  # [B, H, L, h]

# brute force td
td_brute = torch.empty_like(td_out)
for b in range(B):
    for h_ in range(H):
        for c in range(h):
            td_brute[b, h_, :, c] = convfft(u[b, h_, :, c], f_proj[:, c])

print("TD max abs diff:", (td_out - td_brute).abs().max().item())

# Use imported functions
bhld_out = bhld_convfft(f_k, u)  # [B, H, L, K, h]

# brute force bhld
bhld_brute = torch.empty_like(bhld_out)
for b in range(B):
    for h_ in range(H):
        for k_ in range(K):
            for c in range(h):
                bhld_brute[b, h_, :, k_, c] = convfft(u[b, h_, :, c], f_k[:, k_])

print("BHLD max abs diff:", (bhld_out - bhld_brute).abs().max().item())
