import torch

def convfft(signal: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """
    Causal 1D convolution via FFT.
    signal: (..., L)
    filt:   (L,)
    returns: (..., L)
    """
    L = signal.shape[-1]
    n = 2 * L - 1
    # rfft along last dim, broadcasts filt → same shape
    S = torch.fft.rfft(signal, n=n)
    F = torch.fft.rfft(filt,   n=n)
    y = torch.fft.irfft(S * F, n=n)
    return y[..., :L]

def main():
    torch.manual_seed(0)

    # shape parameters
    B, L, K = 2, 64, 5

    # random batch of signals
    u = torch.randn(B, L)

    # K random filters of length L
    phi = torch.randn(K, L)

    # random scalar weights for each filter
    w = torch.randn(K)

    # 1) compute each conv separately, then weight+sum
    convs = [convfft(u, phi[k]) for k in range(K)]
    weighted_sum = sum(w[k] * convs[k] for k in range(K))  # shape [B,L]

    # 2) build one “combined” filter, then conv once
    phi_comb = sum(w[k] * phi[k] for k in range(K))        # shape [L]
    combined_conv = convfft(u, phi_comb)                   # shape [B,L]

    # compare
    diff = (weighted_sum - combined_conv).abs()
    print(f"  max abs diff: {diff.max().item():.3e}")
    print("allclose:", torch.allclose(weighted_sum, combined_conv, atol=1e-6))

if __name__ == "__main__":
    main()
