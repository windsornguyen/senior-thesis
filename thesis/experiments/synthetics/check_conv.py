import torch
import torch.nn.functional as F
import torchaudio.functional as taf
from torchaudio.functional import fftconvolve
import time
import sys  # For flushing output


# First implementation (nested vmap)
def cheby_conv_nested(coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if coeffs.dim() != 1:
        raise ValueError("coeffs must be 1D tensor of shape [K]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")
    inputs_perm = inputs.movedim(2, -1)  # [B, H, D, L]
    causal = lambda sig, ker: taf.convolve(sig, ker, mode="full")[..., : sig.shape[-1]]
    cmap = torch.vmap(causal, in_dims=(0, None), out_dims=0)  # over D
    hmap = torch.vmap(cmap, in_dims=(1, None), out_dims=1)  # over H
    bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)  # over B
    y_perm = bmap(inputs_perm, coeffs)  # [B, H, D, L]
    return y_perm.movedim(-1, 2)  # [B, H, L, D]


# Second implementation (single vmap with reshape)
def cheby_conv_flat(coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if coeffs.dim() != 1:
        raise ValueError("coeffs must be 1D tensor of shape [K]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")
    B, H, L, D = inputs.shape
    x = inputs.transpose(-2, -1).reshape(-1, L)
    causal = lambda sig, ker: taf.convolve(sig, ker, mode="full")[..., : sig.shape[-1]]
    vmap_causal = torch.vmap(causal, in_dims=(0, None), out_dims=0)
    y = vmap_causal(x, coeffs)  # [BHD, L]
    return y.reshape(B, H, D, L).transpose(-2, -1)  # [B, H, L, D]


# Third implementation (fftconvolve)
def cheby_conv_fft(coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if coeffs.dim() != 1:
        raise ValueError("coeffs must be 1D tensor of shape [K]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")
    B, H, L, D = inputs.shape
    x = inputs.transpose(-2, -1).reshape(-1, L)
    causal = lambda sig, ker: fftconvolve(sig, ker, mode="full")[..., : sig.shape[-1]]
    vmap_causal = torch.vmap(causal, in_dims=(0, None), out_dims=0)
    y = vmap_causal(x, coeffs)  # [BHD, L]
    return y.reshape(B, H, D, L).transpose(-2, -1)  # [B, H, L, D]


# Fourth implementation (conv1d)
def cheby_conv_conv1d(coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if coeffs.dim() != 1:
        raise ValueError("coeffs must be 1D tensor of shape [K]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")
    B, H, L, D = inputs.shape
    K = coeffs.shape[0]
    x = inputs.transpose(-2, -1).reshape(-1, L)

    # conv1d expects (N, C_in, L_in) and kernel (C_out, C_in/groups, L_ker)
    # We treat BHD as N, C_in=1, C_out=1
    # Need to flip kernel for convolution vs cross-correlation
    kernel = coeffs.flip(-1).view(1, 1, K)  # Shape [1, 1, K]
    # Pad input for causal convolution: (K-1) on the left
    padding = K - 1  # Corrected padding for conv1d

    def causal_conv1d(sig, ker):
        sig_reshaped = sig.view(1, 1, L)  # Shape [1, 1, L]
        # Apply conv1d
        # The output length will be L_in + 2*padding - dilation*(kernel_size-1) = L + 2*(K-1) - 1*(K-1) = L + K - 1. Incorrect!
        # With padding=P, output L_out = L_in + 2*P - K + 1
        # We want L_out = L. So L = L + 2*P - K + 1 => 2*P = K - 1.
        # Padding must be integer. If K is even, this is not possible with conv1d symmetric padding.
        # Let's stick to torchaudio's definition which uses asymmetric padding.
        # F.pad can handle asymmetric padding.
        sig_padded = F.pad(sig_reshaped, (K - 1, 0))  # Pad K-1 on the left
        # Now call conv1d with padding=0
        output = F.conv1d(sig_padded, ker, padding=0)
        return output.view(L)  # Shape [L]

    # vmap over the BHD dimension
    vmap_conv1d = torch.vmap(causal_conv1d, in_dims=(0, None), out_dims=0)
    y = vmap_conv1d(x, kernel)  # [BHD, L]
    return y.reshape(B, H, D, L).transpose(-2, -1)  # [B, H, L, D]


# Implementation using convolve
def cheby_conv_convolve_impl(coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if coeffs.dim() != 1:
        raise ValueError("coeffs must be 1D tensor of shape [K]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")
    B, H, L, D = inputs.shape
    x = inputs.transpose(-2, -1).reshape(-1, L)
    causal = lambda sig, ker: taf.convolve(sig, ker, mode="full")[..., : sig.shape[-1]]
    vmap_causal = torch.vmap(causal, in_dims=(0, None), out_dims=0)
    y = vmap_causal(x, coeffs)  # [BHD, L]
    return y.reshape(B, H, D, L).transpose(-2, -1)  # [B, H, L, D]


# Implementation using fftconvolve
def cheby_conv_fftconvolve_impl(coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if coeffs.dim() != 1:
        raise ValueError("coeffs must be 1D tensor of shape [K]")
    if inputs.dim() != 4:
        raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")
    B, H, L, D = inputs.shape
    x = inputs.transpose(-2, -1).reshape(-1, L)
    causal = lambda sig, ker: fftconvolve(sig, ker, mode="full")[..., : sig.shape[-1]]
    vmap_causal = torch.vmap(causal, in_dims=(0, None), out_dims=0)
    y = vmap_causal(x, coeffs)  # [BHD, L]
    return y.reshape(B, H, D, L).transpose(-2, -1)  # [B, H, L, D]


# Compile the functions
cheby_conv_convolve = torch.compile(cheby_conv_convolve_impl, mode="reduce-overhead", fullgraph=False)
cheby_conv_fftconvolve = torch.compile(cheby_conv_fftconvolve_impl, mode="reduce-overhead", fullgraph=False)


# Benchmarking function
def benchmark(func, coeffs, inputs, num_iterations=10, num_warmup=3):
    # Warm-up runs (allow for compilation and stabilization)
    for _ in range(num_warmup):
        func(coeffs, inputs)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iterations):
        func(coeffs, inputs)
        torch.cuda.synchronize()  # Ensure GPU work is complete
    end_time = time.time()

    return (end_time - start_time) / num_iterations * 1000  # Time in ms


# Setup
torch.manual_seed(1746)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available")

# Fixed dimensions (except L)
B, H, D, K = 32, 16, 64, 10
num_iterations = 10  # Increase measurement iterations
num_warmup = 3  # Increase warmup iterations

# Sequence lengths to test
L_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

results = []

print(f"Benchmarking torch.compiled convolve vs fftconvolve for B={B}, H={H}, D={D}, K={K}")
print("-" * 60)
print(f"{'L':>8} {'Conv Time (ms)':>18} {'FFT Time (ms)':>18} {'Faster':>10}")
print("-" * 60)

# Run benchmarks for different L
for L in L_values:
    print(f"Running L = {L}...", end=" ")
    sys.stdout.flush()
    coeffs = torch.randn(K).to(device)
    inputs = torch.randn(B, H, L, D).to(device)

    # Benchmark the compiled functions
    conv_time = benchmark(cheby_conv_convolve, coeffs, inputs, num_iterations, num_warmup)
    fft_time = benchmark(cheby_conv_fftconvolve, coeffs, inputs, num_iterations, num_warmup)

    faster = "FFT" if fft_time < conv_time else "Conv"
    results.append((L, conv_time, fft_time, faster))
    print(f"Conv: {conv_time:.2f}ms, FFT: {fft_time:.2f}ms -> {faster}")

# Print summary table
print("\nSummary Table:")
print("-" * 60)
print(f"{'L':>8} {'Conv Time (ms)':>18} {'FFT Time (ms)':>18} {'Faster':>10}")
print("-" * 60)
for L, conv_time, fft_time, faster in results:
    print(f"{L:>8d} {conv_time:>18.3f} {fft_time:>18.3f} {faster:>10}")
print("-" * 60)

# Find crossover point
crossover_L = -1
for i in range(len(results) - 1):
    if results[i][3] == "Conv" and results[i + 1][3] == "FFT":
        crossover_L = results[i + 1][0]
        break
    elif results[i][3] == "FFT" and results[i + 1][3] == "Conv":  # Should not happen typically
        print("Warning: Crossover direction seems reversed?")
        crossover_L = results[i + 1][0]
        break

if results[0][3] == "FFT":
    print("\nFFT was faster even for the smallest L tested.")
elif crossover_L != -1:
    print(f"\nApproximate crossover point: FFT becomes faster around L = {crossover_L}")
elif results[-1][3] == "Conv":
    print("\nConv remained faster for all tested L values.")
else:
    print("\nCould not determine a clear crossover point within the tested range.")


print("\n" + "=" * 60)
print("Replicating specific bare run (L=1024, K=10, no torch.compile)")
print("=" * 60)

# Original parameters for the specific run
B_bare, H_bare, L_bare, D_bare, K_bare = 32, 16, 1024, 64, 10
num_iterations_bare = 4
num_warmup_bare = 1  # Original benchmark had 1 warm-up run implicitly
seed_bare = 1746

torch.manual_seed(seed_bare)
coeffs_bare = torch.randn(K_bare).to(device)
inputs_bare = torch.randn(B_bare, H_bare, L_bare, D_bare).to(device)

print(f"Params: B={B_bare}, H={H_bare}, L={L_bare}, D={D_bare}, K={K_bare}")
print(f"Seed: {seed_bare}, Iterations: {num_iterations_bare}, Warmup: {num_warmup_bare}")

# Benchmark the *non-compiled* implementations
conv_time_bare = benchmark(cheby_conv_convolve_impl, coeffs_bare, inputs_bare, num_iterations_bare, num_warmup_bare)
fft_time_bare = benchmark(cheby_conv_fftconvolve_impl, coeffs_bare, inputs_bare, num_iterations_bare, num_warmup_bare)

print(f"\nBare run convolve time:    {conv_time_bare:.3f} ms")
print(f"Bare run fftconvolve time: {fft_time_bare:.3f} ms")
faster_bare = "FFT" if fft_time_bare < conv_time_bare else "Conv"
print(f"Result: {faster_bare} was faster.")
if faster_bare == "Conv":
    print(
        f"Ratio (Conv / FFT): {conv_time_bare / fft_time_bare:.2f} (FFT is {fft_time_bare / conv_time_bare:.2f}x slower)"
    )
else:
    print(
        f"Ratio (FFT / Conv): {fft_time_bare / conv_time_bare:.2f} (Conv is {conv_time_bare / fft_time_bare:.2f}x slower)"
    )
print("=" * 60)
