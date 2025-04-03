import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
import torchaudio.functional as F
import jax.scipy.signal as jsp
from torch import vmap

torch.set_float32_matmul_precision("high")

# **Configure Devices**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
jax.config.update("jax_platform_name", "gpu" if torch.cuda.is_available() else "cpu")

# **Define Input Shapes**
batch_size = 4
num_heads = 24
sequence_length = 1024
head_dim = 4

# **Generate Random Inputs**
filters_np = np.random.randn(num_heads, sequence_length).astype(np.float32)
k_np = np.random.randn(batch_size, num_heads, sequence_length, head_dim).astype(np.float32)

# Convert to JAX arrays
spectral_basis_jax = jnp.array(filters_np.T)  # [sequence_length, num_heads]
k_jax = jnp.array(k_np)  # [batch_size, num_heads, sequence_length, head_dim]

# Convert to PyTorch tensors
filters_torch = torch.tensor(filters_np).to(device)
k_torch = torch.tensor(k_np).to(device)

# **JAX Implementation**
tr_conv = lambda x, y: jsp.convolve(x, y, method="fft")[: x.shape[0]]


def conv_per_head_jax(f_h, k_h):
    conv_dims = jax.vmap(tr_conv, in_axes=(1, None), out_axes=1)
    return jax.vmap(conv_dims, in_axes=(0, None), out_axes=0)(k_h, f_h)


@jax.jit
def conv_spectral_jax(spectral_basis, k):
    filters = spectral_basis.T
    k_conv = jax.vmap(conv_per_head_jax, in_axes=(0, 1), out_axes=1)(filters, k)
    return k_conv


# **PyTorch torchaudio Implementation**
def conv_per_head_torch_ta(f_h, k_h):
    batch_size, seq_len, head_dim = k_h.shape
    k_h_reshaped = k_h.reshape(-1, seq_len)
    result = F.fftconvolve(k_h_reshaped, f_h.unsqueeze(0), mode="same")
    return result.reshape(batch_size, head_dim, seq_len).permute(0, 2, 1)


conv_spectral_torch_ta_base = vmap(conv_per_head_torch_ta, in_dims=(0, 1), out_dims=1)
conv_spectral_torch_ta = torch.compile(conv_spectral_torch_ta_base)


# **PyTorch FFT Implementation**
def conv_per_head_torch_fft(f_h, k_h):
    n = k_h.shape[1]
    f_h_fft = torch.fft.rfft(f_h, n=n)
    k_h_flat = k_h.reshape(-1, n)
    k_h_fft = torch.fft.rfft(k_h_flat, n=n)
    conv_fft = k_h_fft * f_h_fft.unsqueeze(0)
    result = torch.fft.irfft(conv_fft, n=n)[:, :n]
    return result.reshape(batch_size, n, head_dim)


conv_spectral_torch_fft_base = vmap(conv_per_head_torch_fft, in_dims=(0, 1), out_dims=1)
conv_spectral_torch_fft = torch.compile(conv_spectral_torch_fft_base)

# **Pre-Compile Functions**
_ = conv_spectral_jax(spectral_basis_jax, k_jax).block_until_ready()
_ = conv_spectral_torch_ta(filters_torch, k_torch)
_ = conv_spectral_torch_fft(filters_torch, k_torch)
if device.type == "cuda":
    torch.cuda.synchronize()

# **Equality Check**
jax_result = conv_spectral_jax(spectral_basis_jax, k_jax).block_until_ready()
torch_ta_result = conv_spectral_torch_ta(filters_torch, k_torch)
torch_fft_result = conv_spectral_torch_fft(filters_torch, k_torch)
if device.type == "cuda":
    torch.cuda.synchronize()

# Convert to NumPy for comparison
jax_result_np = np.array(jax_result)
torch_ta_result_np = torch_ta_result.cpu().numpy()
torch_fft_result_np = torch_fft_result.cpu().numpy()

# Print detailed comparison
print("\nDetailed comparison of outputs:")
print(f"Max difference between JAX and torchaudio: {np.max(np.abs(jax_result_np - torch_ta_result_np))}")
print(f"Mean difference between JAX and torchaudio: {np.mean(np.abs(jax_result_np - torch_ta_result_np))}")
print(f"Max difference between JAX and PyTorch FFT: {np.max(np.abs(jax_result_np - torch_fft_result_np))}")
print(f"Mean difference between JAX and PyTorch FFT: {np.mean(np.abs(jax_result_np - torch_fft_result_np))}")
print(f"Max difference between torchaudio and PyTorch FFT: {np.max(np.abs(torch_ta_result_np - torch_fft_result_np))}")
print(
    f"Mean difference between torchaudio and PyTorch FFT: {np.mean(np.abs(torch_ta_result_np - torch_fft_result_np))}"
)

# Check equality within tolerance
tolerance = 1e-5
if not (
    np.allclose(jax_result_np, torch_ta_result_np, atol=tolerance)
    and np.allclose(jax_result_np, torch_fft_result_np, atol=tolerance)
):
    print("Outputs are not equal within tolerance!")
else:
    print("All methods produce equivalent outputs within tolerance.")


# **Timing Functions**
def time_jax():
    start = time.perf_counter()
    result = conv_spectral_jax(spectral_basis_jax, k_jax)
    result.block_until_ready()
    end = time.perf_counter()
    return end - start


def time_torch_ta():
    start = time.perf_counter()
    result = conv_spectral_torch_ta(filters_torch, k_torch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start


def time_torch_fft():
    start = time.perf_counter()
    result = conv_spectral_torch_fft(filters_torch, k_torch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start


# **Warmup Function**
def warmup(method, inputs):
    if method == "jax":
        for _ in range(4):
            _ = conv_spectral_jax(*inputs).block_until_ready()
            time.sleep(0.1)
    elif method == "torch_ta":
        for _ in range(4):
            _ = conv_spectral_torch_ta(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            time.sleep(0.1)
    elif method == "torch_fft":
        for _ in range(4):
            _ = conv_spectral_torch_fft(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            time.sleep(0.1)


# **Benchmarking Loop**
num_repeats = 10
jax_times = []
torch_ta_times = []
torch_fft_times = []

for _ in range(num_repeats):
    # JAX: Warmup then time
    warmup("jax", (spectral_basis_jax, k_jax))
    jax_time = time_jax()
    jax_times.append(jax_time)
    time.sleep(0.1)

    # PyTorch torchaudio: Warmup then time
    warmup("torch_ta", (filters_torch, k_torch))
    torch_ta_time = time_torch_ta()
    torch_ta_times.append(torch_ta_time)
    time.sleep(0.1)

    # PyTorch FFT: Warmup then time
    warmup("torch_fft", (filters_torch, k_torch))
    torch_fft_time = time_torch_fft()
    torch_fft_times.append(torch_fft_time)
    time.sleep(0.1)

# **Compute and Display Results**
avg_jax_time = sum(jax_times) / num_repeats
avg_torch_ta_time = sum(torch_ta_times) / num_repeats
avg_torch_fft_time = sum(torch_fft_times) / num_repeats

print(f"Average JAX FFT Convolution time: {avg_jax_time:.6f} seconds")
print(f"Average PyTorch torchaudio Convolution time: {avg_torch_ta_time:.6f} seconds")
print(f"Average PyTorch FFT Convolution time: {avg_torch_fft_time:.6f} seconds")

fastest = min(avg_jax_time, avg_torch_ta_time, avg_torch_fft_time)
if fastest == avg_jax_time:
    print("JAX is the fastest")
    print(f"  Faster than torchaudio by {avg_torch_ta_time - avg_jax_time:.6f} seconds")
    print(f"  Faster than PyTorch FFT by {avg_torch_fft_time - avg_jax_time:.6f} seconds")
elif fastest == avg_torch_ta_time:
    print("PyTorch torchaudio is the fastest")
    print(f"  Faster than JAX by {avg_jax_time - avg_torch_ta_time:.6f} seconds")
    print(f"  Faster than PyTorch FFT by {avg_torch_fft_time - avg_torch_ta_time:.6f} seconds")
else:
    print("PyTorch FFT is the fastest")
    print(f"  Faster than JAX by {avg_jax_time - avg_torch_fft_time:.6f} seconds")
    print(f"  Faster than torchaudio by {avg_torch_ta_time - avg_torch_fft_time:.6f} seconds")
