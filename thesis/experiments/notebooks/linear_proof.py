import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dummy implementation for get_monic_chebyshev_coeffs
def get_monic_chebyshev_coeffs(n: int) -> torch.Tensor:
    return torch.ones(n + 1, device=device)


# Revised LearnableSpectralFilters that now takes fixed dims (proj_dim, proj_dim)
class LearnableSpectralFilters(nn.Module):
    def __init__(self, d_in: int, d_out: int, use_hankel_L: bool, device):
        super().__init__()
        # Initialize the filters as a learnable weight of shape (proj_dim, proj_dim)
        self.weight = nn.Parameter(torch.randn(d_in, d_out, device=device))

    def forward(self) -> torch.Tensor:
        return self.weight


# Revised SpectralAttention
class SpectralAttention(nn.Module):
    """
    Revised simplified spectral attention:
    - Project input from n_embd to a fixed low dimension (proj_dim) instead of seq_len.
    - Use learnable spectral filters of shape (proj_dim, proj_dim).
    - Perform cumulative linear attention in this fixed space.
    """

    def __init__(self, seq_len: int, n_embd: int, proj_dim: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.proj_dim = proj_dim
        self.p_coeffs = get_monic_chebyshev_coeffs(seq_len - 1)
        # Spectral filters now operate on the fixed low-dim space
        self.Q_filt = LearnableSpectralFilters(proj_dim, proj_dim, use_hankel_L, device)
        self.K_filt = LearnableSpectralFilters(proj_dim, proj_dim, use_hankel_L, device)
        # Project input into fixed low-dim space (n_embd -> proj_dim)
        self.in_proj = nn.Linear(n_embd, proj_dim).to(device)
        self.v_proj = nn.Linear(n_embd, proj_dim).to(device)
        # Map back from proj_dim to model dimension (proj_dim -> n_embd)
        self.o_proj = nn.Linear(proj_dim, n_embd).to(device)
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embd)
        B, T, D = x.shape
        # Project input to fixed low-dim space: (B, T, proj_dim)
        x_proj = self.in_proj(x)
        # Apply spectral filters to x_proj to get Q and K: (B, T, proj_dim)
        Q = x_proj @ self.Q_filt()
        K = x_proj @ self.K_filt()
        # Project x (or x_proj) to get V; here we choose to project x directly:
        V = self.v_proj(x)

        # Compute the outer product at each time step: (B, T, proj_dim, proj_dim)
        Z = torch.einsum("bti, btj -> btij", V, K)
        decay = self.decay.view(1, T, 1, 1)
        Z = Z * decay
        # Cumulative sum over time dimension T: (B, T, proj_dim, proj_dim)
        H = torch.cumsum(Z, dim=1)
        # Multiply Q with H: contract over the proj_dim dimension -> (B, T, proj_dim)
        Y = torch.einsum("bti, btij -> btj", Q, H)
        # Project back to the original model dimension: (B, T, n_embd)
        return self.o_proj(Y)


# Define the doubling experiment parameters
seq_lengths = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]  # Doubling sequence lengths (adjust as per your GPU memory)
times = []

B = 1
n_embd = 32  # input embedding dim
proj_dim = 16  # fixed projection dimension (choose a value much smaller than T)
iters = 5

# Warm-up runs
for T in seq_lengths:
    model = SpectralAttention(seq_len=T, n_embd=n_embd, proj_dim=proj_dim, device=device)
    x = torch.randn(B, T, n_embd, device=device)
    with torch.no_grad():
        _ = model(x)
        torch.cuda.synchronize()

# Timing runs with sleep to avoid throttling
for T in seq_lengths:
    model = SpectralAttention(seq_len=T, n_embd=n_embd, proj_dim=proj_dim, device=device)
    model.to(device)
    x = torch.randn(B, T, n_embd, device=device)
    elapsed = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        elapsed += start.elapsed_time(end)  # Time in ms
        time.sleep(1)  # sleep to reduce throttling effects
    avg_time = elapsed / iters
    times.append(avg_time)
    print(f"Sequence Length: {T}, Avg Inference Time: {avg_time:.2f} ms")

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(seq_lengths, times, marker="o")
plt.xlabel("Sequence Length")
plt.ylabel("Average Inference Time (ms)")
plt.title("Time Complexity Analysis of SpectralAttention")
plt.grid(True)
plt.show()
