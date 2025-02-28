import torch
from hankel_generator import build_hankel_generator, build_hankel_matrix


def low_rank_factorization(Z: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a truncated SVD of matrix Z and returns factors U and V such that Z ≈ U @ V^T.

    Parameters:
        Z (torch.Tensor): The input Hankel matrix.
        rank (int): The target rank (semiseparable rank, as observed experimentally).

    Returns:
        tuple: (U_trunc, V_trunc) where U_trunc has shape (n, rank) and V_trunc has shape (n, rank).
    """
    # Compute the SVD of Z
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    # Truncate to the desired rank
    U_trunc = U[:, :rank] * torch.sqrt(S[:rank])
    V_trunc = Vh[:rank, :].T * torch.sqrt(S[:rank])
    return U_trunc, V_trunc


def fast_multiply(U: torch.Tensor, V: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Multiplies the approximated Hankel matrix Z ≈ U @ V^T by vector x.

    Parameters:
        U (torch.Tensor): Factor matrix of shape (n, r).
        V (torch.Tensor): Factor matrix of shape (n, r).
        x (torch.Tensor): Vector of shape (n,).

    Returns:
        torch.Tensor: The product Zx approximated as U @ (V^T @ x).
    """
    return U @ (V.T @ x)


# get the hankel matrix (og function)
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


# Example usage:
seq_len = 256  # for example
device = "cpu"
a, b, c = build_hankel_generator(seq_len)
Z = build_hankel_matrix(seq_len, a, b, c)

# Use the semiseparable rank from your experiments, say 10:
rank = 10
U_trunc, V_trunc = low_rank_factorization(Z, rank)

# Now, to multiply Z by a vector x fast:
x = torch.randn(seq_len, dtype=torch.float32, device=device)
Zx_fast = fast_multiply(U_trunc, V_trunc, x)
Zx_dense = Z @ x

print("Relative error:", torch.norm(Zx_dense - Zx_fast) / torch.norm(Zx_dense))


def benchmark_semiseparable():
    """
    Comprehensive benchmark comparing semiseparable matrix multiplication against naive matmul.
    Includes proper CUDA warmup, power throttling mitigation, and error analysis.
    """
    import time
    import statistics
    from typing import List, Tuple

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    seq_lengths = [256, 512, 1024, 2048]
    ranks = [4, 8, 16, 32]
    n_warmup = 10
    n_trials = 20
    cooldown_time = 0.1  # seconds between trials to prevent throttling

    def run_trial(seq_len: int, rank: int) -> Tuple[float, float, float, float]:
        # Generate problem instance
        a, b, c = build_hankel_generator(seq_len)
        Z = build_hankel_matrix(seq_len, a, b, c).to(device)
        x = torch.randn(seq_len, dtype=torch.float32, device=device)

        # Get low-rank factors
        U_trunc, V_trunc = low_rank_factorization(Z, rank)
        U_trunc, V_trunc = U_trunc.to(device), V_trunc.to(device)

        # Time fast multiply
        torch.cuda.synchronize()
        start = time.perf_counter()
        Zx_fast = fast_multiply(U_trunc, V_trunc, x)
        torch.cuda.synchronize()
        fast_time = time.perf_counter() - start

        # Time dense multiply
        torch.cuda.synchronize()
        start = time.perf_counter()
        Zx_dense = Z @ x
        torch.cuda.synchronize()
        dense_time = time.perf_counter() - start

        # Compute errors
        rel_error = torch.norm(Zx_dense - Zx_fast) / torch.norm(Zx_dense)

        # Memory usage in MB
        fast_memory = (U_trunc.nelement() + V_trunc.nelement()) * 4 / (1024 * 1024)
        dense_memory = Z.nelement() * 4 / (1024 * 1024)

        return fast_time, dense_time, rel_error.item(), fast_memory / dense_memory

    # Results storage
    results = {}

    print("Starting benchmarks...")
    print(f"Device: {device}")
    print(f"{'Seq Len':>8} {'Rank':>6} {'Speedup':>10} {'Rel Error':>12} {'Mem Ratio':>10}")
    print("-" * 50)

    # Warmup
    print("Warming up CUDA...")
    warmup_len = 512
    warmup_rank = 8
    for _ in range(n_warmup):
        _ = run_trial(warmup_len, warmup_rank)

    # Main benchmark loop
    for seq_len in seq_lengths:
        results[seq_len] = {}
        for rank in ranks:
            fast_times: List[float] = []
            dense_times: List[float] = []
            rel_errors: List[float] = []
            mem_ratios: List[float] = []

            for _ in range(n_trials):
                fast_t, dense_t, rel_err, mem_ratio = run_trial(seq_len, rank)
                fast_times.append(fast_t)
                dense_times.append(dense_t)
                rel_errors.append(rel_err)
                mem_ratios.append(mem_ratio)
                time.sleep(cooldown_time)  # Prevent throttling

            # Compute statistics
            avg_speedup = statistics.mean(dense_times) / statistics.mean(fast_times)
            avg_error = statistics.mean(rel_errors)
            avg_mem_ratio = statistics.mean(mem_ratios)

            results[seq_len][rank] = {"speedup": avg_speedup, "error": avg_error, "mem_ratio": avg_mem_ratio}

            print(f"{seq_len:>8} {rank:>6} {avg_speedup:>10.2f}x {avg_error:>12.2e} {avg_mem_ratio:>10.2f}")

    return results


if __name__ == "__main__":
    results = benchmark_semiseparable()
