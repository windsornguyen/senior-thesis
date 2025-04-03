import time
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Define Helper Functions
# ------------------------------

## Numerically Stable Combine Function (Already in Log Space)
def combine_fn(x, y):
    """
    Combines two sets of values (m, log_N, log_D) in a numerically stable manner in log domain.
    
    Args:
        x: Tuple of (m_x, log_N_x, log_D_x) - previous cumulative values.
        y: Tuple of (m_y, log_N_y, log_D_y) - current values to combine.
    
    Returns:
        Tuple of (m_new, log_N_new, log_D_new) - combined values in log domain.
    """
    m_x, log_N_x, log_D_x = x
    m_y, log_N_y, log_D_y = y

    # Compute new max for stability
    m_new = jnp.maximum(m_x, m_y)

    # Shift values to new log-domain reference
    log_N_x_shifted = log_N_x + (m_x - m_new)[..., None]
    log_N_y_shifted = log_N_y + (m_y - m_new)[..., None]

    log_D_x_shifted = log_D_x + (m_x - m_new)
    log_D_y_shifted = log_D_y + (m_y - m_new)

    # Use logsumexp for stable addition in log domain
    log_N_new = jsp.special.logsumexp(jnp.stack([log_N_x_shifted, log_N_y_shifted]), axis=0)
    log_D_new = jsp.special.logsumexp(jnp.stack([log_D_x_shifted, log_D_y_shifted]), axis=0)

    return m_new, log_N_new, log_D_new

## Scan-Based Attention Function (Corrected for Log Space)
@jax.jit
def scan_attention_fn(Q, K, V):
    """
    Computes attention using a scan-based approach in log space for numerical stability.

    Args:
        Q: Query tensor of shape (B, H, T, P), representing log(Q_feat).
        K: Key tensor of shape (B, H, T, P), representing log(K_feat).
        V: Value tensor of shape (B, H, T, P), in linear space.

    Returns:
        Output tensor of shape (B, H, T, P) - attention-weighted values.
    """
    # Inputs are already in log space
    log_Q = Q  # log(Q_feat)
    log_K = K  # log(K_feat)

    epsilon = 1e-6

    # Initial values for the scan
    m_initial = log_K  # Shape: (B, H, T, P)
    # Numerator: Compute log(V * K_feat) = log(|V|) + log(K_feat), handling V's sign
    log_V = jnp.where(V > 0, jnp.log(jnp.abs(V) + epsilon), jnp.log(jnp.abs(V) + epsilon))
    sign_V = jnp.sign(V)  # Shape: (B, H, T, P)
    # Reshape for broadcasting: (B, H, T, P) -> (B, H, T, P, 1)
    log_V_expanded = log_V[..., None]  # Shape: (B, H, T, P, 1)
    # Reshape log_K: (B, H, T, P) -> (B, H, T, 1, P)
    log_K_expanded = log_K[..., None, :]  # Shape: (B, H, T, 1, P)
    log_num_initial = log_V_expanded + log_K_expanded  # Shape: (B, H, T, P, P)
    log_denom_initial = log_K  # Shape: (B, H, T, P)

    # Perform the associative scan in log space
    tuple_initial = (m_initial, log_num_initial, log_denom_initial)
    m_cum, log_num_cum, log_denom_cum = jax.lax.associative_scan(combine_fn, tuple_initial, axis=2)

    # Compute the output: Y_num / Y_den
    # log(Q_feat * num_cum) = log_Q + log_num_cum
    # log_Q: (B, H, T, P) -> (B, H, T, P, 1)
    log_Q_expanded = log_Q[..., None]  # Shape: (B, H, T, P, 1)
    log_Y_num = log_Q_expanded + log_num_cum  # Shape: (B, H, T, P, P)
    # log_Y_den = log(sum_p(Q_feat * denom_cum))
    log_Q_plus_denom = log_Q + log_denom_cum  # Shape: (B, H, T, P)
    log_Y_den = jsp.special.logsumexp(log_Q_plus_denom, axis=-1)  # Shape: (B, H, T)

    # Compute the final output in linear space
    Y_num = sign_V[..., None] * jnp.exp(log_Y_num)  # Shape: (B, H, T, P, P)
    Y_den = jnp.exp(log_Y_den)  # Shape: (B, H, T)
    # Sum over the second P dimension to get output shape (B, H, T, P)
    output = jnp.sum(Y_num, axis=-2) / (Y_den[..., None] + epsilon)  # Shape: (B, H, T, P)
    return output

## Benchmarking Routine
def benchmark_fn(fn, Q, K, V, n_iters=50, warmup_per_iter=4):
    """
    Benchmarks the given function by running warm-up iterations and timing execution.

    Args:
        fn: Function to benchmark (e.g., scan_attention_fn).
        Q, K, V: Input tensors for the function.
        n_iters: Number of iterations to time.
        warmup_per_iter: Number of warm-up runs per iteration.

    Returns:
        Numpy array of runtimes in seconds for each iteration.
    """
    times = []
    for _ in range(n_iters):
        # Warm-up runs to stabilize performance
        for _ in range(warmup_per_iter):
            fn(Q, K, V).block_until_ready()
        start = time.time()
        out = fn(Q, K, V)
        out.block_until_ready()
        times.append(time.time() - start)
        time.sleep(0.1)  # Prevent power throttling or overheating
    return np.array(times)

# ------------------------------
# 2. Benchmarking Setup
# ------------------------------

if __name__ == "__main__":
    # Display available JAX devices (e.g., CPU, GPU)
    print("JAX devices:", jax.devices())

    # Configuration
    B, H, P = 1, 1, 128  # Batch size, heads, feature dimension
    T_values = [2**i for i in range(7, 19)]  # Sequence lengths: 128 to 131072 (128k)
    n_iters = 50  # Number of iterations for timing

    # Initialize random key for reproducibility
    key = jax.random.PRNGKey(42)

    # Lists to store benchmark results
    scan_means, scan_stds = [], []

    # Benchmark for each sequence length
    for T in T_values:
        print(f"\nBenchmarking for T = {T}")

        # Generate random data (in log space for Q and K)
        key, subkey = jax.random.split(key)
        Q = jax.random.normal(subkey, (B, H, T, P))  # log(Q_feat)
        key, subkey = jax.random.split(key)
        K = jax.random.normal(subkey, (B, H, T, P))  # log(K_feat)
        key, subkey = jax.random.split(key)
        V = jax.random.normal(subkey, (B, H, T, P))  # V in linear space

        # Benchmark scan-based attention
        scan_times = benchmark_fn(scan_attention_fn, Q, K, V, n_iters=n_iters)
        scan_mean = scan_times.mean() * 1000  # Convert to milliseconds
        scan_std = scan_times.std() * 1000    # Convert to milliseconds
        scan_means.append(scan_mean)
        scan_stds.append(scan_std)
        print(f"  Scan-based attention: mean = {scan_mean:.3f} ms, std = {scan_std:.3f} ms")

    # ------------------------------
    # 3. Scaling Analysis
    # ------------------------------
    print("\nScaling Analysis for Scan-Based Attention:")
    for i in range(len(T_values) - 1):
        T = T_values[i]
        k = np.log2(T)  # Since T = 2^k
        measured_ratio = scan_means[i + 1] / scan_means[i]
        expected_log_ratio = 1 + 1 / k  # Expected for O(log T) scaling
        print(
            f"T={T} to T={T_values[i+1]}: measured ratio = {measured_ratio:.3f}, "
            f"expected log scaling = {expected_log_ratio:.3f}, expected linear = 2.000"
        )

    # ------------------------------
    # 4. Plotting Results
    # ------------------------------
    T_values_np = np.array(T_values)

    plt.figure(figsize=(10, 6))
    plt.errorbar(T_values_np, scan_means, yerr=scan_stds, fmt="-o", label="Scan-Based Attention")
    plt.xlabel("Sequence Length (T)")
    plt.ylabel("Runtime per iteration (ms)")
    plt.title("Benchmark: Scan-Based Attention")
    plt.legend()
    plt.xscale("log")  # Logarithmic scale for better visualization
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
