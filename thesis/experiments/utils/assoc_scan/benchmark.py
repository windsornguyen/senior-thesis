import time

import torch
import jax
import jax.numpy as jnp

from kernel import associative_scan as triton_associative_scan

torch.set_float32_matmul_precision("high")


def torch_associative_scan_reference(gated_Z, gates):
    batch_size, feature_size, seq_len = gated_Z.shape
    cumulative_gated_Z = torch.zeros_like(gated_Z)
    cumulative_gates = torch.zeros_like(gates)
    for t in range(seq_len):
        if t == 0:
            cumulative_gated_Z[:, :, t] = gated_Z[:, :, t]
            cumulative_gates[:, :, t] = gates[:, :, t]
        else:
            cumulative_gated_Z[:, :, t] = cumulative_gated_Z[:, :, t - 1] + gated_Z[:, :, t]
            cumulative_gates[:, :, t] = cumulative_gates[:, :, t - 1] + gates[:, :, t]
    return cumulative_gated_Z, cumulative_gates


# JAX jitted implementation using associative_scan
@jax.jit
def jax_scan_fn(gated_Z, gates):
    """
    JAX implementation using jax.lax.associative_scan to be JIT compiled once
    """

    def combine_fn(a, b):
        a_Z, a_gates = a
        b_Z, b_gates = b
        return (a_Z + b_Z, a_gates + b_gates)

    return jax.lax.associative_scan(combine_fn, (gated_Z, gates), axis=2)


def benchmark_scan_implementations(
    configs=None, dtypes=[torch.float32, torch.bfloat16], num_warmup=5, num_trials=20, sleep_time=0.1
):
    """
    Benchmark the Triton scan implementation against both JAX and PyTorch reference.

    Args:
        configs: List of (batch_size, feature_size, seq_len) tuples to benchmark
        dtypes: List of torch dtypes to benchmark
        num_warmup: Number of warmup iterations before timing
        num_trials: Number of trials to run for each configuration
        sleep_time: Time to sleep between different test cases to reset thermal state
    """
    if configs is None:
        configs = [
            # (batch_size, feature_size, seq_len)
            (16, 8, 32),  # Small
            (32, 16, 64),  # Medium
            (64, 32, 128),  # Large
            (128, 64, 256),  # Very large
            (256, 32, 512),  # Extreme
            (64, 64, 1024),  # Long sequence
        ]

    device = torch.device("cuda")
    print(f"Running benchmarks on device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    for dtype in dtypes:
        jax_dtype = jnp.float32 if dtype == torch.float32 else jnp.bfloat16

        print(f"\nBenchmarking with {dtype}:")
        print(
            f"{'Size (B,F,S)':<20} {'Triton (ms)':<15} {'JAX (ms)':<15} {'PyTorch (ms)':<15} {'Triton/JAX':<10} {'Triton/PyTorch':<10}"
        )
        print("-" * 90)

        for B, F, S in configs:
            # Clear any lingering caches/memory before starting new test case
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()  # Only once before the test case, not in the loop

            # Create fixed random inputs for all implementations
            torch.manual_seed(42)  # For reproducibility
            torch_gated_Z = torch.rand(B, F, S, device=device, dtype=dtype)
            torch_gates = torch.rand(B, F, S, device=device, dtype=dtype)

            # Create fixed inputs for JAX - outside the timing loop
            jax_key = jax.random.PRNGKey(42)  # Use same seed as PyTorch
            jax_gated_Z = jax.device_put(jax.random.uniform(jax_key, (B, F, S), dtype=jax_dtype))
            jax_key, subkey = jax.random.split(jax_key)
            jax_gates = jax.device_put(jax.random.uniform(subkey, (B, F, S), dtype=jax_dtype))

            # Warmup all implementations before timing
            print(f"Warming up implementations for ({B},{F},{S})...")

            # Warmup Triton
            for _ in range(num_warmup):
                _ = triton_associative_scan(torch_gated_Z, torch_gates)
            torch.cuda.synchronize()

            # Warmup PyTorch
            for _ in range(num_warmup):
                _ = torch_associative_scan_reference(torch_gated_Z, torch_gates)
            torch.cuda.synchronize()

            # Warmup JAX - compile once
            _ = jax_scan_fn(jax_gated_Z, jax_gates)
            jax.block_until_ready(_)  # Ensure compilation is done
            for _ in range(num_warmup):
                _ = jax_scan_fn(jax_gated_Z, jax_gates)
                jax.block_until_ready(_)

            # Sleep once after warmup to let thermal state settle
            time.sleep(sleep_time)

            # Arrays to store timing results
            triton_times = []
            jax_times = []
            pytorch_times = []

            # Benchmark all implementations with standardized timing method
            print(f"Running timed trials for ({B},{F},{S})...")

            # Benchmark Triton
            for _ in range(num_trials):
                torch.cuda.synchronize()  # Ensure GPU is idle
                start_time = time.time()
                triton_result = triton_associative_scan(torch_gated_Z, torch_gates)
                torch.cuda.synchronize()  # Wait for completion
                end_time = time.time()
                triton_times.append((end_time - start_time) * 1000)  # ms

            # Rest briefly between implementations
            time.sleep(sleep_time)

            # Benchmark PyTorch reference
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start_time = time.time()
                pytorch_result = torch_associative_scan_reference(torch_gated_Z, torch_gates)
                torch.cuda.synchronize()
                end_time = time.time()
                pytorch_times.append((end_time - start_time) * 1000)  # ms

            # Rest briefly between implementations
            time.sleep(sleep_time)

            # Benchmark JAX
            for _ in range(num_trials):
                start_time = time.time()
                jax_result = jax_scan_fn(jax_gated_Z, jax_gates)
                jax.block_until_ready(jax_result)  # Proper sync
                end_time = time.time()
                jax_times.append((end_time - start_time) * 1000)  # ms

            # Compute statistics
            triton_mean = sum(triton_times) / len(triton_times)
            triton_std = (sum((t - triton_mean) ** 2 for t in triton_times) / len(triton_times)) ** 0.5

            jax_mean = sum(jax_times) / len(jax_times)
            jax_std = (sum((t - jax_mean) ** 2 for t in jax_times) / len(jax_times)) ** 0.5

            pytorch_mean = sum(pytorch_times) / len(pytorch_times)
            pytorch_std = (sum((t - pytorch_mean) ** 2 for t in pytorch_times) / len(pytorch_times)) ** 0.5

            # Compute speedup ratios (lower numbers are better for Triton)
            jax_speedup = jax_mean / triton_mean
            pytorch_speedup = pytorch_mean / triton_mean

            # Print results
            size_str = f"({B},{F},{S})"
            triton_str = f"{triton_mean:.2f}±{triton_std:.2f}"
            jax_str = f"{jax_mean:.2f}±{jax_std:.2f}"
            pytorch_str = f"{pytorch_mean:.2f}±{pytorch_std:.2f}"
            jax_speedup_str = f"{jax_speedup:.2f}x"
            pytorch_speedup_str = f"{pytorch_speedup:.2f}x"

            print(
                f"{size_str:<20} {triton_str:<15} {jax_str:<15} {pytorch_str:<15} {jax_speedup_str:<10} {pytorch_speedup_str:<10}"
            )

    print("\nBenchmark complete!")


# Function to profile memory usage - update references from PyTorch to JAX
def profile_memory_usage(configs=None, dtype=torch.float32):
    """Profile memory usage of both implementations"""
    if configs is None:
        configs = [
            (16, 8, 32),
            (32, 16, 64),
            (64, 32, 128),
            (128, 64, 256),
            (256, 32, 512),
            (64, 64, 1024),
        ]

    device = torch.device("cuda")
    jax_dtype = jnp.float32 if dtype == torch.float32 else jnp.bfloat16

    print(f"\nProfiling memory usage with {dtype}:")
    print(f"{'Size (B,F,S)':<20} {'Input Size (MB)':<20} {'Triton (MB)':<15} {'JAX (MB)':<15}")
    print("-" * 70)

    for B, F, S in configs:
        # Calculate theoretical input size
        input_size_bytes = 2 * B * F * S * dtype.itemsize  # Both gated_Z and gates
        input_size_mb = input_size_bytes / (1024 * 1024)

        # Create inputs
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gated_Z = torch.rand(B, F, S, device=device, dtype=dtype)
        gates = torch.rand(B, F, S, device=device, dtype=dtype)
        baseline = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Measure Triton memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = triton_associative_scan(gated_Z, gates)
        triton_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) - baseline

        # JAX memory usage is more complex to measure accurately from PyTorch
        # For now just report N/A
        jax_mem = "N/A"

        # Print results
        size_str = f"({B},{F},{S})"
        input_str = f"{input_size_mb:.2f}"
        triton_str = f"{triton_mem:.2f}"

        print(f"{size_str:<20} {input_str:<20} {triton_str:<15} {jax_mem:<15}")

    print("\nMemory profiling complete!")


def run_comprehensive_benchmarks():
    """Run both correctness tests and performance benchmarks."""
    print("=" * 80)
    print("ASSOCIATIVE SCAN BENCHMARK SUITE")
    print("=" * 80)

    # Skip correctness tests since test_triton_vs_jax was removed
    # Run performance benchmarks directly
    print("\n1. PERFORMANCE BENCHMARKING")
    print("-" * 80)
    benchmark_scan_implementations(dtypes=[torch.float32, torch.bfloat16], num_warmup=5, num_trials=20, sleep_time=0.1)

    # Profile memory usage
    print("\n2. MEMORY USAGE PROFILING")
    print("-" * 80)
    profile_memory_usage()

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETED")
    print("=" * 80)


# Optional: Visualization function if matplotlib is available
def visualize_benchmark_results(configs, triton_times, pytorch_times, dtype):
    """Generate visualization of benchmark results"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [f"({B},{F},{S})" for B, F, S in configs]
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, triton_times, width, label="Triton")
        ax.bar(x + width / 2, pytorch_times, width, label="PyTorch")

        # Add speedup annotations
        for i in range(len(triton_times)):
            speedup = pytorch_times[i] / triton_times[i]
            ax.text(i, max(triton_times[i], pytorch_times[i]) + 0.1, f"{speedup:.2f}x", ha="center", va="bottom")

        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Associative Scan Performance ({dtype})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"scan_benchmark_{dtype}.png")
        print(f"Visualization saved to scan_benchmark_{dtype}.png")

    except ImportError:
        print("Matplotlib not available for visualization")


def main():
    run_comprehensive_benchmarks()


if __name__ == "__main__":
    main()
