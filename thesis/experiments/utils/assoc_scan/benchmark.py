import time
import torch
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import uuid
from datetime import datetime
from kernel_1d import associative_scan as triton_associative_scan
from kernel import associative_scan as assoc_scan_official

torch.set_float32_matmul_precision("high")


def torch_associative_scan_reference(gated_Z: torch.Tensor, gates: torch.Tensor) -> tuple:
    """PyTorch reference implementation of associative scan.

    Args:
        gated_Z: Tensor [B, D, L] of gated values.
        gates: Tensor [B, D, L] of gating coefficients.

    Returns:
        Tuple of cumulative gated_Z and gates, each [B, D, L].
    """

    def combine_fn(carry, next_val):
        carry_gated_Z, carry_gates = carry
        next_gated_Z, next_gates = next_val
        return carry_gated_Z + next_gated_Z, carry_gates + next_gates

    return assoc_scan_official(
        combine_fn=combine_fn, xs=(gated_Z, gates), dim=-1, reverse=False, combine_mode="generic"
    )


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mem=False):
    """Benchmark function execution time and memory usage.

    Args:
        fn: Function to benchmark.
        warmup: Number of warmup iterations.
        rep: Number of benchmarking iterations.
        grad_to_none: List of tensors to reset gradients.
        quantiles: Quantiles for time statistics.
        fast_flush: Whether to flush cache before benchmarking.
        return_mem: Whether to return memory usage.

    Returns:
        Median time (ms) or (median time, median memory in MB) if return_mem is True.
    """
    quantiles = quantiles or [0.5]
    device = torch.device("cuda")

    try:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Cache flush
        if fast_flush:
            torch.empty(int(256e6) // 4, dtype=torch.float32, device=device).normal_()
            torch.cuda.synchronize()

        # Benchmark
        times, mem_usages = [], []
        for _ in range(rep):
            if grad_to_none:
                for x in grad_to_none:
                    x.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
            mem_usages.append(torch.cuda.max_memory_allocated() / (1024 * 1024))

        times = torch.tensor(times, dtype=torch.float32)
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float32)).tolist()
        median_time = ret[0]
        median_mem = np.median(mem_usages)

        return (median_time, median_mem) if return_mem else median_time
    except Exception as e:
        print(f"Benchmarking failed: {e}")
        return (float("nan"), float("nan")) if return_mem else float("nan")


@jit
def combine_fn(a, b):
    a_Z, a_gates = a
    b_Z, b_gates = b
    return a_Z + b_Z, a_gates + b_gates


@partial(jit, static_argnums=(2,))
def jax_scan_fn(gated_Z, gates, axis=2):
    return jax.lax.associative_scan(combine_fn, (gated_Z, gates), axis=axis)


def jax_bench(fn, rep=100, warmup=25):
    """Benchmark JAX function with CUDA events.

    Args:
        fn: JAX function to benchmark.
        rep: Number of benchmarking iterations.
        warmup: Number of warmup iterations.

    Returns:
        Median execution time (ms).
    """
    try:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(warmup):
            jax.block_until_ready(fn())
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(rep):
            torch.cuda.synchronize()
            start_event.record()
            jax.block_until_ready(fn())
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        return np.median(times)
    except Exception as e:
        print(f"JAX benchmarking failed: {e}")
        return float("nan")


def benchmark_scan_implementations(
    configs=None, dtypes=[torch.float32, torch.bfloat16], num_warmup=25, num_trials=100, sleep_time=0.1
):
    """Benchmark associative scan implementations across frameworks.

    Args:
        configs: List of (batch, features, sequence length) tuples.
        dtypes: Data types to test.
        num_warmup: Warmup iterations.
        num_trials: Benchmarking iterations.
        sleep_time: Sleep time between runs.
    """
    configs = configs or [
        (16, 8, 32),
        (32, 16, 64),
        (64, 32, 128),
        (128, 64, 256),
        (256, 32, 512),
        (64, 64, 1024),
        (16, 32, 8192),
    ]
    device = torch.device("cuda")
    jax_device = jax.devices("gpu")[0] if jax.devices("gpu") else None

    print(f"\nBenchmarking on {torch.cuda.get_device_name()} (CUDA {torch.version.cuda})")
    print(f"JAX Device: {jax_device or 'default'}")

    for dtype in dtypes:
        jax_dtype = jnp.float32 if dtype == torch.float32 else jnp.bfloat16
        print(f"\nBenchmarking {dtype}:")
        header = (
            f"{'Shape (B,F,S)':<18} | {'Triton Fwd':^12} {'Triton F+B':^12} {'Triton x':^10} | "
            f"{'JAX Fwd':^12} {'JAX F+B':^12} {'JAX x':^10} | "
            f"{'PyTorch Fwd':^12} {'PyTorch F+B':^12} {'PyTorch x':^10}"
        )
        print(header)
        print("-" * len(header))

        for B, F, S in configs:
            torch.cuda.empty_cache()
            jax.clear_caches()

            # Initialize inputs
            torch.manual_seed(42)
            torch_gated_Z = torch.rand(B, F, S, device=device, dtype=dtype, requires_grad=True)
            torch_gates = torch.rand(B, F, S, device=device, dtype=dtype, requires_grad=True)

            jax_key = jax.random.PRNGKey(42)
            jax_gated_Z = jax.device_put(jax.random.uniform(jax_key, (B, F, S), dtype=jax_dtype), device=jax_device)
            jax_key, subkey = jax.random.split(jax_key)
            jax_gates = jax.device_put(jax.random.uniform(subkey, (B, F, S), dtype=jax_dtype), device=jax_device)

            # Triton benchmarks
            triton_fwd_fn = lambda: triton_associative_scan(torch_gated_Z, torch_gates)
            triton_fwd_time = do_bench(triton_fwd_fn, warmup=num_warmup, rep=num_trials, fast_flush=False)

            def triton_fwd_bwd_fn():
                out_z, out_g = triton_associative_scan(torch_gated_Z, torch_gates)
                loss = (out_z.sum() + out_g.sum()) * 1.0
                loss.backward()

            triton_fwd_bwd_time = do_bench(
                triton_fwd_bwd_fn,
                grad_to_none=[torch_gated_Z, torch_gates],
                warmup=num_warmup,
                rep=num_trials,
                fast_flush=False,
            )
            triton_multiplier = triton_fwd_bwd_time / triton_fwd_time if triton_fwd_time > 0 else float("inf")

            # JAX benchmarks
            jax_fwd_fn = lambda: jax_scan_fn(jax_gated_Z, jax_gates)
            jax_fwd_time = jax_bench(jax_fwd_fn, warmup=num_warmup, rep=num_trials)

            def jax_loss_fn(z, g):
                out_z, out_g = jax_scan_fn(z, g)
                return jnp.sum(out_z) + jnp.sum(out_g)

            jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1))

            def jax_fwd_bwd_fn():
                fwd_result = jax_fwd_fn()
                grad_result = jax_grad_fn(jax_gated_Z, jax_gates)
                return fwd_result, grad_result

            jax_fwd_bwd_time = jax_bench(jax_fwd_bwd_fn, warmup=num_warmup, rep=num_trials)
            jax_multiplier = jax_fwd_bwd_time / jax_fwd_time if jax_fwd_time > 0 else float("inf")

            # PyTorch benchmarks
            pytorch_fwd_fn = lambda: torch_associative_scan_reference(torch_gated_Z, torch_gates)
            pytorch_fwd_time = do_bench(pytorch_fwd_fn, warmup=num_warmup, rep=num_trials, fast_flush=False)

            def pytorch_fwd_bwd_fn():
                out_z, out_g = torch_associative_scan_reference(torch_gated_Z, torch_gates)
                loss = (out_z.sum() + out_g.sum()) * 1.0
                loss.backward()

            pytorch_fwd_bwd_time = do_bench(
                pytorch_fwd_bwd_fn,
                grad_to_none=[torch_gated_Z, torch_gates],
                warmup=num_warmup,
                rep=num_trials,
                fast_flush=False,
            )
            pytorch_multiplier = pytorch_fwd_bwd_time / pytorch_fwd_time if pytorch_fwd_time > 0 else float("inf")

            # Revised results print statement
            results_line = (
                f"({B},{F},{S})".ljust(18)
                + "| "
                + f"{triton_fwd_time:^12.2f}{triton_fwd_bwd_time:^12.2f}{triton_multiplier:^10.2f} | "
                + f"{jax_fwd_time:^12.2f}{jax_fwd_bwd_time:^12.2f}{jax_multiplier:^10.2f} | "
                + f"{pytorch_fwd_time:^12.2f}{pytorch_fwd_bwd_time:^12.2f}{pytorch_multiplier:^10.2f}"
            )
            print(results_line)
            time.sleep(sleep_time)

    print("\nBenchmark complete!")


def profile_memory_usage(configs=None, dtype=torch.float32):
    """Profile memory usage of associative scan implementations.

    Args:
        configs: List of (batch, features, sequence length) tuples.
        dtype: Data type to test.
    """
    configs = configs or [
        (16, 8, 32),
        (32, 16, 64),
        (64, 32, 128),
        (128, 64, 256),
        (256, 32, 512),
        (64, 64, 1024),
        (16, 32, 8192),
    ]
    device = torch.device("cuda")
    jax_device = jax.devices("gpu")[0] if jax.devices("gpu") else None
    jax_dtype = jnp.float32 if dtype == torch.float32 else jnp.bfloat16

    print(f"\nMemory Profiling ({dtype}):")
    print(f"{'Shape (B,F,S)':<20} {'Input (MB)':<15} {'Triton (MB)':<15} {'JAX (MB)':<15} {'PyTorch (MB)':<15}")
    print("-" * 80)

    for B, F, S in configs:
        input_size_mb = 2 * B * F * S * dtype.itemsize / (1024 * 1024)

        # Triton memory
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        gated_Z = torch.rand(B, F, S, device=device, dtype=dtype)
        gates = torch.rand(B, F, S, device=device, dtype=dtype)
        torch.cuda.synchronize()
        baseline = torch.cuda.max_memory_allocated() / (1024 * 1024)
        triton_fn = lambda: triton_associative_scan(gated_Z, gates)
        _, triton_mem = do_bench(triton_fn, warmup=25, rep=100, fast_flush=False, return_mem=True)
        triton_mem = max(0, triton_mem - baseline)

        # JAX memory
        jax.clear_caches()
        jax_gated_Z = jax.device_put(
            jax.random.uniform(jax.random.PRNGKey(42), (B, F, S), dtype=jax_dtype), device=jax_device
        )
        jax_gates = jax.device_put(
            jax.random.uniform(jax.random.PRNGKey(43), (B, F, S), dtype=jax_dtype), device=jax_device
        )
        baseline_mem = (
            jax.device_get(jax.devices("gpu")[0].memory_stats())["bytes_in_use"] / (1024 * 1024) if jax_device else 0
        )
        with jax.profiler.TraceAnnotation("jax_scan"):
            result = jax_scan_fn(jax_gated_Z, jax_gates)
            jax.block_until_ready(result)
        peak_mem = (
            jax.device_get(jax.devices("gpu")[0].memory_stats())["bytes_in_use"] / (1024 * 1024) if jax_device else 0
        )
        jax_mem = max(0, peak_mem - baseline_mem)

        # PyTorch memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.max_memory_allocated() / (1024 * 1024)
        pytorch_fn = lambda: torch_associative_scan_reference(gated_Z, gates)
        _, pytorch_mem = do_bench(pytorch_fn, warmup=25, rep=100, fast_flush=False, return_mem=True)
        pytorch_mem = max(0, pytorch_mem - baseline)

        print(
            f"({B},{F},{S})".ljust(20)
            + f"{input_size_mb:<15.2f}{triton_mem:<15.2f}{jax_mem:<15.2f}{pytorch_mem:<15.2f}"
        )


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks for associative scan implementations."""
    print(f"\n{'=' * 80}\nAssociative Scan Benchmarks ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n{'=' * 80}")
    print("\n1. Performance Benchmarks")
    print("-" * 50)
    benchmark_scan_implementations(
        dtypes=[torch.float32, torch.bfloat16], num_warmup=25, num_trials=100, sleep_time=0.1
    )
    print("\n2. Memory Usage Benchmarks")
    print("-" * 50)
    profile_memory_usage()
    print(f"\n{'=' * 80}\nBenchmarks Completed\n{'=' * 80}")


def main():
    run_comprehensive_benchmarks()


if __name__ == "__main__":
    main()
