import torch
import time
from typing import Tuple

# Define a simple scan_fn for demonstration
def scan_fn(sim: torch.Tensor, v_norm: torch.Tensor, gated_Z: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Dummy scan_fn: adds inputs and scales by gates."""
    return sim + v_norm + gated_Z * gates

# Original batched_scan_fn with nested vmaps
def batched_scan_fn_nested(sim: torch.Tensor, v_norm: torch.Tensor, gated_Z: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    hmap = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)  # over H
    bmap = torch.vmap(hmap, in_dims=(0, 0, 0, 0), out_dims=0)  # over B
    return bmap(sim, v_norm, gated_Z, gates)

# Optimized batched_scan_fn with single vmap
def batched_scan_fn_single(sim: torch.Tensor, v_norm: torch.Tensor, gated_Z: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    B, H = sim.shape[0], sim.shape[1]
    sim = sim.reshape(B * H, *sim.shape[2:])
    v_norm = v_norm.reshape(B * H, *v_norm.shape[2:])
    gated_Z = gated_Z.reshape(B * H, *gated_Z.shape[2:])
    gates = gates.reshape(B * H, *gates.shape[2:])
    
    vmap = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    result = vmap(sim, v_norm, gated_Z, gates)
    
    return result.reshape(B, H, *result.shape[1:])

# Benchmark function
def benchmark_functions(B: int, H: int, D: int, num_iterations: int = 100, warmup_iterations: int = 10):
    print(f"\nBenchmarking with B={B}, H={H}, D={D}")
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create random input tensors
    torch.manual_seed(42)
    sim = torch.randn(B, H, D, device=device)
    v_norm = torch.randn(B, H, D, device=device)
    gated_Z = torch.randn(B, H, D, device=device)
    gates = torch.randn(B, H, D, device=device)
    
    # Warm-up runs to initialize GPU and cache
    for _ in range(warmup_iterations):
        _ = batched_scan_fn_nested(sim, v_norm, gated_Z, gates)
        _ = batched_scan_fn_single(sim, v_norm, gated_Z, gates)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark nested vmap
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output_nested = batched_scan_fn_nested(sim, v_norm, gated_Z, gates)
        if device.type == "cuda":
            torch.cuda.synchronize()
    nested_time = (time.perf_counter() - start_time) / num_iterations * 1000  # ms per iteration
    
    # Benchmark single vmap
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output_single = batched_scan_fn_single(sim, v_norm, gated_Z, gates)
        if device.type == "cuda":
            torch.cuda.synchronize()
    single_time = (time.perf_counter() - start_time) / num_iterations * 1000  # ms per iteration
    
    # Verify outputs are equal
    is_equal = torch.allclose(output_nested, output_single, atol=1e-6)
    
    print(f"Nested vmap time: {nested_time:.3f} ms per iteration")
    print(f"Single vmap time: {single_time:.3f} ms per iteration")
    print(f"Speedup (nested/single): {nested_time/single_time:.3f}x")
    print(f"Outputs are equal: {is_equal}")

if __name__ == "__main__":
    # Test multiple input sizes
    input_configs = [
        {"B": 16, "H": 8, "D": 64},    # Small
        {"B": 64, "H": 16, "D": 128},  # Medium
        {"B": 128, "H": 32, "D": 256}, # Large
    ]
    
    for config in input_configs:
        benchmark_functions(**config, num_iterations=100, warmup_iterations=10)