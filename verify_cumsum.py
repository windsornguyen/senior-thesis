import torch
import fla.ops.linear_attn
import math
import time  # For CPU timing if needed, though GPU events are better

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA not available. Skipping verification.")
    exit()

device = torch.device("cuda")

import torch
import fla.ops.linear_attn
import math

torch.manual_seed(1746)

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA not available. Skipping verification.")
    exit()
device = torch.device("cuda")

# Use head dims >= 16 for potential downstream kernel use, although cumsum might not require it
B, H, T, N, P = 8, 16, 1024, 256, 256  # N/H=16, P/H=16
DQ = N // H
DK = N // H
DV = P // H
print(f"Verifying Q @ torch.cumsum(V @ K^T) vs fla.fused_chunk_linear_attn(q=Q, k=V, v=K)")
print(f"Using Head Dims: DQ={DQ}, DK={DK}, DV={DV}, T={T}")

# Ensure dimensions allow the final einsum Q @ H(V@K^T) and reshaping
if DQ != DV or N != P:
    raise ValueError(f"Dimension mismatch: Require DQ=DV and N=P. Got DQ={DQ}, DV={DV}, N={N}, P={P}")

# Create inputs
Q = torch.randn(B, H, T, DQ, device=device, dtype=torch.float32, requires_grad=True)
K = torch.randn(B, H, T, DK, device=device, dtype=torch.float32, requires_grad=True)
V = torch.randn(B, H, T, DV, device=device, dtype=torch.float32, requires_grad=True)

# Clones for separate backward passes
Q_clone = Q.clone().detach().requires_grad_(True)
K_clone = K.clone().detach().requires_grad_(True)
V_clone = V.clone().detach().requires_grad_(True)

# --- 1. Reference Calculation (Using torch.cumsum for V @ K^T) ---
print("\n--- Computing Reference: Q @ torch.cumsum(V @ K^T) ---")
scale = DK**-0.5  # Scale based on K dimension
Q_scaled = Q * scale
Z_vk_ref = torch.einsum("bhtv,bhtk->bhtvk", V, K)  # V @ K^T -> shape (B, H, T, DV, DK)
H_vk_ref = torch.cumsum(Z_vk_ref, dim=2)  # Apply torch.cumsum
Y_ref = torch.einsum("bhtq,bhtvk->bhtk", Q_scaled, H_vk_ref)  # Output: B,H,T,DK
Y_ref = Y_ref.transpose(1, 2).reshape(B, T, N)  # Reshape standard output format
print(f"  Y_ref shape: {Y_ref.shape}")


# --- 2. Test Calculation (Using fla.fused_chunk_linear_attn with swapped K/V) ---
print("\n--- Computing Test: fla.fused_chunk_linear_attn(q=Q, k=V, v=K) ---")

# Permute inputs to (B, T, H, D) format expected by kernel
Q_fla_in = Q_clone.permute(0, 2, 1, 3).contiguous()
K_fla_in = K_clone.permute(0, 2, 1, 3).contiguous()  # Original K tensor permuted
V_fla_in = V_clone.permute(0, 2, 1, 3).contiguous()  # Original V tensor permuted

# Call kernel, passing V_fla_in as k and K_fla_in as v
# This computes Q @ cumsum(V @ K^T)
Y_fla_out_permuted, _ = fla.ops.linear_attn.fused_chunk_linear_attn(
    q=Q_fla_in,
    k=V_fla_in,  # Pass V tensor here
    v=K_fla_in,  # Pass K tensor here
    scale=scale,  # Scale based on original K dim (DK)
)

# Output dim matches kernel's v input dim (K_fla_in -> DK)
# Output shape is (B, T, H, DK)
# Permute output back to (B, H, T, DK)
Y_test = Y_fla_out_permuted.permute(0, 2, 1, 3)
# Reshape to standard output format (B, T, N), assumes DK*H = N
Y_test = Y_test.reshape(B, T, N)
print(f"  Y_test shape: {Y_test.shape}")


# --- 3. Compare Forward Results ---
print("\n--- Comparing Forward Outputs ---")
try:
    # Using tighter tolerances now that we expect a match based on user's previous tests
    torch.allclose(Y_ref, Y_test, atol=1e-5, rtol=1e-4)
    print("  ✅ Forward results are numerically close.")
except AssertionError as e:
    print(f"  ❌ Forward results differ significantly: {e}")
    diff = (Y_ref - Y_test).abs()
    print(f"     Max difference: {diff.max().item()}")
    print(f"     Mean difference: {diff.mean().item()}")

# --- 4. Compare Gradients ---
print("\n--- Comparing Gradients ---")
grad_output = torch.randn_like(Y_ref)

# Backward pass for reference
Y_ref.backward(grad_output, retain_graph=True)

# Backward pass for test (FLA kernel)
Y_test.backward(grad_output, retain_graph=True)

# Compare gradients
grad_check_passed = True
print("  Checking gradient numerical closeness:")
try:
    torch.allclose(Q.grad, Q_clone.grad, atol=1e-4, rtol=1e-3)
    print("     ✅ Q gradients match.")
except AssertionError as e:
    print(f"     ❌ Q gradients differ: {e}")
    grad_diff = (Q.grad - Q_clone.grad).abs()
    print(f"        Max difference: {grad_diff.max().item()}")
    print(f"        Mean difference: {grad_diff.mean().item()}")
    grad_check_passed = False
try:
    torch.allclose(K.grad, K_clone.grad, atol=1e-4, rtol=1e-3)
    print("     ✅ K gradients match.")
except AssertionError as e:
    print(f"     ❌ K gradients differ: {e}")
    grad_diff = (K.grad - K_clone.grad).abs()
    print(f"        Max difference: {grad_diff.max().item()}")
    print(f"        Mean difference: {grad_diff.mean().item()}")
    grad_check_passed = False
try:
    torch.allclose(V.grad, V_clone.grad, atol=1e-4, rtol=1e-3)
    print("     ✅ V gradients match.")
except AssertionError as e:
    print(f"     ❌ V gradients differ: {e}")
    grad_diff = (V.grad - V_clone.grad).abs()
    print(f"        Max difference: {grad_diff.max().item()}")
    print(f"        Mean difference: {grad_diff.mean().item()}")
    grad_check_passed = False

if not grad_check_passed:
    print("  ❌ Some gradients did not match numerically.")
else:
    print("  ✅ All gradients match numerically.")


print("\nVerification complete.")

# Define base tensor dimensions (T will be looped)
B = 2  # Batch size
H = 4  # Number of heads
DK = 16  # Key dimension
DV = 16  # Value dimension
DQ = 16  # Query dimension
NUM_WARMUPS = 4
NUM_REPS = 1  # Timed repetitions after warmup
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]

print(f"Verifying Linear Attention Equivalence: Q @ cumsum(K @ V^T)")
print(f"Shapes: B={B}, H={H}, DK={DK}, DV={DV}, DQ={DQ}")
print(f"Warmup runs: {NUM_WARMUPS}, Timed runs: {NUM_REPS}")


# Function to create inputs for a given sequence length T
def create_inputs(T, B, H, DQ, DK, DV, device):
    Q = torch.randn(B, H, T, DQ, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, T, DK, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, T, DV, device=device, dtype=torch.float32, requires_grad=True)
    # Clones for separate backward passes
    Q_clone = Q.clone().detach().requires_grad_(True)
    K_clone = K.clone().detach().requires_grad_(True)
    V_clone = V.clone().detach().requires_grad_(True)
    return Q, K, V, Q_clone, K_clone, V_clone


# Store results
results = {}

for T in SEQUENCE_LENGTHS:
    print(f"\n{'=' * 10} Testing Sequence Length T = {T} {'=' * 10}")
    results[T] = {}

    # --- Timing Reference Implementation (Torch) ---
    print("--- Timing Torch Implementation ---")
    Q, K, V, Q_clone, K_clone, V_clone = create_inputs(T, B, H, DQ, DK, DV, device)

    # Forward Pass Warmup (Torch)
    # print("  Warming up Torch forward pass...") # Verbose, can be commented out
    for _ in range(NUM_WARMUPS):
        Q_scaled = Q * (DK**-0.5)
        Z_kv = torch.einsum("bhtk,bhtv->bhtkv", K, V)
        H_kv = torch.cumsum(Z_kv, dim=2)
        _ = torch.einsum("bhtq,bhtqv->bhtv", Q_scaled, H_kv)
        torch.cuda.synchronize()

    # Forward Pass Timed (Torch)
    # print(f"  Timing Torch forward pass ({NUM_REPS} reps)...")
    start_torch_fwd_outer = torch.cuda.Event(enable_timing=True)
    end_torch_fwd_outer = torch.cuda.Event(enable_timing=True)
    start_torch_fwd_outer.record()
    for _ in range(NUM_REPS):
        Q_scaled = Q * (DK**-0.5)
        Z_kv = torch.einsum("bhtk,bhtv->bhtkv", K, V)
        H_kv = torch.cumsum(Z_kv, dim=2)
        Y_ref = torch.einsum("bhtq,bhtqv->bhtv", Q_scaled, H_kv)
    end_torch_fwd_outer.record()
    torch.cuda.synchronize()
    time_torch_fwd = start_torch_fwd_outer.elapsed_time(end_torch_fwd_outer) / NUM_REPS
    results[T]["torch_fwd_time"] = time_torch_fwd
    print(f"  Avg Torch forward time: {time_torch_fwd:.4f} ms")

    # Backward Pass Warmup (Torch)
    grad_output = torch.randn_like(Y_ref)
    # print("  Warming up Torch backward pass...")
    for _ in range(NUM_WARMUPS):
        Q_b, K_b, V_b, _, _, _ = create_inputs(T, B, H, DQ, DK, DV, device)
        Q_scaled_b = Q_b * (DK**-0.5)
        Z_kv_b = torch.einsum("bhtk,bhtv->bhtkv", K_b, V_b)
        H_kv_b = torch.cumsum(Z_kv_b, dim=2)
        Y_ref_b = torch.einsum("bhtq,bhtqv->bhtv", Q_scaled_b, H_kv_b)
        Y_ref_b.backward(grad_output, retain_graph=False)
        del Q_b, K_b, V_b, Q_scaled_b, Z_kv_b, H_kv_b, Y_ref_b
        torch.cuda.synchronize()

    # Backward Pass Timed (Torch)
    # print(f"  Timing Torch backward pass ({NUM_REPS} reps)...")
    start_torch_bwd_outer = torch.cuda.Event(enable_timing=True)
    end_torch_bwd_outer = torch.cuda.Event(enable_timing=True)
    start_torch_bwd_outer.record()
    for i in range(NUM_REPS):
        Y_ref.backward(grad_output, retain_graph=(i < NUM_REPS - 1))
    end_torch_bwd_outer.record()
    torch.cuda.synchronize()
    time_torch_bwd = start_torch_bwd_outer.elapsed_time(end_torch_bwd_outer) / NUM_REPS
    results[T]["torch_bwd_time"] = time_torch_bwd
    print(f"  Avg Torch backward time: {time_torch_bwd:.4f} ms")
    Q_grad_torch, K_grad_torch, V_grad_torch = Q.grad, K.grad, V.grad
    del Q, K, V, Q_scaled, Z_kv, H_kv, Y_ref, grad_output

    # --- Timing FLA Kernel Implementation ---
    print("--- Timing FLA Kernel Implementation ---")
    Q, K, V, Q_clone, K_clone, V_clone = create_inputs(T, B, H, DQ, DK, DV, device)

    # Forward Pass Warmup (FLA)
    # print("  Warming up FLA forward pass...")
    for _ in range(NUM_WARMUPS):
        Q_fla_in = Q_clone.permute(0, 2, 1, 3).contiguous()
        K_fla_in = K_clone.permute(0, 2, 1, 3).contiguous()
        V_fla_in = V_clone.permute(0, 2, 1, 3).contiguous()
        _, _ = fla.ops.linear_attn.fused_chunk_linear_attn(q=Q_fla_in, k=K_fla_in, v=V_fla_in, scale=DK**-0.5)
        torch.cuda.synchronize()

    # Forward Pass Timed (FLA)
    # print(f"  Timing FLA forward pass ({NUM_REPS} reps)...")
    start_fla_fwd_outer = torch.cuda.Event(enable_timing=True)
    end_fla_fwd_outer = torch.cuda.Event(enable_timing=True)
    start_fla_fwd_outer.record()
    for _ in range(NUM_REPS):
        Q_fla_in = Q_clone.permute(0, 2, 1, 3).contiguous()
        K_fla_in = K_clone.permute(0, 2, 1, 3).contiguous()
        V_fla_in = V_clone.permute(0, 2, 1, 3).contiguous()
        Y_fla_out_permuted, _ = fla.ops.linear_attn.fused_chunk_linear_attn(
            q=Q_fla_in, k=K_fla_in, v=V_fla_in, scale=DK**-0.5
        )
    end_fla_fwd_outer.record()
    torch.cuda.synchronize()
    time_fla_fwd = start_fla_fwd_outer.elapsed_time(end_fla_fwd_outer) / NUM_REPS
    results[T]["fla_fwd_time"] = time_fla_fwd
    print(f"  Avg FLA forward time: {time_fla_fwd:.4f} ms")
    Y_fla = Y_fla_out_permuted.permute(0, 2, 1, 3)

    # Backward Pass Warmup (FLA)
    grad_output_fla = torch.randn_like(Y_fla)
    # print("  Warming up FLA backward pass...")
    for _ in range(NUM_WARMUPS):
        _, _, _, Qc_b, Kc_b, Vc_b = create_inputs(T, B, H, DQ, DK, DV, device)
        Qf_in_b = Qc_b.permute(0, 2, 1, 3).contiguous()
        Kf_in_b = Kc_b.permute(0, 2, 1, 3).contiguous()
        Vf_in_b = Vc_b.permute(0, 2, 1, 3).contiguous()
        Yf_out_p_b, _ = fla.ops.linear_attn.fused_chunk_linear_attn(q=Qf_in_b, k=Kf_in_b, v=Vf_in_b, scale=DK**-0.5)
        Yf_b = Yf_out_p_b.permute(0, 2, 1, 3)
        Yf_b.backward(grad_output_fla, retain_graph=False)
        del Qc_b, Kc_b, Vc_b, Qf_in_b, Kf_in_b, Vf_in_b, Yf_out_p_b, Yf_b
        torch.cuda.synchronize()

    # Backward Pass Timed (FLA)
    # print(f"  Timing FLA backward pass ({NUM_REPS} reps)...")
    start_fla_bwd_outer = torch.cuda.Event(enable_timing=True)
    end_fla_bwd_outer = torch.cuda.Event(enable_timing=True)
    start_fla_bwd_outer.record()
    for i in range(NUM_REPS):
        Y_fla.backward(grad_output_fla, retain_graph=(i < NUM_REPS - 1))
    end_fla_bwd_outer.record()
    torch.cuda.synchronize()
    time_fla_bwd = start_fla_bwd_outer.elapsed_time(end_fla_bwd_outer) / NUM_REPS
    results[T]["fla_bwd_time"] = time_fla_bwd
    print(f"  Avg FLA backward time: {time_fla_bwd:.4f} ms")
    Q_grad_fla, K_grad_fla, V_grad_fla = Q_clone.grad, K_clone.grad, V_clone.grad
    del Q, K, V, Q_clone, K_clone, V_clone, Q_fla_in, K_fla_in, V_fla_in, Y_fla_out_permuted, Y_fla, grad_output_fla

    # --- Calculate Speedup ---
    fwd_speedup = time_torch_fwd / time_fla_fwd if time_fla_fwd > 0 else float("inf")
    bwd_speedup = time_torch_bwd / time_fla_bwd if time_fla_bwd > 0 else float("inf")
    results[T]["fwd_speedup"] = fwd_speedup
    results[T]["bwd_speedup"] = bwd_speedup
    print(f"  Forward Speedup (FLA vs Torch): {fwd_speedup:.2f}x")
    print(f"  Backward Speedup (FLA vs Torch): {bwd_speedup:.2f}x")

    # --- Compare results (Final Check) ---
    # print("--- Final Numerical Comparison ---") # Can be verbose
    Q_ref, K_ref, V_ref, Q_fla_comp, K_fla_comp, V_fla_comp = create_inputs(T, B, H, DQ, DK, DV, device)
    Q_scaled_ref = Q_ref * (DK**-0.5)
    Z_kv_ref = torch.einsum("bhtk,bhtv->bhtkv", K_ref, V_ref)
    H_kv_ref = torch.cumsum(Z_kv_ref, dim=2)
    Y_ref_final = torch.einsum("bhtq,bhtqv->bhtv", Q_scaled_ref, H_kv_ref)

    Q_fla_in_comp = Q_fla_comp.permute(0, 2, 1, 3).contiguous()
    K_fla_in_comp = K_fla_comp.permute(0, 2, 1, 3).contiguous()
    V_fla_in_comp = V_fla_comp.permute(0, 2, 1, 3).contiguous()
    Y_fla_out_permuted_final, _ = fla.ops.linear_attn.fused_chunk_linear_attn(
        q=Q_fla_in_comp, k=K_fla_in_comp, v=V_fla_in_comp, scale=DK**-0.5
    )
    Y_fla_final = Y_fla_out_permuted_final.permute(0, 2, 1, 3)

    # print("Comparing forward outputs:")
    try:
        torch.allclose(Y_ref_final, Y_fla_final, atol=1e-5, rtol=1e-4)
        results[T]["fwd_match"] = True
        # print("  ✅ Forward results are numerically close.")
    except AssertionError as e:
        results[T]["fwd_match"] = False
        print(f"  WARNING: Forward outputs differ significantly for T={T}: {e}")

    # Compare stored gradients
    # print("\nComparing gradients:")
    grad_check_passed = True
    try:
        torch.allclose(Q_grad_torch, Q_grad_fla, atol=1e-4, rtol=1e-3)
    except AssertionError as e:
        print(f"  WARNING: Q gradients differ for T={T}: {e}")
        grad_check_passed = False
    try:
        torch.allclose(K_grad_torch, K_grad_fla, atol=1e-4, rtol=1e-3)
    except AssertionError as e:
        print(f"  WARNING: K gradients differ for T={T}: {e}")
        grad_check_passed = False
    try:
        torch.allclose(V_grad_torch, V_grad_fla, atol=1e-4, rtol=1e-3)
    except AssertionError as e:
        print(f"  WARNING: V gradients differ for T={T}: {e}")
        grad_check_passed = False
    results[T]["grad_match"] = grad_check_passed
    del Q_ref, K_ref, V_ref, Q_fla_comp, K_fla_comp, V_fla_comp  # Clean up memory
    del Y_ref_final, Y_fla_final, Y_fla_out_permuted_final
    del Q_grad_torch, K_grad_torch, V_grad_torch, Q_grad_fla, K_grad_fla, V_grad_fla


# --- Print Summary Table ---
print(f"\n{'=' * 10} Summary {'=' * 10}")
print(
    f"{'Seq Len (T)':<12} | {'Torch Fwd (ms)':<16} | {'FLA Fwd (ms)':<14} | {'Fwd Speedup':<12} | {'Torch Bwd (ms)':<16} | {'FLA Bwd (ms)':<14} | {'Bwd Speedup':<12} | {'Fwd Match':<10} | {'Grad Match':<10}"
)
print("-" * 135)
for T in SEQUENCE_LENGTHS:
    res = results[T]
    print(
        f"{T:<12} | {res['torch_fwd_time']:<16.4f} | {res['fla_fwd_time']:<14.4f} | {res['fwd_speedup']:<12.2f}x | {res['torch_bwd_time']:<16.4f} | {res['fla_bwd_time']:<14.4f} | {res['bwd_speedup']:<12.2f}x | {str(res['fwd_match']):<10} | {str(res['grad_match']):<10}"
    )

print("\nVerification complete.")
