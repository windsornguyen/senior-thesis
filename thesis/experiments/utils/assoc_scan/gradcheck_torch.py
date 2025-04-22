import time
import torch

from torch.autograd import gradcheck

# Import assoc_scan_official from kernel.py
from kernel import associative_scan as assoc_scan_official

# Import AssociativeScan from kernel_1d.py
from kernel_1d import AssociativeScan

def torch_associative_scan_reference(gated_Z: torch.Tensor, gates: torch.Tensor):
    """
    PyTorch reference implementation of associative scan using assoc_scan_official.
    
    Args:
        gated_Z: Tensor of shape [B, D, L] or [batch_size, B, D, L] representing the gated values.
        gates: Tensor of shape [B, D, L] or [batch_size, B, D, L] representing the gating coefficients.
    
    Returns:
        A tuple (cumulative_gated_Z, cumulative_gates), each of shape [B, D, L] or [batch_size, B, D, L],
        corresponding to the cumulative associative scan results along the sequence dimension.
    """
    def combine_fn(carry, next_val):
        carry_gated_Z, carry_gates = carry
        next_gated_Z, next_gates = next_val
        return (carry_gated_Z + next_gated_Z, carry_gates + next_gates)
    
    xs = (gated_Z, gates)
    return assoc_scan_official(combine_fn=combine_fn, xs=xs, dim=-1, reverse=False, combine_mode="generic")

def run_grad_check(B, F, S, dtype=torch.float64, eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5):
    torch.manual_seed(42 + B + F + S)
    gated_Z = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)
    gates = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)

    try:
        triton_success = gradcheck(
            AssociativeScan.apply, (gated_Z, gates), eps=eps, atol=atol, rtol=rtol, nondet_tol=nondet_tol, fast_mode=True
        )
    except Exception as e:
        print(f"Triton gradcheck error for shape ({B},{F},{S}): {e}")
        triton_success = False

    try:
        reference_success = gradcheck(
            torch_associative_scan_reference, (gated_Z, gates), eps=eps, atol=atol, rtol=rtol, nondet_tol=nondet_tol, fast_mode=True
        )
    except Exception as e:
        print(f"Reference gradcheck error for shape ({B},{F},{S}): {e}")
        reference_success = False

    return triton_success, reference_success

def check_grad_equivalence(B, F, S, dtype=torch.float64, atol=1e-5, vmap=False, batch_size=4):
    # Relax tolerances for long sequences
    atol = 1e-5 if S <= 256 else 1e-3
    rtol = 1e-5 if S <= 256 else 1e-3

    torch.manual_seed(42 + B + F + S)
    if vmap:
        # Add extra batch dimension for vmap
        gated_Z_triton = torch.randn(batch_size, B, F, S, dtype=dtype, device="cuda", requires_grad=True)
        gates_triton = torch.randn(batch_size, B, F, S, dtype=dtype, device="cuda", requires_grad=True)
    else:
        gated_Z_triton = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)
        gates_triton = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)

    gated_Z_ref = gated_Z_triton.detach().clone().requires_grad_(True)
    gates_ref = gates_triton.detach().clone().requires_grad_(True)

    if vmap:
        vmap_scan = torch.vmap(AssociativeScan.apply, in_dims=(0, 0), out_dims=0)
        vmap_ref = torch.vmap(torch_associative_scan_reference, in_dims=(0, 0), out_dims=0)
        triton_Z, triton_gates = vmap_scan(gated_Z_triton, gates_triton)
        ref_Z, ref_gates = vmap_ref(gated_Z_ref, gates_ref)
    else:
        triton_Z, triton_gates = AssociativeScan.apply(gated_Z_triton, gates_triton)
        ref_Z, ref_gates = torch_associative_scan_reference(gated_Z_ref, gates_ref)

    torch.manual_seed(43 + B + F + S)
    grad_Z = torch.randn_like(triton_Z)
    grad_gates = torch.randn_like(triton_gates)

    triton_loss = (triton_Z * grad_Z).sum() + (triton_gates * grad_gates).sum()
    triton_loss.backward()

    ref_loss = (ref_Z * grad_Z).sum() + (ref_gates * grad_gates).sum()
    ref_loss.backward()
    grad_Z_ref = gated_Z_ref.grad
    grad_gates_ref = gates_ref.grad

    grad_match_Z = torch.allclose(gated_Z_triton.grad, grad_Z_ref, atol=atol, rtol=rtol)
    grad_match_gates = torch.allclose(gates_triton.grad, grad_gates_ref, atol=atol, rtol=rtol)

    if not grad_match_Z or not grad_match_gates:
        if not grad_match_Z:
            diff_Z = (gated_Z_triton.grad - grad_Z_ref).abs()
            print(f"  Z gradient mismatch for shape ({batch_size if vmap else ''}{B},{F},{S}): max={diff_Z.max():.6f}, mean={diff_Z.mean():.6f}")
        if not grad_match_gates:
            diff_gates = (gates_triton.grad - grad_gates_ref).abs()
            print(f"  Gates gradient mismatch for shape ({batch_size if vmap else ''}{B},{F},{S}): max={diff_gates.max():.6f}, mean={diff_gates.mean():.6f}")

    return grad_match_Z and grad_match_gates

def test_correctness():
    print("=" * 80)
    print("Associative Scan Correctness Tests")
    print("=" * 80)

    test_shapes = [
        (2, 3, 8),
        (4, 16, 32),
        (1, 10, 15),
        (3, 7, 64),
        (2, 8, 2048),
        (1, 4, 16384),
    ]

    # Non-vmap correctness tests
    for shape in test_shapes:
        B, D, L = shape
        gated_Z_torch = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
        gates_torch = torch.randn(B, D, L, dtype=torch.float32, device="cuda")

        ref_Z, ref_gates = torch_associative_scan_reference(gated_Z_torch, gates_torch)
        triton_Z, triton_gates = AssociativeScan.apply(gated_Z_torch, gates_torch)

        if not torch.allclose(triton_Z, ref_Z, rtol=1e-3, atol=1e-3):
            print(f"  Mismatch in cumulative_gated_Z for shape {shape}")
        if not torch.allclose(triton_gates, ref_gates, rtol=1e-3, atol=1e-3):
            print(f"  Mismatch in cumulative_gates for shape {shape}")
        else:
            print(f"Shape {shape}: ✓ PASS")

    # Vmap correctness tests
    print("\nVmap Correctness Tests")
    print("-" * 50)
    vmap_shapes = [
        (2, 3, 8, 4),  # (B, D, L, batch_size)
        (4, 16, 32, 2),
        (1, 10, 15, 8),
    ]

    for B, D, L, batch_size in vmap_shapes:
        gated_Z_torch = torch.randn(batch_size, B, D, L, dtype=torch.float32, device="cuda")
        gates_torch = torch.randn(batch_size, B, D, L, dtype=torch.float32, device="cuda")

        vmap_scan = torch.vmap(AssociativeScan.apply, in_dims=(0, 0), out_dims=0)
        vmap_ref = torch.vmap(torch_associative_scan_reference, in_dims=(0, 0), out_dims=0)

        triton_Z, triton_gates = vmap_scan(gated_Z_torch, gates_torch)
        ref_Z, ref_gates = vmap_ref(gated_Z_torch, gates_torch)

        if not torch.allclose(triton_Z, ref_Z, rtol=1e-3, atol=1e-3):
            print(f"  Mismatch in cumulative_gated_Z for vmap shape ({batch_size},{B},{D},{L})")
        if not torch.allclose(triton_gates, ref_gates, rtol=1e-3, atol=1e-3):
            print(f"  Mismatch in cumulative_gates for vmap shape ({batch_size},{B},{D},{L})")
        else:
            print(f"Vmap shape ({batch_size},{B},{D},{L}): ✓ PASS")

def comprehensive_gradient_testing():
    print("=" * 80)
    print("Associative Scan Gradient Tests")
    print("=" * 80)

    grad_equiv_configs = [
        (2, 2, 4),
        (4, 8, 16),
        (8, 16, 32),
        (16, 32, 64),
        (32, 32, 128),
        (1, 1, 1),
        (3, 5, 7),
        (32, 2, 256),
        (64, 1, 32),
        (2, 64, 32),
        (16, 32, 2048),
        (8, 16, 16384),
    ]

    vmap_grad_equiv_configs = [
        (2, 2, 4, 4),  # (B, F, S, batch_size)
        (4, 8, 16, 2),
        (3, 5, 7, 8),
    ]

    numerical_gradcheck_configs = [
        (2, 2, 4),
        (3, 5, 7),
        (4, 4, 8),
        (1, 1, 1),
    ]

    print("\n1. Gradient Equivalence (float64)")
    print("-" * 50)
    print(f"{'Shape (B,F,S)':<20} {'Status':<10} {'Time (s)':<12}")

    all_equiv_passed = True

    for B, F, S in grad_equiv_configs:
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64, vmap=False)
        elapsed = time.time() - start_time

        size_str = f"({B},{F},{S})"
        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        time_str = f"{elapsed:.2f}"

        print(f"{size_str:<20} {result_str:<10} {time_str:<12}")

        if not grad_match:
            all_equiv_passed = False

    print("\n2. Vmap Gradient Equivalence (float64)")
    print("-" * 50)
    print(f"{'Shape (batch_size,B,F,S)':<30} {'Status':<10} {'Time (s)':<12}")

    for B, F, S, batch_size in vmap_grad_equiv_configs:
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64, vmap=True, batch_size=batch_size)
        elapsed = time.time() - start_time

        size_str = f"({batch_size},{B},{F},{S})"
        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        time_str = f"{elapsed:.2f}"

        print(f"{size_str:<30} {result_str:<10} {time_str:<12}")

        if not grad_match:
            all_equiv_passed = False

    print("\n3. Numerical Gradient Check (float64)")
    print("-" * 50)
    print(f"{'Shape (B,F,S)':<20} {'Triton':<10} {'Ref':<10} {'Time (s)':<10}")

    all_gradcheck_passed = True

    for B, F, S in numerical_gradcheck_configs:
        start_time = time.time()
        triton_success, ref_success = run_grad_check(B, F, S)
        elapsed = time.time() - start_time

        size_str = f"({B},{F},{S})"
        triton_str = "✓ PASS" if triton_success else "✗ FAIL"
        ref_str = "✓ PASS" if ref_success else "✗ FAIL"
        time_str = f"{elapsed:.2f}"

        print(f"{size_str:<20} {triton_str:<10} {ref_str:<10} {time_str:<10}")

        if not (triton_success and ref_success):
            all_gradcheck_passed = False

    print("\n4. Heavy Workload Tests (float64)")
    print("-" * 50)

    heavy_configs = [
        (64, 32, 256),
        (128, 16, 512),
        (32, 64, 128),
        (16, 32, 8192),
        (8, 16, 16384),
    ]

    for B, F, S in heavy_configs:
        print(f"Shape ({B},{F},{S}):")
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64, vmap=False)
        elapsed = time.time() - start_time

        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        print(f"  Gradient equivalence: {result_str} ({elapsed:.2f}s)")

        if not grad_match:
            all_equiv_passed = False

    print("\n" + "=" * 80)
    print("Gradient Test Summary")
    print("=" * 80)
    print(f"Gradient equivalence: {'✓ PASSED' if all_equiv_passed else '✗ FAILED'}")
    print(f"Numerical gradcheck: {'✓ PASSED' if all_gradcheck_passed else '✗ FAILED'}")
    print("=" * 80)

    return all_equiv_passed and all_gradcheck_passed

if __name__ == "__main__":
    test_correctness()
    comprehensive_gradient_testing()
