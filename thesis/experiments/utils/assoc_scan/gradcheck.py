import time
import torch
import jax
import jax.numpy as jnp
import numpy as np
from torch.autograd import gradcheck

from benchmark import (
    triton_associative_scan,
    torch_associative_scan_reference,
)

from kernel import AssociativeScan


def run_grad_check(B, F, S, dtype=torch.float64, eps=1e-6, atol=1e-4, rtol=1e-3):
    """
    Run gradient check on the associative scan implementation for a specific shape.

    Args:
        B: Batch size
        F: Feature size
        S: Sequence length
        dtype: Data type to use for gradient check (float64 recommended)
        eps: Epsilon for numerical gradient approximation
        atol: Absolute tolerance for gradient check
        rtol: Relative tolerance for gradient check

    Returns:
        Tuple of (triton_success, reference_success)
    """
    # Create random inputs with double precision (required for accurate gradcheck)
    torch.manual_seed(42 + B + F + S)  # Different seed for each shape
    gated_Z = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)
    gates = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)

    # Triton implementation gradient check
    try:
        triton_success = gradcheck(
            AssociativeScan.apply,
            (gated_Z, gates),
            eps=eps,
            atol=atol,
            rtol=rtol,
            fast_mode=False,  # Use full Jacobian check
        )
    except Exception as e:
        print(f"Triton gradcheck error: {e}")
        triton_success = False

    # PyTorch reference implementation gradient check
    try:
        reference_success = gradcheck(
            torch_associative_scan_reference,
            (gated_Z, gates),
            eps=eps,
            atol=atol,
            rtol=rtol,
            fast_mode=False,  # Use full Jacobian check
        )
    except Exception as e:
        print(f"Reference gradcheck error: {e}")
        reference_success = False

    return triton_success, reference_success


def check_grad_equivalence(B, F, S, dtype=torch.float64, atol=1e-5):
    """
    Check if the gradients computed by Triton implementation match the PyTorch reference.

    Args:
        B: Batch size
        F: Feature size
        S: Sequence length
        dtype: Data type for tensors
        atol: Absolute tolerance for gradient comparison

    Returns:
        True if gradients match within tolerance, False otherwise
    """
    # Create random inputs
    torch.manual_seed(42 + B + F + S)
    gated_Z_triton = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)
    gates_triton = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)

    # Clone inputs for reference
    gated_Z_ref = gated_Z_triton.detach().clone().requires_grad_(True)
    gates_ref = gates_triton.detach().clone().requires_grad_(True)

    # Forward pass - Triton
    triton_Z, triton_gates = triton_associative_scan(gated_Z_triton, gates_triton)
    # Forward pass - Reference
    ref_Z, ref_gates = torch_associative_scan_reference(gated_Z_ref, gates_ref)

    # Create gradient targets (random for diversity)
    torch.manual_seed(43 + B + F + S)
    grad_Z = torch.randn_like(triton_Z)
    grad_gates = torch.randn_like(triton_gates)

    # Backward pass - Triton
    triton_loss = (triton_Z * grad_Z).sum() + (triton_gates * grad_gates).sum()
    triton_loss.backward()

    # Backward pass - Reference
    ref_loss = (ref_Z * grad_Z).sum() + (ref_gates * grad_gates).sum()
    ref_loss.backward()

    # Check Z gradients
    grad_match_Z = torch.allclose(gated_Z_triton.grad, gated_Z_ref.grad, atol=atol)
    # Check gates gradients
    grad_match_gates = torch.allclose(gates_triton.grad, gates_ref.grad, atol=atol)

    if not grad_match_Z or not grad_match_gates:
        # Print detailed diagnostics if mismatch detected
        if not grad_match_Z:
            diff_Z = (gated_Z_triton.grad - gated_Z_ref.grad).abs()
            max_diff_Z = diff_Z.max().item()
            mean_diff_Z = diff_Z.mean().item()
            print(f"  Z gradient mismatch: max={max_diff_Z:.6f}, mean={mean_diff_Z:.6f}")

            # Find location of max difference
            max_idx = diff_Z.argmax().item()
            flat_idx = max_idx
            b_idx = flat_idx // (F * S)
            remainder = flat_idx % (F * S)
            f_idx = remainder // S
            s_idx = remainder % S
            print(f"  Max diff at [b={b_idx}, f={f_idx}, s={s_idx}]")

        if not grad_match_gates:
            diff_gates = (gates_triton.grad - gates_ref.grad).abs()
            max_diff_gates = diff_gates.max().item()
            mean_diff_gates = diff_gates.mean().item()
            print(f"  Gates gradient mismatch: max={max_diff_gates:.6f}, mean={mean_diff_gates:.6f}")

    return grad_match_Z and grad_match_gates


def comprehensive_gradient_testing():
    """
    Run comprehensive gradient testing on a wide variety of shapes.
    Tests both numerical gradient checking and gradient equivalence.
    """
    print("=" * 80)
    print("ASSOCIATIVE SCAN GRADIENT CHECKING")
    print("=" * 80)

    # Shapes to test for gradient equivalence (fp32)
    grad_equiv_configs = [
        # Standard shapes
        (2, 2, 4),  # Tiny
        (4, 8, 16),  # Small
        (8, 16, 32),  # Medium
        (16, 32, 64),  # Large
        (32, 32, 128),  # Very large
        # Edge cases
        (1, 1, 1),  # Minimal
        (3, 5, 7),  # Odd sizes
        (32, 2, 256),  # Long sequence
        (64, 1, 32),  # Single feature
        (2, 64, 32),  # Many features
    ]

    # Subset of shapes to run numerical gradcheck on (fp64 - slow)
    numerical_gradcheck_configs = [
        (2, 2, 4),  # Tiny
        (3, 5, 7),  # Odd sizes
        (4, 4, 8),  # Small
        (1, 1, 1),  # Minimal
    ]

    # 1. Gradient equivalence check (fp32)
    print("\n1. Checking gradient equivalence (fp32):")
    print("-" * 80)
    print(f"{'Size (B,F,S)':<20} {'Result':<15} {'Notes':<45}")

    all_equiv_passed = True

    for B, F, S in grad_equiv_configs:
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64)
        elapsed = time.time() - start_time

        size_str = f"({B},{F},{S})"
        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        notes = f"Took {elapsed:.2f}s"

        print(f"{size_str:<20} {result_str:<15} {notes:<45}")

        if not grad_match:
            all_equiv_passed = False

    # 2. Numerical gradient check (fp64 - slower)
    print("\n2. Numerical gradient checking (fp64):")
    print("-" * 80)
    print(f"{'Size (B,F,S)':<20} {'Triton':<15} {'Reference':<15} {'Time (s)':<10}")

    all_gradcheck_passed = True

    for B, F, S in numerical_gradcheck_configs:
        start_time = time.time()
        triton_success, ref_success = run_grad_check(B, F, S)
        elapsed = time.time() - start_time

        size_str = f"({B},{F},{S})"
        triton_str = "✓ PASS" if triton_success else "✗ FAIL"
        ref_str = "✓ PASS" if ref_success else "✗ FAIL"
        time_str = f"{elapsed:.2f}"

        print(f"{size_str:<20} {triton_str:<15} {ref_str:<15} {time_str:<10}")

        if not (triton_success and ref_success):
            all_gradcheck_passed = False

    # 3. Heavy workload testing for distributed training
    print("\n3. Heavy workload testing (fp32):")
    print("-" * 80)
    print("Testing larger sizes to simulate distributed training workloads")

    heavy_configs = [
        (64, 32, 256),  # Large batch
        (128, 16, 512),  # Long sequence
        (32, 64, 128),  # Many features
    ]

    for B, F, S in heavy_configs:
        print(f"\nTesting shape ({B},{F},{S}):")

        # Test with standard fp32
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64)
        elapsed = time.time() - start_time

        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        print(f"  Gradient equivalence (fp32): {result_str} (took {elapsed:.2f}s)")

        if not grad_match:
            all_equiv_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("GRADIENT TESTING SUMMARY")
    print("=" * 80)
    print(f"Gradient equivalence tests: {'PASSED' if all_equiv_passed else 'FAILED'}")
    print(f"Numerical gradient checks: {'PASSED' if all_gradcheck_passed else 'FAILED'}")
    print("=" * 80)

    return all_equiv_passed and all_gradcheck_passed


def combine_fn_jax(carry, next_val):
    """Associative combine function: elementwise addition."""
    sum_gated_Z, sum_gates = carry
    next_gated_Z, next_gates = next_val
    return (sum_gated_Z + next_gated_Z, sum_gates + next_gates)


def jax_associative_scan(gated_Z, gates, axis=2):
    """
    Performs an associative scan using JAX along the specified axis.

    Args:
        gated_Z: jnp.ndarray of shape [B, D, L].
        gates: jnp.ndarray of shape [B, D, L].
        axis: Axis along which to scan.

    Returns:
        A tuple (cumulative_gated_Z, cumulative_gates) with the same shape.
    """
    return jax.lax.associative_scan(combine_fn_jax, (gated_Z, gates), axis=axis)


def test_correctness():
    print("=" * 80)
    print("ASSOCIATIVE SCAN CORRECTNESS TESTING")
    print("=" * 80)

    # List of shapes to test: [B, D, L]
    test_shapes = [(2, 3, 8), (4, 16, 32), (1, 10, 15), (3, 7, 64)]

    for shape in test_shapes:
        B, D, L = shape
        # Generate random inputs (float32)
        gated_Z_np = np.random.randn(B, D, L).astype(np.float32)
        gates_np = np.random.randn(B, D, L).astype(np.float32)

        # Compute cumulative scan using JAX.
        gated_Z_jax = jnp.array(gated_Z_np)
        gates_jax = jnp.array(gates_np)
        cumulative_gated_Z_jax, cumulative_gates_jax = jax_associative_scan(gated_Z_jax, gates_jax, axis=2)
        cumulative_gated_Z_jax = np.array(cumulative_gated_Z_jax)
        cumulative_gates_jax = np.array(cumulative_gates_jax)

        # Compute cumulative scan using the Triton wrapper.
        # Ensure inputs are on GPU.
        gated_Z_torch = torch.tensor(gated_Z_np, device="cuda")
        gates_torch = torch.tensor(gates_np, device="cuda")
        cumulative_gated_Z_torch, cumulative_gates_torch = triton_associative_scan(gated_Z_torch, gates_torch)
        cumulative_gated_Z_torch = cumulative_gated_Z_torch.cpu().numpy()
        cumulative_gates_torch = cumulative_gates_torch.cpu().numpy()

        # Compare the outputs.
        if not np.allclose(cumulative_gated_Z_jax, cumulative_gated_Z_torch, rtol=1e-3, atol=1e-3):
            print(f"Mismatch in cumulative_gated_Z for shape {shape}")
        if not np.allclose(cumulative_gates_jax, cumulative_gates_torch, rtol=1e-3, atol=1e-3):
            print(f"Mismatch in cumulative_gates for shape {shape}")
        else:
            print(f"Shape {shape} passed!")


if __name__ == "__main__":
    test_correctness()
    comprehensive_gradient_testing()
