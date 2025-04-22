import time
import torch
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from torch.autograd import gradcheck
from jax import jit, grad

from benchmark import (
    triton_associative_scan,
    torch_associative_scan_reference,
)

from kernel_simple import AssociativeScan

jax.config.update("jax_enable_x64", True)

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


def check_grad_equivalence(B, F, S, dtype=torch.float64, atol=1e-5, use_jax_reference=False):
    torch.manual_seed(42 + B + F + S)
    gated_Z_triton = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)
    gates_triton = torch.randn(B, F, S, dtype=dtype, device="cuda", requires_grad=True)
    gated_Z_ref = gated_Z_triton.detach().clone().requires_grad_(True)
    gates_ref = gates_triton.detach().clone().requires_grad_(True)

    triton_Z, triton_gates = triton_associative_scan(gated_Z_triton, gates_triton)

    if use_jax_reference:
        gated_Z_jax = jnp.array(gated_Z_ref.cpu().detach().numpy(), dtype=jnp.float64)
        gates_jax = jnp.array(gates_ref.cpu().detach().numpy(), dtype=jnp.float64)
        ref_Z_jax, ref_gates_jax = jax_associative_scan(gated_Z_jax, gates_jax, axis=2)
        ref_Z = torch.tensor(np.array(ref_Z_jax), device="cuda", dtype=dtype)
        ref_gates = torch.tensor(np.array(ref_gates_jax), device="cuda", dtype=dtype)
    else:
        ref_Z, ref_gates = torch_associative_scan_reference(gated_Z_ref, gates_ref)

    torch.manual_seed(43 + B + F + S)
    grad_Z = torch.randn_like(triton_Z)
    grad_gates = torch.randn_like(triton_gates)

    triton_loss = (triton_Z * grad_Z).sum() + (triton_gates * grad_gates).sum()
    triton_loss.backward()

    if use_jax_reference:
        def loss_fn(z, g, grad_z, grad_g):
            out_z, out_g = jax_associative_scan(z, g, axis=2)
            return (out_z * grad_z).sum() + (out_g * grad_g).sum()

        grad_fn = grad(loss_fn, argnums=(0, 1))
        grad_Z_jax, grad_gates_jax = grad_fn(
            gated_Z_jax, gates_jax,
            jnp.array(grad_Z.cpu().numpy(), dtype=jnp.float64),
            jnp.array(grad_gates.cpu().numpy(), dtype=jnp.float64)
        )
        grad_Z_ref = torch.tensor(np.array(grad_Z_jax), device="cuda", dtype=dtype)
        grad_gates_ref = torch.tensor(np.array(grad_gates_jax), device="cuda", dtype=dtype)
    else:
        ref_loss = (ref_Z * grad_Z).sum() + (ref_gates * grad_gates).sum()
        ref_loss.backward()
        grad_Z_ref = gated_Z_ref.grad
        grad_gates_ref = gates_ref.grad

    grad_match_Z = torch.allclose(gated_Z_triton.grad, grad_Z_ref, atol=atol)
    grad_match_gates = torch.allclose(gates_triton.grad, grad_gates_ref, atol=atol)

    if not grad_match_Z or not grad_match_gates:
        if not grad_match_Z:
            diff_Z = (gated_Z_triton.grad - grad_Z_ref).abs()
            print(f"  Z gradient mismatch for shape ({B},{F},{S}): max={diff_Z.max():.6f}, mean={diff_Z.mean():.6f}")
        if not grad_match_gates:
            diff_gates = (gates_triton.grad - grad_gates_ref).abs()
            print(f"  Gates gradient mismatch for shape ({B},{F},{S}): max={diff_gates.max():.6f}, mean={diff_gates.mean():.6f}")

    return grad_match_Z and grad_match_gates


def verify_jax_reference(B, F, S, dtype=torch.float64, atol=1e-5):
    torch.manual_seed(42 + B + F + S)
    gated_Z_torch = torch.randn(B, F, S, dtype=dtype, device="cuda")
    gates_torch = torch.randn(B, F, S, dtype=dtype, device="cuda")

    ref_Z, ref_gates = torch_associative_scan_reference(gated_Z_torch, gates_torch)
    
    gated_Z_jax = jnp.array(gated_Z_torch.cpu().numpy(), dtype=jnp.float64)
    gates_jax = jnp.array(gates_torch.cpu().numpy(), dtype=jnp.float64)
    jax_Z, jax_gates = jax_associative_scan(gated_Z_jax, gates_jax, axis=2)
    
    jax_Z = torch.tensor(np.array(jax_Z), device="cuda", dtype=dtype)
    jax_gates = torch.tensor(np.array(jax_gates), device="cuda", dtype=dtype)

    return torch.allclose(ref_Z, jax_Z, atol=atol) and torch.allclose(ref_gates, jax_gates, atol=atol)


def comprehensive_gradient_testing():
    print("=" * 80)
    print("Associative Scan Gradient Tests")
    print("=" * 80)

    short_seq_shape = (4, 8, 16)
    jax_reference_valid = verify_jax_reference(*short_seq_shape)
    print(f"JAX vs PyTorch reference (short sequences): {'✓ PASS' if jax_reference_valid else '✗ FAIL'}")

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
        use_jax = jax_reference_valid and S > 128
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64, use_jax_reference=use_jax)
        elapsed = time.time() - start_time

        size_str = f"({B},{F},{S})"
        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        time_str = f"{elapsed:.2f}{' (JAX)' if use_jax else ''}"

        print(f"{size_str:<20} {result_str:<10} {time_str:<12}")

        if not grad_match:
            all_equiv_passed = False

    print("\n2. Numerical Gradient Check (float64)")
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

    print("\n3. Heavy Workload Tests (float64)")
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
        use_jax = jax_reference_valid and S > 128
        start_time = time.time()
        grad_match = check_grad_equivalence(B, F, S, dtype=torch.float64, use_jax_reference=use_jax)
        elapsed = time.time() - start_time

        result_str = "✓ PASS" if grad_match else "✗ FAIL"
        print(f"  Gradient equivalence: {result_str} ({elapsed:.2f}s{' (JAX)' if use_jax else ''})")

        if not grad_match:
            all_equiv_passed = False

    print("\n" + "=" * 80)
    print("Gradient Test Summary")
    print("=" * 80)
    print(f"Gradient equivalence: {'✓ PASSED' if all_equiv_passed else '✗ FAILED'}")
    print(f"Numerical gradcheck: {'✓ PASSED' if all_gradcheck_passed else '✗ FAILED'}")
    print("=" * 80)

    return all_equiv_passed and all_gradcheck_passed


@jit
def combine_fn_jax(carry, next_val):
    sum_gated_Z, sum_gates = carry
    next_gated_Z, next_gates = next_val
    return (sum_gated_Z + next_gated_Z, sum_gates + next_gates)


@partial(jit, static_argnums=(2,))
def jax_associative_scan(gated_Z, gates, axis=2):
    return jax.lax.associative_scan(combine_fn_jax, (gated_Z, gates), axis=axis)


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

    for shape in test_shapes:
        B, D, L = shape
        gated_Z_np = np.random.randn(B, D, L).astype(np.float32)
        gates_np = np.random.randn(B, D, L).astype(np.float32)

        gated_Z_jax = jnp.array(gated_Z_np)
        gates_jax = jnp.array(gates_np)
        cumulative_gated_Z_jax, cumulative_gates_jax = jax_associative_scan(gated_Z_jax, gates_jax, axis=2)
        cumulative_gated_Z_jax = np.array(cumulative_gated_Z_jax)
        cumulative_gates_jax = np.array(cumulative_gates_jax)

        gated_Z_torch = torch.tensor(gated_Z_np, device="cuda")
        gates_torch = torch.tensor(gates_np, device="cuda")
        cumulative_gated_Z_torch, cumulative_gates_torch = triton_associative_scan(gated_Z_torch, gates_torch)
        cumulative_gated_Z_torch = cumulative_gated_Z_torch.cpu().numpy()
        cumulative_gates_torch = cumulative_gates_torch.cpu().numpy()

        if not np.allclose(cumulative_gated_Z_jax, cumulative_gated_Z_torch, rtol=1e-3, atol=1e-3):
            print(f"  Mismatch in cumulative_gated_Z for shape {shape}")
        if not np.allclose(cumulative_gates_jax, cumulative_gates_torch, rtol=1e-3, atol=1e-3):
            print(f"  Mismatch in cumulative_gates for shape {shape}")
        else:
            print(f"Shape {shape}: ✓ PASS")


if __name__ == "__main__":
    test_correctness()
    comprehensive_gradient_testing()
