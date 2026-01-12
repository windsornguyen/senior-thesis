# -*- Author: Windsor Nguyen -*-
"""
Implements an associative scan using Triton with autotuning and vmap support.
Performs prefix sums over the sequence dimension of input tensors using elementwise addition.
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import torch.nn.functional as F
from typing import Tuple


# Data type mapping for Triton
dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def combine_fn(Z_x, g_x, g_norm_x, Z_y, g_y, g_norm_y):
    """Combine function for associative scan: elementwise addition."""
    return Z_x + Z_y, g_x + g_y, g_norm_x + g_norm_y


def get_scan_configs():
    """Define autotune configurations for Triton kernel."""
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
        )
    return configs


def keep_config(conf, seq_len=None):
    """Filter out invalid autotune configurations."""
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    if seq_len is not None and BLOCK_SIZE > seq_len:
        return False
    return True


@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def fwd_scan_kernel(
    Z_ptr,
    g_ptr,
    g_norm_ptr,
    out_Z_ptr,
    out_g_ptr,
    out_g_norm_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_f: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Forward associative scan kernel."""
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    offset = pid_batch * stride_b + pid_feature * stride_f
    offs = offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < seq_len

    Z = tl.load(Z_ptr + offs * stride_l, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs * stride_l, mask=mask, other=0.0)
    g_norm = tl.load(g_norm_ptr + offs * stride_l, mask=mask, other=0.0)

    res_Z, res_g, res_g_norm = tl.associative_scan((Z, g, g_norm), axis=0, combine_fn=combine_fn, reverse=False)

    tl.store(out_Z_ptr + offs * stride_l, res_Z, mask=mask)
    tl.store(out_g_ptr + offs * stride_l, res_g, mask=mask)
    tl.store(out_g_norm_ptr + offs * stride_l, res_g_norm, mask=mask)


@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def bwd_scan_kernel(
    grad_cumul_Z_ptr,
    grad_cumul_g_ptr,
    grad_cumul_g_norm_ptr,
    grad_Z_ptr,
    grad_g_ptr,
    grad_g_norm_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_d: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Backward associative scan kernel (reverse scan)."""
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    base = pid_batch * stride_b + pid_feature * stride_d

    carry_Z = tl.zeros([1], dtype=DTYPE)
    carry_g = tl.zeros([1], dtype=DTYPE)
    carry_g_norm = tl.zeros([1], dtype=DTYPE)

    for start in tl.range(0, seq_len, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < seq_len
        rev_idx = seq_len - 1 - idx
        safe_idx = tl.where(mask, rev_idx, 0)
        offs = base + safe_idx * stride_l

        gZ = tl.load(grad_cumul_Z_ptr + offs, mask=mask, other=0.0)
        gg = tl.load(grad_cumul_g_ptr + offs, mask=mask, other=0.0)
        gg_norm = tl.load(grad_cumul_g_norm_ptr + offs, mask=mask, other=0.0)

        res_Z, res_g, res_g_norm = tl.associative_scan((gZ, gg, gg_norm), axis=0, combine_fn=combine_fn, reverse=False)

        res_Z += carry_Z
        res_g += carry_g
        res_g_norm += carry_g_norm

        tl.store(grad_Z_ptr + offs, res_Z, mask=mask)
        tl.store(grad_g_ptr + offs, res_g, mask=mask)
        tl.store(grad_g_norm_ptr + offs, res_g_norm, mask=mask)

        last_valid = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        if last_valid >= 0:
            lv_mask = tl.arange(0, BLOCK_SIZE) == last_valid
            carry_Z = tl.sum(tl.where(lv_mask, res_Z, 0), axis=0, keep_dims=True)
            carry_g = tl.sum(tl.where(lv_mask, res_g, 0), axis=0, keep_dims=True)
            carry_g_norm = tl.sum(tl.where(lv_mask, res_g_norm, 0), axis=0, keep_dims=True)


class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for associative scan using Triton."""

    @staticmethod
    def forward(ctx, Z: torch.Tensor, g: torch.Tensor, g_norm: torch.Tensor):
        batch_size, feature_size, seq_len = Z.shape
        Z, g, g_norm = Z.contiguous().cuda(), g.contiguous().cuda(), g_norm.contiguous().cuda()
        out_Z, out_g, out_g_norm = torch.empty_like(Z), torch.empty_like(g), torch.empty_like(g_norm)

        grid = (batch_size, feature_size)
        triton_dtype = dtype_map.get(Z.dtype, tl.float32)

        fwd_scan_kernel[grid](
            Z,
            g,
            g_norm,
            out_Z,
            out_g,
            out_g_norm,
            batch_size,
            feature_size,
            seq_len,
            Z.stride(0),
            Z.stride(1),
            Z.stride(2),
            DTYPE=triton_dtype,
        )

        ctx.save_for_backward(Z, g, g_norm)
        ctx.triton_dtype = triton_dtype
        return out_Z, out_g, out_g_norm

    @staticmethod
    def backward(ctx, grad_out_Z: torch.Tensor, grad_out_g: torch.Tensor, grad_out_g_norm: torch.Tensor):
        Z, g, g_norm = ctx.saved_tensors
        batch_size, feature_size, seq_len = Z.shape

        grad_out_Z, grad_out_g, grad_out_g_norm = (
            grad_out_Z.contiguous().cuda(),
            grad_out_g.contiguous().cuda(),
            grad_out_g_norm.contiguous().cuda(),
        )
        grad_Z, grad_g, grad_g_norm = torch.empty_like(Z), torch.empty_like(g), torch.empty_like(g_norm)

        grid = (batch_size, feature_size)

        bwd_scan_kernel[grid](
            grad_out_Z,
            grad_out_g,
            grad_out_g_norm,
            grad_Z,
            grad_g,
            grad_g_norm,
            batch_size,
            feature_size,
            seq_len,
            grad_out_Z.stride(0),
            grad_out_Z.stride(1),
            grad_out_Z.stride(2),
            DTYPE=ctx.triton_dtype,
        )

        return grad_Z, grad_g, grad_g_norm

    @staticmethod
    def _move_bdim_to_front(x, bdim):
        """Move vmap dim to dim-0."""
        if bdim is None:
            return x, None
        return x.movedim(bdim, 0), 0

    @staticmethod
    def _flatten_leading_dims(x, keep=2):
        """Collapse all dims except the last `keep` dims into one."""
        batch_shape = x.shape[:-keep]
        flat = x.reshape(-1, *x.shape[-keep:]) if batch_shape else x
        return flat, batch_shape

    @staticmethod
    def vmap(info, in_dims, Z, g, g_norm):
        """Batched rule for torch.vmap."""
        bdim = in_dims[0]
        assert bdim == in_dims[1] == in_dims[2], "Z, g, and g_norm must share the same vmap dim"

        Z, _ = AssociativeScan._move_bdim_to_front(Z, bdim)
        g, _ = AssociativeScan._move_bdim_to_front(g, bdim)
        g_norm, _ = AssociativeScan._move_bdim_to_front(g_norm, bdim)

        Z, batch_shape = AssociativeScan._flatten_leading_dims(Z, keep=2)
        g, _ = AssociativeScan._flatten_leading_dims(g, keep=2)
        g_norm, _ = AssociativeScan._flatten_leading_dims(g_norm, keep=2)

        Z_cumul, g_cumul, g_norm_cumul = AssociativeScan.apply(Z, g, g_norm)

        if batch_shape:
            Z_cumul = Z_cumul.view(*batch_shape, *Z_cumul.shape[-2:])
            g_cumul = g_cumul.view(*batch_shape, *g_cumul.shape[-2:])
            g_norm_cumul = g_norm_cumul.view(*batch_shape, *g_norm_cumul.shape[-2:])

        if bdim is not None and bdim != 0:
            Z_cumul = Z_cumul.movedim(0, bdim)
            g_cumul = g_cumul.movedim(0, bdim)
            g_norm_cumul = g_norm_cumul.movedim(0, bdim)

        return (Z_cumul, g_cumul, g_norm_cumul), (bdim, bdim, bdim)


def associative_scan(Z: torch.Tensor, g: torch.Tensor, g_norm: torch.Tensor):
    """Executes an associative scan on the provided tensors."""
    return AssociativeScan.apply(Z, g, g_norm)


# Reference Implementation
from thesis.experiments.utils.assoc_scan.kernel import associative_scan as assoc_scan_ref


def combine_fn_ref(x, y):
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y
    m_new = torch.maximum(m_x, m_y)
    exp_x = torch.exp(m_x - m_new)
    exp_y = torch.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x.unsqueeze(-1) + n_y * exp_y.unsqueeze(-1)
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new


def scan_fn(
    qk_slice: torch.Tensor, v_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs associative scan over one (B,H) stream."""
    leaves = (qk_slice, torch.ones_like(qk_slice), v_slice, Z_slice, g_slice)
    return assoc_scan_ref(combine_fn=combine_fn_ref, xs=leaves, dim=0, combine_mode="generic")


def batched_scan_fn(
    sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs scan_fn independently for every (B,H) stream."""
    B, H, L, h = v.shape
    sim_bh = sim.reshape(-1, L)
    v_bh = v.reshape(-1, L, h)
    gated_Z_bh = gated_Z.reshape(-1, L, h, h)
    gates_z_bh = gates_z.reshape(-1, L)

    scan_vmapped = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    m_res, s_res, n_res, Z_res, g_res = scan_vmapped(sim_bh, v_bh, gated_Z_bh, gates_z_bh)

    return (
        m_res.reshape(B, H, L),
        s_res.reshape(B, H, L),
        n_res.reshape(B, H, L, h),
        Z_res.reshape(B, H, L, h, h),
        g_res.reshape(B, H, L),
    )


def lrelu2(x: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
    """Squared leaky ReLU activation."""
    out = torch.where(x > 0, x, alpha * x)
    return out * out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-6

    # Test Dimensions
    B_test, H_test, L_test, h_test = 2, 4, 8, 16
    D_test = H_test * h_test

    # Create Test Inputs
    x_tilde = torch.randn(B_test, L_test, D_test, device=device)
    wq = torch.randn(D_test, D_test, device=device)
    wk = torch.randn(D_test, D_test, device=device)
    wv = torch.randn(D_test, D_test, device=device)
    wg_z = torch.randn(h_test * h_test, 1, device=device)

    # Compute q, k, v
    q = torch.matmul(x_tilde, wq).reshape(B_test, L_test, H_test, h_test).transpose(1, 2)
    k = torch.matmul(x_tilde, wk).reshape(B_test, L_test, H_test, h_test).transpose(1, 2)
    v = torch.matmul(x_tilde, wv).reshape(B_test, L_test, H_test, h_test).transpose(1, 2)
    q, k, v = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(v, dim=-1)

    # Compute Similarity and Z
    sim = torch.einsum("bhld,bhld->bhl", q, k) * (h_test**-0.5)
    Z = torch.einsum("bhsn,bhsp->bhspn", k, v)

    # Compute Gates
    gate_input_z = Z.reshape(B_test, H_test, L_test, -1)
    gates_logits_z = torch.matmul(gate_input_z, wg_z)
    gates_z = lrelu2(gates_logits_z) + eps
    gates_z = gates_z.squeeze(-1)

    # Apply Gates to Z
    gated_Z = Z * gates_z.unsqueeze(-1).unsqueeze(-1)

    # Prepare Inputs for Scan
    gated_Z_flat = gated_Z.reshape(B_test * H_test, L_test, h_test * h_test).transpose(1, 2).contiguous()
    gates_z_flat = gates_z.reshape(B_test * H_test, 1, L_test).expand(-1, h_test * h_test, -1)

    # Run Triton Forward Pass
    print("\nRunning Triton forward pass...")
    Z_cumul_flat_triton, gate_cumul_flat_triton, _ = associative_scan(gated_Z_flat, gates_z_flat, gates_z_flat)
    Z_cumul_triton = Z_cumul_flat_triton.transpose(1, 2).reshape(B_test, H_test, L_test, h_test, h_test)
    gate_cumul_triton = gate_cumul_flat_triton[:, 0, :].reshape(B_test, H_test, L_test)

    # Run Reference Forward Pass
    print("Running reference forward pass...")
    max_cumul_ref, norm_cumul_ref, v_cumul_ref, Z_cumul_ref, gate_cumul_ref = batched_scan_fn(sim, v, gated_Z, gates_z)

    # Compare Components
    print("\nComparing forward pass components...")
    try:
        triton.testing.assert_close(Z_cumul_triton, Z_cumul_ref, rtol=1e-5, atol=1e-5)
        print("Z_cumul components MATCH!")
        triton.testing.assert_close(gate_cumul_triton, gate_cumul_ref, rtol=1e-5, atol=1e-5)
        print("gate_cumul components MATCH!")
        H_triton = Z_cumul_triton / (gate_cumul_triton.unsqueeze(-1).unsqueeze(-1) + eps)
        H_ref = Z_cumul_ref / (gate_cumul_ref.unsqueeze(-1).unsqueeze(-1) + eps)
        triton.testing.assert_close(H_triton, H_ref, rtol=1e-5, atol=1e-5)
        print("Final H MATCHES!")
    except AssertionError as e:
        print("Components DO NOT MATCH:", e)
        print("\nTriton Outputs:")
        print(
            f"Z_cumul_triton shape: {Z_cumul_triton.shape}, min: {Z_cumul_triton.min().item():.6f}, max: {Z_cumul_triton.max().item():.6f}"
        )
        print(
            f"gate_cumul_triton shape: {gate_cumul_triton.shape}, min: {gate_cumul_triton.min().item():.6f}, max: {gate_cumul_triton.max().item():.6f}"
        )
        print("\nReference Outputs:")
        print(
            f"Z_cumul_ref shape: {Z_cumul_ref.shape}, min: {Z_cumul_ref.min().item():.6f}, max: {Z_cumul_ref.max().item():.6f}"
        )
        print(
            f"gate_cumul_ref shape: {gate_cumul_ref.shape}, min: {gate_cumul_ref.min().item():.6f}, max: {gate_cumul_ref.max().item():.6f}"
        )

    # Gradient Check
    print("\n--- Gradient Check ---")
    B_grad, H_grad, L_grad, h_grad = 2, 2, 4, 8
    h2_grad = h_grad * h_grad
    gated_Z_grad = torch.randn(
        B_grad * H_grad, h2_grad, L_grad, device=device, dtype=torch.float64, requires_grad=True
    )
    gates_z_grad = torch.randn(
        B_grad * H_grad, h2_grad, L_grad, device=device, dtype=torch.float64, requires_grad=True
    )

    def f(gated_Z, gates_z):
        Z_cumul, gate_cumul, _ = associative_scan(gated_Z, gates_z, gates_z)
        return Z_cumul + gate_cumul

    print("Running gradcheck...")
    assert torch.autograd.gradcheck(f, (gated_Z_grad, gates_z_grad), eps=1e-6, atol=1e-4, rtol=1e-3), (
        "gradcheck failed!"
    )
    print("Gradcheck passed!")

    # Benchmark
    print("\n--- Benchmarking Forward Pass ---")
    B_bench, H_bench, L_bench, h_bench = 4, 2, 1024, 128
    bench_dtype = torch.float32
    print(f"Benchmarking with: B={B_bench}, H={H_bench}, L={L_bench}, h={h_bench}, dtype={bench_dtype}")

    # Create Benchmark Inputs
    x_tilde_bench = torch.randn(B_bench, L_bench, H_bench * h_bench, dtype=bench_dtype, device=device)
    wq_bench = torch.randn(H_bench * h_bench, H_bench * h_bench, dtype=bench_dtype, device=device)
    wk_bench = torch.randn(H_bench * h_bench, H_bench * h_bench, dtype=bench_dtype, device=device)
    wv_bench = torch.randn(H_bench * h_bench, H_bench * h_bench, dtype=bench_dtype, device=device)
    wg_z_bench = torch.randn(h_bench * h_bench, 1, dtype=bench_dtype, device=device)

    # Compute q, k, v
    q_bench = torch.matmul(x_tilde_bench, wq_bench).reshape(B_bench, L_bench, H_bench, h_bench).transpose(1, 2)
    k_bench = torch.matmul(x_tilde_bench, wk_bench).reshape(B_bench, L_bench, H_bench, h_bench).transpose(1, 2)
    v_bench = torch.matmul(x_tilde_bench, wv_bench).reshape(B_bench, L_bench, H_bench, h_bench).transpose(1, 2)
    q_bench, k_bench, v_bench = (
        F.normalize(q_bench, dim=-1),
        F.normalize(k_bench, dim=-1),
        F.normalize(v_bench, dim=-1),
    )

    # Compute Similarity and Z
    sim_bench = torch.einsum("bhld,bhld->bhl", q_bench, k_bench) * (h_bench**-0.5)
    Z_bench = torch.einsum("bhsn,bhsp->bhspn", k_bench, v_bench)

    # Compute Gates
    gate_input_z_bench = Z_bench.reshape(B_bench, H_bench, L_bench, -1)
    gates_logits_z_bench = torch.matmul(gate_input_z_bench, wg_z_bench)
    gates_z_bench = lrelu2(gates_logits_z_bench) + eps
    gates_z_bench = gates_z_bench.squeeze(-1)

    # Apply Gates to Z
    gated_Z_bench = Z_bench * gates_z_bench.unsqueeze(-1).unsqueeze(-1)

    # Prepare Inputs for Triton Kernel
    gated_Z_flat_bench = (
        gated_Z_bench.reshape(B_bench * H_bench, L_bench, h_bench * h_bench).transpose(1, 2).contiguous()
    )
    gates_z_flat_bench = gates_z_bench.reshape(B_bench * H_bench, 1, L_bench).expand(-1, h_bench * h_bench, -1)

    bench_fn = lambda: associative_scan(gated_Z_flat_bench, gates_z_flat_bench, gates_z_flat_bench)
    bench_fn_ref = lambda: batched_scan_fn(sim_bench, v_bench, gated_Z_bench, gates_z_bench)

    print("Running initial call for compilation/autotuning...")
    _ = bench_fn()
    _ = bench_fn_ref()
    print("Starting benchmark runs...")
    time_triton = do_bench(bench_fn)
    time_ref = do_bench(bench_fn_ref)
    print(f"Time Triton Forward: {time_triton:.4f} ms")
    print(f"Time Reference Forward: {time_ref:.4f} ms")

    bytes_input = sum(t.numel() * t.element_size() for t in [gated_Z_flat_bench, gates_z_flat_bench])
    bytes_output = bytes_input
    total_bytes = bytes_input + bytes_output
    gb_transferred = total_bytes / (1024**3)
    print(f"Total memory transfer: {gb_transferred:.2f} GB")

    h100_bw_gbs = 3350
    ideal_time_ms = (total_bytes / (h100_bw_gbs * (1024**3))) * 1000
    print(f"Ideal time on H100 ({h100_bw_gbs} GB/s): {ideal_time_ms:.4f} ms")
    print(f"Bandwidth util of Triton Forward: {ideal_time_ms / time_triton * 100:.2f} %")
    print(f"Bandwidth util of Reference Forward: {ideal_time_ms / time_ref * 100:.2f} %")
