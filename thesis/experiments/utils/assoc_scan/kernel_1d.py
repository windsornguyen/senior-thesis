# -*- Author: Windsor Nguyen -*-

"""
Implements an associative scan using Triton with autotuning and vmap support.
Performs prefix sums over the sequence dimension of input tensors using elementwise addition.
"""

import torch
import triton
import triton.language as tl


dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def combine_fn(carry_gated_Z, carry_gates, next_gated_Z, next_gates):
    """Combine function for the associative scan: elementwise addition."""
    return (carry_gated_Z + next_gated_Z, carry_gates + next_gates)


# Define autotune configurations, inspired by 05-layer-norm.py and 06-fused-attention.py
def get_scan_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024]
        for s in [2, 3, 4]  # Vary pipeline stages for latency hiding
        for w in [4, 8]  # Common warp counts for CUDA
    ]
    # Add HIP-specific configs for AMD GPUs, as in 06-fused-attention.py
    # if triton.runtime.driver.active.get_current_target().backend == "hip":
    #     configs.extend(
    #         [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
    #     )
    return configs


# Filter configs to avoid bad combinations, as in 06-fused-attention.py
def keep_config(conf, seq_len=None):
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    # Ensure larger block sizes use sufficient warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    # Filter out configs with too small block sizes for large warps
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    # Ensure BLOCK_SIZE <= seq_len
    if seq_len is not None and BLOCK_SIZE > seq_len:
        return False
    return True

@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def fwd_scan_kernel(
    gated_Z_ptr, gates_ptr, out_gated_Z_ptr, out_gates_ptr,
    batch_size: int, feature_size: int, seq_len: int,
    stride_b: int, stride_d: int, stride_l: int,
    BLOCK_SIZE: tl.constexpr, DTYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d

    cumulative_gated_Z = tl.zeros([1], dtype=DTYPE)
    cumulative_gates = tl.zeros([1], dtype=DTYPE)

    for start_idx in tl.range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        indices = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = indices < seq_len
        offsets = start_offset + indices * stride_l

        gated_Z = tl.load(gated_Z_ptr + offsets, mask=mask, other=0.0)
        gates = tl.load(gates_ptr + offsets, mask=mask, other=0.0)

        result_gated_Z, result_gates = tl.associative_scan(
            (gated_Z, gates), axis=0, combine_fn=combine_fn, reverse=False
        )

        result_gated_Z += cumulative_gated_Z
        result_gates += cumulative_gates
        tl.store(out_gated_Z_ptr + offsets, result_gated_Z, mask=mask)
        tl.store(out_gates_ptr + offsets, result_gates, mask=mask)

        last_valid_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_idx - 1)
        if last_valid_idx >= 0:
            last_valid_mask = tl.arange(0, BLOCK_SIZE) == last_valid_idx
            cumulative_gated_Z = tl.sum(tl.where(last_valid_mask, result_gated_Z, 0.0), axis=0, keep_dims=True)
            cumulative_gates = tl.sum(tl.where(last_valid_mask, result_gates, 0.0), axis=0, keep_dims=True)

@triton.autotune(
    configs=list(filter(lambda conf: keep_config(conf), get_scan_configs())),
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def bwd_scan_kernel(
    grad_cumul_Z_ptr,
    grad_cumul_g_ptr,
    grad_Z_ptr,
    grad_g_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_d: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Right‑to‑left prefix‑sum of gradients (reverse scan)."""
    
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    base = pid_batch * stride_b + pid_feature * stride_d

    # running carry (Z, g) coming from the *right* of the current block
    carry_Z = tl.zeros([1], dtype=DTYPE)
    carry_g = tl.zeros([1], dtype=DTYPE)

    # iterate over chunks from *left* to *right*, but address memory reversed
    for start in tl.range(0, seq_len, BLOCK_SIZE):
        # logical (left‑to‑right) indices of this chunk
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < seq_len
        # map to physical indices counted from the end
        rev_idx = seq_len - 1 - idx
        safe_idx = tl.where(mask, rev_idx, 0)  # clamp for OOB safety
        offs = base + safe_idx * stride_l

        # --- load reversed gradients -----------------------------------
        gZ = tl.load(grad_cumul_Z_ptr + offs, mask=mask, other=0.0)
        gg = tl.load(grad_cumul_g_ptr + offs, mask=mask, other=0.0)

        # --- intra‑block forward scan (because we already reversed order)
        res_Z, res_g = tl.associative_scan((gZ, gg), axis=0, combine_fn=combine_fn, reverse=False)

        # --- add carry from the right ----------------------------------
        res_Z += carry_Z
        res_g += carry_g

        # --- store back to the same reversed locations -----------------
        tl.store(grad_Z_ptr + offs, res_Z, mask=mask)
        tl.store(grad_g_ptr + offs, res_g, mask=mask)

        # --- update carry with the *last* valid element in this chunk --
        last_valid = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        if last_valid >= 0:
            lv_mask = tl.arange(0, BLOCK_SIZE) == last_valid
            carry_Z = tl.sum(tl.where(lv_mask, res_Z, 0), axis=0, keep_dims=True)
            carry_g = tl.sum(tl.where(lv_mask, res_g, 0), axis=0, keep_dims=True)


class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for an associative scan using Triton."""

    @staticmethod
    def forward(gated_Z: torch.Tensor, gates: torch.Tensor):
        batch_size, feature_size, seq_len = gated_Z.shape
        gated_Z = gated_Z.contiguous().cuda()
        gates = gates.contiguous().cuda()
        cumulative_gated_Z = torch.empty_like(gated_Z)
        cumulative_gates = torch.empty_like(gates)

        grid = (batch_size, feature_size, 1)
        triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)

        fwd_scan_kernel[grid](
            gated_Z,
            gates,
            cumulative_gated_Z,
            cumulative_gates,
            batch_size,
            feature_size,
            seq_len,
            gated_Z.stride(0),
            gated_Z.stride(1),
            gated_Z.stride(2),
            DTYPE=triton_dtype,
        )

        return cumulative_gated_Z, cumulative_gates

    @staticmethod
    def setup_context(ctx, inputs, output):
        gated_Z, gates = inputs
        cumulative_gated_Z, cumulative_gates = output
        ctx.save_for_backward(gated_Z, gates)
        ctx.triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)

    @staticmethod
    def backward(ctx, grad_cumulative_gated_Z: torch.Tensor, grad_cumulative_gates: torch.Tensor):
        gated_Z, gates = ctx.saved_tensors
        batch_size, feature_size, seq_len = gated_Z.shape
        grad_cumulative_gated_Z = grad_cumulative_gated_Z.contiguous().cuda()
        grad_cumulative_gates = grad_cumulative_gates.contiguous().cuda()
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates = torch.empty_like(gates)

        grid = (batch_size, feature_size, 1)

        bwd_scan_kernel[grid](
            grad_cumulative_gated_Z,
            grad_cumulative_gates,
            grad_gated_Z,
            grad_gates,
            batch_size,
            feature_size,
            seq_len,
            grad_cumulative_gated_Z.stride(0),
            grad_cumulative_gated_Z.stride(1),
            grad_cumulative_gated_Z.stride(2),
            DTYPE=ctx.triton_dtype,
        )

        return grad_gated_Z, grad_gates

    @staticmethod
    def _move_bdim_to_front(x, bdim):
        """Move the vmap dim to dim‑0; return (tensor, new_bdim)."""
        if bdim is None:
            return x, None
        return x.movedim(bdim, 0), 0

    @staticmethod
    def _flatten_leading_dims(x, keep=2):
        """
        Collapse all dims *except* the last `keep` dims into one.
        Returns (flat, original_batch_shape).
        """
        batch_shape = x.shape[:-keep]
        if batch_shape:  # non‑empty ⇒ need flatten
            flat = x.reshape(-1, *x.shape[-keep:])
        else:  # already rank‑`keep`
            flat = x
        return flat, batch_shape

    @staticmethod
    def vmap(info, in_dims, gated_Z, gates):
        """Batched rule: works for torch.vmap / torch.func.vmap."""
        bdim = in_dims[0]
        assert bdim == in_dims[1], "`gated_Z` and `gates` must share the same vmap dim"

        # 1. put the vmap dim in front (if any)
        gated_Z, _ = AssociativeScan._move_bdim_to_front(gated_Z, bdim)
        gates, _ = AssociativeScan._move_bdim_to_front(gates, bdim)

        # 2. collapse *all* leading batch axes into one
        gated_Z, batch_shape = AssociativeScan._flatten_leading_dims(gated_Z, keep=2)
        gates, _ = AssociativeScan._flatten_leading_dims(gates, keep=2)

        # 3. run the (non‑batched) kernel
        Z_cumul, g_cumul = AssociativeScan.apply(gated_Z, gates)

        # 4. un‑flatten to original batch shape and restore dim order
        if batch_shape:
            Z_cumul = Z_cumul.view(*batch_shape, *Z_cumul.shape[-2:])
            g_cumul = g_cumul.view(*batch_shape, *g_cumul.shape[-2:])

        if bdim is not None and bdim != 0:
            Z_cumul = Z_cumul.movedim(0, bdim)
            g_cumul = g_cumul.movedim(0, bdim)

        # out_dims must mirror the positions we put the batch dim back to
        return (Z_cumul, g_cumul), (bdim, bdim)


def associative_scan(gated_Z: torch.Tensor, gates: torch.Tensor):
    """Executes an associative scan on the provided tensors."""
    return AssociativeScan.apply(gated_Z, gates)


# Self-test
if __name__ == "__main__":
    import torch
    from torch.autograd import gradcheck

    torch.manual_seed(0)
    device = "cuda"

    # ---------------------------------------------------------------------
    # Large‑shape forward & backward sanity check
    # ---------------------------------------------------------------------
    B, D, L, batch = 2, 3, 8193, 4  # shapes match your original test

    Z = torch.randn(batch, B, D, L, dtype=torch.float64, device=device, requires_grad=True)
    g = torch.randn(batch, B, D, L, dtype=torch.float64, device=device, requires_grad=True)

    # reference = simple cumulative sum
    ref_Z = Z.cumsum(-1)
    ref_g = g.cumsum(-1)

    # associative_scan vmapped over leading batch dimension
    vmap_scan = torch.vmap(associative_scan, in_dims=(0, 0), out_dims=0)
    tri_Z, tri_g = vmap_scan(Z, g)

    # forward agreement
    torch.testing.assert_close(tri_Z, ref_Z)
    torch.testing.assert_close(tri_g, ref_g)

    # compare input‑gradients (avoid non‑leaf .grad)
    grad_Z_ref, grad_g_ref = torch.autograd.grad(ref_Z.sum() + ref_g.sum(), (Z, g), retain_graph=True)
    grad_Z_tri, grad_g_tri = torch.autograd.grad(tri_Z.sum() + tri_g.sum(), (Z, g))

    torch.testing.assert_close(grad_Z_tri, grad_Z_ref)
    torch.testing.assert_close(grad_g_tri, grad_g_ref)

    # ---------------------------------------------------------------------
    # Numerical Jacobian check with gradcheck (smaller size for speed)
    # ---------------------------------------------------------------------
    L_small = 33
    Z_s = torch.randn(batch, B, D, L_small, dtype=torch.float64, device=device, requires_grad=True)
    g_s = torch.randn(batch, B, D, L_small, dtype=torch.float64, device=device, requires_grad=True)

    vmap_scan_small = torch.vmap(associative_scan, in_dims=(0, 0), out_dims=0)

    def f(z_in, g_in):
        z_out, g_out = vmap_scan_small(z_in, g_in)
        return z_out + g_out  # any differentiable tensor output is fine

    assert gradcheck(f, (Z_s, g_s), eps=1e-6, atol=1e-4, rtol=1e-3), "gradcheck failed!"

    print("All checks passed ✓")
