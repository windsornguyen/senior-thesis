# -*- Author: Windsor Nguyen -*-

"""
Implements an associative scan using Triton with autotuning and vmap support.
Performs prefix sums over the sequence dimension of input tensors using elementwise addition.
Separates scans for Z (matrix) and g (scalar) components.
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
def combine_add_scalars(a, b):
    """Combine function for scalar addition."""
    return a + b


@triton.jit
def combine_add_matrices(a, b):
    """Combine function for matrix element-wise addition."""
    # Assumes a and b have shape [BLOCK_SIZE, h, h] or compatible
    return a + b


# Define autotune configurations, inspired by 05-layer-norm.py and 06-fused-attention.py
def get_scan_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024]
        for s in [2, 3, 4]  # Vary pipeline stages for latency hiding
        for w in [4, 8]  # Common warp counts for CUDA
    ]
    # Add HIP-specific configs for AMD GPUs, as in 06-fused-attention.py
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
        )
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
    # Need separate keys if H dimension affects performance significantly
    key=["total_rows", "seq_len", "H_CONST"],
)
@triton.jit
def fwd_scan_kernel(
    # Pointers for Z
    gated_Z_ptr,
    out_gated_Z_ptr,
    # Pointers for g
    gates_ptr,
    out_gates_ptr,
    # Dimensions
    # batch_size and feature_size are combined into the grid dim 0
    total_rows: int,
    seq_len: int,
    H_CONST: tl.constexpr,
    # Strides for Z (assuming TOTAL_ROWS, L, H1, H2 layout after flattening)
    Z_stride_row: int,
    Z_stride_l: int,
    Z_stride_h1: int,
    Z_stride_h2: int,
    # Strides for g (assuming TOTAL_ROWS, L layout after flattening)
    g_stride_row: int,
    g_stride_l: int,
    # Kernel Params
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)  # This iterates through the collapsed batch/feature dims

    # Base pointers for the current row (collapsed batch/feature element)
    Z_base_ptr = gated_Z_ptr + pid_row * Z_stride_row
    g_base_ptr = gates_ptr + pid_row * g_stride_row
    out_Z_base_ptr = out_gated_Z_ptr + pid_row * Z_stride_row
    out_g_base_ptr = out_gates_ptr + pid_row * g_stride_row

    # --- State carried between blocks ---
    cumulative_Z = tl.zeros((H_CONST, H_CONST), dtype=DTYPE)  # Shape [h, h]
    cumulative_g = tl.zeros((1,), dtype=DTYPE)  # Shape [1] (scalar)

    # --- Define block shapes ---
    # Block shape for Z: [BLOCK_SIZE, H_CONST, H_CONST]
    Z_block_shape = (BLOCK_SIZE, H_CONST, H_CONST)
    # Block shape for g: [BLOCK_SIZE]
    g_block_shape = (BLOCK_SIZE,)

    # --- Define parent tensor shapes and strides for make_block_ptr ---
    # Parent shape Z: (seq_len, H_CONST, H_CONST) relative to Z_base_ptr
    Z_parent_shape = (seq_len, H_CONST, H_CONST)
    Z_parent_strides = (Z_stride_l, Z_stride_h1, Z_stride_h2)
    # Parent shape g: (seq_len,) relative to g_base_ptr
    g_parent_shape = (seq_len,)
    g_parent_strides = (g_stride_l,)

    # --- Define order for memory coalescing (tune if needed) ---
    # Assuming seq_len is the slowest moving dim for Z within a feature
    Z_order = (2, 1, 0)  # Order H2, H1, L
    g_order = (0,)  # Order L

    for start_idx in tl.range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        chunk_start = start_idx * BLOCK_SIZE

        # --- Create block pointers ---
        # Offsets Z: start at (chunk_start, 0, 0) in parent tensor
        Z_offs = (chunk_start, 0, 0)
        Z_blk_ptr = tl.make_block_ptr(
            base=Z_base_ptr,
            shape=Z_parent_shape,
            strides=Z_parent_strides,
            offsets=Z_offs,
            block_shape=Z_block_shape,
            order=Z_order,
        )
        out_Z_blk_ptr = tl.make_block_ptr(
            base=out_Z_base_ptr,
            shape=Z_parent_shape,
            strides=Z_parent_strides,
            offsets=Z_offs,
            block_shape=Z_block_shape,
            order=Z_order,
        )
        # Offsets g: start at (chunk_start,) in parent tensor
        g_offs = (chunk_start,)
        g_blk_ptr = tl.make_block_ptr(
            base=g_base_ptr,
            shape=g_parent_shape,
            strides=g_parent_strides,
            offsets=g_offs,
            block_shape=g_block_shape,
            order=g_order,
        )
        out_g_blk_ptr = tl.make_block_ptr(
            base=out_g_base_ptr,
            shape=g_parent_shape,
            strides=g_parent_strides,
            offsets=g_offs,
            block_shape=g_block_shape,
            order=g_order,
        )

        # --- Load chunk data ---
        # Boundary check on axis 0 (sequence dimension)
        gated_Z = tl.load(Z_blk_ptr, boundary_check=(0,), padding_option="zero")
        gates = tl.load(g_blk_ptr, boundary_check=(0,), padding_option="zero")

        # --- Intra-block scan for Z ---
        (scanned_Z,) = tl.associative_scan((gated_Z,), axis=0, combine_fn=combine_add_matrices, reverse=False)

        # --- Intra-block scan for g ---
        (scanned_g,) = tl.associative_scan((gates,), axis=0, combine_fn=combine_add_scalars, reverse=False)

        # --- Apply carry-in from previous block ---
        result_Z = scanned_Z + cumulative_Z[None, :, :]  # Broadcast carry [h,h] to [1,h,h]
        result_g = scanned_g + cumulative_g  # Broadcast carry [1] to [BLOCK_SIZE]

        # --- Store results for this block ---
        # Boundary check on axis 0 (sequence dimension)
        tl.store(out_Z_blk_ptr, result_Z, boundary_check=(0,))
        tl.store(out_g_blk_ptr, result_g, boundary_check=(0,))

        # --- Update carry-out state for next block ---
        # Need the state from the last valid element *after* applying carry-in
        chunk_size = tl.minimum(BLOCK_SIZE, seq_len - chunk_start)
        last_idx_in_block = chunk_size - 1
        if last_idx_in_block >= 0:
            # We need to extract the slice corresponding to the last element
            # Using manual indexing here might be simpler than creating a tiny block pointer
            block_indices = tl.arange(0, BLOCK_SIZE)
            last_mask_Z = (block_indices == last_idx_in_block)[:, None, None]
            last_mask_g = block_indices == last_idx_in_block

            # Reduce along block dim (axis=0) to get carry shape
            cumulative_Z = tl.sum(tl.where(last_mask_Z, result_Z, 0.0), axis=0)  # Shape [h, h]
            cumulative_g = tl.sum(tl.where(last_mask_g, result_g, 0.0), axis=0, keep_dims=True)  # Shape [1]


@triton.autotune(
    configs=list(filter(lambda conf: keep_config(conf), get_scan_configs())),
    key=["total_rows", "seq_len", "H_CONST"],
)
@triton.jit
def bwd_scan_kernel(
    # Grad Z inputs/outputs
    grad_cumul_Z_ptr,
    grad_Z_ptr,
    # Grad g inputs/outputs
    grad_cumul_g_ptr,
    grad_g_ptr,
    # Dimensions
    total_rows: int,
    seq_len: int,
    H_CONST: tl.constexpr,
    # Strides Z (assuming TOTAL_ROWS, L, H1, H2 layout)
    Z_stride_row: int,
    Z_stride_l: int,
    Z_stride_h1: int,
    Z_stride_h2: int,
    # Strides g (assuming TOTAL_ROWS, L layout)
    g_stride_row: int,
    g_stride_l: int,
    # Kernel Params
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Right‑to‑left prefix‑sum of gradients (reverse scan)."""
    pid_row = tl.program_id(0)  # Collapsed batch/feature index

    # Base pointers for the current row
    in_Z_base_ptr = grad_cumul_Z_ptr + pid_row * Z_stride_row
    in_g_base_ptr = grad_cumul_g_ptr + pid_row * g_stride_row
    out_Z_base_ptr = grad_Z_ptr + pid_row * Z_stride_row
    out_g_base_ptr = grad_g_ptr + pid_row * g_stride_row

    # --- State carried between blocks (from right to left) ---
    carry_Z = tl.zeros((H_CONST, H_CONST), dtype=DTYPE)  # Shape [h, h]
    carry_g = tl.zeros((1,), dtype=DTYPE)  # Shape [1]

    # --- Define block shapes ---
    Z_block_shape = (BLOCK_SIZE, H_CONST, H_CONST)
    g_block_shape = (BLOCK_SIZE,)

    # --- Define parent tensor shapes and strides ---
    Z_parent_shape = (seq_len, H_CONST, H_CONST)
    Z_parent_strides = (Z_stride_l, Z_stride_h1, Z_stride_h2)
    g_parent_shape = (seq_len,)
    g_parent_strides = (g_stride_l,)

    # --- Define order ---
    Z_order = (2, 1, 0)
    g_order = (0,)

    # Iterate block-wise, loading data in reverse sequence order
    for start_idx in tl.range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        chunk_start_fwd = start_idx * BLOCK_SIZE  # Start index if scanning forward
        chunk_end_fwd = tl.minimum((start_idx + 1) * BLOCK_SIZE, seq_len)
        chunk_size = chunk_end_fwd - chunk_start_fwd

        # Calculate the starting sequence index for the *reversed* block
        # The block corresponds to forward indices [chunk_start_fwd, chunk_end_fwd)
        # The reversed indices are [seq_len - chunk_end_fwd, seq_len - chunk_start_fwd)
        rev_chunk_start_seq_idx = seq_len - chunk_end_fwd

        # --- Create block pointers based on reversed start index ---
        # Offsets Z: start at (rev_chunk_start_seq_idx, 0, 0)
        Z_offs = (rev_chunk_start_seq_idx, 0, 0)
        in_Z_blk_ptr = tl.make_block_ptr(
            base=in_Z_base_ptr,
            shape=Z_parent_shape,
            strides=Z_parent_strides,
            offsets=Z_offs,
            block_shape=Z_block_shape,
            order=Z_order,
        )
        out_Z_blk_ptr = tl.make_block_ptr(
            base=out_Z_base_ptr,
            shape=Z_parent_shape,
            strides=Z_parent_strides,
            offsets=Z_offs,
            block_shape=Z_block_shape,
            order=Z_order,
        )
        # Offsets g: start at (rev_chunk_start_seq_idx,)
        g_offs = (rev_chunk_start_seq_idx,)
        in_g_blk_ptr = tl.make_block_ptr(
            base=in_g_base_ptr,
            shape=g_parent_shape,
            strides=g_parent_strides,
            offsets=g_offs,
            block_shape=g_block_shape,
            order=g_order,
        )
        out_g_blk_ptr = tl.make_block_ptr(
            base=out_g_base_ptr,
            shape=g_parent_shape,
            strides=g_parent_strides,
            offsets=g_offs,
            block_shape=g_block_shape,
            order=g_order,
        )

        # --- Load reversed gradients for the block ---
        # Boundary check needed as block might go past start of reversed sequence
        grad_Z_in = tl.load(in_Z_blk_ptr, boundary_check=(0,), padding_option="zero")
        grad_g_in = tl.load(in_g_blk_ptr, boundary_check=(0,), padding_option="zero")

        # --- Intra-block forward scan (on reversed Z data) ---
        (scanned_grad_Z,) = tl.associative_scan((grad_Z_in,), axis=0, combine_fn=combine_add_matrices, reverse=False)

        # --- Intra-block forward scan (on reversed g data) ---
        (scanned_grad_g,) = tl.associative_scan((grad_g_in,), axis=0, combine_fn=combine_add_scalars, reverse=False)

        # --- Add carry-in (from the block to the right) ---
        result_grad_Z = scanned_grad_Z + carry_Z[None, :, :]  # Broadcast carry [h,h]
        result_grad_g = scanned_grad_g + carry_g  # Broadcast carry [1]

        # --- Store results back to reversed locations ---
        tl.store(out_Z_blk_ptr, result_grad_Z, boundary_check=(0,))
        tl.store(out_g_blk_ptr, result_grad_g, boundary_check=(0,))

        # --- Update carry-out for next block (to the left) ---
        # We need the state corresponding to the *first* logical index processed
        # in this reversed block, which is the *last* element loaded/scanned.
        last_idx_in_block = chunk_size - 1  # Index within the loaded block [0..BLOCK_SIZE-1]
        if last_idx_in_block >= 0:
            block_indices = tl.arange(0, BLOCK_SIZE)
            last_mask_Z = (block_indices == last_idx_in_block)[:, None, None]
            last_mask_g = block_indices == last_idx_in_block

            carry_Z = tl.sum(tl.where(last_mask_Z, result_grad_Z, 0.0), axis=0)  # Shape [h, h]
            carry_g = tl.sum(tl.where(last_mask_g, result_grad_g, 0.0), axis=0, keep_dims=True)  # Shape [1]


class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for an associative scan using Triton."""

    @staticmethod
    def forward(gated_Z: torch.Tensor, gates: torch.Tensor):
        # Z: [TOTAL_ROWS, L, H, H], g: [TOTAL_ROWS, L]
        # Where TOTAL_ROWS is the collapsed batch/feature dimensions
        assert gated_Z.ndim == 4, "Expected gated_Z to be 4D (collapsed batch/feature, L, H, H)"
        assert gates.ndim == 2, "Expected gates to be 2D (collapsed batch/feature, L)"

        collapsed_batch_dim, seq_len, H_CONST, _ = gated_Z.shape
        gated_Z = gated_Z.contiguous().cuda()
        gates = gates.contiguous().cuda()
        cumulative_gated_Z = torch.empty_like(gated_Z)
        cumulative_gates = torch.empty_like(gates)

        assert gates.shape == (collapsed_batch_dim, seq_len), "gates shape mismatch"

        grid = (collapsed_batch_dim,)  # Launch one kernel thread per collapsed row
        triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)

        fwd_scan_kernel[grid](
            # Z args
            gated_Z,
            cumulative_gated_Z,
            # g args
            gates,
            cumulative_gates,
            # Dims
            collapsed_batch_dim,
            seq_len,
            H_CONST,
            # Z strides
            gated_Z.stride(0),
            gated_Z.stride(1),
            gated_Z.stride(2),
            gated_Z.stride(3),
            # g strides
            gates.stride(0),
            gates.stride(1),
            # Kernel params
            DTYPE=triton_dtype,
        )

        return cumulative_gated_Z, cumulative_gates

    @staticmethod
    def setup_context(ctx, inputs, output):
        gated_Z, gates = inputs
        # Keep outputs if needed for backward? Maybe not for simple prefix sum.
        ctx.save_for_backward(gated_Z, gates)  # Save inputs needed for backward
        ctx.triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)
        # Store shapes/strides from the FLATTENED inputs
        ctx.collapsed_batch_dim = gated_Z.shape[0]
        ctx.seq_len = gated_Z.shape[1]
        ctx.H_CONST = gated_Z.shape[2]
        ctx.Z_strides = (gated_Z.stride(0), gated_Z.stride(1), gated_Z.stride(2), gated_Z.stride(3))
        ctx.g_strides = (gates.stride(0), gates.stride(1))

    @staticmethod
    def backward(ctx, grad_cumulative_gated_Z: torch.Tensor, grad_cumulative_gates: torch.Tensor):
        gated_Z, gates = ctx.saved_tensors  # Retrieve potentially needed inputs
        # Z: [TOTAL_ROWS, L, H, H], g: [TOTAL_ROWS, L]
        grad_cumulative_gated_Z = grad_cumulative_gated_Z.contiguous().cuda()
        grad_cumulative_gates = grad_cumulative_gates.contiguous().cuda()
        # Gradients w.r.t. inputs
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates = torch.empty_like(gates)

        assert grad_cumulative_gates.shape == (ctx.collapsed_batch_dim, ctx.seq_len), "grad_gates shape mismatch"
        assert grad_cumulative_gated_Z.shape == (ctx.collapsed_batch_dim, ctx.seq_len, ctx.H_CONST, ctx.H_CONST), (
            "grad_Z shape mismatch"
        )

        grid = (ctx.collapsed_batch_dim,)

        bwd_scan_kernel[grid](
            # Grad Z args
            grad_cumulative_gated_Z,
            grad_gated_Z,
            # Grad g args
            grad_cumulative_gates,
            grad_gates,
            # Dims
            ctx.collapsed_batch_dim,
            ctx.seq_len,
            ctx.H_CONST,
            # Z strides (use strides of grad_Z output which match input Z)
            ctx.Z_strides[0],
            ctx.Z_strides[1],
            ctx.Z_strides[2],
            ctx.Z_strides[3],
            # g strides (use strides of grad_g output which match input g)
            ctx.g_strides[0],
            ctx.g_strides[1],
            # Kernel params
            DTYPE=ctx.triton_dtype,
        )

        # The backward of a prefix sum returns the suffix sum of the incoming gradients
        # The kernel computes this directly.
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
        # Adjust keep based on tensor rank
        actual_keep = min(keep, x.ndim)
        batch_shape = x.shape[:-actual_keep]
        if batch_shape:  # non‑empty ⇒ need flatten
            flat = x.reshape(-1, *x.shape[-actual_keep:])
        else:  # already rank‑`keep` or less
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
        # Keep last 3 dims for Z (L, H, H), last 1 for g (L)
        gated_Z, batch_shape = AssociativeScan._flatten_leading_dims(gated_Z, keep=3)
        gates, _ = AssociativeScan._flatten_leading_dims(gates, keep=1)
        # Reshape gates to match Z's collapsed batch dim + L
        if batch_shape:
            gates = gates.reshape(gated_Z.shape[0], -1)

        # 3. run the (non‑batched) kernel
        # The kernel expects [B_flat, D_flat, L, ...] -> need to adjust kernel launch / input prep
        # For simplicity here, assume kernel handles collapsed B*D as the first dim
        # Need B_flat * D_flat for grid dim 0
        num_features = gated_Z.shape[0]  # Collapsed B*D
        Z_cumul, g_cumul = AssociativeScan.apply(
            gated_Z, gates
        )  # Apply might need adjustment if it assumes unflattened

        # 4. un‑flatten to original batch shape and restore dim order
        if batch_shape:
            Z_cumul = Z_cumul.view(*batch_shape, *Z_cumul.shape[-3:])  # Unflatten Z
            g_cumul = g_cumul.view(*batch_shape, *g_cumul.shape[-1:])  # Unflatten g

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
    B, D, L, H = 2, 3, 8193, 4  # Example H=4
    batch = 4

    Z_in = torch.randn(batch, B, D, L, H, H, dtype=torch.float64, device=device, requires_grad=True)
    g_in = torch.randn(batch, B, D, L, dtype=torch.float64, device=device, requires_grad=True)

    # reference = simple cumulative sum
    ref_Z = Z_in.cumsum(-3)  # Scan along L dim
    ref_g = g_in.cumsum(-1)  # Scan along L dim

    # associative_scan vmapped over leading batch dimension
    # Need to handle the nested B, D dimensions correctly for vmap or flatten manually
    Z_flat = Z_in.flatten(0, 2)  # Flatten batch, B, D -> (batch*B*D)
    g_flat = g_in.flatten(0, 2)  # Flatten batch, B, D -> (batch*B*D)
    # Now apply scan directly as the kernel handles the first two grid dims
    tri_Z_flat, tri_g_flat = associative_scan(Z_flat, g_flat)

    # Unflatten results
    tri_Z = tri_Z_flat.view(batch, B, D, L, H, H)
    tri_g = tri_g_flat.view(batch, B, D, L)

    # forward agreement
    torch.testing.assert_close(tri_Z, ref_Z, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tri_g, ref_g, rtol=1e-5, atol=1e-5)
    print("Forward check passed ✓")

    # compare input‑gradients (avoid non‑leaf .grad)
    # Need to use flattened inputs for autograd grad call matching the function input
    grad_Z_ref, grad_g_ref = torch.autograd.grad(ref_Z.sum() + ref_g.sum(), (Z_in, g_in), retain_graph=True)
    grad_Z_tri_flat, grad_g_tri_flat = torch.autograd.grad(tri_Z_flat.sum() + tri_g_flat.sum(), (Z_flat, g_flat))

    # Unflatten gradients for comparison
    grad_Z_tri = grad_Z_tri_flat.view(batch, B, D, L, H, H)
    grad_g_tri = grad_g_tri_flat.view(batch, B, D, L)

    torch.testing.assert_close(grad_Z_tri, grad_Z_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_g_tri, grad_g_ref, rtol=1e-5, atol=1e-5)
    print("Backward check passed ✓")

    # ---------------------------------------------------------------------
    # Numerical Jacobian check with gradcheck (smaller size for speed)
    # ---------------------------------------------------------------------
    L_small = 33
    H_small = 2
    Z_s = torch.randn(
        1, 1, 1, L_small, H_small, H_small, dtype=torch.float64, device=device, requires_grad=True
    )  # Use simple batching
    g_s = torch.randn(1, 1, 1, L_small, dtype=torch.float64, device=device, requires_grad=True)

    # Flatten for input to associative_scan
    Z_s_flat = Z_s.flatten(0, 2)
    g_s_flat = g_s.flatten(0, 2)

    # Define function for gradcheck
    def f(z_in_flat, g_in_flat):
        z_out_flat, g_out_flat = associative_scan(z_in_flat, g_in_flat)
        # Unflatten for consistent output shape if needed by loss, or sum flattened
        return z_out_flat.sum() + g_out_flat.sum()  # Sum directly on potentially flattened output

    # Perform gradcheck
    gradcheck_passed = gradcheck(f, (Z_s_flat, g_s_flat), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert gradcheck_passed, "Gradcheck failed!"

    print("Gradcheck passed ✓")
    print("All checks passed ✓")
