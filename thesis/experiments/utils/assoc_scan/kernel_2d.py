"""2D tiling"""

import torch
import triton
import triton.language as tl


@triton.jit
def combine_fn(carry_gated_Z, carry_gates, next_gated_Z, next_gates):
    """Combine function for the associative scan: elementwise addition."""
    return (carry_gated_Z + next_gated_Z, carry_gates + next_gates)


# Define autotune configurations, inspired by 09-persistent-matmul.py
def get_scan_configs():
    configs = [
        triton.Config({"BLOCK_SIZE_L": bsl, "BLOCK_SIZE_D": bsd}, num_stages=s, num_warps=w)
        for bsl in [64, 128, 256, 512]  # Sequence block sizes
        for bsd in [1, 4, 8, 16]  # Feature block sizes
        for s in [2, 3, 4]  # Pipeline stages
        for w in [4, 8]  # Warp counts
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [
                triton.Config({"BLOCK_SIZE_L": bsl, "BLOCK_SIZE_D": bsd}, num_stages=1, num_warps=4, waves_per_eu=3)
                for bsl in [64, 128, 256]
                for bsd in [1, 4]
            ]
        )
    return configs


# Filter configs to ensure efficiency, inspired by 06-fused-attention.py
def keep_config(conf):
    BLOCK_SIZE_L = conf.kwargs["BLOCK_SIZE_L"]
    BLOCK_SIZE_D = conf.kwargs["BLOCK_SIZE_D"]
    num_warps = conf.num_warps
    # Avoid large blocks with few warps or small blocks with many warps
    if BLOCK_SIZE_L >= 256 and num_warps < 8:
        return False
    if BLOCK_SIZE_L * BLOCK_SIZE_D < 128 and num_warps > 4:
        return False
    return True


@triton.autotune(configs=list(filter(keep_config, get_scan_configs())), key=["batch_size", "feature_size", "seq_len"])
@triton.jit
def fwd_scan_kernel(
    gated_Z_ptr,
    gates_ptr,
    out_gated_Z_ptr,
    out_gates_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_d: int,
    stride_l: int,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Forward kernel for cumulative associative scan with 2D tiling."""
    tl.assume(seq_len > 0)

    pid_batch = tl.program_id(0)
    pid_feature_block = tl.program_id(1)
    feature_start = pid_feature_block * BLOCK_SIZE_D
    feature_indices = feature_start + tl.arange(0, BLOCK_SIZE_D)
    mask_d = feature_indices < feature_size

    # Initialize cumulative state for the scan
    cumulative_gated_Z = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)
    cumulative_gates = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)

    # Process sequence in chunks
    for start_idx in tl.range(0, seq_len, BLOCK_SIZE_L):
        indices_l = start_idx + tl.arange(0, BLOCK_SIZE_L)
        indices_l = tl.multiple_of(indices_l, 16)
        indices_l = tl.max_contiguous(indices_l, BLOCK_SIZE_L)
        mask_l = indices_l < seq_len
        offsets = pid_batch * stride_b + feature_indices[:, None] * stride_d + indices_l[None, :] * stride_l
        mask = mask_d[:, None] & mask_l[None, :]

        # Load input chunk
        gated_Z = tl.load(gated_Z_ptr + offsets, mask=mask, other=0.0)
        gates = tl.load(gates_ptr + offsets, mask=mask, other=0.0)

        # Perform associative scan within the chunk along sequence dimension
        result_gated_Z, result_gates = tl.associative_scan(
            (gated_Z, gates), axis=1, combine_fn=combine_fn, reverse=False
        )

        # Add cumulative state from previous chunks
        result_gated_Z += cumulative_gated_Z[:, None]
        result_gates += cumulative_gates[:, None]

        # Store results
        tl.store(out_gated_Z_ptr + offsets, result_gated_Z, mask=mask)
        tl.store(out_gates_ptr + offsets, result_gates, mask=mask)

        # Update cumulative state with the last valid sequence element
        last_valid_idx = tl.minimum(BLOCK_SIZE_L - 1, seq_len - start_idx - 1)
        if last_valid_idx >= 0:
            last_valid_mask = tl.arange(0, BLOCK_SIZE_L) == last_valid_idx
            # Use masking to select the last valid element
            cumulative_gated_Z = tl.where(
                mask_d, 
                tl.sum(result_gated_Z * last_valid_mask[None, :], axis=1), 
                cumulative_gated_Z
            )
            cumulative_gates = tl.where(
                mask_d, 
                tl.sum(result_gates * last_valid_mask[None, :], axis=1), 
                cumulative_gates
            )

@triton.autotune(configs=list(filter(keep_config, get_scan_configs())), key=["batch_size", "feature_size", "seq_len"])
@triton.jit
def bwd_scan_kernel(
    grad_cumulative_gated_Z_ptr,
    grad_cumulative_gates_ptr,
    grad_gated_Z_ptr,
    grad_gates_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_d: int,
    stride_l: int,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Backward kernel for associative scan with 2D tiling."""
    tl.assume(seq_len > 0)

    pid_batch = tl.program_id(0)
    pid_feature_block = tl.program_id(1)
    feature_start = pid_feature_block * BLOCK_SIZE_D
    feature_indices = feature_start + tl.arange(0, BLOCK_SIZE_D)
    mask_d = feature_indices < feature_size

    # Initialize cumulative gradient state
    cumulative_grad_Z = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)
    cumulative_grad_gates = tl.zeros([BLOCK_SIZE_D], dtype=DTYPE)

    # Process sequence in forward order, accessing gradients in reverse
    for start_idx in tl.range(0, seq_len, BLOCK_SIZE_L):
        indices_l = tl.arange(0, BLOCK_SIZE_L)
        indices_l = tl.multiple_of(indices_l, 16)
        indices_l = tl.max_contiguous(indices_l, BLOCK_SIZE_L)
        rev_indices = seq_len - 1 - (start_idx + indices_l)
        rev_offsets = pid_batch * stride_b + feature_indices[:, None] * stride_d + rev_indices[None, :] * stride_l
        mask_l = rev_indices >= 0
        mask = mask_d[:, None] & mask_l[None, :]

        # Load reversed input gradients
        grad_in_Z_rev = tl.load(grad_cumulative_gated_Z_ptr + rev_offsets, mask=mask, other=0.0)
        grad_in_gates_rev = tl.load(grad_cumulative_gates_ptr + rev_offsets, mask=mask, other=0.0)

        # Perform associative scan within the chunk
        result_grad_Z_rev, result_grad_gates_rev = tl.associative_scan(
            (grad_in_Z_rev, grad_in_gates_rev), axis=1, combine_fn=combine_fn, reverse=False
        )

        # Add cumulative gradient from previous chunks
        result_grad_Z_rev += cumulative_grad_Z[:, None]
        result_grad_gates_rev += cumulative_grad_gates[:, None]

        # Store results
        tl.store(grad_gated_Z_ptr + rev_offsets, result_grad_Z_rev, mask=mask)
        tl.store(grad_gates_ptr + rev_offsets, result_grad_gates_rev, mask=mask)

        # Update cumulative gradient with the last valid element
        last_valid_idx = tl.minimum(BLOCK_SIZE_L - 1, seq_len - start_idx - 1)
        if last_valid_idx >= 0:
            last_valid_mask = tl.arange(0, BLOCK_SIZE_L) == last_valid_idx
            # Use masking to select the last valid element
            cumulative_grad_Z = tl.where(
                mask_d, 
                tl.sum(result_grad_Z_rev * last_valid_mask[None, :], axis=1), 
                cumulative_grad_Z
            )
            cumulative_grad_gates = tl.where(
                mask_d, 
                tl.sum(result_grad_gates_rev * last_valid_mask[None, :], axis=1), 
                cumulative_grad_gates
            )

class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for an associative scan using Triton."""

    @staticmethod
    def forward(ctx, gated_Z: torch.Tensor, gates: torch.Tensor):
        batch_size, feature_size, seq_len = gated_Z.shape
        gated_Z = gated_Z.contiguous().cuda()
        gates = gates.contiguous().cuda()
        cumulative_gated_Z = torch.empty_like(gated_Z)
        cumulative_gates = torch.empty_like(gates)

        # Constrain grid to NUM_SMS for batch dimension, tile features
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = lambda meta: (min(NUM_SMS, batch_size), triton.cdiv(feature_size, meta["BLOCK_SIZE_D"]), 1)

        dtype_map = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
            torch.float64: tl.float64,
        }
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
            enable_fp_fusion=False,
        )

        ctx.save_for_backward(gated_Z, gates)
        ctx.triton_dtype = triton_dtype
        return cumulative_gated_Z, cumulative_gates

    @staticmethod
    def backward(ctx, grad_cumulative_gated_Z: torch.Tensor, grad_cumulative_gates: torch.Tensor):
        gated_Z, gates = ctx.saved_tensors
        batch_size, feature_size, seq_len = gated_Z.shape
        grad_cumulative_gated_Z = grad_cumulative_gated_Z.contiguous().cuda()
        grad_cumulative_gates = grad_cumulative_gates.contiguous().cuda()
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates = torch.empty_like(gates)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = lambda meta: (min(NUM_SMS, batch_size), triton.cdiv(feature_size, meta["BLOCK_SIZE_D"]), 1)

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
            enable_fp_fusion=False,
        )

        return grad_gated_Z, grad_gates


def associative_scan(gated_Z: torch.Tensor, gates: torch.Tensor):
    """Executes an associative scan on the provided tensors."""
    return AssociativeScan.apply(gated_Z, gates)


# Self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, L = 2, 3, 8193  # Test with long sequence length
    Z = torch.randn(B, D, L, dtype=torch.float64, device="cuda")
    g = torch.randn(B, D, L, dtype=torch.float64, device="cuda")
    Z.requires_grad_()
    g.requires_grad_()

    ref_Z = Z.cumsum(-1)
    ref_g = g.cumsum(-1)
    tri_Z, tri_g = associative_scan(Z, g)
    torch.testing.assert_close(tri_Z, ref_Z)
    torch.testing.assert_close(tri_g, ref_g)

    loss_ref = ref_Z.sum() + ref_g.sum()
    loss_tri = tri_Z.sum() + tri_g.sum()
    loss_ref.backward()
    loss_tri.backward()
    torch.testing.assert_close(Z.grad, Z.grad.clone())  # Sanity check
    torch.testing.assert_close(g.grad, g.grad.clone())
    print("All checks passed ✓")
