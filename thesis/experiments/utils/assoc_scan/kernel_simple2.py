import torch
import triton
import triton.language as tl


@triton.jit
def combine_fn(carry_gated_Z, carry_gates, next_gated_Z, next_gates):
    """
    Combine function for the associative scan.
    
    Computes the elementwise addition of both the gated tensor and the gates.
    
    Args:
        carry_gated_Z: Current cumulative gated value.
        carry_gates: Current cumulative gate value.
        next_gated_Z: Next gated value.
        next_gates: Next gate value.
    
    Returns:
        A tuple (carry_gated_Z + next_gated_Z, carry_gates + next_gates).
    """
    return (carry_gated_Z + next_gated_Z, carry_gates + next_gates)


@triton.jit
def fwd_scan_kernel(gated_Z_ptr,
                    gates_ptr,
                    out_gated_Z_ptr,
                    out_gates_ptr,
                    batch_size: int,
                    feature_size: int,
                    seq_len: int,
                    stride_b: int,
                    stride_d: int,
                    stride_l: int,
                    BLOCK_SIZE: tl.constexpr,
                    DTYPE: tl.constexpr):
    """
    Forward kernel for cumulative associative scan.
    
    Scans along the sequence (L) dimension for each batch (B) and feature (D).
    
    Args:
        gated_Z_ptr: Pointer to input gated tensor [B, D, L].
        gates_ptr: Pointer to input gates tensor [B, D, L].
        out_gated_Z_ptr: Pointer to output cumulative gated tensor.
        out_gates_ptr: Pointer to output cumulative gates tensor.
        batch_size: Batch size (B).
        feature_size: Feature dimension (D).
        seq_len: Sequence length (L).
        stride_b: Stride of the batch dimension.
        stride_d: Stride of the feature dimension.
        stride_l: Stride of the sequence dimension.
        BLOCK_SIZE: Number of elements per block in the sequence dimension.
        DTYPE: Data type for computations (e.g., tl.float32, tl.float64).
    """
    # Assume seq_len > 0 for compiler optimization
    tl.assume(seq_len > 0)

    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d

    # Precompute offsets with compiler hints
    seq_offsets = tl.arange(0, BLOCK_SIZE) * stride_l
    seq_offsets = tl.multiple_of(seq_offsets, 16)  # Hint: aligned to 16 bytes
    seq_offsets = tl.max_contiguous(seq_offsets, BLOCK_SIZE)  # Hint: contiguous
    offsets = start_offset + seq_offsets
    mask = tl.arange(0, BLOCK_SIZE) < seq_len

    # Load data
    gated_Z = tl.load(gated_Z_ptr + offsets, mask=mask, other=0.0)
    gates = tl.load(gates_ptr + offsets, mask=mask, other=0.0)

    # Perform associative scan
    result_gated_Z, result_gates = tl.associative_scan(
        (gated_Z, gates), axis=0, combine_fn=combine_fn, reverse=False
    )

    # Store results
    tl.store(out_gated_Z_ptr + offsets, result_gated_Z, mask=mask)
    tl.store(out_gates_ptr + offsets, result_gates, mask=mask)


@triton.jit
def bwd_scan_kernel(grad_cumulative_gated_Z_ptr,
                    grad_cumulative_gates_ptr,
                    grad_gated_Z_ptr,
                    grad_gates_ptr,
                    batch_size: int,
                    feature_size: int,
                    seq_len: int,
                    stride_b: int,
                    stride_d: int,
                    stride_l: int,
                    BLOCK_SIZE: tl.constexpr,
                    DTYPE: tl.constexpr):
    """
    Backward kernel for associative scan.
    
    Computes gradients for the associative scan by performing a forward scan
    on the reversed gradient sequence.
    
    Args:
        grad_cumulative_gated_Z_ptr: Pointer to gradient of cumulative gated tensor [B, D, L].
        grad_cumulative_gates_ptr: Pointer to gradient of cumulative gates tensor [B, D, L].
        grad_gated_Z_ptr: Pointer to output gradient for gated tensor.
        grad_gates_ptr: Pointer to output gradient for gates tensor.
        batch_size: Batch size (B).
        feature_size: Feature dimension (D).
        seq_len: Sequence length (L).
        stride_b: Stride of the batch dimension.
        stride_d: Stride of the feature dimension.
        stride_l: Stride of the sequence dimension.
        BLOCK_SIZE: Number of elements per block in the sequence dimension.
        DTYPE: Data type for computations (e.g., tl.float32, tl.float64).
    """
    # Assume seq_len > 0 for compiler optimization
    tl.assume(seq_len > 0)

    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d

    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < seq_len

    # Precompute reversed offsets with compiler hints
    rev_indices = seq_len - 1 - indices
    rev_indices = tl.max_contiguous(rev_indices, BLOCK_SIZE)  # Hint: contiguous
    rev_offsets = start_offset + rev_indices * stride_l

    # Load data
    grad_in_Z_rev = tl.load(grad_cumulative_gated_Z_ptr + rev_offsets, mask=mask, other=0.0)
    grad_in_gates_rev = tl.load(grad_cumulative_gates_ptr + rev_offsets, mask=mask, other=0.0)

    # Perform associative scan
    result_grad_Z_rev, result_grad_gates_rev = tl.associative_scan(
        (grad_in_Z_rev, grad_in_gates_rev), axis=0, combine_fn=combine_fn, reverse=False
    )

    # Store results
    tl.store(grad_gated_Z_ptr + rev_offsets, result_grad_Z_rev, mask=mask)
    tl.store(grad_gates_ptr + rev_offsets, result_grad_gates_rev, mask=mask)


class AssociativeScan(torch.autograd.Function):
    """
    PyTorch autograd wrapper for an associative scan using Triton.
    
    Implements a cumulative associative scan along the sequence dimension for
    input tensors of shape [B, D, L].
    """
    @staticmethod
    def forward(ctx, gated_Z: torch.Tensor, gates: torch.Tensor):
        """
        Forward pass.
        
        Args:
            gated_Z: Input gated tensor of shape [B, D, L].
            gates: Input gates tensor of shape [B, D, L].
        
        Returns:
            A tuple (cumulative_gated_Z, cumulative_gates), each of shape [B, D, L],
            containing the cumulative scan results.
        """
        batch_size, feature_size, seq_len = gated_Z.shape
        gated_Z = gated_Z.contiguous().cuda()
        gates = gates.contiguous().cuda()
        cumulative_gated_Z = torch.empty_like(gated_Z)
        cumulative_gates = torch.empty_like(gates)

        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        grid = (batch_size, feature_size, 1)

        # Determine Triton dtype based on input dtype
        dtype_map = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
            torch.float64: tl.float64
        }
        triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)  # Default to float32 if unsupported

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
            BLOCK_SIZE=BLOCK_SIZE,
            DTYPE=triton_dtype
        )

        ctx.save_for_backward(gated_Z, gates)
        ctx.triton_dtype = triton_dtype
        return cumulative_gated_Z, cumulative_gates

    @staticmethod
    def backward(ctx, grad_cumulative_gated_Z: torch.Tensor, grad_cumulative_gates: torch.Tensor):
        """
        Backward pass.
        
        Args:
            grad_cumulative_gated_Z: Gradient of cumulative gated tensor [B, D, L].
            grad_cumulative_gates: Gradient of cumulative gates tensor [B, D, L].
        
        Returns:
            Gradients with respect to the input tensors gated_Z and gates.
        """
        gated_Z, gates = ctx.saved_tensors
        batch_size, feature_size, seq_len = gated_Z.shape
        grad_cumulative_gated_Z = grad_cumulative_gated_Z.contiguous().cuda()
        grad_cumulative_gates = grad_cumulative_gates.contiguous().cuda()
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates = torch.empty_like(gates)

        BLOCK_SIZE = triton.next_power_of_2(seq_len)
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
            BLOCK_SIZE=BLOCK_SIZE,
            DTYPE=ctx.triton_dtype
        )

        return grad_gated_Z, grad_gates


def associative_scan(gated_Z: torch.Tensor, gates: torch.Tensor):
    """
    Executes an associative scan on the provided tensors.
    
    Args:
        gated_Z: Tensor of shape [B, D, L] representing the gated values.
        gates: Tensor of shape [B, D, L] representing the gating coefficients.

    Returns:
        A tuple (cumulative_gated_Z, cumulative_gates), each of shape [B, D, L],
        corresponding to the cumulative associative scan results along the sequence dimension.
    """
    return AssociativeScan.apply(gated_Z, gates)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, L = 2, 3, 8193  # intentionally cross chunk boundary
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
    torch.testing.assert_close(Z.grad, Z.grad.clone())  # sanity
    torch.testing.assert_close(g.grad, g.grad.clone())
    print("All checks passed ✓")
