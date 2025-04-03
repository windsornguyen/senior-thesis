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
                    BLOCK_SIZE: tl.constexpr):
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
    """
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d
    offsets = start_offset + tl.arange(0, BLOCK_SIZE) * stride_l
    mask = tl.arange(0, BLOCK_SIZE) < seq_len

    gated_Z = tl.load(gated_Z_ptr + offsets, mask=mask, other=0.0)
    gates = tl.load(gates_ptr + offsets, mask=mask, other=0.0)

    result_gated_Z, result_gates = tl.associative_scan(
        (gated_Z, gates), axis=0, combine_fn=combine_fn, reverse=False
    )

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
                    BLOCK_SIZE: tl.constexpr):
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
    """
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d

    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < seq_len

    # Compute reversed offsets to capture the triangular Jacobian pattern.
    rev_offsets = start_offset + (seq_len - 1 - indices) * stride_l

    grad_in_Z_rev = tl.load(grad_cumulative_gated_Z_ptr + rev_offsets, mask=mask, other=0.0)
    grad_in_gates_rev = tl.load(grad_cumulative_gates_ptr + rev_offsets, mask=mask, other=0.0)

    result_grad_Z_rev, result_grad_gates_rev = tl.associative_scan(
        (grad_in_Z_rev, grad_in_gates_rev), axis=0, combine_fn=combine_fn, reverse=False
    )

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
        )

        ctx.save_for_backward(gated_Z, gates)
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
