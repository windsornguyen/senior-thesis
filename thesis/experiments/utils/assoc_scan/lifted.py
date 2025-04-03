import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops import associative_scan

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class LiftedGatedScanV4_Final(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z: torch.Tensor, gates: torch.Tensor, dim: int, combine_mode: str) -> torch.Tensor:
        """
        Forward pass (in “lifted” coordinates):
          For each time step t (along the specified axis), we form a block matrix
              T[t] = [[1,      Z[t], 0],
                      [0,  gates[t], 0],
                      [0,       0,   1]]
          Then we compute the cumulative product along the recurrence axis:
              cumulative[t] = T[t] @ T[t-1] @ ... @ T[0]
          With s₀ = [0, 1, 1]ᵀ (broadcasted along the batch axes),
          the output is defined as
              output[t] = (cumulative[t] @ s₀)[0]
          
        This implementation first permutes the input so that the recurrence dimension is last,
        constructs the 3×3 block matrices (each “entry” being a tensor with the remaining shape),
        runs an associative scan along that axis, and finally inverts the permutation.
        """
        # --- Permute inputs so that the recurrence (time) axis is last.
        orig_ndim = Z.ndim
        if dim != orig_ndim - 1:
            # Build permutation: move axis "dim" to the end.
            perm = [i for i in range(orig_ndim) if i != dim] + [dim]
            Z = Z.permute(*perm)
            gates = gates.permute(*perm)
            # Compute the inverse permutation (length = orig_ndim)
            inv_perm = [0] * orig_ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            ctx.permute_inv = inv_perm
        else:
            ctx.permute_inv = None
        # Store original shape (so we can later unpermute the output)
        ctx.orig_shape = Z.shape  # after permutation
        # Now Z (and gates) have shape: (batch_dims, T)
        *batch, T = Z.shape
        # For our recurrence, we want to build a 3x3 block for each element along T.
        # Here one and zero have the same shape as Z.
        one = torch.ones_like(Z)
        zero = torch.zeros_like(Z)
        # Construct block rows. (These “stack” operations add a new dimension.)
        # T_top will have shape: (*batch, T, 3) with the three “blocks” [1, Z, 0]
        T_top = torch.stack([one, Z, zero], dim=-1)
        T_mid = torch.stack([zero, gates, zero], dim=-1)
        T_bot = torch.stack([zero, zero, one], dim=-1)
        # Now stack the three rows to get a 3x3 block matrix:
        # T_tensor has shape: (*batch, T, 3, 3)
        T_tensor = torch.stack([T_top, T_mid, T_bot], dim=-2)
        # In the new T_tensor, the recurrence dimension is at index -3.
        new_T_index = -3  # because T_tensor.shape = (*batch, T, 3, 3)
        
        ctx.dim = new_T_index  # store for backward
        ctx.combine_mode = combine_mode

        # Build s₀ vector (same for every t and batch element):
        # We want s0 of shape: (*batch, T, 3, 1) so that it broadcasts along batch and time.
        s0 = torch.tensor([0.0, 1.0, 1.0], dtype=Z.dtype, device=Z.device)
        s0 = s0.view(*([1] * (Z.ndim)), 3, 1).expand(*Z.shape, 3, 1)
        ctx.s0 = s0  # save for backward if needed

        # Define the combine function. Note we use a reversed order (matmul(b, a))
        def forward_combine(a, b):
            return torch.matmul(b, a)

        # Perform the associative scan along the recurrence axis (which is at index new_T_index)
        cumulative = associative_scan(forward_combine, T_tensor, dim=new_T_index, combine_mode=combine_mode)
        ctx.save_for_backward(T_tensor, cumulative)

        # Multiply cumulative matrices by s0:
        s = torch.matmul(cumulative, s0)  # shape: (*batch, T, 3, 1)
        # Our output is the first “block” (row 0) of the result.
        output = s[..., 0, 0]  # shape: (*batch, T)

        # Invert the permutation (if one was applied) so that the output matches the original order.
        if ctx.permute_inv is not None:
            # The output currently is in permuted order (last dim is T).
            # We invert the permutation (which is a list of length = orig_ndim) to put T back.
            output = output.permute(ctx.permute_inv)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        T_tensor, cumulative = ctx.saved_tensors
        combine_mode = ctx.combine_mode
        device = T_tensor.device
        dtype = T_tensor.dtype

        # In the forward, after permutation, T_tensor has shape: (*batch, T, 3, 3)
        # The recurrence axis is at index ctx.dim (which we set to -3).
        new_T_index = ctx.dim
        T_len = T_tensor.shape[new_T_index]

        # In the permuted space, grad_output has shape (*batch, T) (with T in the last dimension if no permutation was needed)
        # If a permutation was applied in forward, grad_output is still in the original (non-permuted) order,
        # so we need to permute it as we did for the inputs.
        if ctx.permute_inv is not None:
            # Permute grad_output with the same permutation as in forward
            perm = ctx.permute_inv  # note: inverse of inverse is the original permutation
            grad_output = grad_output.permute(perm)

        # We need to build dC (local derivatives) with the same shape as T_tensor.
        dC = torch.zeros_like(T_tensor)
        # Our forward pass computed output = (cumulative @ s0)[..., 0, 0]
        # So only the (0,:) row of each T_tensor matters.
        # Multiply grad_output (shape: (*batch, T)) by the s0 vector (which is [0,1,1])
        s0_vec = torch.tensor([0.0, 1.0, 1.0], dtype=dtype, device=device)
        # Set dC[..., 0, :] = grad_output[..., None] * s0_vec (broadcasted appropriately)
        dC[..., 0, :] = grad_output.unsqueeze(-1) * s0_vec.view(*([1] * (grad_output.ndim)), 3)

        # Now, perform a manual reverse accumulation along the recurrence axis.
        total_dC = torch.zeros_like(T_tensor)
        # To index along the recurrence dimension (at index new_T_index), use a list of slices.
        idx = [slice(None)] * T_tensor.ndim
        idx[new_T_index] = T_len - 1
        total_dC[tuple(idx)] = dC[tuple(idx)]
        # Loop backwards along the recurrence axis.
        for t in range(T_len - 2, -1, -1):
            idx_current = [slice(None)] * T_tensor.ndim
            idx_next = [slice(None)] * T_tensor.ndim
            idx_current[new_T_index] = t
            idx_next[new_T_index] = t + 1
            total_dC[tuple(idx_current)] = dC[tuple(idx_current)] + torch.matmul(
                T_tensor[tuple(idx_next)].transpose(-2, -1),
                total_dC[tuple(idx_next)]
            )
        dC_result = total_dC

        # Build C_prev: the cumulative product up to (t-1). We prepend an identity.
        batch_ndim = T_tensor.ndim - 3  # all dims except (T,3,3)
        I = torch.eye(3, device=device, dtype=dtype).view(*([1] * batch_ndim), 1, 3, 3)
        I = I.expand(*T_tensor.shape[:batch_ndim], 1, 3, 3)
        if T_len > 1:
            C_prev = torch.cat([I, cumulative.narrow(new_T_index, 0, T_len - 1)], dim=new_T_index)
        else:
            C_prev = I
        C_prev_T = C_prev.transpose(-2, -1)

        # Compute dT_tensor:
        dT_tensor = torch.matmul(dC_result, C_prev_T)
        # Gradients for Z and gates come from the (0,1) and (1,1) blocks, respectively.
        dZ = dT_tensor[..., 0, 1]
        d_gates = dT_tensor[..., 1, 1]

        # Invert the permutation on the gradients if needed.
        if ctx.permute_inv is not None:
            dZ = dZ.permute(ctx.permute_inv)
            d_gates = d_gates.permute(ctx.permute_inv)
        return dZ, d_gates, None, None

def lifted_gated_scan_v4_final(Z: torch.Tensor, gates: torch.Tensor, dim: int = 2, combine_mode: str = "generic") -> torch.Tensor:
    return LiftedGatedScanV4_Final.apply(Z, gates, dim, combine_mode)

def _gradcheck_wrapper(Z, gates):
    # Only pass tensors; fix dim and combine_mode.
    return lifted_gated_scan_v4_final(Z, gates, 2, "generic")

def run_gradcheck():
    torch.manual_seed(0)
    B, C, T = 2, 3, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use double precision for gradcheck.
    Z = torch.randn(B, C, T, device=device, dtype=torch.float64, requires_grad=True)
    G = torch.randn(B, C, T, device=device, dtype=torch.float64, requires_grad=True)
    test = torch.autograd.gradcheck(_gradcheck_wrapper, (Z, G), eps=1e-6, atol=1e-4)
    print("Gradcheck passed?", test)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T = 2, 3, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Try with a 3D input (for example)
    Z = torch.randn(B, C, T, device=device, requires_grad=True, dtype=torch.float32)
    gates = torch.randn(B, C, T, device=device, requires_grad=True, dtype=torch.float32)
    
    out = lifted_gated_scan_v4_final(Z, gates, 2, "generic")
    loss = out.pow(2).sum()
    loss.backward()
    print("Output (V4 Final):", out)
    print("Gradients for Z (V4 Final):", Z.grad)
    print("Gradients for gates (V4 Final):", gates.grad)
    
    print("\nRunning gradcheck in double precision...")
    run_gradcheck()
