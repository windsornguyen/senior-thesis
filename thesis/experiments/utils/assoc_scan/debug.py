"""
debug_lifted_gated_scan.py

This file implements a lifted gated scan in two ways:
1. A brute-force version that uses an explicit loop and autograd.
2. A vectorized version that uses a custom autograd function with associative_scan.

We compute the loss (sum of squares of outputs) and call backward on both.
Finally, we print the outputs and the gradients for Z and gates, along with
their differences, so you can see where the vectorized backward might be deviating.
"""

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops import associative_scan

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

############################################
# Vectorized Version (Custom Autograd Func)
############################################

class LiftedGatedScanV4_Final(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z: torch.Tensor, gates: torch.Tensor, dim: int, combine_mode: str) -> torch.Tensor:
        """
        For each time step t, form:
            T[t] = [[1,      Z[t], 0],
                    [0,  gates[t], 0],
                    [0,       0,   1]]
        Then compute cumulative[t] = T[t] @ ... @ T[0] (via an associative scan with
        custom ordering) and output H[t] = (cumulative[t] @ s0)[0] where s0 = [0,1,1]^T.
        """
        B, C, T = Z.shape
        one = torch.ones_like(Z)
        zero = torch.zeros_like(Z)
        # Build the 3×3 matrices for every time step:
        T_top = torch.stack([one, Z, zero], dim=-1)      # (B, C, T, 3)
        T_mid = torch.stack([zero, gates, zero], dim=-1)   # (B, C, T, 3)
        T_bot = torch.stack([zero, zero, one], dim=-1)     # (B, C, T, 3)
        T_tensor = torch.stack([T_top, T_mid, T_bot], dim=-2)  # (B, C, T, 3, 3)
        
        ctx.dim = dim
        ctx.combine_mode = combine_mode
        
        # Expand s0 to (B, C, T, 3, 1)
        s0 = torch.tensor([0.0, 1.0, 1.0], dtype=Z.dtype, device=Z.device).view(1, 1, 1, 3, 1).expand(B, C, T, 3, 1)
        ctx.s0 = s0
        
        # The forward combine reverses the order: cumulative[t] = T[t] @ cumulative[t-1]
        def forward_combine(a, b):
            return torch.matmul(b, a)
        
        cumulative = associative_scan(forward_combine, T_tensor, dim=dim, combine_mode=combine_mode)
        ctx.save_for_backward(T_tensor, cumulative)
        
        s = torch.matmul(cumulative, s0).squeeze(-1)  # (B, C, T, 3)
        output = s[..., 0]  # Only the first entry matters.
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        T_tensor, cumulative = ctx.saved_tensors
        dim = ctx.dim
        combine_mode = ctx.combine_mode
        B, C, T, _, _ = T_tensor.shape
        device = T_tensor.device
        dtype = T_tensor.dtype
        
        # s0 vector for the forward pass
        s0_vec = torch.tensor([0.0, 1.0, 1.0], dtype=dtype, device=device)
        
        # Local derivative dC: only the first row of cumulative contributes to the output.
        dC = torch.zeros_like(T_tensor)
        dC[..., 0, :] = grad_output.unsqueeze(-1) * s0_vec.view(1, 1, 1, 3)
        
        # We use the transposed T_tensor for the reverse pass.
        A_trans = T_tensor.transpose(-2, -1)
        state_pair = (A_trans, dC)
        flat_state, state_spec = pytree.tree_flatten(state_pair)
        
        def flat_combine_fn(flat_x, flat_y):
            pair_x = pytree.tree_unflatten(flat_x, state_spec)
            pair_y = pytree.tree_unflatten(flat_y, state_spec)
            A_x, dC_x = pair_x
            A_y, dC_y = pair_y
            newA = torch.matmul(A_x, A_y)
            newdC = dC_x + torch.matmul(A_y, dC_y)
            new_pair = (newA, newdC)
            flat_new, _ = pytree.tree_flatten(new_pair)
            return flat_new
        
        # Run the reverse scan (accumulating gradients in reverse order)
        flat_result = associative_scan(flat_combine_fn, flat_state, dim=dim, reverse=True, combine_mode=combine_mode)
        result_pair = pytree.tree_unflatten(flat_result, state_spec)
        _, dC_result = result_pair
        
        # Build the cumulative product up to t-1.
        I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3).expand(B, C, 1, 3, 3)
        if T > 1:
            C_prev = torch.cat([I, cumulative[..., :-1, :, :]], dim=2)
        else:
            C_prev = I
        # We need the transpose of the previous cumulative matrix.
        C_prev_T = C_prev.transpose(-2, -1)
        
        # Key: dT_tensor = dC_result @ (C_prev)^T
        dT_tensor = torch.matmul(dC_result, C_prev_T)
        
        # Recover gradients for Z and gates from the structure of T_tensor:
        # T[t] = [[1, Z[t], 0],
        #         [0, gates[t], 0],
        #         [0, 0, 1]]
        dZ = dT_tensor[..., 0, 1]
        d_gates = dT_tensor[..., 1, 1]
        print("dC_result:", dC_result)
        print("dT_tensor:", dT_tensor)
        print("dZ:", dZ)
        print("d_gates:", d_gates)
        
        return dZ, d_gates, None, None

def lifted_gated_scan_vectorized(Z: torch.Tensor, gates: torch.Tensor, dim: int = 2, combine_mode: str = "generic") -> torch.Tensor:
    return LiftedGatedScanV4_Final.apply(Z, gates, dim, combine_mode)

############################################
# Brute Force Version (Loop + Autograd)
############################################

def lifted_gated_scan_bruteforce(Z: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """
    Brute-force forward pass that computes the cumulative matrix product in a loop.
    This version is fully differentiable via autograd.
    """
    B, C, T = Z.shape
    cumulative_list = []
    for t in range(T):
        one = torch.ones_like(Z[..., t])
        zero = torch.zeros_like(Z[..., t])
        T_t = torch.stack([
            torch.stack([one, Z[..., t], zero], dim=-1),
            torch.stack([zero, gates[..., t], zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1)
        ], dim=-2)  # (B, C, 3, 3)
        if t == 0:
            cum = T_t
        else:
            cum = torch.matmul(T_t, cumulative_list[-1])
        cumulative_list.append(cum)
    
    # s0 vector is [0,1,1]^T
    s0 = torch.tensor([0.0, 1.0, 1.0], dtype=Z.dtype, device=Z.device).view(1,1,3,1).expand(B, C, 3, 1)
    outputs = []
    for t in range(T):
        s = torch.matmul(cumulative_list[t], s0).squeeze(-1)  # (B, C, 3)
        out = s[..., 0]  # Only the first element
        outputs.append(out)
    outputs = torch.stack(outputs, dim=-1)  # (B, C, T)
    return outputs

############################################
# Main: Compare the Two Implementations
############################################

def main():
    torch.manual_seed(42)
    B, C, T = 2, 3, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Create identical inputs for both versions.
    Z_init = torch.randn(B, C, T, device=device, dtype=dtype)
    gates_init = torch.randn(B, C, T, device=device, dtype=dtype)
    
    # For brute force, make copies that require grad.
    Z_bf = Z_init.clone().detach().requires_grad_(True)
    gates_bf = gates_init.clone().detach().requires_grad_(True)
    # For vectorized version:
    Z_vec = Z_init.clone().detach().requires_grad_(True)
    gates_vec = gates_init.clone().detach().requires_grad_(True)

    # Brute-force forward and backward:
    out_bf = lifted_gated_scan_bruteforce(Z_bf, gates_bf)
    loss_bf = out_bf.pow(2).sum()
    loss_bf.backward()
    
    # Vectorized (custom autograd) forward and backward:
    out_vec = lifted_gated_scan_vectorized(Z_vec, gates_vec, dim=2, combine_mode="generic")
    loss_vec = out_vec.pow(2).sum()
    loss_vec.backward()

    # Print outputs and gradients for both versions.
    print("===== Brute Force Version =====")
    print("Output:\n", out_bf)
    print("Gradients for Z:\n", Z_bf.grad)
    print("Gradients for gates:\n", gates_bf.grad)

    print("\n===== Vectorized Version =====")
    print("Output:\n", out_vec)
    print("Gradients for Z:\n", Z_vec.grad)
    print("Gradients for gates:\n", gates_vec.grad)
    
    # Compare gradients
    print("\n===== Gradient Differences (Vectorized - Brute Force) =====")
    z_diff = Z_vec.grad - Z_bf.grad
    gates_diff = gates_vec.grad - gates_bf.grad
    print("Difference for Z:\n", z_diff)
    print("Difference for gates:\n", gates_diff)
    
    # Check if implementations match
    output_match = torch.allclose(out_vec, out_bf, rtol=1e-5, atol=1e-5)
    grad_z_match = torch.allclose(Z_vec.grad, Z_bf.grad, rtol=1e-5, atol=1e-5)
    grad_gates_match = torch.allclose(gates_vec.grad, gates_bf.grad, rtol=1e-5, atol=1e-5)
    
    print("\n===== Implementation Check =====")
    print(f"Outputs match: {output_match}")
    print(f"Z gradients match: {grad_z_match}")
    print(f"Gates gradients match: {grad_gates_match}")
    print(f"All checks passed: {output_match and grad_z_match and grad_gates_match}")

if __name__ == "__main__":
    main()
