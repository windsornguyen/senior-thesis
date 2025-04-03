import torch
from torch._higher_order_ops import associative_scan

class GatedScanVectorizedBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z: torch.Tensor, gates: torch.Tensor, dim: int = 2, combine_mode: str = "generic") -> torch.Tensor:
        # Pack the two channels into a state tensor of shape (..., 2)
        # (We unsqueeze both so that both tensors become (..., 1) and then concatenate.)
        state = torch.cat([Z.unsqueeze(-1), gates.unsqueeze(-1)], dim=-1)
        ctx.dim = dim
        ctx.combine_mode = combine_mode

        def combine_fn(state_a, state_b):
            # Both state_a and state_b have shape (..., 2). We slice them to keep an extra dim.
            Z_a = state_a[..., 0:1]
            gate_a = state_a[..., 1:2]
            Z_b = state_b[..., 0:1]
            gate_b = state_b[..., 1:2]
            Z_combined = Z_a + gate_a * Z_b
            gate_combined = gate_a * gate_b
            return torch.cat([Z_combined, gate_combined], dim=-1)

        # Run the associative scan along the given dimension.
        result = associative_scan(combine_fn, state, dim=dim, combine_mode=combine_mode)
        ctx.save_for_backward(state, result)
        # We return only the Z channel (i.e. the first channel of the last dimension).
        return result[..., 0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state, result = ctx.saved_tensors
        dim = ctx.dim
        # For simplicity, we only support dim==2 in this example.
        if dim != 2:
            raise NotImplementedError("gated_scan currently supports dim=2 only.")
        B, C, T, _ = state.shape
        device, dtype = state.device, state.dtype

        # --- Step 1: Set up the reverse recurrence ---
        # Let dY[t] denote the gradient with respect to the cumulative state at time t.
        # Since the forward pass outputs only the Z channel, we initialize:
        dY_final = torch.zeros(B, C, 2, device=device, dtype=dtype)
        dY_final[..., 0] = grad_output[..., -1]  # For t=T-1, (grad_output, 0)

        # --- Step 2: Build the 2x2 matrices M[t] for t=1,...,T-1 ---
        # For each t>=1, define:
        #   M[t] = [[1, 0],
        #           [ state[t]_Z, state[t]_gate ]]
        ones = torch.ones(B, C, T - 1, device=device, dtype=dtype)
        zeros = torch.zeros(B, C, T - 1, device=device, dtype=dtype)
        M_bottom_left = state[:, :, 1:, 0]   # shape: (B, C, T-1)
        M_bottom_right = state[:, :, 1:, 1]  # shape: (B, C, T-1)
        M_top = torch.stack([ones, zeros], dim=-1)          # shape: (B, C, T-1, 2)
        M_bottom = torch.stack([M_bottom_left, M_bottom_right], dim=-1)  # shape: (B, C, T-1, 2)
        M = torch.stack([M_top, M_bottom], dim=-2)  # shape: (B, C, T-1, 2, 2)

        # --- Step 3: Compute cumulative products in reverse using an associative scan ---
        def matmul_combine(A, B):
            return torch.matmul(A, B)
        P = associative_scan(matmul_combine, M, dim=2, reverse=True, combine_mode=ctx.combine_mode)
        # P[..., t] will be the product M[t] @ M[t+1] @ ... @ M[T-2]

        # --- Step 4: Recover all dY ---
        # For t in 0,...,T-2: dY[t] = P[..., t] @ dY_final.
        dY_mid = torch.matmul(P, dY_final.unsqueeze(-1)).squeeze(-1)  # shape: (B, C, T-1, 2)
        dY = torch.cat([dY_mid, dY_final.unsqueeze(2)], dim=2)         # shape: (B, C, T, 2)

        # --- Step 5: Compute gradients for the input state ---
        # For t = 0, the cumulative state is just the first state element.
        d_state = torch.empty_like(state)
        d_state[:, :, 0, :] = dY[:, :, 0, :]
        # For t >= 1, the gradient for the "right" input (i.e. state[t]) is given by:
        #    d_state[t] = (result[..., t-1, 1]) * dY[t]
        # where result[..., t-1, 1] is the cumulative gate from the previous time step.
        gate_prev = result[:, :, :-1, 1]            # shape: (B, C, T-1)
        gate_prev_exp = gate_prev.unsqueeze(-1)      # shape: (B, C, T-1, 1)
        d_state[:, :, 1:, :] = gate_prev_exp * dY[:, :, 1:, :]

        # Unpack the gradients for the two original inputs.
        dZ_input = d_state[..., 0]         # shape: (B, C, T)
        dGates_input = d_state[..., 1]       # shape: (B, C, T)
        return dZ_input, dGates_input, None, None

def gated_scan(Z: torch.Tensor, gates: torch.Tensor, dim: int = 2, combine_mode: str = "generic") -> torch.Tensor:
    """
    Computes a token-dependent gated accumulation:
      Z_out[t] = Z[t] accumulated via:
          Z_out[t] = Z_out[t-1] + gate[t-1] * Z[t],
      with gate_out[t] = gate[t-1] * gate[t],
    and returns the accumulated Z values.

    Args:
      Z (torch.Tensor): Input tensor with shape (..., T, ...), where T is the scan dimension.
      gates (torch.Tensor): Tensor of the same shape as Z.
      dim (int): The dimension along which to scan. (Default is 2.)
      combine_mode (str): The combine mode for the associative scan (default: "generic").

    Returns:
      torch.Tensor: The accumulated result (with the same shape as Z).
    """
    return GatedScanVectorizedBwd.apply(Z, gates, dim, combine_mode)

def run_gradcheck():
    torch.manual_seed(0)
    B, C, T = 2, 3, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use double precision for gradcheck.
    Z = torch.randn(B, C, T, device=device, dtype=torch.float64, requires_grad=True)
    gates = torch.randn(B, C, T, device=device, dtype=torch.float64, requires_grad=True)
    test = torch.autograd.gradcheck(gated_scan, (Z, gates, 2, "generic"), eps=1e-6, atol=1e-4)
    print("Gradcheck passed?", test)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T = 2, 3, 8
    device = "cuda"  # (Only CUDA is supported at the moment.)
    Z = torch.randn(B, C, T, device=device, requires_grad=True)
    gates = torch.randn(B, C, T, device=device, requires_grad=True)
    
    out = gated_scan(Z, gates, dim=2, combine_mode="generic")
    loss = out.pow(2).sum()
    loss.backward()
    print("Output (Z scan):", out)
    print("Gradients for Z:", Z.grad)
    print("Gradients for gates:", gates.grad)
    
    print("Running gradcheck...")
    run_gradcheck()
