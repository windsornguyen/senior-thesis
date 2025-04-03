import torch
from torch._higher_order_ops import associative_scan

class AdditiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int = -1, combine_mode: str = "pointwise") -> torch.Tensor:
        # Save parameters for backward
        ctx.dim = dim
        ctx.combine_mode = combine_mode
        # The forward pass is just an associative scan with addition.
        out = associative_scan(lambda a, b: a + b, x, dim=dim, combine_mode=combine_mode)
        # Nothing special to save, as the derivative is 1 everywhere.
        ctx.save_for_backward(torch.tensor(1.0, device=x.device, dtype=x.dtype))
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve dimension (combine_mode isn't needed in backward here)
        dim = ctx.dim
        # For y[i] = sum_{j=0}^{i} x[j], each x[j] gets the sum of all grad_output from j to end.
        # We compute that via a reversed cumulative sum along the specified dimension.
        grad_x = torch.flip(torch.cumsum(torch.flip(grad_output, dims=[dim]), dim=dim), dims=[dim])
        # Return gradients for each argument. Non-Tensor args get None.
        return grad_x, None, None

def additive_scan(x: torch.Tensor, dim: int = -1, combine_mode: str = "pointwise") -> torch.Tensor:
    """
    Computes the cumulative sum of `x` along a specified dimension using an
    associative scan.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to compute the cumulative sum.
        combine_mode (str): The combine_mode to use for the associative scan
            (e.g., "pointwise").

    Returns:
        torch.Tensor: The cumulative sum of `x` along `dim`.
    """
    return AdditiveScan.apply(x, dim, combine_mode)

def run_gradcheck():
    # Use double precision for gradcheck.
    T, D = 16, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(T, D, device=device, dtype=torch.float64, requires_grad=True)
    test = torch.autograd.gradcheck(additive_scan, (x, -1, "pointwise"), eps=1e-6, atol=1e-4)
    print("Gradcheck passed?", test)

if __name__ == "__main__":
    T, D = 16, 8
    device = "cuda"  # Only CUDA is supported for now (associative_scan still very experimental)
    x = torch.randn(T, D, requires_grad=True, device=device, dtype=torch.float32)
    y = additive_scan(x, dim=-1, combine_mode="pointwise")
    loss = y.pow(2).sum()
    loss.backward()
    print("Input:", x)
    print("Cumulative sum output:", y)
    print("Gradient w.r.t. x:", x.grad)

    print("Running gradcheck...")
    run_gradcheck()
