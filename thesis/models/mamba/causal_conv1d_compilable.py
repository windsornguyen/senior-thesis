from typing import Optional, Tuple
import torch
import causal_conv1d_cuda

########################################
# Custom ops registration
########################################


# Causal Conv1D Forward Function
@torch.library.custom_op(
    "mamba_causal_conv1d::causal_conv1d_fwd",
    mutates_args=(),
    device_types="cuda",
)
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    # Ensure activation is valid
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # Ensure x is contiguous
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()

    # Make bias and seq_idx contiguous if they exist
    bias = bias.contiguous() if bias is not None else None
    seq_idx = seq_idx.contiguous() if seq_idx is not None else None

    # Translate activation to bool for custom CUDA kernel
    use_activation = activation in ["silu", "swish"]

    # Call custom CUDA kernel for forward pass
    out = causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, seq_idx, None, None, use_activation)
    return out


# Register a fake forward pass for tracing
@causal_conv1d_fwd.register_fake
def _causal_conv1d_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    torch._check(x.shape[-2] == weight.shape[0])
    return torch.empty_like(x)


# Causal Conv1D Backward Function
@torch.library.custom_op(
    "mamba_causal_conv1d::causal_conv1d_bwd",
    mutates_args=(),
    device_types="cuda",
)
def causal_conv1d_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    activation: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Ensure dout is contiguous
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()

    # Call custom CUDA kernel for backward pass
    dx, dweight, dbias, _ = causal_conv1d_cuda.causal_conv1d_bwd(
        x, weight, bias, dout, seq_idx, None, None, None, False, activation
    )

    # If there was no bias originally, we can ignore dbias
    dbias = dbias if bias is not None else torch.empty((0,), device=dout.device)
    return dx, dweight, dbias


# Register a fake backward pass for tracing
@causal_conv1d_bwd.register_fake
def _causal_conv1d_bwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    activation: bool,
):
    return (
        torch.empty_like(x),
        torch.empty_like(weight),
        torch.empty_like(bias) if bias is not None else None,
    )


# Setup context for autograd
def causal_conv1d_setup_context(ctx, inputs, output):
    x, weight, bias, seq_idx, activation = inputs
    ctx.activation = activation in ["silu", "swish"]
    ctx.save_for_backward(x, weight, bias, seq_idx)


# Bridge for backward pass in autograd
def causal_conv1d_bwd_bridge(ctx, dout):
    x, weight, bias, seq_idx = ctx.saved_tensors
    dx, dweight, dbias = causal_conv1d_bwd(x, weight, bias, dout, seq_idx, ctx.activation)
    # If original bias was None, dbias will remain None
    dbias = dbias if bias is not None else None
    return dx, dweight, dbias, None, None


# Register custom autograd function
torch.library.register_autograd(
    "mamba_causal_conv1d::causal_conv1d_fwd",
    causal_conv1d_bwd_bridge,
    setup_context=causal_conv1d_setup_context,
)


# Define a higher-level function to invoke the custom op
def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None):
    return causal_conv1d_fwd(x, weight, bias, seq_idx, activation)


########################################
# causal_conv1d_update op
########################################


@torch.library.custom_op(
    "mamba_causal_conv1d::causal_conv1d_update",
    mutates_args=(),
    device_types="cuda",
)
def causal_conv1d_update_fwd(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    x: (batch, dim, seqlen) or (batch, dim)
    conv_state: (batch, dim, state_len)
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,) or None
    out: same shape as x
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    do_activation = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    out = causal_conv1d_cuda.causal_conv1d_update(x, conv_state, weight, bias, do_activation, cache_seqlens)
    if unsqueeze:
        out = out.squeeze(-1)
    return out


@causal_conv1d_update_fwd.register_fake
def _causal_conv1d_update_fwd(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(x)


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    return causal_conv1d_update_fwd(x, conv_state, weight, bias, activation, cache_seqlens)


def print_tensor_stats(t: torch.Tensor, label: str) -> None:
    """Print shape, min, max, mean, std of a tensor."""
    if t is None or t.numel() == 0:
        print(f"{label} is None or empty.")
        return
    print(
        f"{label}: shape={tuple(t.shape)}, "
        f"min={t.min():.4f}, max={t.max():.4f}, "
        f"mean={t.mean():.4f}, std={t.std():.4f}"
    )


def test_conv(label: str, fn, x: torch.Tensor, weight: torch.Tensor, bias, activation="silu"):
    """
    Runs forward and backward for a given function `fn`,
    prints stats of the output and gradients.
    """
    print(f"\n=== {label} ===")
    # Forward
    out = fn(x, weight, bias, activation=activation)
    print_tensor_stats(out, "Out")

    # Backward
    out.sum().backward()
    print_tensor_stats(x.grad, "x.grad")
    print_tensor_stats(weight.grad, "weight.grad")
    if bias is not None:
        print_tensor_stats(bias.grad, "bias.grad")

    return out


# Test the implementation
if __name__ == "__main__":
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_ref

    torch.manual_seed(1746)
    device = "cuda"

    # Prepare input
    x = torch.randn(8, 32, 16, device=device, requires_grad=True)
    weight = torch.randn(32, 3, device=device, requires_grad=True)
    # Uncomment if you want bias:
    # bias = torch.randn(32, device=device, requires_grad=True)
    bias = None

    # 1) Uncompiled Implementation
    x.grad, weight.grad = None, None
    if bias is not None:
        bias.grad = None
    test_conv("Uncompiled Implementation", causal_conv1d_fn, x, weight, bias, "silu")

    # 2) Compiled Implementation
    x.grad, weight.grad = None, None
    if bias is not None:
        bias.grad = None

    compiled_conv1d = torch.compile(causal_conv1d_fn)
    print(f"\nCompiled function object: {compiled_conv1d}")
    test_conv("Compiled Implementation", compiled_conv1d, x, weight, bias, "silu")

    # 3) Reference Implementation
    x.grad, weight.grad = None, None
    if bias is not None:
        bias.grad = None
    test_conv("Reference Implementation", causal_conv1d_fn_ref, x, weight, bias, "silu")
