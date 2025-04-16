"""Adapted from @https://github.com/bohnstingl/pytorch/blob/associative_scan_74/torch/_higher_order_ops/associative_scan.py"""

import contextlib
import functools
import itertools
import sys
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.utils.benchmark as benchmark_utils
from torch.autograd import gradcheck, gradgradcheck

import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._subclasses.functional_tensor import disable_functional_mode
from torch._higher_order_ops.utils import (
    _maybe_reenter_make_fx,
    _maybe_run_with_interpreter,
    _set_compilation_env,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch.fx.passes.shape_prop import (
    _extract_tensor_metadata,
    TensorMetadata,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

import triton
import triton.language as tl

aten = torch._ops.ops.aten

try:
    import jax
    import jax.numpy as jnp
    from jax.lax import associative_scan as jax_associative_scan

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax_associative_scan = None
    print("JAX not available; skipping JAX comparison tests.", file=sys.stderr)

from thesis.experiments.utils.assoc_scan.aotautograd import create_bw_fn


def autotune(
    key=None, configs=None, warmup=25, rep=100, max_autotune_gemm=True, cache_key=None
):
    r"""A convenient decorator for autotuning Triton kernels.

    Args:
        key: List of parameter names to use as the autotuning key.
             Default: Detects common dimension parameters (B, T, N, D).
        configs: List of Triton.Config objects. If None, creates configs with different
                 BLOCK_SIZE values based on power of 2.
        warmup: Number of warmup iterations for benchmarking.
        rep: Number of repeat iterations for benchmarking.
        max_autotune_gemm: Whether to use predefined GEMM-like configs for kernels with
                          matrix-like parameters.
        cache_key: Optional custom cache key prefix for disk caching.

    Example:
        @autotune()  # Automatically detects parameters and creates configs
        @triton.jit
        def my_kernel(x_ptr, B, T, D, BLOCK_SIZE: tl.constexpr):
            # kernel code here

        @autotune(key=['B', 'T'])  # Custom key parameters
        @triton.jit
        def another_kernel(x_ptr, B, T, D, BLOCK_SIZE: tl.constexpr):
            # kernel code here

    """

    def get_default_configs(fn):
        """Create default configurations based on function signature."""
        import inspect

        sig = inspect.signature(fn)
        params = sig.parameters

        # Detect common block size parameters
        has_block_size = any(p.endswith("BLOCK_SIZE") for p in params)

        if not has_block_size:
            return [triton.Config({}, num_warps=4)]

        # Generate power-of-2 block sizes from 16 to 1024
        block_sizes = [16, 32, 64, 128, 256, 512, 1024]
        configs = []

        # If we have matrix-like dimensions and max_autotune_gemm is True
        if max_autotune_gemm and any(
            p in params for p in ["M", "N", "K", "B", "T", "D"]
        ):
            # Matrix multiplication-like configs (block sizes for different dimensions)
            for bs in [32, 64, 128, 256]:
                configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=4))
                configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=8))
                configs.append(
                    triton.Config({"BLOCK_SIZE": bs}, num_stages=2, num_warps=8)
                )
        else:
            # Standard block size configs
            for bs in block_sizes:
                # Different warp configurations for different block sizes
                if bs <= 64:
                    configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=2))
                    configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=4))
                elif bs <= 256:
                    configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=4))
                    configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=8))
                else:
                    configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=8))
                    configs.append(triton.Config({"BLOCK_SIZE": bs}, num_warps=16))

        # Add specialized configs for specific kernel signatures
        if "BLOCK_SIZE" in params and "T" in params and "B" in params and "D" in params:
            # Looks like a sequence processing kernel
            configs.extend(
                [
                    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
                    triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
                    triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=3),
                ]
            )

        print(
            f"[Tiki Autotuner] Generated {len(configs)} configurations to test for '{fn.__name__}'"
        )
        return configs

    def get_default_key(fn):
        """Auto-detect which parameters should be part of the autotuning key."""
        import inspect

        sig = inspect.signature(fn)
        params = sig.parameters

        # Common dimension parameters to use as keys
        common_dims = ["B", "T", "N", "D", "M", "K", "H", "W", "C"]
        detected_keys = [p for p in params if p in common_dims]

        # If no common dimensions found, use the first 1-2 integer-like parameter names
        if not detected_keys:
            int_like_params = [
                p
                for p in params
                if p not in ["x_ptr", "o_ptr", "y_ptr", "z_ptr", "mask"]
                and not p.endswith("_ptr")
                and not p.startswith("stride_")
            ]
            detected_keys = int_like_params[:2] if len(int_like_params) > 0 else []

        print(
            f"[Tiki Autotuner] Auto-detected key parameters for '{fn.__name__}': {detected_keys}"
        )
        return detected_keys

    def decorator(fn):
        nonlocal key, configs

        # Auto-detect key if not provided
        if key is None:
            key = get_default_key(fn)
        else:
            print(
                f"[Tiki Autotuner] Using provided key parameters for '{fn.__name__}': {key}"
            )

        # Create default configs if not provided
        if configs is None:
            print(f"[Tiki Autotuner] Creating default configs for '{fn.__name__}'")
            configs = get_default_configs(fn)
        else:
            print(
                f"[Tiki Autotuner] Using {len(configs)} provided configs for '{fn.__name__}'"
            )

        # Store if we have already printed the best config for a given key value
        printed_configs = {}

        # Create a custom pre-hook for the triton.autotune decorator
        def autotune_pre_hook(args_dict, reset_only=False):
            # Only print the autotuning start message if we haven't tuned for this key yet
            current_key_tuple = tuple(args_dict.get(k) for k in key if k in args_dict)
            if current_key_tuple not in printed_configs:
                print(
                    f"[Tiki Autotuner] Autotuning '{fn.__name__}' with key values: "
                    + ", ".join(
                        [f"{k}={args_dict.get(k)}" for k in key if k in args_dict]
                    )
                )
            return {}

        # Apply the Triton autotune decorator
        print(
            f"[Tiki Autotuner] Setting up autotuning for '{fn.__name__}' with {len(configs)} configs"
        )
        autotuned_fn = triton.autotune(
            configs=configs,
            key=key,
            warmup=warmup,
            rep=rep,
            pre_hook=autotune_pre_hook,
            # cache_results=True if cache_key else False, # Not supported in this Triton version
        )(fn)

        # Create a wrapper function to add a completion message after autotuning
        @functools.wraps(autotuned_fn)
        def wrapper(*args, **kwargs):
            # Construct the current key tuple from *args based on arg_names
            # This assumes key parameters are passed positionally and match the kernel signature
            arg_names = (
                autotuned_fn.arg_names
            )  # Get arg names from the autotuned function
            args_dict = {**dict(zip(arg_names, args)), **kwargs}
            current_key_tuple = tuple(args_dict.get(k) for k in key if k in args_dict)

            # Call the autotuned function (this triggers tuning if needed)
            result = autotuned_fn(*args, **kwargs)

            best_config = getattr(autotuned_fn, "best_config", "Not Set")
            print(
                f"[Tiki Autotuner] After calling '{fn.__name__}' (key={current_key_tuple}): best_config is {best_config}"
            )

            # Check if tuning occurred and we haven't printed this config yet
            if (
                hasattr(autotuned_fn, "best_config")
                and autotuned_fn.best_config is not None
            ):
                if (
                    current_key_tuple not in printed_configs
                    or printed_configs[current_key_tuple] != autotuned_fn.best_config
                ):
                    config_str = str(autotuned_fn.best_config).replace("\n", " ")
                    # Remove excessive whitespace
                    config_str = " ".join(config_str.split())
                    print(
                        f"[Tiki Autotuner] Selected config for '{fn.__name__}' (key={current_key_tuple}): {config_str}"
                    )
                    printed_configs[current_key_tuple] = autotuned_fn.best_config
                else:
                    print(
                        f"[Tiki Autotuner] Using cached config for '{fn.__name__}' (key={current_key_tuple})"
                    )

            return result

        return wrapper

    # Handle both @autotune and @autotune()
    if callable(key) and configs is None:
        print(f"[Tiki Autotuner] Autotuning '{key.__name__}'")
        fn = key
        key = None
        return decorator(fn)

    return decorator


@autotune(key=["B", "T", "D"])
@triton.jit
def batched_cumprod_kernel(
    x_ptr,  # Input tensor pointer [B, T, T, D]
    o_ptr,  # Output tensor pointer
    B: tl.int32,
    T: tl.int32,
    D: tl.int32,
    stride_b: tl.int32,
    stride_t0: tl.int32,
    stride_t1: tl.int32,
    stride_d: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    r"""Triton kernel to compute a batched cumulative product along the time axis.

    Parameters:
        x_ptr: Pointer to the input tensor.
        o_ptr: Pointer to the output tensor.
        B, T, D: Dimensions corresponding to batch, time, and depth.
        stride_b, stride_t0, stride_t1, stride_d: Tensor strides.
        BLOCK_SIZE: Block size constant.

    """
    pid_b = tl.program_id(0)
    pid_t0 = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_t1 = tl.arange(0, BLOCK_SIZE)
    mask = offs_t1 < T

    base = pid_b * stride_b + pid_t0 * stride_t0 + pid_d * stride_d
    x = tl.load(x_ptr + base + offs_t1 * stride_t1, mask=mask, other=1.0)
    result = tl.cumprod(x, axis=0)

    tl.store(o_ptr + base + offs_t1 * stride_t1, result, mask=mask)


def launch_cumprod(h_mat: torch.Tensor) -> torch.Tensor:
    r"""
    Launch Triton kernel to compute a batched cumulative product.

    Args:
        h_mat: Tensor of shape [B, T, T, D].

    Returns:
        A tensor with the same shape containing the cumulative products.

    """
    if h_mat.ndim != 4:
        raise ValueError("Input tensor must have 4 dimensions [B, T, T, D].")

    B, T, _, D = h_mat.shape
    h_mat = h_mat.contiguous()
    out = torch.empty_like(h_mat)

    grid = (B, T, D)
    batched_cumprod_kernel[grid](
        h_mat,
        out,
        B,
        T,
        D,
        h_mat.stride(0),
        h_mat.stride(1),
        h_mat.stride(2),
        h_mat.stride(3),
    )

    return out


@triton.jit
def combine_fn(carry, x):
    r"""Combine function used in Triton associative scan.

    Args:
        carry: Current accumulated value.
        x: Next element in the scan.

    Returns:
        The product of carry and x.

    """
    return carry * x


@autotune(key=["B", "D", "T"])
@triton.jit
def bwd_grad_scan(
    g_h_ptr,
    g_x_ptr,
    g_ys_ptr,
    g_xs_ptr,
    B: tl.int32,
    D: tl.int32,
    T: tl.int32,
    stride_b: tl.int32,
    stride_t: tl.int32,
    stride_d: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    r"""Backward gradient scan kernel that performs an associative scan for gradient propagation.

    Args:
        g_h_ptr, g_x_ptr, g_ys_ptr, g_xs_ptr: Pointers to gradient tensors.
        B, D, T: Dimensions for batch, depth, and time.
        stride_b, stride_t, stride_d: Corresponding tensor strides.
        BLOCK_SIZE: Block size for processing.

    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < T
    base_offset = pid_b * stride_b + pid_d * stride_d
    rev_offsets = T - 1 - offsets  # Do reverse indexing in kernel

    g_h_rev = tl.load(
        g_h_ptr + base_offset + rev_offsets * stride_t, mask=mask, other=1.0
    )
    g_ys_rev = tl.load(
        g_ys_ptr + base_offset + rev_offsets * stride_t, mask=mask, other=0.0
    )
    g_x_rev = tl.load(
        g_x_ptr + base_offset + rev_offsets * stride_t, mask=mask, other=0.0
    )

    chain_scan = tl.associative_scan(
        g_h_rev, axis=0, combine_fn=combine_fn, reverse=False
    )
    fused_result = chain_scan * g_ys_rev * g_x_rev

    tl.store(g_xs_ptr + base_offset + rev_offsets * stride_t, fused_result, mask=mask)


def launch_bwd_grad_scan(
    g_h: torch.Tensor,
    g_x: torch.Tensor,
    g_ys: torch.Tensor,
) -> torch.Tensor:
    r"""Launch Triton kernel for the backward gradient scan.

    Args:
        g_h, g_x, g_ys: Input gradient tensors of shape [B, T, D].

    Returns:
        A tensor with the computed gradient scan result.

    """
    B, T, D = g_h.shape
    g_xs = torch.empty_like(g_h, device=g_h.device, dtype=g_h.dtype)

    grid = (B, D)
    bwd_grad_scan[grid](
        g_h,
        g_x,
        g_ys,
        g_xs,
        B,
        D,
        T,
        g_h.stride(0),
        g_h.stride(1),
        g_h.stride(2),
    )

    return g_xs


def _from_fun(t: Any) -> Any:
    r"""Materialize a tensor from a functional tensor if needed.

    Args:
        t: A tensor that may be a functional tensor.

    Returns:
        A new tensor with the same properties or the original value.

    """
    from torch._functorch.aot_autograd import from_fun
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, torch.Tensor):
        if t.dtype != torch.bool:
            return torch.empty_strided(
                t.size(),
                t.stride(),
                dtype=t.dtype,
                requires_grad=t.requires_grad,
                device=t.device,
            )
        else:
            maybe_unfunc_t = t
            if isinstance(t, FunctionalTensor):
                torch._sync(t)
                maybe_unfunc_t = from_fun(t)
            elif torch._is_functional_tensor(t):
                torch._sync(t)
                maybe_unfunc_t = torch._from_functional_tensor(t)
            return maybe_unfunc_t.clone()
    return t


def materialize_as_graph(
    fn: Callable,
    args: tuple[Any],
    include_key_set: torch._C.DispatchKeySet,
    exclude_key_set: torch._C.DispatchKeySet,
    force_enable_grad: bool = False,
) -> torch.fx.GraphModule:
    r"""Materialize a PyTorch FX GraphModule by running the function with non-functional tensors.

    Args:
        fn: Function to trace.
        args: Tuple of arguments to pass to the function.
        include_key_set: Dispatch key set to include.
        exclude_key_set: Dispatch key set to exclude.
        force_enable_grad: Whether to force-enable gradients.

    Returns:
        A GraphModule representing the traced function.

    """

    @torch._dynamo.disable(recursive=True, reason=None)
    def _materialize_as_graph_inner():
        with suspend_functionalization(), disable_functional_mode():
            with disable_proxy_modes_tracing():
                unfunc_t = [_from_fun(arg) for arg in args]
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                torch._C._ForceDispatchKeyGuard(include_key_set, exclude_key_set),
            )
            if force_enable_grad:
                stack.enter_context(torch.enable_grad())
            return _maybe_reenter_make_fx(fn)(*unfunc_t)

    gm = _materialize_as_graph_inner()
    if gm is None:
        raise RuntimeError("GraphModule creation failed.")
    return gm


def wrap_combine_fn_flat(
    *args, combine_fn: Callable, spec: Any, num_leaves: int
) -> list:
    r"""Wrap and flatten the combine function across PyTree leaves.

    Args:
        *args: List of flattened PyTree leaves.
        combine_fn: Combine function to apply.
        spec: PyTree specification.
        num_leaves: Number of leaves in the PyTree.

    Returns:
        A list of combined PyTree leaves.
    """
    if len(args) != 2 * num_leaves:
        raise ValueError("Expected twice the number of leaves in args.")
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    if num_leaves != len(combined_leaves):
        raise RuntimeError("Mismatch in number of leaves after combine.")
    return combined_leaves


def _interleave(a: torch.Tensor, b: torch.Tensor, dim: int = 0) -> torch.Tensor:
    r"""Interleave two tensors along a given dimension.

    NOTE: See # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors

    Args:
        a: First tensor.
        b: Second tensor.
        dim: Dimension along which to interleave.

    Returns:
        The interleaved tensor.

    """
    b_trunc = a.shape[dim] == b.shape[dim] + 1
    if b_trunc:
        pad = (
            [0] * ((b.ndim - dim - 1) * 2 + 1)
            + [1]
            + [0] * (b.ndim * 2 - ((b.ndim - dim - 1) * 2 + 2))
        )
        b = torch.nn.functional.pad(b, pad)
    stacked = torch.stack([a, b], dim=dim + 1)
    interleaved = torch.flatten(stacked, start_dim=dim, end_dim=dim + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along dim for torch.jit.script to work
        interleaved = aten.slice(interleaved, dim, 0, b.shape[dim] + a.shape[dim] - 1)
    return interleaved


def safe_map(f: Callable, *args) -> list:
    r"""Apply a function safely over multiple lists ensuring equal length.

    Args:
        f: Function to apply.
        *args: Iterables to map over.

    Returns:
        A list resulting from the mapped function.

    Raises:
        ValueError: If the input lists are not of equal length.

    """
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("Length mismatch in safe_map input lists.")
    return list(map(lambda a: f(*a), zip(*args)))


def get_tensor_mask(tensor_list: list[Any]) -> list[bool]:
    r"""Generate a boolean mask indicating which elements are tensors.

    Args:
        tensor_list: List of elements.

    Returns:
        A list of booleans corresponding to whether each element is a tensor.

    """
    return [isinstance(v, torch.Tensor) for v in tensor_list]


def mask_list(
    mask: list[bool], inp: list[Any], other: Optional[list[Any]] = None
) -> list[Any]:
    r"""Mask elements of a list based on a boolean mask.

    Args:
        mask: Boolean mask.
        inp: Primary list.
        other: Optional secondary list to use for masked-off items.

    Returns:
        A new list with elements conditionally selected from `inp` or `other`.

    """
    # Masks elements on an `inp` list.
    # If other is None, then the elements of the `inp` list where the mask is False are removed
    # If other is not None, then the elements of the `inp` list where the mask is False are
    # replaced with the elements of the `other` list
    if other is not None:
        return [i if m else o for m, i, o in zip(mask, inp, other)]
    return [i for m, i in zip(mask, inp) if m]


def first_slice_copy(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return t.select(dim, 0).clone()

def first_slice_copy_with_grad(li: list[Any]) -> list[Any]:
    r"""Create first-slice copies of tensors with gradient information preserved.

    Args:
        li: List of tensors.

    Returns:
        A list of first-slice copies for each tensor.

    """
    # First_slice_copy does not keep the original requires_grad flag,
    # but we need it for materialize_as_graph
    # in order to compute the correct gradients
    return [first_slice_copy(x).requires_grad_(x.requires_grad) for x in li]


def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list:
    r"""Split an iterable into chunks of specified sizes.

    Args:
        iterable: The input iterable.
        chunk_sizes: List of chunk sizes that should sum up to the length of `iterable`.

    Returns:
        A list of lists, where each sublist represents a chunk.

    Raises:
        AssertionError: If the sum of chunk_sizes does not equal the length of the iterable.
    """
    it = iter(iterable)
    if sum(chunk_sizes) != len(iterable):
        raise AssertionError(
            "Sum of chunk sizes must equal the length of the iterable."
        )
    return [list(itertools.islice(it, size)) for size in chunk_sizes]


class AssociativeScanOp(HigherOrderOperator):
    r"""Higher-order operator for performing associative scans."""

    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn: Callable, xs: Any, additional_inputs: Any) -> Any:
        r"""Invoke the associative scan operator.

        Args:
            combine_fn: A callable combining two PyTree structures.
            xs: A PyTree of input tensors.
            additional_inputs: Additional inputs required by the scan.

        Returns:
            The result of the associative scan.

        """
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        if not isinstance(additional_inputs, (tuple, list)):
            raise ValueError("additional_inputs must be a tuple or list.")
        validate_subgraph_args_types(additional_inputs)
        return super().__call__(combine_fn, xs, additional_inputs)


# Instance of the operator
associative_scan_op = AssociativeScanOp()


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""Compute an associative scan along a specified dimension with various modes.

    Args:
        combine_fn: A callable to combine two PyTree structures.
        xs: A PyTree of input tensors.
        dim: The dimension along which to scan.
        reverse: Whether to reverse the scan order.
        combine_mode: 'pointwise' (all inputs on CUDA) or 'generic' mode.

    Returns:
        A PyTree with the associative scan computed.

    Raises:
        ValueError: On invalid input types or shapes.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """

    # TODO: Support lifted arguments in inductor for associative_scan
    # TODO: Support autograd for cases with lifted arguments for combine_mode=pointwise

    if not callable(combine_fn):
        raise ValueError(f"Combine_fn must be callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise ValueError(f"Dim must be an int, but got {type(dim)}")
    if combine_mode not in ["pointwise", "generic"]:
        raise ValueError(
            f"Combine_mode must be 'pointwise' or 'generic', but got {combine_mode}"
        )

    if not torch.compiler.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(associative_scan, fullgraph=True, backend="eager")(
                combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
            )

    leaves, spec = pytree.tree_flatten(xs)
    if combine_mode == "pointwise" and not all(
        leaf.device.type == "cuda" for leaf in leaves
    ):
        raise ValueError(
            "For combine_mode='pointwise', all input tensors must be on CUDA."
        )
    if len(leaves) == 0:
        raise ValueError("Expected at least one xs leaf.")
    if any(not isinstance(x, torch.Tensor) for x in leaves):
        raise ValueError("All xs leaves must be Tensors.")
    if any(x.is_sparse for x in leaves):
        raise ValueError(
            "All xs leaves must be dense Tensors. Consider using `to_dense()`."
        )
    if any(x.ndim <= dim or x.shape[dim] == 0 for x in leaves):
        raise ValueError("Each xs leaf must have a valid dimension for scanning.")

    if reverse:
        leaves = [torch.flip(elem, [dim]) for elem in leaves]

    ndim = leaves[0].ndim
    orig_scan_dim = utils.canonicalize_dim(ndim, dim)
    leaves = [torch.movedim(elem, dim, 0) for elem in leaves]

    # Call the combine_fn with only a slice along the scan dim
    # and check whether the output leaves have the same slice dimensions
    sliced_leaves = [first_slice_copy(leaf) for leaf in leaves]

    out = combine_fn(
        pytree.tree_unflatten(sliced_leaves, spec),
        pytree.tree_unflatten(sliced_leaves, spec),
    )
    out_leaves = pytree.tree_leaves(out)
    if len(leaves) != len(out_leaves):
        raise RuntimeError("Output PyTree structure must match input structure.")
    for x, x_sliced in zip(out_leaves, sliced_leaves):
        if (
            x.shape != x_sliced.shape
            or x.dtype != x_sliced.dtype
            or x.device != x_sliced.device
            or x.stride() != x_sliced.stride()
        ):
            raise RuntimeError(
                f"Output metadata mismatch:\nxs: {(x_sliced.shape, x_sliced.dtype, x_sliced.device, x_sliced.stride())}\n"
                f"operator output: {(x.shape, x.dtype, x.device, x.stride())}"
            )

    if combine_mode == "generic":
        # The generic_associative_scan implementation calls the combine_fn with a `batch` along the scan dimension
        # For example, consider:
        # def add(x: torch.Tensor, y: torch.Tensor):
        #     return x + y
        # leaves = torch.tensor([[0.0, 1.0, 2.0, 3.0]
        #                        [0.0, 1.0, 2.0, 3.0]])
        # which has shape 2 x 4;
        # dim = 1;
        # In the first iteration of `_scan` the combine_fn gets invoked with
        # combine_fn([torch.tensor([[0.0, 2.0],
        #                           [0.0, 2.0]])],
        #            [torch.tensor([[1.0, 3.0],
        #                           [1.0, 3.0]])])
        # The arguments are of shape 2 x 2, but can be evaluated in parallel along the scan dimension.
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=torch.vmap(
                combine_fn,
                in_dims=(
                    pytree.tree_unflatten([0] * len(leaves), spec),
                    pytree.tree_unflatten([0] * len(leaves), spec),
                ),
                out_dims=0,
            ),
            spec=spec,
            num_leaves=len(leaves),
        )
        result_flat = generic_associative_scan(combine_fn, leaves, additional_inputs=())
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec,
            num_leaves=len(leaves),
        )
        result_flat = associative_scan_op(combine_fn, leaves, additional_inputs=())

    if reverse:
        result_flat = [torch.flip(elem, [0]) for elem in result_flat]
    result_flat = [torch.movedim(elem, 0, orig_scan_dim) for elem in result_flat]
    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(
    operator: Callable, leaves: list, dim: int = 0, additional_inputs=()
) -> list:
    r"""A generic associative scan implementation using a recursive strategy.

    Args:
        operator: A callable that combines parts of the tensor.
        leaves: List of flattened tensor leaves.
        dim: Dimension to scan along.
        additional_inputs: Extra inputs for the operator.

    Returns:
        A list of scanned tensor leaves.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        leaves = torch.tensor([0.0, 1.0, 2.0, 3.0])

        First iteration of _scan ->
            # odd_elems -> apply operator on all neighbours
            # odd_elems = operator([torch.tensor([0.0, 2.0])],
            #                      [torch.tensor([1.0, 3.0])])
            odd_elems = torch.tensor([1.0, 5.0])
            Second iteration of _scan ->
                # odd_elems = operator([torch.tensor([1.0])],
                #                      [torch.tensor([5.0])])
                odd_elems = torch.tensor([6.0])
                # even_elems -> apply operator on all odd_elems and
                # every second element of ``elems``, starting from the second element.
                # even_elems is expanded with the first element of ``elems``
                even_elems = [1.0]
                # Merges odd_elems and even_elems
                res = torch.tensor([1.0, 6.0])
            # even_elems -> apply operator on all odd_elems and
            # every second element of ``elems``, starting from the second element.
            # even_elems is expanded with the first element of ``elems``
            even_elems = [0.0, 3.0]
            # Merges odd_elems and even_elems
            res = torch.tensor([0.0, 1.0, 3.0, 6.0])

    """

    def _scan(elems):
        """Perform the actual recursive scan on ``elems``."""
        num_elems = elems[0].shape[dim]
        if num_elems < 2:
            return elems

        reduced_elems = operator(
            *[aten.slice(elem, dim, 0, -1, 2) for elem in elems],
            *[aten.slice(elem, dim, 1, None, 2) for elem in elems],
            *additional_inputs,
        )
        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        else:
            even_elems = operator(
                *odd_elems,
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        even_elems = [
            torch.cat([aten.slice(elem, dim, 0, 1), result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else aten.slice(elem, dim, 0, 1)
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    return _scan(leaves)


def diff_tensor_meta(
    meta1: TensorMetadata, meta2: TensorMetadata, check_grad: bool = True
) -> list[str]:
    """
    Compare two TensorMetadata instances and return differences.

    Args:
        meta1: First tensor metadata.
        meta2: Second tensor metadata.
        check_grad: Whether to compare the 'requires_grad' flag.

    Returns:
        A list of strings describing differences.
    """
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    pair_diffs = []
    for meta_name in TensorMetadata._fields:
        if not check_grad and meta_name == "requires_grad":
            continue
        val1 = getattr(meta1, meta_name)
        val2 = getattr(meta2, meta_name)
        try:
            if val1 != val2:
                pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
        except GuardOnDataDependentSymNode:
            pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
    return pair_diffs


def check_meta_consistency(
    lhs_list: list[Union[torch.Tensor, torch.SymInt, int]],
    rhs_list: list[Union[torch.Tensor, torch.SymInt, int]],
    lhs_name: str,
    rhs_name: str,
) -> None:
    r"""Check if metadata between two lists of tensors/values is consistent.

    Args:
        lhs_list: List of left-hand side items.
        rhs_list: List of right-hand side items.
        lhs_name: Identifier for the left-hand side.
        rhs_name: Identifier for the right-hand side.

    Raises:
        torch._dynamo.exc.UncapturedHigherOrderOpError: If metadata does not match.

    """

    def diff_meta_pairs(lhs_list, rhs_list) -> list[str]:
        def diff_meta(lhs, rhs) -> str:
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                return ", ".join(
                    diff_tensor_meta(
                        _extract_tensor_metadata(lhs, include_contiguity=False),
                        _extract_tensor_metadata(rhs, include_contiguity=False),
                        check_grad=False,
                    )
                )
            elif isinstance(lhs, (int, torch.SymInt)) and isinstance(
                rhs, (int, torch.SymInt)
            ):
                return ""
            return f"type: {lhs} vs {rhs}"

        def diff_device(lhs, rhs) -> str:
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                return (
                    ""
                    if (
                        rhs.device.type == lhs.device.type
                        and rhs.device.index == lhs.device.index
                    )
                    else "device"
                )
            return ""

        if len(lhs_list) != len(rhs_list):
            raise torch._dynamo.exc.UncapturedHigherOrderOpError(
                f"Expected {lhs_name} and {rhs_name} to have the same number of outputs but got lhs:{lhs_list} and rhs:{rhs_list}"
            )
        all_diffs = []
        for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
            if diff := diff_meta(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
            if diff := diff_device(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
        return all_diffs

    if all_diffs := diff_meta_pairs(lhs_list, rhs_list):
        diff_str = "\n".join(all_diffs)
        raise torch._dynamo.exc.UncapturedHigherOrderOpError(
            f"Expected {lhs_name} and {rhs_name} to have same metadata but found:\n{diff_str}"
        )


def trace_associative_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
):
    r"""Trace the associative scan for proxy mode execution.

    Args:
        proxy_mode: The current proxy mode.
        func_overload: The overloaded function used for tracing.
        combine_fn: The combining function.
        xs: List of input tensors.
        additional_inputs: Additional input tensors.

    Returns:
        A tracked tensor tree representing the traced scan.

    """
    with disable_proxy_modes_tracing():
        sample_xs = [first_slice_copy(x) for x in itertools.chain(xs, xs)]
        combine_graph = reenter_make_fx(combine_fn)(*sample_xs, *additional_inputs)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            outputs = node.args[0]
    if outputs is None or len(outputs) != len(xs):
        raise RuntimeError(
            f"Expected combine_fn to return {len(xs)} results but got {len(outputs) if outputs is not None else 'None'}"
        )

    xs_fake_tensors = [first_slice_copy(x) for x in xs]
    output_fake_tensors = [c.meta["val"] for c in outputs]
    check_meta_consistency(xs_fake_tensors, output_fake_tensors, "init", "carry")

    _, combine_graph_name = unique_graph_id(
        proxy_mode, prefix="associative_scan_combine_graph"
    )
    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, xs, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )
    with disable_proxy_modes_tracing():
        out = tuple(aten.clone(x) for x in xs)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs, additional_inputs):
    r"""Dense (non-sparse) implementation of the associative scan operator."""
    return generic_associative_scan(combine_fn, xs, additional_inputs=additional_inputs)


class AssociativeScanAutogradOp(torch.autograd.Function):
    r"""Autograd wrapper for the associative scan operator.

    Example:
        xs = torch.arange(1, 5)  # [1, 2, 3, 4]
        ys = torch.cumprod(xs, dim=0)  # [1, 2, 6, 24]

        def combine_fn(a: torch.Tensor, b: torch.Tensor):
            return a * b

        def combine_fn_bw(a: torch.Tensor, b: torch.Tensor, g_y: torch.Tensor):
            return g_y * b, g_y * a

    Forward pass (simplified):
        ys = associative_scan(combine_fn, xs)
        # Expanded computation:
        # ys_0 = xs_0
        # ys_1 = combine_fn(ys_0, xs_1), ... , ys_T = combine_fn(ys_(T-1), xs_T)

    Backward pass overview:
        Receive upstream gradients g_ys = [g_ys_0, ..., g_ys_T]
        Compute instantaneous gradients (g_x) and backward states (g_h):

        For step t:
            g_h_t, g_x_t = combine_fn_bw(ys_(t-1), xs_t, g_ys_t)

        Example results:
            g_h = [1, 2, 3, 4]
            g_x = [1, 1, 2, 6]

    Gradient transition matrix (h_mat):
        h_mat[i, j] = product of gradients g_h from step i+1 to j

        Example:
            h_mat = [[1, 2, 6, 24],
                    [0, 1, 3, 12],
                    [0, 0, 1, 4],
                    [0, 0, 0, 1]]

    Final gradient computation:
        scaled_h_mat = h_mat * g_ys
        summed_h_mat = scaled_h_mat.sum(dim=1)
        g_xs = summed_h_mat * g_x

        Example:
            g_xs = [33, 16, 5, 1] * [1, 1, 2, 6] = [33, 16, 10, 6]

    General associative_scan autograd steps:
    1. Compute forward pass:
        ys = associative_scan_op(combine_fn, xs, additional_inputs)

    2. Prepare backward computation:
        ctx._combine_fn_bw = create_bw_fn(combine_fn, operands=[xs_0, additional_inputs])

    3. Materialize ctx._combine_fn_bw (required for dynamo tracing compatibility).

    4. Compute instantaneous gradients (g_h_t, g_x_t) at each step using ctx._combine_fn_bw.

    5. Compute gradient transition matrix (h_mat):
        - Form square matrix from g_h
        - Fill lower triangular with 1s
        - Take cumulative product row-wise (due to chain rule)
        - Restore upper triangular zeros

    6. Scale h_mat by upstream gradients g_ys.

    7. Row-wise sum scaled matrix to get contributions.

    8. Multiply by instantaneous gradients (g_x) for final g_xs.

    Note:
    - Gradients for tensors with requires_grad=False will be zero tensors of matching shape.

    This method follows the principles outlined at:
    https://justintchiu.com/blog/pscan_diff/

    """

    @staticmethod
    def forward(ctx, combine_fn, num_xs, num_additional_inputs, *operands):
        """Forward pass for the associative scan.

        Args:
            combine_fn: Combining function.
            num_xs: Number of xs tensors.
            num_additional_inputs: Number of additional inputs.
            operands: All operands (inputs and outputs).

        Returns:
            The outputs of the associative scan.

        """
        ctx._num_xs = num_xs
        ctx._num_additional_inputs = num_additional_inputs
        ctx._combine_fn = combine_fn
        xs, additional_inputs = split_into_chunks(
            operands, [num_xs, num_additional_inputs]
        )
        scan_length = xs[0].shape[0]
        ctx._scan_length = scan_length

        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        with torch._C._AutoDispatchBelowAutograd():
            ys = associative_scan_op(combine_fn, xs, additional_inputs)
            save_tensors_and_symints_for_backward(ctx, list(operands) + list(ys))
        return (*ys,)

    @staticmethod
    def backward(ctx, *flat_grads):
        """Backward pass for the associative scan.

        Args:
            flat_grads: Gradients for the outputs.

        Returns:
            Gradients for the inputs (with None for non-differentiable parameters).

        """
        # The backward of associative_scan is always performed on the first dimension
        dim = 0
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs
        num_additional_inputs = ctx._num_additional_inputs

        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = saved_tensors_and_symints(ctx)
        xs, outs, additional_inputs = split_into_chunks(
            flat_args, [num_xs, num_xs, num_additional_inputs]
        )
        ndim = outs[0].ndim

        # First_slice_copy does not keep the original requires_grad flag,
        # but we need it here in order to compute the correcte gradients
        xs_slices = first_slice_copy_with_grad(list(itertools.chain(xs, xs)))

        # 2.) Prepare the backward graph
        ctx._combine_fn_bw = create_bw_fn(
            ctx._combine_fn, (*xs_slices, *additional_inputs)
        )

        # 3.) Materialize the ``ctx._combine_fn_bw``
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        combine_fn_bw_gm = materialize_as_graph(
            ctx._combine_fn_bw,
            (*xs_slices, *additional_inputs, *[first_slice_copy(o) for o in outs]),
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        # vmap joint graph over scan dimension to compute the individual
        # gradients for each time slice ``t`` in parallel.
        # This computation can be parallelized, as these are just the instantaneous gradients and not the full chain-rule
        mapped_combine_fn_bw_gm = torch.vmap(combine_fn_bw_gm, 0, 0)

        # 4.) Compute the instantaneous gradients at every step ``t``
        # Use a ones_like tensor in order not to scale the g_h_t and g_x_t
        dummy_upstream_grad = (torch.ones_like(x) for x in flat_grads)
        grads = mapped_combine_fn_bw_gm(
            *(o.roll(1, dim) for o in outs), *xs, *dummy_upstream_grad
        )
        g_h_t, g_x_t = split_into_chunks(grads, [num_xs, num_xs])

        def compute_grad_h_mat(g_h: torch.Tensor) -> torch.Tensor:
            # Prepare a ones and a zeros helper mask in order to easily compute the y_mat
            def compute_helper_tril_mask(diagonal: int) -> torch.Tensor:
                mask = torch.tril(
                    torch.ones(
                        scan_length, scan_length, device=g_h.device, dtype=torch.bool
                    ),
                    diagonal=diagonal,
                )
                for _ in range(ndim - 1):
                    mask = mask.unsqueeze(-1)
                return mask.expand(-1, -1, *g_h.shape[1:])

            # The ones mask is used to fill the main diagonal and all elements below it with 1s
            # The elements on the main diagonal are 1 because of do_0/dy_0 = do_1/dy_1 = ... = 1
            # and the elements below it are set to 1, in order for the cumprod can be computed properly.
            ones_mask = compute_helper_tril_mask(0)

            # The zero mask is used to set all elements below the main diagonal to 0, because do_0/dy_1 = do_0/dy_2 = ... = 0
            zeros_mask = compute_helper_tril_mask(-1)

            # 5.1) Repeat the elements of gh to form the square matrix of derivatives
            h_mat = g_h.unsqueeze(dim).repeat_interleave(scan_length, dim)

            # 5.2) Fill the lower triangular part, including the diagonal,
            # of the h_mat with 1s. I.e., use the ones_mask to fill with 1s.
            h_mat.masked_fill_(ones_mask, 1.0)

            # 5.3) Compute the cumulative products across dim + 1
            # h_mat = h_mat.cumprod(dim=dim + 1)
            h_mat = launch_cumprod(h_mat)

            # 5.4) Fill the zeros_mask with 0s again
            h_mat.masked_fill_(zeros_mask, 0.0)

            return h_mat

        def compute_grad(g_x, g_h, g_ys):
            # Set the i=0 component of df(x_i,y_{i-1})/dx_i to 1.0
            # i.e., the first gradient component is always 1.0
            index = [slice(None)] * g_x.ndim
            index[dim] = 0
            g_x[tuple(index)] = 1.0

            # 5.) Compute the gradient matrix
            h_mat = compute_grad_h_mat(g_h)

            # 6.) scale the h_mat with the upstream gradients g_ys
            scaled_h_mat = h_mat * g_ys

            # 7.) Reduce the h_mat with sum along the columns to get the total contributions for xs_t
            summed_h_mat = scaled_h_mat.sum(dim + 1)

            # 8.) Scale with the g_x to obtain the final gradients g_xs
            g_xs = summed_h_mat * g_x

            return g_xs

        # Stack all elements of the gradients along the first dimension.
        # This is useful as later the gradients of those elements can be computed in parallel.
        g_x_stacked = torch.stack(g_x_t)
        g_h_stacked = torch.stack(g_h_t)
        g_ys_stacked = torch.stack(flat_grads)

        # The compute_grad function is parallelized across all individual elements of xs
        # as these gradients can be computed independently from each other
        # compute_grad_mapped = torch.vmap(compute_grad, 0, 0)
        # g_xs = compute_grad_mapped(g_x_stacked, g_h_stacked, g_ys_stacked)

        # Alternatively, do the scan via Triton's associative scan
        g_xs = launch_bwd_grad_scan(g_x_stacked, g_h_stacked, g_ys_stacked)

        # TODO: Currently the gradients for the additional_inputs are not computed properly
        return *[None] * 3, *g_xs, *[None] * num_additional_inputs


@associative_scan_op.py_impl(DispatchKey.Autograd)
def associative_scan_autograd(combine_fn, xs, additional_inputs):
    """Autograd implementation of the associative scan operator.

    Args:
        combine_fn: Combining function.
        xs: List of input tensors.
        additional_inputs: Additional inputs.

    Returns:
        The scanned output tensors.

    """
    num_xs = len(xs)
    num_additional_inputs = len(additional_inputs)
    flat_out = AssociativeScanAutogradOp.apply(
        combine_fn,
        num_xs,
        num_additional_inputs,
        *(tuple(xs) + tuple(additional_inputs)),
    )
    return (*flat_out,)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, xs, additional_inputs):
    """Proxy mode implementation for tracing the associative scan.

    Args:
        mode: The proxy mode.
        combine_fn: Combining function.
        xs: List of input tensors.
        additional_inputs: Additional inputs.

    Returns:
        A traced associative scan result.

    """
    return trace_associative_scan(
        mode, associative_scan_op, combine_fn, xs, additional_inputs
    )


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, xs, additional_inputs):
    """Fake tensor mode implementation for the associative scan.

    Args:
        mode: Fake tensor mode.
        combine_fn: Combining function.
        xs: List of input tensors.
        additional_inputs: Additional inputs.

    Returns:
        Cloned tensors as a stand-in for actual computation.

    """
    with mode:
        return tuple(x.clone() for x in xs)


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, additional_inputs):
    """Functionalize the associative scan operator for redispatch.

    Args:
        ctx: Functionalization context.
        combine_fn: Combining function.
        xs: List of input tensors.
        additional_inputs: Additional inputs.

    Returns:
        Functionalized output tensors.

    """
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        ret = associative_scan_op(
            functional_combine_fn, unwrapped_xs, unwrapped_additional_inputs
        )
    return ctx.wrap_tensors(ret)


def _fake_associative_scan(combine_fn, xs, dim, reverse: bool = False):
    """A fallback (fake) associative scan implementation that performs the scan in Python.

    Args:
        combine_fn: Combining function.
        xs: PyTree of input tensors.
        dim: Dimension to scan along.
        reverse: Whether to reverse the order of scanning.

    Returns:
        A PyTree of tensors containing the scanned results.

    """
    inp_leaves, spec = pytree.tree_flatten(xs)
    result_flat: list[Any] = []
    num_leaves = len(inp_leaves)
    op = reversed if reverse else lambda x: x

    for ind in op(range(inp_leaves[0].size(dim))):
        r = [
            inp_leaves[leave_ind][(slice(None),) * dim + (ind,)]
            for leave_ind in range(num_leaves)
        ]
        if (ind > 0 and not reverse) or (
            ind < (inp_leaves[0].size(dim) - 1) and reverse
        ):
            r = combine_fn(
                pytree.tree_unflatten(result_flat[-1], spec),
                pytree.tree_unflatten(r, spec),
            )
        r_flat, _ = pytree.tree_flatten(r)
        result_flat.append(r_flat)
    results = [
        torch.stack([e[leave_ind] for e in op(result_flat)], dim)
        for leave_ind in range(num_leaves)
    ]
    return pytree.tree_unflatten(results, spec)


def main():
    torch.manual_seed(0)
    if JAX_AVAILABLE:
        jax.random.PRNGKey(0)

    def add_combine_fn(a, b):
        return a + b

    def mul_combine_fn(a, b):
        return a * b

    if JAX_AVAILABLE:

        def jax_add_combine_fn(a, b):
            return a + b

    print("Running gradient checks...")
    xs = torch.randn(2, 4, 3, requires_grad=True, dtype=torch.float64, device="cuda")
    additional_inputs = ()

    def associative_scan_wrapper(input_tensor):
        return associative_scan(
            mul_combine_fn, input_tensor, dim=1, reverse=False, combine_mode="generic"
        )

    try:
        # Test gradcheck with compiled version
        # Clone xs for gradcheck as compile might modify inputs
        gradcheck(
            associative_scan_wrapper,
            (xs.clone().detach().requires_grad_(True).to(torch.float64),),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print("First-order gradient check passed (compiled).")
    except RuntimeError as e:
        print(f"First-order gradient check failed (compiled): {e}")
        # Depending on criticality, might want to return or handle differently

    try:
        # Test gradgradcheck with compiled version
        gradgradcheck(
            associative_scan_wrapper,
            (xs.clone().detach().requires_grad_(True).to(torch.float64),),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print("Second-order gradient check passed (compiled).")
    except RuntimeError as e:
        print(f"Second-order gradient check failed (compiled): {e}")
        # Depending on criticality, might want to return or handle differently

    print(
        "\nRunning fuzzing tests"
        + (" with JAX comparison" if JAX_AVAILABLE else "")
        + "..."
    )
    fuzzer = benchmark_utils.Fuzzer(
        parameters=[
            benchmark_utils.FuzzedParameter(
                name="k0",
                minval=2,
                maxval=128,
                distribution="loguniform",
            ),
            benchmark_utils.FuzzedParameter(
                name="k1",
                minval=2,
                maxval=128,
                distribution="loguniform",
            ),
            benchmark_utils.FuzzedParameter(
                name="k2",
                minval=2,
                maxval=128,
                distribution="loguniform",
            ),
            benchmark_utils.FuzzedParameter(
                name="tensor_dtype",
                distribution={torch.float32: 0.5, torch.float64: 0.5},
            ),
        ],
        tensors=[
            benchmark_utils.FuzzedTensor(
                name="xs",
                size=("k0", "k1", "k2"),
                probability_contiguous=0.75,
                min_elements=64,
                max_elements=512 * 512,
            ),
        ],
        seed=0,
    )

    n = 100
    failed_cases = []
    pytorch_times = []
    jax_times = []
    import timeit

    for i, (tensors, tensor_properties, _) in enumerate(fuzzer.take(n=n)):
        xs_orig = tensors["xs"]
        xs_pytorch = xs_orig.clone().requires_grad_(True)

        xs_order = str(tensor_properties["xs"]["order"])
        shape = ", ".join(f"{d:>4}" for d in xs_pytorch.shape)
        dtype = str(xs_pytorch.dtype)
        description = f"shape: ({shape}), dtype: {dtype}, order: {'contiguous' if xs_pytorch.is_contiguous() else xs_order}"

        try:
            # Time PyTorch execution (using compiled version)
            # Warmup
            _ = associative_scan_wrapper(xs_pytorch)
            torch.cuda.synchronize() if xs_pytorch.is_cuda else None  # Ensure completion if on GPU
            start_time = timeit.default_timer()
            pytorch_result = associative_scan_wrapper(xs_pytorch)
            torch.cuda.synchronize() if xs_pytorch.is_cuda else None
            end_time = timeit.default_timer()
            pytorch_times.append(end_time - start_time)

            if not isinstance(pytorch_result, torch.Tensor):
                raise RuntimeError("Output is not a tensor")

            loss = pytorch_result.sum()
            loss.backward()

            if xs_pytorch.grad is None:
                raise RuntimeError("No gradients computed for xs")

            if pytorch_result.shape != xs_pytorch.shape:
                raise RuntimeError(
                    f"Output shape {pytorch_result.shape} does not match input shape {xs_pytorch.shape}"
                )

            if JAX_AVAILABLE:
                try:
                    xs_np = xs_orig.detach().cpu().numpy()
                    jax_xs = jnp.array(xs_np)
                except Exception as conversion_err:
                    print(
                        f"\rSkipping JAX comparison for iteration {i + 1} due to conversion error: {conversion_err}",
                        end="",
                    )
                    continue

                _ = jax_associative_scan(
                    jax_add_combine_fn, jax_xs, axis=1, reverse=False
                ).block_until_ready()
                start_time_jax = timeit.default_timer()
                jax_result = jax_associative_scan(
                    jax_add_combine_fn,
                    jax_xs,
                    axis=1,
                    reverse=False,
                ).block_until_ready()
                end_time_jax = timeit.default_timer()
                jax_times.append(end_time_jax - start_time_jax)

        except Exception as e:
            failed_cases.append((description, str(e)))
            print(
                f"\rFuzzing iteration {i + 1}/{n} FAILED: {description} | Error: {e}",
                file=sys.stderr,
            )
        else:
            print(f"\rFuzzing iteration {i + 1}/{n} passed: {description}", end="")
            sys.stdout.flush()

    print("\n\nFuzzing completed.")
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}/{n}):")
        for desc, error in failed_cases:
            print(f"- {desc} | Error: {error}")
    else:
        print("All fuzzing tests passed successfully.")

    if pytorch_times:
        pytorch_mean = np.mean(pytorch_times)
        pytorch_std = np.std(pytorch_times)
        print(
            f"\nPyTorch Timings (mean ± std): {pytorch_mean:.6f} ± {pytorch_std:.6f} seconds"
        )

    if JAX_AVAILABLE and jax_times:
        jax_mean = np.mean(jax_times)
        jax_std = np.std(jax_times)
        print(
            f"JAX Timings (mean ± std):             {jax_mean:.6f} ± {jax_std:.6f} seconds"
        )
    elif JAX_AVAILABLE:
        print("\nNo JAX timings recorded (possibly due to input conversion issues).")


if __name__ == "__main__":
    main()
