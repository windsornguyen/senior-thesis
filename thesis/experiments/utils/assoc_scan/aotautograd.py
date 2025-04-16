# ===----------------------------------------------------------------------===##
#  File: grad_aot.py
#  Author(s): Windsor Nguyen, gemini-2.5-pro-exp-03-25
#  Description:
#    Uses AOTAutograd to capture the gradients of the backward pass.
# ===----------------------------------------------------------------------===##

import torch

# Try importing the default compiler (Inductor)
try:
    # Note: Adjust import path if necessary depending on PyTorch version
    from torch._inductor import compile as inductor_compile

    DEFAULT_COMPILER = inductor_compile
    print("Using Inductor as the default compiler backend.")
except ImportError:
    print(
        "Warning: Inductor compiler not found. Defaulting to no compiler (eager execution)."
    )
    DEFAULT_COMPILER = None

from torch._functorch.aot_autograd import AOTConfig, create_joint
from typing import Tuple, Callable, Any, Optional
import torch.utils._pytree as pytree  # For potential future generalization


# ===----------------------------------------------------------------------===##
#                              Function Definitions
# ===----------------------------------------------------------------------===##
def attention_combine_fn_torch(inputs):
    """PyTorch version of the attention combine function that takes a single tuple argument."""
    # Unpack the single input which contains both x and y
    x, y = inputs

    # Now unpack x and y
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y

    # Compute new maximum
    m_new = torch.maximum(m_x, m_y)

    # Scale factors
    exp_x = torch.exp(m_x - m_new)
    exp_y = torch.exp(m_y - m_new)

    # Update softmax components
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x.unsqueeze(-1) + n_y * exp_y.unsqueeze(-1)

    # Update gated Z and gate accumulation
    Z_new = Z_x + Z_y
    g_new = g_x + g_y

    return (m_new, s_new, n_new, Z_new, g_new)


# ===----------------------------------------------------------------------===##
#                              AOTAutograd Setup
# ===----------------------------------------------------------------------===##
def aot_wrapper(fn):
    """Wrapper that adapts the function interface for AOTAutograd."""

    # Note: Assumes fn takes two arguments based on attention_combine_fn_torch
    def wrapped_fn(x, y):
        return fn((x, y))

    return wrapped_fn


default_no_recompute_ops = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.c10d_functional.reduce_scatter_tensor.default,
    # torch.ops.xformers_flash.flash_fwd.default, # Commented out if xformers not guaranteed
    # torch.ops.xformers.efficient_attention_forward_cutlass.default,
}


def create_bw_fn(
    fn: Callable,
    args: Tuple[Any],
    fw_compiler: Optional[Callable] = DEFAULT_COMPILER,
    bw_compiler: Optional[Callable] = DEFAULT_COMPILER,
    partition_fn: Optional[Callable] = None,
    decompositions: Optional[dict] = None,
    aot_id: int = 0,
) -> Callable:
    r"""Create an optimized backward function for a given forward function using AOT Autograd.

    This function traces the forward and backward graphs of `fn` and uses the provided
    compilers (`fw_compiler`, `bw_compiler`) and partitioner (`partition_fn`) to
    potentially optimize the gradient computation. It returns a callable function
    that computes the gradients of the inputs to `fn`.

    Args:
        fn: The forward function for which to generate the backward pass. It should
            take PyTorch tensors or tuples/lists/structs thereof as input.
        args: Example arguments (matching the structure expected by `fn`) used for
              tracing the function graph. The shapes and dtypes should be representative.
        fw_compiler: The compiler backend to use for the forward graph fragment.
                     Defaults to `torch._inductor.compile` if available, otherwise None.
        bw_compiler: The compiler backend to use for the backward graph fragment.
                     Defaults to `torch._inductor.compile` if available, otherwise None.
        partition_fn: An optional function to control activation recomputation (checkpointing).
                      If None (default), standard partitioning is used.
        decompositions: An optional dictionary of decompositions to apply during tracing.
        aot_id: An identifier for the AOT compilation instance.

    Returns:
        A callable function `backward_callable(*primals, *tangents)` that takes the
        original primal inputs (*primals, matching the structure of `args`) and
        the output gradients (*tangents, matching the structure of `fn`'s outputs)
        and returns a list of gradients corresponding to the primal inputs.

    """
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions if decompositions is not None else {},
        num_params_buffers=0,  # Assuming no parameters/buffers in fn
        aot_id=aot_id,
        keep_inference_input_mutations=False,
        # no_tangents=False, # Might be needed depending on usage
    )

    print(f"Creating AOT backward function (id={aot_id}):")
    print(f"  fw_compiler: {fw_compiler.__name__ if fw_compiler else 'None'}")
    print(f"  bw_compiler: {bw_compiler.__name__ if bw_compiler else 'None'}")
    print(f"  partition_fn: {partition_fn.__name__ if partition_fn else 'None'}")

    # --- Start: Core logic from the working version ---
    # Create a wrapped version for AOTAutograd that expects structured (x, y)
    wrapped_fn = aot_wrapper(fn)

    # Function that accepts flat primals, reconstructs x, y, and calls wrapped_fn
    # This function will be traced by create_joint
    def reconstruct_and_call_fn(*flat_primals):
        # Hardcoded based on attention_combine_fn_torch expecting (x, y) where x, y are 5-tuples
        # TODO: Generalize reconstruction based on pytree structure of `args`
        expected_flat_len = 10
        if len(flat_primals) != expected_flat_len:
            raise ValueError(
                f"[reconstruct_and_call_fn] Expected {expected_flat_len} flat primals based on hardcoded structure, got {len(flat_primals)}."
            )
        num_x_elements = 5
        x_recon = flat_primals[:num_x_elements]
        y_recon = flat_primals[num_x_elements:]

        # Call the original wrapped function
        outputs = wrapped_fn(x_recon, y_recon)

        # create_joint expects the function to return (outputs, tangent_mask)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        tangent_mask = [True] * len(outputs)  # Assume all outputs need tangents
        return outputs, tangent_mask

    # Create joint forward-backward function using the config and reconstructing fn
    try:
        joint_fn = create_joint(reconstruct_and_call_fn, aot_config=aot_config)
    except Exception as e:
        print(f"Error during create_joint: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise the exception after printing details

    # Function returned to the user, handles the actual backward call
    def flat_fn(*args_and_grad_outs):
        # Assumes args_and_grad_outs starts with the primals matching the structure of `args` used in create_bw_fn
        num_original_args = len(args)  # e.g., 2 for (x, y)
        primals_in = args_and_grad_outs[:num_original_args]
        tangents = args_and_grad_outs[num_original_args:]

        # Flatten the runtime primals based on the structure of `args`
        # Hardcoded based on attention_combine_fn_torch expecting (x, y) where x, y are tuples
        # TODO: Generalize flattening based on pytree structure of `args`
        flat_primals = []
        for arg in primals_in:
            if isinstance(arg, (tuple, list)):
                flat_primals.extend(arg)
            else:
                # This case assumes a single tensor argument, adjust if structure varies
                flat_primals.append(arg)

        # Call the AOT-generated joint function
        try:
            _fw_outputs, grad_args = joint_fn(flat_primals, tangents)
        except Exception as e:
            print(f"Error during compiled joint_fn execution: {e}")
            print(
                f"  flat_primals ({len(flat_primals)}): {[type(p).__name__ for p in flat_primals]}"
            )
            print(
                f"  tangents ({len(tangents)}): {[type(t).__name__ for t in tangents]}"
            )
            import traceback

            traceback.print_exc()
            raise

        num_expected_grads = len(flat_primals)
        if len(grad_args) != num_expected_grads:
            # This might happen legitimately if some inputs don't require grad
            print(
                f"Info: Joint function returned {len(grad_args)} gradients, expected {num_expected_grads} based on flattened primals."
            )

        # print(f"Joint function returned {len(grad_args)} gradient arguments") # Optional debug print
        return grad_args

    # --- End: Core logic from the working version ---

    return flat_fn


# ===----------------------------------------------------------------------===##
#                              Utility Functions
# ===----------------------------------------------------------------------===##
def first_slice_copy(tensor):
    """Create a first-slice copy of a tensor with gradient information preserved."""
    if isinstance(tensor, torch.Tensor):
        # Ensure requires_grad status is copied
        return tensor[0:1].clone().detach().requires_grad_(tensor.requires_grad)
    return tensor


# ===----------------------------------------------------------------------===##
#                              Comparison Function
# ===----------------------------------------------------------------------===##
def compare_aot_with_autograd(aot_grads, standard_grads, names):
    """Compare AOTAutograd gradients with standard autograd gradients."""
    print("\n----- Comparing AOTAutograd and Standard Autograd -----")
    if len(aot_grads) != len(standard_grads):
        print(
            f"❌ Gradient list length mismatch: AOT ({len(aot_grads)}) vs Standard ({len(standard_grads)})"
        )
        return
    if len(aot_grads) != len(names):
        print(
            f"❌ Name list length ({len(names)}) mismatch with gradient list length ({len(aot_grads)})"
        )
        names = [f"grad_{i}" for i in range(len(aot_grads))]  # Fallback names

    all_match = True
    for i, (aot_grad, std_grad, name) in enumerate(
        zip(aot_grads, standard_grads, names)
    ):
        match = False
        if aot_grad is None and std_grad is None:
            print(f"✅ {name}: Both gradients are None")
            match = True
        elif aot_grad is None or std_grad is None:
            print(
                f"❌ {name}: One gradient is None (AOT: {aot_grad is None}, Standard: {std_grad is None}) - MISMATCH"
            )
        elif aot_grad.shape != std_grad.shape:
            print(
                f"❌ {name} shape mismatch: AOT {aot_grad.shape} vs Standard {std_grad.shape}"
            )
        else:
            try:
                # Use appropriate tolerances for comparison
                torch.testing.assert_close(aot_grad, std_grad, rtol=1e-5, atol=1e-5)
                print(f"✅ {name} MATCHES")
                match = True
            except AssertionError as e:  # Catch specific assertion error
                print(f"❌ {name} MISMATCH: {e}")
                # Print some stats for debugging
                print(
                    f"   AOT mean: {aot_grad.mean().item():.6f}, std: {aot_grad.std().item():.6f}"
                )
                print(
                    f"   STD mean: {std_grad.mean().item():.6f}, std: {std_grad.std().item():.6f}"
                )
        if not match:
            all_match = False

    print("----- Comparison Summary -----")
    if all_match:
        print("✅ All gradients match!")
    else:
        print("❌ Some gradients mismatched.")


# ===----------------------------------------------------------------------===##
#                              Main Execution
# ===----------------------------------------------------------------------===##
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create sample tensors with gradients enabled
    # Use CUDA if available for more realistic performance comparison
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16  # Slightly larger batch
    seq_len = 64
    hidden_dim = 128

    # Define inputs x and y as tuples of tensors
    m_x = torch.randn(batch_size, seq_len, requires_grad=True, device=device)
    s_x = torch.randn(batch_size, seq_len, requires_grad=True, device=device)
    n_x = torch.randn(
        batch_size, seq_len, hidden_dim, requires_grad=True, device=device
    )
    Z_x = torch.randn(
        batch_size, seq_len, hidden_dim, requires_grad=True, device=device
    )
    g_x = torch.randn(batch_size, seq_len, requires_grad=True, device=device)
    x = (m_x, s_x, n_x, Z_x, g_x)

    m_y = torch.randn(batch_size, seq_len, requires_grad=True, device=device)
    s_y = torch.randn(batch_size, seq_len, requires_grad=True, device=device)
    n_y = torch.randn(
        batch_size, seq_len, hidden_dim, requires_grad=True, device=device
    )
    Z_y = torch.randn(
        batch_size, seq_len, hidden_dim, requires_grad=True, device=device
    )
    g_y = torch.randn(batch_size, seq_len, requires_grad=True, device=device)
    y = (m_y, s_y, n_y, Z_y, g_y)

    example_args = (x, y)  # Define example args structure used for tracing

    # --- 1. AOTAutograd ---
    print("\n----- AOTAutograd Test (Default Compiler: Inductor) -----")

    # Create the backward function using the full tensors and default compiler
    # This triggers compilation if DEFAULT_COMPILER is set
    bw_fn_compiled = create_bw_fn(attention_combine_fn_torch, example_args)

    # Forward pass with full data to get outputs for grad_outputs
    result_full = attention_combine_fn_torch(example_args)

    # Create dummy gradient outputs matching the full output shapes
    grad_outputs = tuple(torch.ones_like(t, device=device) for t in result_full)

    # --- Warmup Run (Important for Compiled Code Timing) ---
    print("Warmup run for compiled backward function...")
    aot_grads = []  # Initialize as list to allow timed run check
    try:
        # Call with the original example_args structure
        _ = bw_fn_compiled(*example_args, *grad_outputs)
        print("Warmup complete.")
    except Exception as e:
        print(f"Error during warmup: {e}")
        # Don't run timed section if warmup fails
        aot_grads = None  # Set to None to skip timing and comparison
        # Decide if error is critical

    # --- Timed Run ---
    aot_time = float("inf")
    if aot_grads is not None:  # Only run if warmup succeeded
        print("Applying compiled AOT backward function...")
        try:
            start_time = None
            end_time = None
            if device.type == "cuda":
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()

            # Call the backward function with full x and y
            aot_grads = bw_fn_compiled(*example_args, *grad_outputs)

            if device.type == "cuda":
                end_time.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded!
                aot_time = start_time.elapsed_time(end_time) / 1000.0  # Time in seconds
                print(f"Compiled AOTAutograd backward took: {aot_time:.6f} seconds")
            else:
                # Basic timing for CPU (less precise)
                import time

                start = time.perf_counter()
                aot_grads = bw_fn_compiled(*example_args, *grad_outputs)
                end = time.perf_counter()
                aot_time = end - start
                print(
                    f"Compiled AOTAutograd backward took (CPU time): {aot_time:.6f} seconds"
                )

            print(
                f"AOTAutograd produced {len(aot_grads)} gradient arguments"
            )  # Changed print message slightly

        except Exception as e:
            print(f"Error in compiled AOTAutograd run: {e}")
            import traceback

            traceback.print_exc()
            print("Skipping comparisons due to compiled AOTAutograd error")
            aot_grads = None
            aot_time = float("inf")

    # --- 2. Standard Autograd ---
    print("\n----- Standard Autograd Test -----")

    # Reset gradients on original tensors
    # Need to iterate through the actual tensors in the tuples
    for arg_tuple in example_args:
        for tensor in arg_tuple:
            if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                tensor.grad.zero_()

    # --- Warmup ---
    print("Warmup run for standard autograd...")
    std_grads = []  # Initialize std_grads here, will be overwritten if successful
    try:
        result_std_warmup = attention_combine_fn_torch(example_args)
        loss_std_warmup = sum(r.sum() for r in result_std_warmup)
        loss_std_warmup.backward()
        # Reset grads after warmup backward
        for arg_tuple in example_args:
            for tensor in arg_tuple:
                if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                    tensor.grad.zero_()
        print("Warmup complete.")
    except Exception as e:
        print(f"Error during standard autograd warmup: {e}")
        std_grads = None  # Ensure comparison is skipped if warmup fails

    # --- Timed Run ---
    std_time = float("inf")
    # Only run timed section if warmup didn't set std_grads to None
    if std_grads is not None:
        print("Performing standard forward and backward pass...")
        try:
            start_time_std, end_time_std = None, None
            if device.type == "cuda":
                start_time_std = torch.cuda.Event(enable_timing=True)
                end_time_std = torch.cuda.Event(enable_timing=True)
                start_time_std.record()

            result_std = attention_combine_fn_torch(example_args)
            loss_std = sum(r.sum() for r in result_std)
            loss_std.backward()

            if device.type == "cuda":
                end_time_std.record()
                torch.cuda.synchronize()
                std_time = start_time_std.elapsed_time(end_time_std) / 1000.0
                print(f"Standard Autograd backward took: {std_time:.6f} seconds")
            else:
                import time

                # Need to re-run for CPU timing as backward() modifies grads in place
                # Reset grads first
                for arg_tuple in example_args:
                    for tensor in arg_tuple:
                        if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                            tensor.grad.zero_()
                start_std = time.perf_counter()
                # Re-run forward and backward for timing
                result_std = attention_combine_fn_torch(example_args)
                loss_std = sum(r.sum() for r in result_std)
                loss_std.backward()
                end_std = time.perf_counter()
                std_time = end_std - start_std
                print(
                    f"Standard Autograd backward took (CPU time): {std_time:.6f} seconds"
                )

            # Collect standard autograd gradients AFTER timing
            std_grads = [
                t.grad.clone() if t.grad is not None else None
                for arg_tuple in example_args
                for t in arg_tuple
            ]
            print("Standard autograd backward pass complete.")

        except Exception as e:
            print(f"Error during standard autograd run: {e}")
            import traceback

            traceback.print_exc()
            std_grads = None  # Ensure comparison is skipped

    # --- 3. Comparison ---
    if aot_grads is not None and std_grads is not None:
        # Prepare the tensor names for comparison output
        tensor_names = [
            "m_x",
            "s_x",
            "n_x",
            "Z_x",
            "g_x",
            "m_y",
            "s_y",
            "n_y",
            "Z_y",
            "g_y",
        ]
        compare_aot_with_autograd(aot_grads, std_grads, tensor_names)
        if (
            aot_time != float("inf") and std_time != float("inf") and aot_time > 1e-9
        ):  # Avoid division by zero
            print(f"\nSpeedup Factor (Standard / AOT): {std_time / aot_time:.2f}x")
        else:
            print("\nCould not compute speedup factor (timing error or zero AOT time).")


if __name__ == "__main__":
    main()
