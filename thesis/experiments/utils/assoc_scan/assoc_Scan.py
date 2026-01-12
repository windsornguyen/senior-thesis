import torch
import triton
import triton.language as tl

# Experiment 1: Basic Single Tensor Scan


@triton.jit
def combine_add(left: tl.tensor, right: tl.tensor) -> tl.tensor:
    """Simple addition combine function."""
    return left + right


@triton.jit
def kernel_exp1(input_ptr, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel for basic single tensor scan."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)  # Use 0.0 as identity for addition

    # Perform associative scan
    # Input: single tensor `x`
    # Combine function: `combine_add` expects 2 args (left, right)
    y = tl.associative_scan(x, axis=0, combine_fn=combine_add)

    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)


def run_experiment1():
    print("--- Experiment 1: Basic Single Tensor Scan ---")
    N = 10
    BLOCK_SIZE = 16  # Ensure BLOCK_SIZE >= N for simplicity
    x_torch = torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    y_torch = torch.empty_like(x_torch)

    print(f"Input Tensor: {x_torch.cpu().numpy()}")

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel_exp1[grid](x_torch, y_torch, N=N, BLOCK_SIZE=BLOCK_SIZE)

    expected_result = torch.cumsum(x_torch, dim=0)
    print(f"Triton Output: {y_torch.cpu().numpy()}")
    print(f"Expected Output (cumsum): {expected_result.cpu().numpy()}")
    assert torch.allclose(y_torch, expected_result), "Experiment 1 Failed!"
    print("Experiment 1 Passed!\n")


# Experiment 2: Tuple of Tensors Scan


@triton.jit
def combine_tuple_add(a_left, b_left, a_right, b_right):
    """Combine function expecting unpacked elements from two tuples.
    ((a_left, b_left), (a_right, b_right))
    """
    a_new = a_left + a_right
    b_new = b_left + b_right
    return a_new, b_new


@triton.jit
def kernel_exp2(a_in_ptr, b_in_ptr, a_out_ptr, b_out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel for tuple tensor scan."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data for two tensors
    a_in = tl.load(a_in_ptr + offsets, mask=mask, other=0.0)
    b_in = tl.load(b_in_ptr + offsets, mask=mask, other=0.0)

    # Perform associative scan on a tuple
    # Input: tuple (a_in, b_in)
    # Combine function: `combine_tuple_add` expects 4 args (a_left, b_left, a_right, b_right)
    a_out, b_out = tl.associative_scan((a_in, b_in), axis=0, combine_fn=combine_tuple_add)

    # Store results
    tl.store(a_out_ptr + offsets, a_out, mask=mask)
    tl.store(b_out_ptr + offsets, b_out, mask=mask)


def run_experiment2():
    print("--- Experiment 2: Tuple of Tensors Scan ---")
    N = 10
    BLOCK_SIZE = 16
    a_in_torch = torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    b_in_torch = torch.arange(101, 101 + N, dtype=torch.float32, device="cuda")
    a_out_torch = torch.empty_like(a_in_torch)
    b_out_torch = torch.empty_like(b_in_torch)

    print(f"Input Tensor A: {a_in_torch.cpu().numpy()}")
    print(f"Input Tensor B: {b_in_torch.cpu().numpy()}")

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel_exp2[grid](a_in_torch, b_in_torch, a_out_torch, b_out_torch, N=N, BLOCK_SIZE=BLOCK_SIZE)

    expected_a = torch.cumsum(a_in_torch, dim=0)
    expected_b = torch.cumsum(b_in_torch, dim=0)
    print(f"Triton Output A: {a_out_torch.cpu().numpy()}")
    print(f"Triton Output B: {b_out_torch.cpu().numpy()}")
    print(f"Expected Output A: {expected_a.cpu().numpy()}")
    print(f"Expected Output B: {expected_b.cpu().numpy()}")
    assert torch.allclose(a_out_torch, expected_a), "Experiment 2 Failed (Tensor A)!"
    assert torch.allclose(b_out_torch, expected_b), "Experiment 2 Failed (Tensor B)!"
    print("Experiment 2 Passed!\n")


# Experiment 3: Tuple Scan with Signature Mismatch (Error Check)


@triton.jit
def combine_tuple_wrong_sig(left_state, right_state):
    """Combine function incorrectly expecting two tuples."""
    # This will likely cause a TypeError because scan passes unpacked elements
    a_left, b_left = left_state
    a_right, b_right = right_state
    a_new = a_left + a_right
    b_new = b_left + b_right
    return a_new, b_new


@triton.jit
def kernel_exp3(a_in_ptr, b_in_ptr, a_out_ptr, b_out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel designed to trigger TypeError due to combine_fn signature."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a_in = tl.load(a_in_ptr + offsets, mask=mask, other=0.0)
    b_in = tl.load(b_in_ptr + offsets, mask=mask, other=0.0)

    # This call should fail during compilation or runtime
    scan_results = tl.associative_scan((a_in, b_in), axis=0, combine_fn=combine_tuple_wrong_sig)

    a_out, b_out = scan_results
    tl.store(a_out_ptr + offsets, a_out, mask=mask)
    tl.store(b_out_ptr + offsets, b_out, mask=mask)


def run_experiment3():
    print("--- Experiment 3: Tuple Scan with Signature Mismatch ---")
    N = 10
    BLOCK_SIZE = 16
    a_in_torch = torch.ones(N, dtype=torch.float32, device="cuda")  # Simple inputs
    b_in_torch = torch.ones(N, dtype=torch.float32, device="cuda")
    a_out_torch = torch.empty_like(a_in_torch)
    b_out_torch = torch.empty_like(b_in_torch)

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    try:
        kernel_exp3[grid](a_in_torch, b_in_torch, a_out_torch, b_out_torch, N=N, BLOCK_SIZE=BLOCK_SIZE)
        print("Experiment 3 UNEXPECTEDLY Passed (Should have raised TypeError)!\n")
        raise AssertionError("Experiment 3 did not raise TypeError as expected.")
    except TypeError as e:
        print(f"Experiment 3 Correctly raised TypeError: {e}")
        # Check if the error message is as expected (optional but good)
        if "takes 2 positional arguments but 4 were given" in str(e):
            print("Error message matches expected pattern.")
        else:
            print(f"Error message differs from expected: {str(e)}")
        print("Experiment 3 Passed (Error Check)!\n")
    except Exception as e:
        print(f"Experiment 3 raised an unexpected error: {type(e).__name__}: {e}\n")
        raise e


# Experiment 4: Tuple Scan with `constexpr` (Implicit Passing Check)


@triton.jit
def combine_tuple_with_constexpr(a_left, b_left, a_right, b_right, factor: tl.constexpr):
    """Combine function expecting unpacked elements and an implicit constexpr."""
    a_new = (a_left + a_right) * factor
    b_new = (b_left + b_right) * factor
    return a_new, b_new


@triton.jit
def kernel_exp4(a_in_ptr, b_in_ptr, a_out_ptr, b_out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel for tuple scan testing implicit constexpr passing."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Define constexpr within kernel scope
    FACTOR: tl.constexpr = 2

    a_in = tl.load(a_in_ptr + offsets, mask=mask, other=0.0)
    b_in = tl.load(b_in_ptr + offsets, mask=mask, other=0.0)

    # Call scan passing only the state tuple (a_in, b_in)
    # Hypothesis: FACTOR might be implicitly passed to combine_fn
    scan_results = tl.associative_scan(
        (a_in, b_in),
        axis=0,
        combine_fn=combine_tuple_with_constexpr,  # Expects 5 args (4 state + 1 constexpr)
    )

    a_out, b_out = scan_results
    tl.store(a_out_ptr + offsets, a_out, mask=mask)
    tl.store(b_out_ptr + offsets, b_out, mask=mask)


def run_experiment4():
    print("--- Experiment 4: Tuple Scan with `constexpr` ---")
    N = 10
    BLOCK_SIZE = 16
    a_in_torch = torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    b_in_torch = torch.arange(101, 101 + N, dtype=torch.float32, device="cuda")
    a_out_torch = torch.empty_like(a_in_torch)
    b_out_torch = torch.empty_like(b_in_torch)

    print(f"Input Tensor A: {a_in_torch.cpu().numpy()}")
    print(f"Input Tensor B: {b_in_torch.cpu().numpy()}")

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    passed = False
    try:
        # If this works, constexpr was passed implicitly
        kernel_exp4[grid](a_in_torch, b_in_torch, a_out_torch, b_out_torch, N=N, BLOCK_SIZE=BLOCK_SIZE)

        # Verify results (should be cumulative sum * FACTOR)
        factor = 2
        expected_a = torch.cumsum(a_in_torch, dim=0) * factor
        expected_b = torch.cumsum(b_in_torch, dim=0) * factor
        print(f"Triton Output A: {a_out_torch.cpu().numpy()}")
        print(f"Triton Output B: {b_out_torch.cpu().numpy()}")
        print(f"Expected Output A (cumsum*2): {expected_a.cpu().numpy()}")
        print(f"Expected Output B (cumsum*2): {expected_b.cpu().numpy()}")
        assert torch.allclose(a_out_torch, expected_a, atol=1e-6), "Experiment 4 Failed (Tensor A)!"
        assert torch.allclose(b_out_torch, expected_b, atol=1e-6), "Experiment 4 Failed (Tensor B)!"
        print("Experiment 4 Passed! (`constexpr` was likely passed implicitly)\n")
        passed = True
    except TypeError as e:
        print(f"Experiment 4 raised TypeError: {e}")
        print("This likely means `constexpr` was NOT passed implicitly.")
        print("Experiment 4 Conclusion: Implicit constexpr passing FAILED.\n")
        # Check if the error indicates the wrong number of arguments
        if "takes 5 positional arguments but 4 were given" in str(e):
            print("(Error message confirms missing argument as expected)")
        passed = True  # Test still 'passes' in the sense that it gave expected outcome
    except Exception as e:
        print(f"Experiment 4 raised an unexpected error: {type(e).__name__}: {e}\n")
        raise e
    # assert passed, "Experiment 4 check failed unexpectedly."


# Experiment 5: Silent Signature Mismatch Check


@triton.jit
def combine_fn_extra_args(a_left, b_left, a_right, b_right, extra_arg):
    """Combine function expecting 4 state args + 1 extra arg."""
    # If extra_arg is silently defaulted (e.g., to 0), the result might seem correct.
    # If it's ignored, this function might even compile.
    a_new = a_left + a_right + extra_arg  # Use the extra arg
    b_new = b_left + b_right + extra_arg  # Use the extra arg
    return a_new, b_new


@triton.jit
def kernel_exp5(a_in_ptr, b_in_ptr, a_out_ptr, b_out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel testing combine_fn with more args than scan provides."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a_in = tl.load(a_in_ptr + offsets, mask=mask, other=0.0)
    b_in = tl.load(b_in_ptr + offsets, mask=mask, other=0.0)

    # Scan provides 4 args based on (a_in, b_in)
    # Combine function expects 5 args.
    # EXPECTATION: Should raise TypeError.
    # TEST: Does it run silently?
    scan_results = tl.associative_scan((a_in, b_in), axis=0, combine_fn=combine_fn_extra_args)

    a_out, b_out = scan_results
    tl.store(a_out_ptr + offsets, a_out, mask=mask)
    tl.store(b_out_ptr + offsets, b_out, mask=mask)


def run_experiment5():
    print("--- Experiment 5: Silent Signature Mismatch Check ---")
    N = 10
    BLOCK_SIZE = 16
    a_in_torch = torch.ones(N, dtype=torch.float32, device="cuda")
    b_in_torch = torch.ones(N, dtype=torch.float32, device="cuda")
    a_out_torch = torch.empty_like(a_in_torch)
    b_out_torch = torch.empty_like(b_in_torch)

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    raised_correct_error = False
    ran_silently = False
    try:
        kernel_exp5[grid](a_in_torch, b_in_torch, a_out_torch, b_out_torch, N=N, BLOCK_SIZE=BLOCK_SIZE)
        # If we reach here, it ran without error!
        ran_silently = True
        print("!!! Experiment 5 Kernel RAN SILENTLY without expected TypeError !!!")

        # Check the output. Expected basic cumsum (if extra_arg was ignored or 0?)
        expected_a = torch.cumsum(a_in_torch, dim=0)
        expected_b = torch.cumsum(b_in_torch, dim=0)
        print(f"Triton Output A: {a_out_torch.cpu().numpy()}")
        print(f"Triton Output B: {b_out_torch.cpu().numpy()}")
        print(f"Expected Output A (basic cumsum): {expected_a.cpu().numpy()}")
        print(f"Expected Output B (basic cumsum): {expected_b.cpu().numpy()}")

        if torch.allclose(a_out_torch, expected_a) and torch.allclose(b_out_torch, expected_b):
            print("Output matches basic cumsum - extra arg might have been ignored or defaulted to 0.")
        else:
            print("Output DOES NOT match basic cumsum - behavior is unexpected.")
        print("Experiment 5 Conclusion: Potential SILENT FAILURE mode detected.\n")

    except TypeError as e:
        print(f"Experiment 5 Correctly raised TypeError: {e}")
        if "takes 5 positional arguments but 4 were given" in str(e):
            print("Error message confirms expected signature mismatch.")
            raised_correct_error = True
        else:
            print(f"Error message differs from expected: {str(e)}")
        print("Experiment 5 Conclusion: Scan correctly enforces signature (No silent failure).\n")
    except Exception as e:
        print(f"Experiment 5 raised an unexpected error: {type(e).__name__}: {e}\n")
        # raise e # Comment out to allow other tests to run

    # assert not ran_silently, "Experiment 5 ran silently when TypeError was expected!"
    # assert raised_correct_error, "Experiment 5 did not raise the correct TypeError."


# Experiment 6: Testing "Combined State" Scan Pattern


@triton.jit
def combine_exp6(b_l, a_l, zb_l, adup_l, b_r, a_r, zb_r, adup_r):
    """Combines the 8 unpacked args corresponding to the 4-element input tuple.
    Input tuple: (b, a, zero_b, a_dup)
    Args received: (b_l, a_l, zb_l, adup_l, b_r, a_r, zb_r, adup_r)
    """
    b_new = b_l + b_r
    a_new = a_l + a_r
    zb_new = zb_l + zb_r  # Should remain zero
    adup_new = adup_l + adup_r  # Should match a_new
    return b_new, a_new, zb_new, adup_new


@triton.jit
def kernel_exp6(
    a_in_ptr, b_in_ptr, a_out_ptr, b_out_ptr, zb_out_ptr, adup_out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Kernel testing the combined state scan pattern."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load 'value' and 'gradient'
    a_in = tl.load(a_in_ptr + offsets, mask=mask, other=0.0)
    b_in = tl.load(b_in_ptr + offsets, mask=mask, other=0.0)

    # Create the 'zero gradient' part
    zb_in = tl.zeros_like(b_in)

    # Perform scan and directly unpack the 4 expected results
    b_out, a_out, zb_out, adup_out = tl.associative_scan(
        (b_in, a_in, zb_in, a_in),  # (grad, val, zero_grad, val_dup)
        axis=0,
        combine_fn=combine_exp6,  # Expects 8 args
    )

    # Store results
    tl.store(b_out_ptr + offsets, b_out, mask=mask)
    tl.store(a_out_ptr + offsets, a_out, mask=mask)
    tl.store(zb_out_ptr + offsets, zb_out, mask=mask)
    tl.store(adup_out_ptr + offsets, adup_out, mask=mask)


def run_experiment6():
    print("--- Experiment 6: Testing Combined State Scan Pattern ---")
    N = 10
    BLOCK_SIZE = 16
    a_in_torch = torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    b_in_torch = torch.arange(101, 101 + N, dtype=torch.float32, device="cuda")
    a_out_torch = torch.empty_like(a_in_torch)
    b_out_torch = torch.empty_like(b_in_torch)
    zb_out_torch = torch.empty_like(b_in_torch)  # For the zero grad output
    adup_out_torch = torch.empty_like(a_in_torch)  # For the duplicated value output

    print(f"Input Tensor A (Value): {a_in_torch.cpu().numpy()}")
    print(f"Input Tensor B (Gradient): {b_in_torch.cpu().numpy()}")

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    try:
        kernel_exp6[grid](
            a_in_torch, b_in_torch, a_out_torch, b_out_torch, zb_out_torch, adup_out_torch, N=N, BLOCK_SIZE=BLOCK_SIZE
        )

        expected_a = torch.cumsum(a_in_torch, dim=0)
        expected_b = torch.cumsum(b_in_torch, dim=0)
        expected_zb = torch.zeros_like(b_in_torch)

        print(f"Triton Output B (Grad): {b_out_torch.cpu().numpy()}")
        print(f"Triton Output A (Value): {a_out_torch.cpu().numpy()}")
        print(f"Triton Output ZB (ZeroGrad): {zb_out_torch.cpu().numpy()}")
        print(f"Triton Output ADUP (ValueDup): {adup_out_torch.cpu().numpy()}")

        print(f"Expected Output B (Grad): {expected_b.cpu().numpy()}")
        print(f"Expected Output A (Value): {expected_a.cpu().numpy()}")
        print(f"Expected Output ZB (ZeroGrad): {expected_zb.cpu().numpy()}")
        print(f"Expected Output ADUP (ValueDup): {expected_a.cpu().numpy()}")

        assert torch.allclose(b_out_torch, expected_b), "Experiment 6 Failed (Gradient B)!"
        assert torch.allclose(a_out_torch, expected_a), "Experiment 6 Failed (Value A)!"
        assert torch.allclose(zb_out_torch, expected_zb), "Experiment 6 Failed (Zero Grad ZB)!"
        assert torch.allclose(adup_out_torch, expected_a), "Experiment 6 Failed (Value Dup ADUP)!"
        print("Experiment 6 Passed! Scan handled the 4-element tuple correctly.\n")

    except Exception as e:
        print(f"Experiment 6 Failed with error: {type(e).__name__}: {e}\n")
        # raise e


if __name__ == "__main__":
    # run_experiment1()
    # run_experiment2()
    # run_experiment3()
    # run_experiment4()
    # run_experiment5()
    run_experiment6()
