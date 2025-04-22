import torch
from typing import Tuple, Callable, Any
import torch.utils._pytree as pytree


# Simplified Scan (Loop-based) - enough to show type promotion
def simple_associative_scan(combine_fn: Callable, xs: Tuple, dim: int = 0) -> Tuple:
    """
    Performs a sequential scan using the combine_fn.
    Mimics the core operation relevant to type promotion issues.
    Assumes input xs is a tuple of tensors for simplicity in MRE.
    """
    if not xs or not all(isinstance(t, torch.Tensor) for t in xs):
        raise ValueError("Expected xs to be a non-empty tuple of Tensors for this MRE.")

    # Assuming dim=0 for simplicity (scan along sequence length L)
    num_leaves = len(xs)
    seq_len = xs[0].shape[dim]

    # Get the first slice as the initial state
    current_state = tuple(leaf.select(dim, 0) for leaf in xs)
    results_acc = [[state_leaf.clone() for state_leaf in current_state]]  # Store results per step, clone initial

    for i in range(1, seq_len):
        next_element = tuple(leaf.select(dim, i) for leaf in xs)
        current_state = combine_fn(current_state, next_element)
        results_acc.append(list(current_state))  # Store intermediate states

    # Stack results along the scan dimension
    final_results = []
    for j in range(num_leaves):  # Iterate through elements of the state tuple (m, s, n, Z, g)
        # Stack the j-th element from each step's result list
        final_results.append(torch.stack([step_result[j] for step_result in results_acc], dim=dim))

    return tuple(final_results)


# Combine function from SpectralAttention
def combine_fn(x: Tuple, y: Tuple) -> Tuple:
    """Combine function causing the type promotion."""
    # Assumes x and y are tuples of tensors: (m, s, n, Z, g)
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y

    m = torch.maximum(m_x, m_y)

    # Use float32 for exp calculation for stability
    m_f = m.to(torch.float32)
    m_x_f = m_x.to(torch.float32)
    m_y_f = m_y.to(torch.float32)
    exp_x, exp_y = torch.exp(m_x_f - m_f), torch.exp(m_y_f - m_f)

    # Type promotion happens here: bfloat16 * float32 -> float32
    s = s_x * exp_x + s_y * exp_y
    # Ensure n calculation handles dimensions correctly (n is [h], exp is scalar)
    n = n_x * exp_x.unsqueeze(-1) + n_y * exp_y.unsqueeze(-1)

    Z = Z_x + Z_y
    g = g_x + g_y

    return m, s, n, Z, g


# Scan function from SpectralAttention (modified for metadata check)
def scan_fn(qk_slice: torch.Tensor, v_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor) -> Tuple:
    """Prepares leaves and calls the scan, then checks metadata."""

    print(f"\n--- Running scan_fn for slice ---")
    L, h = v_slice.shape
    print(f"Input shapes: qk=[{L}], v=[{L},{h}], Z=[{L},{h},{h}], g=[{L}]")
    print(f"Input dtypes: qk={qk_slice.dtype}, v={v_slice.dtype}, Z={Z_slice.dtype}, g={g_slice.dtype}")

    # Prepare initial leaves - FORCE BFLOAT16 for inputs that become s and n
    initial_m = qk_slice.to(torch.bfloat16)
    initial_s = torch.ones_like(qk_slice, dtype=torch.bfloat16)  # s starts as bfloat16
    initial_n = v_slice.to(torch.bfloat16)  # n starts as bfloat16
    initial_Z = Z_slice.to(torch.bfloat16)
    initial_g = g_slice.to(torch.bfloat16)

    leaves = (initial_m, initial_s, initial_n, initial_Z, initial_g)
    leaf_names = ["m (qk)", "s (ones)", "n (v)", "Z", "g"]

    print("\nInitial Leaves Metadata:")
    for i, leaf in enumerate(leaves):
        print(f"  Leaf {i} ({leaf_names[i]}): shape={leaf.shape}, dtype={leaf.dtype}, device={leaf.device}")

    # Use the simplified scan for MRE
    print("\nCalling simplified_associative_scan...")
    result = simple_associative_scan(combine_fn=combine_fn, xs=leaves, dim=0)  # Scan along L
    print("Scan finished.")

    print("\nResult Metadata (after scan):")
    mismatched = False
    for i, (res, initial_leaf) in enumerate(zip(result, leaves)):
        print(f"  Result {i} ({leaf_names[i]}): shape={res.shape}, dtype={res.dtype}, device={res.device}")
        if res.dtype != initial_leaf.dtype:
            print(f"    -> Dtype MISMATCH! Initial was {initial_leaf.dtype}")
            mismatched = True
        # Shape check (should match if scan is correct)
        if res.shape != initial_leaf.shape:
            print(f"    -> Shape MISMATCH! Initial was {initial_leaf.shape}")
            mismatched = True
        # Device check
        if res.device != initial_leaf.device:
            print(f"    -> Device MISMATCH! Initial was {initial_leaf.device}")
            mismatched = True

    if mismatched:
        print("\n*** Metadata mismatch detected! This would cause the error in the original kernel. ***")
    else:
        print("\nMetadata consistent.")

    # In the real kernel, an error would be raised here due to the mismatch.
    # We just print the warning above.

    return result


# Batched scan function (simplified for MRE)
def batched_scan_fn(sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor) -> Tuple:
    """Selects one slice and calls scan_fn for MRE."""
    # Select the first slice [0, 0, :, ...] for demonstration
    sim_slice = sim[0, 0]
    v_slice = v[0, 0]
    gated_Z_slice = gated_Z[0, 0]
    gates_z_slice = gates_z[0, 0]

    # Directly call the modified scan_fn for the single slice
    return scan_fn(sim_slice, v_slice, gated_Z_slice, gates_z_slice)


# Main execution for MRE
if __name__ == "__main__":
    B, H, L, h = 1, 1, 8, 4  # Small dimensions for MRE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("Warning: CUDA or bfloat16 not supported, using CPU and float32 for MRE.")
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        print("CUDA and bfloat16 supported, using CUDA and bfloat16 for MRE.")
        dtype = torch.bfloat16

    print(f"Using device: {device}")

    # Create dummy input tensors with the target dtype
    sim = torch.randn(B, H, L, device=device, dtype=dtype)
    v = torch.randn(B, H, L, h, device=device, dtype=dtype)
    gated_Z = torch.randn(B, H, L, h, h, device=device, dtype=dtype)
    gates_z = torch.randn(B, H, L, device=device, dtype=dtype)

    print("\nStarting MRE execution...")
    try:
        # Call the batched scan function
        final_result = batched_scan_fn(sim, v, gated_Z, gates_z)
        print("\nMRE finished.")

    except Exception as e:
        print(f"\n--- MRE terminated with unhandled exception ---")
        import traceback

        traceback.print_exc()
        print("-----------------------------------------------")
