# Create file: thesis/experiments/synthetics/data_debug.py

import numpy as np
import typing as tp
from functools import partial

# === Constants ===
IGNORE_IDX = -100
L = 128  # Example Sequence Length (must be even for the function)
V = 256  # Example Vocab Size
COPY_PREFIX = V - 1 # Based on author's code non_special_vocab_size logic
SEED = 1746

# === Helper Function (from author's code context) ===
def exists(val):
    return val is not None

# === Author's Data Generation Function ===
def generate_in_context_recall_instance(
    vocab_size: int = V,
    seq_len: int = L,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    multi_query: bool = False,
    noise_vocab_size: int = 16, # Example value
    frac_noise: float = 0.,     # Example value
    *args, **kwargs
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Generate an instance of the in-context recall task. (Author's version)
    """
    if not exists(rng):
        rng = np.random.default_rng()

    # generate keys and values:
    # copy_prefix = vocab_size - 1 # Defined globally now
    non_special_vocab_size = vocab_size - 1 if not multi_query else vocab_size
    non_special_vocab_size -= noise_vocab_size
    key_vocab = np.arange(non_special_vocab_size//2)
    value_vocab = np.arange(non_special_vocab_size//2, non_special_vocab_size)

    # generate noise vocab:
    assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)

    # generate inputs/targets:
    kv_map = {}
    inputs, targets = [], []
    keys_presented = {}
    kv_motif_size = 2
    assert seq_len % kv_motif_size == 0, "seq_len must be an even number"
    num_kv_pairs = seq_len // kv_motif_size
    not_noise_idx = rng.choice(num_kv_pairs) # make sure we have at least one key-value pair in the sequence
    for i in range(num_kv_pairs-1): # subtract one to account for final key-value pair

        # determine if noise or kv pair collected:
        is_noise = rng.random()<frac_noise if i!=not_noise_idx and frac_noise>0 else False

        # collect noise:
        if is_noise:
            noise = rng.choice(noise_vocab, size=kv_motif_size, replace=True)
            inputs += list(noise)
            targets += [target_ignore_idx]*kv_motif_size

        # collect kv pair:
        else:
            # key:
            k = rng.choice(key_vocab)
            # value:
            if k not in kv_map:
                v = rng.choice(value_vocab)
                kv_map[k] = v
            else:
                v = kv_map[k]

            inputs.append(k)
            inputs.append(v)

            targets.append(target_ignore_idx)
            if k not in keys_presented:
                targets.append(target_ignore_idx)
            else:
                if multi_query:
                    targets.append(v) # probe value if key has been presented before
                else:
                    targets.append(target_ignore_idx)

            keys_presented[k] = v

    # add a final key-value pair to the sequence as well as a copy-prefix:
    k_probe = rng.choice(list(keys_presented.keys()))
    v_probe = keys_presented[k_probe]

    if not multi_query:
        inputs.append(COPY_PREFIX) # Usually vocab_size - 1
    inputs.append(k_probe)
    inputs.append(v_probe)

    if not multi_query:
        targets.append(target_ignore_idx) # copy prefix target
    targets.append(target_ignore_idx) # k_probe target
    targets.append(v_probe)           # v_probe target

    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    # Final slicing based on training/evaluation mode
    if is_training:
        # autoregressive shift
        return inputs[:-1], inputs[1:] # use shifted inputs as targets for training
    else:
        # Return inputs shifted, and targets shifted
        return inputs[:-1], targets[1:]


# === NumPy Accuracy Calculation Logic ===
def numpy_compute_accuracy(predictions: np.ndarray, targets: np.ndarray, ignore_index: int = IGNORE_IDX) -> tp.Tuple[float, np.ndarray, int, int]:
    """Computes accuracy over a single sequence, ignoring specified index."""
    # Assumes predictions and targets are 1D arrays of the same length
    if predictions.shape != targets.shape:
        print(f"[Error] Shape mismatch: preds {predictions.shape}, targets {targets.shape}")
        return 0.0, np.array([]), 0, 0

    valid_mask = targets != ignore_index
    correct = (predictions == targets) & valid_mask

    num_correct = np.sum(correct)
    num_valid = np.sum(valid_mask)

    accuracy = num_correct / np.maximum(num_valid, 1e-9)
    return accuracy, valid_mask, num_correct, num_valid


# === Main Debugging Block ===
if __name__ == "__main__":
    print("--- ICR Data Generation & Evaluation Debug ---")
    rng = np.random.default_rng(SEED)
    num_examples_to_check = 5

    for i in range(num_examples_to_check):
        print(f"\n--- Example {i+1} ---")

        # --- Step 1: Generate Raw Data (Evaluation Mode) ---
        inputs_raw, targets_raw = generate_in_context_recall_instance(
            is_training=False, rng=rng, multi_query=False
        )
        print(f"Raw inputs shape : {inputs_raw.shape}, Raw targets shape : {targets_raw.shape}")
        print(f"Raw inputs (last 5) : {inputs_raw[-5:]}")
        print(f"Raw targets(last 5) : {targets_raw[-5:]}")

        # --- Step 2: Simulate Slicing for Eval ---
        # The author's code returns inputs[:-1], targets[1:] for eval
        inputs_eval = inputs_raw # No, inputs are inputs[:-1] from raw
        targets_eval = targets_raw # No, targets are targets[1:] from raw

        # Correction based on function return statement for is_training=False
        # The function *already performs* the slicing.
        # So inputs_raw, targets_raw ARE the sliced versions.

        print(f"Eval inputs shape  : {inputs_eval.shape}, Eval targets shape: {targets_eval.shape}")
        print(f"Eval inputs (last 5) : {inputs_eval[-5:]}") # This is what model gets
        print(f"Eval targets(last 5) : {targets_eval[-5:]}") # This is used for accuracy check

        # --- Step 3: Check Final Target Element ---
        final_target_eval = targets_eval[-1]
        is_ignored = final_target_eval == IGNORE_IDX
        print(f"Final eval target    : {final_target_eval}")
        print(f"Is final target ignored? : {is_ignored}")
        # Also print the one before it, where the value *might* be
        if len(targets_eval) > 1:
            print(f"Eval target at [-2]  : {targets_eval[-2]}")

        # --- Step 4: Simulate Accuracy Calculation ---
        # Create dummy predictions (e.g., just predict the input token itself)
        # Note: Real model predicts based on inputs_eval[t] to get prediction for targets_eval[t]
        # For simplicity, let's make dummy predictions = inputs_eval
        dummy_predictions = inputs_eval

        # Add dummy predictions for shape check
        if dummy_predictions.shape != targets_eval.shape:
             print(f"[Warning] Dummy prediction shape {dummy_predictions.shape} != Target shape {targets_eval.shape}. Skipping accuracy calc.")
        else:
            accuracy, mask, num_correct, num_valid = numpy_compute_accuracy(dummy_predictions, targets_eval, IGNORE_IDX)
            print(f"Accuracy (dummy preds): {accuracy:.4f} ({num_correct}/{num_valid})")
            print(f"Mask (last 5)        : {mask[-5:]}")

        print("-" * 20)

    print("\nDebug script finished.")
