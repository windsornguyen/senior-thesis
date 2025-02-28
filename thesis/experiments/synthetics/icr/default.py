import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Union, Optional


def generate_icr(
    num_examples: int = 1000,
    vocab_size: int = 16,
    seq_len: int = 128,
    is_training: bool = True,
    rng: Optional[Union[np.random.Generator, int]] = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
) -> TensorDataset:
    """
    Generate synthetic sequences for testing a model's ability to learn and recall key-value associations
    within the context of a single sequence (in-context learning).

    Task Description:
        - Each sequence consists of key-value pairs, where keys and values are drawn from separate vocabularies
        - Keys can appear multiple times, always paired with their consistent values
        - The task tests whether a model can:
            1. Learn key-value associations on the fly within a sequence
            2. Maintain consistency (same key should predict same value)
            3. Handle interference from noise tokens (if noise is enabled)

    Sequence Structure:
        - Format: [k1,v1, k2,v2, k1,v1, k3,v3, ..., COPY,k_probe,v_probe]
        - Each key-value pair takes 2 positions in the sequence
        - Optional noise tokens can be interspersed between pairs
        - Sequence ends with a probe pair to test recall

    Training vs Testing:
        - Training (is_training=True):
            * Targets are shifted inputs (standard autoregressive training)
            * Model learns to predict the next token at each position
        - Testing (is_training=False):
            * Only value positions after repeated keys are used as targets
            * All other positions are masked with target_ignore_idx

    Args:
        num_examples (int): Number of sequences to generate
        vocab_size (int): Total vocabulary size. The vocab is divided into:
                         - First half: keys
                         - Second half: values
                         - Last token: COPY token (if not multi_query)
        seq_len (int): Length of each sequence (must be even)
        is_training (bool): Whether to generate training or testing sequences
        rng (Generator|int|None): Random number generator or seed
        target_ignore_idx (int): Value used to mask non-target positions
        multi_query (bool): If True, test value recall at every repeated key
                          If False, only test at the final probe pair
        noise_vocab_size (int): Size of vocabulary for noise tokens
        frac_noise (float): Fraction of positions to fill with noise [0,1)
        device (torch.device): Device to place tensors on
        dtype (torch.dtype): Data type for tensors

    Returns:
        TensorDataset: Contains:
            - inputs: (num_examples, seq_len-1) tensor of input tokens
            - targets: (num_examples, seq_len-1) tensor of target tokens
                      For testing, most positions are target_ignore_idx
                      except positions where value recall is tested

    Example:
        A typical sequence might look like:
        Inputs:  [k1,v1, k2,v2, k1,v1, k3,v3, COPY,k1,v1]
        Targets: [-,-,  -,-,   -,v1,  -,-,   -,  -,v1]
        where - represents target_ignore_idx
    """

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    all_inputs = []
    all_targets = []

    for _ in range(num_examples):
        # generate keys and values:
        copy_prefix = vocab_size - 1
        non_special_vocab_size = vocab_size - 1 if not multi_query else vocab_size
        non_special_vocab_size -= noise_vocab_size
        key_vocab = np.arange(non_special_vocab_size // 2)
        value_vocab = np.arange(non_special_vocab_size // 2, non_special_vocab_size)

        # generate noise vocab:
        assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
        if frac_noise > 0:
            assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
            noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size + noise_vocab_size)

        # generate inputs/targets:
        kv_map = {}
        inputs, targets = [], []
        keys_presented = {}
        kv_motif_size = 2
        assert seq_len % kv_motif_size == 0, "seq_len must be an even number"
        num_kv_pairs = seq_len // kv_motif_size
        not_noise_idx = rng.choice(num_kv_pairs)  # make sure we have at least one key-value pair in the sequence

        for i in range(num_kv_pairs - 1):  # subtract one to account for final key-value pair
            # determine if noise or kv pair collected:
            is_noise = rng.random() < frac_noise if i != not_noise_idx and frac_noise > 0 else False

            # collect noise:
            if is_noise:
                noise = rng.choice(noise_vocab, size=kv_motif_size, replace=True)
                inputs += list(noise)
                targets += [target_ignore_idx] * kv_motif_size

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
                        targets.append(v)  # probe value if key has been presented before
                    else:
                        targets.append(target_ignore_idx)

                keys_presented[k] = v

        # add a final key-value pair to the sequence as well as a copy-prefix:
        k_probe = rng.choice(list(keys_presented.keys()))
        v_probe = keys_presented[k_probe]

        if not multi_query:
            inputs.append(copy_prefix)
        inputs.append(k_probe)
        inputs.append(v_probe)

        if not multi_query:
            targets.append(target_ignore_idx)  # copy prefix
        targets.append(target_ignore_idx)  # k_probe
        targets.append(v_probe)

        inputs = np.array(inputs).astype(int)
        targets = np.array(targets).astype(int)

        if is_training:
            # autoregressive shift
            all_inputs.append(inputs[:-1])
            all_targets.append(inputs[1:])  # use shifted inputs as targets for training
        else:
            all_inputs.append(inputs[:-1])
            all_targets.append(targets[1:])

    # convert to tensors
    inputs_tensor = torch.tensor(np.stack(all_inputs), dtype=torch.long)
    targets_tensor = torch.tensor(np.stack(all_targets), dtype=torch.long)

    if device is not None:
        inputs_tensor = inputs_tensor.to(device)
        targets_tensor = targets_tensor.to(device)

    return TensorDataset(inputs_tensor, targets_tensor)
