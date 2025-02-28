import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Union, Optional
from itertools import permutations


def exists(obj):
    return obj is not None and obj != ""


def generate_vocab_permutations(vocab, token_motif_size: int = 1, rng=None, *args, **kwargs):
    """
    Generate all possible permutations for a given vocabulary.

    Args:
        vocab (list): The vocabulary.
        token_motif_size (int): The size of the token motif. Defaults to 1.
        rng (np.random.Generator, optional): If given, generated tokens are shuffled.

    Returns:
        list: A list of all possible permutations of the vocabulary.
    """
    values = list(permutations(vocab, token_motif_size))
    if exists(rng):
        rng.shuffle(values)
    return values


def generate_fuzzy_icr(
    num_examples: int = 1000,
    vocab_size: int = 16,
    seq_len: int = 128,
    k_motif_size: int = 3,
    v_motif_size: int = 3,
    is_training: bool = True,
    rng: Optional[Union[np.random.Generator, int]] = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
) -> TensorDataset:
    """
    Generate an instance of the fuzzy in-context recall task.

    Args:
        num_examples (int, optional): Number of examples to generate.
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        k_motif_size (int, optional): The maximum number of adjacent tokens used to represent a key.
        v_motif_size (int, optional): The maximum number of adjacent tokens used to represent a value.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (Union[np.random.Generator, int], optional): The random number generator or seed to use.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        device (torch.device, optional): Device to place tensors on.
        dtype (torch.dtype, optional): Data type for tensors.

    Returns:
        TensorDataset: Dataset containing inputs and targets tensors.
    """

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    all_inputs = []
    all_targets = []

    for _ in range(num_examples):
        # generate keys and values
        copy_prefix = vocab_size - 1
        pad_token = vocab_size - 2 if not multi_query else vocab_size - 1
        non_special_vocab_size = vocab_size - 2 if not multi_query else vocab_size - 1
        non_special_vocab_size -= noise_vocab_size
        key_vocab = np.arange(non_special_vocab_size // 2)
        value_vocab = np.arange(non_special_vocab_size // 2, non_special_vocab_size)

        if is_training:
            # generate keys and values of variable motif sizes
            keys = {}
            for motif_size in range(1, k_motif_size + 1):
                keys_size = generate_vocab_permutations(key_vocab, token_motif_size=motif_size, rng=rng)
                keys[motif_size] = keys_size
            values = {}
            for motif_size in range(1, v_motif_size + 1):
                values_size = generate_vocab_permutations(value_vocab, token_motif_size=motif_size, rng=rng)
                values[motif_size] = values_size
        else:
            # we always prompt at the maximum key motif size
            keys = {k_motif_size: generate_vocab_permutations(key_vocab, token_motif_size=k_motif_size, rng=rng)}
            values = {}
            for motif_size in range(1, v_motif_size + 1):
                values_size = generate_vocab_permutations(value_vocab, token_motif_size=motif_size, rng=rng)
                values[motif_size] = values_size

        # generate noise vocab, if needed:
        assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
        if frac_noise > 0:
            assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
            noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size + noise_vocab_size)

        # generate key-value probe:
        k_probe_size = rng.choice(list(keys.keys())) if is_training else k_motif_size
        v_probe_size = rng.choice(list(values.keys()))
        k_probe = tuple(rng.choice(keys[k_probe_size]))
        v_probe = tuple(rng.choice(values[v_probe_size]))
        kv_probe_size = k_probe_size + v_probe_size
        probe_idx = rng.choice(
            seq_len - kv_probe_size - kv_probe_size
        )  # make sure we add key-value pair to the non-probe part of the sequence
        probe_added = False  # flag indicating whether key-value probe has been added to the sequence

        # generate inputs/targets:
        kv_map = {s: {} for s in range(1, k_motif_size + 1)}
        inputs, targets = [], []
        keys_presented = {}

        # make sure we dont generate too long sequences
        # we pad later to make sure outputs are of length input_seq_len
        while len(inputs) < seq_len - kv_probe_size - (k_motif_size + v_motif_size):
            # determine key-value motif sizes:
            k_size = rng.choice(list(keys.keys())) if is_training else k_motif_size
            v_size = rng.choice(list(values.keys()))

            # make sure key-value probe is in the sequence:
            if len(inputs) >= probe_idx and not probe_added:
                inputs.extend(k_probe)
                inputs.extend(v_probe)
                targets.extend(tuple([target_ignore_idx] * (k_probe_size + len(v_probe))))
                kv_map[k_probe_size][k_probe] = v_probe
                keys_presented[k_probe] = v_probe
                probe_added = True
                continue

            # determine if noise or key-value pair collected:
            is_noise = rng.random() < frac_noise if frac_noise > 0 else False

            # collect noise:
            if is_noise:
                noise_size = k_size + v_size
                noise = rng.choice(noise_vocab, size=noise_size, replace=True)
                inputs.extend(noise)
                targets.extend(tuple([target_ignore_idx] * noise_size))

            # collect key-value pair:
            else:
                # key:
                k = tuple(rng.choice(keys[k_size]))
                inputs.extend(k)

                # value:
                if k == k_probe:
                    v = v_probe
                    probe_added = True
                else:
                    if k not in kv_map[k_size]:
                        v = tuple(rng.choice(values[v_size]))
                        kv_map[k_size][k] = v
                    else:
                        v = kv_map[k_size][k]
                inputs.extend(v)

                # determine targets:
                targets.extend(tuple([target_ignore_idx] * k_size))
                if k not in keys_presented:
                    targets.extend(tuple([target_ignore_idx] * len(v)))
                else:
                    if multi_query:
                        targets.extend(v)  # probe value if key has been presented before
                    else:
                        targets.extend(tuple([target_ignore_idx] * len(v)))

                keys_presented[k] = v

        # add a final key-value pair to the sequence as well as a copy-prefix:
        if not multi_query:
            inputs.extend(tuple([copy_prefix]))
        inputs.extend(k_probe)
        inputs.extend(v_probe)

        if not multi_query:
            targets.extend(tuple([target_ignore_idx]))
        targets.extend(tuple([target_ignore_idx] * k_probe_size))
        targets.extend(v_probe)

        inputs = np.array(inputs).astype(int)
        targets = np.array(targets).astype(int)

        # pad inputs/targets to seq_len:
        if len(inputs) < (seq_len + 1):  # add one to account for autoregressive shift
            n_pad = seq_len + 1 - len(inputs)
            inputs = np.concatenate([np.array([pad_token] * n_pad), inputs])
            targets = np.concatenate([np.array([target_ignore_idx] * n_pad), targets])

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
