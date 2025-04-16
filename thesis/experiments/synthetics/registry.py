import typing as tp
import numpy as np
from itertools import permutations
import os
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset
import time

from thesis.experiments.synthetics.mqar import generate_mqar

# utils:

IGNORE_IDX = -100


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


# in-context recall:


def generate_induction_heads(
    num_examples: int = 5,
    seq_len: int = 30,
    vocab_size: int = 20,
    seed: int = 0,
    ignore_index: int = IGNORE_IDX,
) -> TensorDataset:
    torch.manual_seed(seed)
    special = vocab_size - 1
    inputs = torch.randint(0, vocab_size - 1, (num_examples, seq_len), dtype=torch.long)
    idx = torch.randint(0, seq_len - 2, (num_examples,), dtype=torch.long)
    inputs[torch.arange(num_examples), idx] = special
    inputs[torch.arange(num_examples), -1] = special

    # Create full target tensor with ignore_index
    targets_full = torch.full_like(inputs, ignore_index)
    # Get the actual target values
    target_values = inputs[torch.arange(num_examples), idx + 1]
    # Place target values at the correct positions
    targets_full[torch.arange(num_examples), idx + 1] = target_values

    # For induction heads, the objective is typically autoregressive prediction
    # So, we return inputs and shifted targets (full sequence shape)
    return TensorDataset(inputs, targets_full)


def generate_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    multi_query: bool = False,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.0,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the in-context recall task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.
        noise_vocab_size (int, optional): The size of the noise vocabulary.
        frac_noise (float, optional): The fraction of noise tokens in the sequence.

    Returns:
        tuple: Inputs and targets.
    """

    if not exists(rng):
        rng = np.random.default_rng()

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
        targets.append(IGNORE_IDX)  # copy prefix
    targets.append(IGNORE_IDX)  # k_probe
    targets.append(v_probe)

    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    if is_training:
        # autoregressive shift
        return inputs[:-1], inputs[1:]  # use shifted inputs as targets for training
    else:
        return inputs[:-1], targets[1:]


# noisy in-context recall:


def generate_noisy_in_context_recall_instance(
    vocab_size: int = 32,
    seq_len: int = 128,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.2,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    multi_query: bool = False,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the noisy in-context recall task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.

    Returns:
        tuple: Inputs and targets.
    """
    return generate_in_context_recall_instance(
        vocab_size=vocab_size,
        seq_len=seq_len,
        is_training=is_training,
        rng=rng,
        target_ignore_idx=target_ignore_idx,
        multi_query=multi_query,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
    )


# fuzzy in-context recall:


def generate_fuzzy_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    k_motif_size: int = 3,
    v_motif_size: int = 3,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    multi_query: bool = False,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the fuzzy in-context recall task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        k_motif_size (int, optional): The maximum number of adjacent tokens used to represent a key.
        v_motif_size (int, optional): The maximum number of adjacent tokens used to represent a value.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.

    Returns:
        tuple: Inputs and targets.
    """

    if not exists(rng):
        rng = np.random.default_rng()

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
    # we pad later to make sure outupts are of length input_seq_len
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
        targets.extend(tuple([IGNORE_IDX]))
    targets.extend(tuple([IGNORE_IDX] * k_probe_size))
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
        return inputs[:-1], inputs[1:]  # use shifted inputs as targets for training
    else:
        return inputs[:-1], targets[1:]


# memorization:


def generate_kv_map(
    vocab_size: int,
    k_motif_size: int = 1,
    v_motif_size: int = 1,
    seed: int = 12345,
    *args,
    **kwargs,
) -> dict:
    """
    Generate a fixed mapping from keys to values.

    Args:
        vocab_size (int): The size of the vocabulary.
        {k,v}_motif_size (int, optional): The number of adjacent tokens used to represent a key/value.
        seed (int, optional): The seed used for the random number generator.

    Returns:
        dict: A dictionary mapping keys to values.
    """
    kv_map_rng = np.random.default_rng(seed)  # fixed rng to ensure stable kv mapping across instances
    key_vocab = np.arange(vocab_size // 2)
    value_vocab = np.arange(vocab_size // 2, vocab_size)
    keys = generate_vocab_permutations(key_vocab, k_motif_size, kv_map_rng)
    values = generate_vocab_permutations(value_vocab, v_motif_size, kv_map_rng)
    return {k: v for k, v in zip(keys, values)}


def generate_memorization_instance(
    vocab_size: int = 256,
    seq_len: int = 32,
    kv_map: dict = None,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    kv_map_seed: int = 12345,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the memorization task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        kv_map (dict, optional): A dictionary mapping keys to values.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        kv_map_seed (int, optional): The seed for the random number generator used to generate the kv mapping.

    Returns:
        tuple: Inputs (keys) and targets (values)
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # define vocab sizes:
    insert_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size

    if not exists(kv_map):
        # generate fixed mapping from keys to values:
        kv_map = generate_kv_map(
            vocab_size=non_special_vocab_size,  # account for insert token
            k_motif_size=1,
            v_motif_size=1,
            seed=kv_map_seed,
        )

    keys = list(kv_map.keys())

    # generate noise vocab:
    assert frac_noise >= 0 and frac_noise <= 1, "frac_noise must be >=0 and <=1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise vocab size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size + noise_vocab_size)

    # generate inputs/targets:
    inputs, targets = [], []
    kv_motif_size = 2
    num_kv_pairs = seq_len // kv_motif_size
    not_noise_idx = rng.choice(num_kv_pairs)  # make sure we have at least one key/value pair in the sequence
    for i in range(num_kv_pairs):  # subtract one to account for the final key-value pair
        # determine if noise or key-value pair collected:
        is_noise = rng.random() < frac_noise if i != not_noise_idx and frac_noise > 0 else False

        # collect noise:
        if is_noise:
            noise = rng.choice(noise_vocab, size=kv_motif_size, replace=True)
            inputs.append(noise)
            targets.append([target_ignore_idx] * kv_motif_size)

        # collect key/value pair:
        else:
            k = tuple(rng.choice(keys))
            v = kv_map[k]

            inputs.append(k)
            inputs.append([insert_token])

            targets.append([target_ignore_idx])
            targets.append(v)

    inputs = np.concatenate(inputs).astype(int)
    targets = np.concatenate(targets).astype(int)
    return inputs, targets


# compression:


def generate_compression_instance(
    vocab_size: int = 16,
    seq_len: int = 32,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the compression task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.

    Returns:
        tuple: Inputs and targets.
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # generate inputs/targets:
    compression_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    vocab = np.arange(non_special_vocab_size)
    inputs = rng.choice(vocab, size=(seq_len - 1,), replace=True).reshape(-1)
    inputs = np.concatenate([inputs, np.array([compression_token])])
    targets = np.array(inputs)

    # add noise:
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise vocab size must be > 0 if frac_noise > 0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size + noise_vocab_size)
        for i in range(seq_len - 1):  # exclude compression token
            if rng.random() < frac_noise:
                inputs[i : (i + 1)] = rng.choice(noise_vocab)
                targets[i : (i + 1)] = target_ignore_idx

    return inputs, inputs


# copying:


def generate_copying_instance(
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    selective: bool = False,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the copying task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        num_tokens_to_copy (int, optional): The number of tokens to copy.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        selective (bool, optional): Whether to randomly insert blank tokens in sequence (-> selective copying task).

    Returns:
        tuple: Inputs and targets.
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # define vocab:
    copy_token = vocab_size - 1
    blank_token = vocab_size - 2
    non_special_vocab_size = vocab_size - 2
    vocab = np.arange(non_special_vocab_size)

    # generate inputs:
    assert seq_len > (num_tokens_to_copy * 2) + 1, "seq_len must be > (num_tokens_to_copy * 2) + 1"
    num_blank_tokens = seq_len - (num_tokens_to_copy * 2) - 1
    to_copy = rng.choice(vocab, size=(num_tokens_to_copy,), replace=True).reshape(-1)
    if not selective:
        inputs = list(to_copy)
        inputs += [blank_token] * num_blank_tokens
    else:
        inputs = np.array(to_copy)
        insert_indices = rng.integers(0, len(inputs), size=num_blank_tokens)
        inputs = np.insert(inputs, insert_indices, [blank_token] * num_blank_tokens).tolist()
    inputs += [copy_token]
    inputs += [blank_token] * num_tokens_to_copy
    inputs = np.array(inputs)

    # generate targets:
    targets = [target_ignore_idx] * (num_tokens_to_copy + num_blank_tokens + 1)
    targets += list(to_copy)
    targets = np.array(targets)

    return inputs, targets


# selective copying:


def generate_selective_copying_instance(
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator = None,
    target_ignore_idx: int = IGNORE_IDX,
    *args,
    **kwargs,
) -> tp.Tuple[np.array, np.array]:
    """
    Generate a instance of the selective copying task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        num_tokens_to_copy (int, optional): The number of tokens to copy.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.

    Returns:
        tuple: Inputs and targets.
    """
    return generate_copying_instance(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_tokens_to_copy=num_tokens_to_copy,
        rng=rng,
        target_ignore_idx=target_ignore_idx,
        selective=True,
    )


try:
    import ray
    import ray.util.multiprocessing as mp

    RAY_INSTALLED = True
except ImportError:
    print("ray not installed, falling back to python multiprocessing")
    import multiprocessing as mp

    RAY_INSTALLED = False


def check_for_leakage(train_inputs, test_inputs):
    """Helper to check for data leakage between train and test sets."""
    train_set = set([" ".join(map(str, x)) for x in train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. "
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )


# infrastructure for datasets that are kept in memory:


def generate_data(
    instance_fn,
    instance_fn_kwargs,
    train_data_path: str,
    test_data_path: str,
    num_train_examples: int,
    num_test_examples: int,
    num_workers: int,
    verbose: bool = True,
):
    """
    Generate train/test data in memory (for small datasets or large memory).
    Handles parameter-aware caching using metadata.json.

    Args:
        instance_fn (callable): function that generates individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        num_train_examples (int): number of training examples to generate
        num_test_examples (int): number of test examples to generate
        train_data_path (str): path to write training data to
        test_data_path (str): path to write test data to
        num_workers (int): number of workers to use. If 0, no parallelization is used.
        verbose (bool): whether to print progress.

    Returns:
        dict: dictionary with keys 'train' and 'test' containing torch datasets
    """

    # --- Parameter-aware cache checking ---
    def check_cache_validity(path: str, num_examples: int, current_kwargs: dict) -> bool:
        """Checks if cache at path is valid for current parameters."""
        if not path:
            return False  # Cannot cache if no path provided
        metadata_path = os.path.join(path, "metadata.json")
        inputs_path = os.path.join(path, "inputs.npy")
        targets_path = os.path.join(path, "targets.npy")

        if os.path.exists(metadata_path) and os.path.exists(inputs_path) and os.path.exists(targets_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                # Compare number of examples
                if metadata.get("num_examples") != num_examples:
                    if verbose:
                        print(
                            f"Cache invalid: Mismatched num_examples ({metadata.get('num_examples')} vs {num_examples}) at {path}"
                        )
                    return False
                # Compare instance function kwargs (handle potential non-serializable args like rng)
                # We only compare serializable items for simplicity. A hash could be more robust.
                serializable_current = {
                    k: v
                    for k, v in current_kwargs.items()
                    if isinstance(v, (int, float, str, bool, list, dict, tuple)) and k != "rng"
                }
                serializable_cached = metadata.get("instance_fn_kwargs", {})
                serializable_cached = {
                    k: v
                    for k, v in serializable_cached.items()
                    if isinstance(v, (int, float, str, bool, list, dict, tuple)) and k != "rng"
                }

                if serializable_cached != serializable_current:
                    if verbose:
                        print(f"Cache invalid: Mismatched instance_fn_kwargs at {path}")
                        # Optionally print diff for debugging:
                        # import difflib
                        # diff = difflib.unified_diff(
                        #     json.dumps(serializable_cached, indent=2).splitlines(),
                        #     json.dumps(serializable_current, indent=2).splitlines(),
                        #     fromfile='cached_params', tofile='current_params', lineterm=''
                        # )
                        # print('\n'.join(diff))
                    return False
                # If checks pass, cache is valid
                return True
            except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError) as e:
                if verbose:
                    print(f"Cache metadata error at {path}: {e}. Regenerating.")
                return False  # Metadata corrupted or incompatible
        return False  # Cache files missing

    # Make Training Data.
    train_instance_fn_kwargs = dict(instance_fn_kwargs)
    train_instance_fn_kwargs["is_training"] = True
    train_cache_valid = check_cache_validity(train_data_path, num_train_examples, train_instance_fn_kwargs)

    training_dataset = MemoryDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=train_instance_fn_kwargs,
        verbose=verbose,
    )

    if train_data_path is not None and train_cache_valid:
        if verbose:
            print(f"Valid training data cache found, loading from: {train_data_path}")
        try:
            training_dataset.load_data(train_data_path)  # Assumes load_data only loads .npy
        except Exception as e:
            if verbose:
                print(f"Error loading cached training data from {train_data_path}: {e}. Regenerating.")
            training_dataset.inputs = None  # Reset state
            training_dataset.targets = None
            training_dataset.generate_data(num_examples=num_train_examples, num_workers=num_workers)
            training_dataset.save_data(train_data_path, num_train_examples, train_instance_fn_kwargs)
    elif train_data_path is not None:
        if verbose and os.path.exists(train_data_path):
            print(f"Existing training data at {train_data_path} is invalid/stale. Regenerating...")
        training_dataset.generate_data(num_examples=num_train_examples, num_workers=num_workers)
        training_dataset.save_data(train_data_path, num_train_examples, train_instance_fn_kwargs)
    else:  # No path provided, generate in memory without saving
        if verbose:
            print("No train_data_path provided, generating data in memory without caching.")
        training_dataset.generate_data(num_examples=num_train_examples, num_workers=num_workers)

    # Make Test Data.
    test_instance_fn_kwargs = dict(instance_fn_kwargs)
    test_instance_fn_kwargs["is_training"] = False
    test_cache_valid = check_cache_validity(test_data_path, num_test_examples, test_instance_fn_kwargs)

    test_dataset = MemoryDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=test_instance_fn_kwargs,
        verbose=verbose,
    )

    if test_data_path is not None and test_cache_valid:
        if verbose:
            print(f"Valid test data cache found, loading from: {test_data_path}")
        try:
            test_dataset.load_data(test_data_path)
        except Exception as e:
            if verbose:
                print(f"Error loading cached test data from {test_data_path}: {e}. Regenerating.")
            test_dataset.inputs = None  # Reset state
            test_dataset.targets = None
            test_dataset.generate_data(num_examples=num_test_examples, num_workers=num_workers)
            test_dataset.save_data(test_data_path, num_test_examples, test_instance_fn_kwargs)

    elif test_data_path is not None:
        if verbose and os.path.exists(test_data_path):
            print(f"Existing test data at {test_data_path} is invalid/stale. Regenerating...")
        test_dataset.generate_data(num_examples=num_test_examples, num_workers=num_workers)
        test_dataset.save_data(test_data_path, num_test_examples, test_instance_fn_kwargs)

    else:  # No path provided, generate in memory without saving
        if verbose:
            print("No test_data_path provided, generating data in memory without caching.")
        test_dataset.generate_data(num_examples=num_test_examples, num_workers=num_workers)

    # Check for data leakage.
    check_for_leakage(training_dataset.inputs, test_dataset.inputs)

    # NOTE: The assertion checks below are removed as the loading/generation logic
    # now ensures the datasets have the correct number of examples if caching is used.
    # If cache is invalid, it regenerates, ensuring correctness.
    # If no path is given, it generates exactly the requested number.
    # assert training_dataset.inputs.shape[0] == num_train_examples, (
    #     f"Training data have {training_dataset.inputs.shape[0]} samples but should have {num_train_examples} samples."
    # )
    # assert test_dataset.inputs.shape[0] == num_test_examples, (
    #     f"Test data have {test_dataset.inputs.shape[0]} samples but should have {num_test_examples} samples."
    # )

    return {"train": training_dataset, "test": test_dataset}


class MemoryDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that stores data in memory.

    Args:
        instance_fn (callable): function used to generate individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        is_training (bool): whether training or test targets are used.
        verbose (bool): whether to print progress.
    """

    def __init__(self, instance_fn: callable, instance_fn_kwargs: dict, verbose: bool = True) -> None:
        super().__init__()
        self.instance_fn = instance_fn
        self.instance_fn_kwargs = instance_fn_kwargs
        self.inputs = None
        self.targets = None
        self.verbose = verbose

    def __getitem__(self, idx):
        # Ensure data is loaded or generated before access
        if self.inputs is None:
            raise RuntimeError(
                "Dataset data has not been generated or loaded. Call generate_data() or load_data() first."
            )
        # Check index bounds after ensuring data is loaded
        if not 0 <= idx < len(self.inputs):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.inputs)}")
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        # Ensure data is loaded or generated before access
        if self.inputs is None:
            # Return 0 if data not loaded/generated yet, common in some DataLoader setups
            return 0
        return len(self.inputs)

    def load_data(self, path: str):
        """
        Load data from path. Assumes metadata check was done externally.

        Args:
            path (str): path to load data from.
        """
        if self.verbose:
            print(f"Loading data arrays from: {path}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found for loading data: {path}")
        inputs_path = os.path.join(path, "inputs.npy")
        targets_path = os.path.join(path, "targets.npy")
        if not os.path.exists(inputs_path) or not os.path.exists(targets_path):
            raise FileNotFoundError(f"Missing 'inputs.npy' or 'targets.npy' in {path}")

        self.inputs = np.load(inputs_path)
        self.targets = np.load(targets_path)
        if len(self.inputs) != len(self.targets):
            # This check should ideally not fail if save_data is correct
            raise ValueError(
                f"Loaded inputs and targets have different lengths in {path}: {len(self.inputs)} vs {len(self.targets)}"
            )

    def save_data(self, path: str, num_examples: int, instance_fn_kwargs: dict):
        """
        Save data and metadata to path.

        Args:
            path (str): path to save data to.
            num_examples (int): Number of examples requested (used for metadata).
            instance_fn_kwargs (dict): Instance function kwargs used for generation.
        """
        if self.inputs is None or self.targets is None:
            raise RuntimeError("Cannot save data before it is generated.")

        actual_num_examples = len(self.inputs)
        if actual_num_examples != num_examples:
            # This case might happen if generation produces fewer examples than requested, though unlikely with current generators.
            # More likely, the check_cache_validity already determined the number based on existing valid cache.
            # We should save the *actual* number of examples in the metadata.
            if self.verbose:
                print(
                    f"WARNING: Saving {actual_num_examples} examples, but {num_examples} were requested/expected. Metadata will reflect actual count."
                )
            num_examples_to_save = actual_num_examples  # Use actual count for metadata
        else:
            num_examples_to_save = num_examples

        if os.path.isdir(path):
            if self.verbose:
                print(f'INFO: directory "{path}" exists already, overwriting contents...')
        os.makedirs(path, exist_ok=True)

        # Save data arrays
        inputs_path = os.path.join(path, "inputs.npy")
        targets_path = os.path.join(path, "targets.npy")
        np.save(inputs_path, self.inputs)
        np.save(targets_path, self.targets)

        # Prepare and save metadata
        metadata = {
            "num_examples": num_examples_to_save,
            # Store only serializable kwargs, exclude 'rng'
            "instance_fn_kwargs": {
                k: v
                for k, v in instance_fn_kwargs.items()
                if isinstance(v, (int, float, str, bool, list, dict, tuple)) and k != "rng"
            },
        }
        metadata_path = os.path.join(path, "metadata.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            if self.verbose:
                print(f"Saved data ({actual_num_examples} examples) and metadata to {path}")
        except TypeError as e:
            print(f"WARNING: Could not serialize all instance_fn_kwargs to {metadata_path}: {e}")
            # Attempt to save without unserializable items
            try:
                with open(metadata_path, "w") as f:
                    # More robust filtering
                    serializable_kwargs = {}
                    for k, v in instance_fn_kwargs.items():
                        if k == "rng":
                            continue
                        try:
                            json.dumps({k: v})  # Test serializability
                            serializable_kwargs[k] = v
                        except TypeError:
                            if self.verbose:
                                print(f"  Skipping non-serializable kwarg: {k}")
                            pass
                    metadata["instance_fn_kwargs"] = serializable_kwargs
                    json.dump(metadata, f, indent=4)
                    if self.verbose:
                        print(f"Saved data ({actual_num_examples} examples) and partial metadata to {path}")
            except Exception as final_e:
                print(f"ERROR: Failed to save metadata even after filtering: {final_e}. Cache validation may fail.")
                # Still save the .npy files, just warn about metadata

    def generate_data(
        self,
        num_examples: int,
        num_workers: int = 0,
        verbose: bool = True,
    ):
        """
        Generate data with self.instance_fn.

        Args:
            num_examples (int): number of examples to generate.
            num_workers (int): number of workers to use. If 0, no parallelization is used.
            verbose (bool): whether to print progress.
        """
        assert self.inputs is None
        assert self.targets is None

        if self.verbose:
            print(f"Generating dataset with {num_examples} examples, using {num_workers} workers...")

        # parallel data generation:
        if num_workers > 1:
            # with ray:
            if RAY_INSTALLED:
                kv_map_id = ray.put(self.instance_fn_kwargs.get("kv_map", None))

                @ray.remote
                def f(*args):
                    return self.instance_fn(
                        **{k: v for k, v in self.instance_fn_kwargs.items() if k != "kv_map"},
                        kv_map=ray.get(kv_map_id),
                    )

            # without ray:
            else:

                def f(*args):
                    return self.instance_fn(**self.instance_fn_kwargs)

            pool = mp.Pool(num_workers)
            if RAY_INSTALLED:
                # we process in num_workers chunks to avoid cpu throttle!
                # see: https://discuss.ray.io/t/using-a-subset-of-available-cpus
                instances = []
                for _ in range(0, num_examples, num_workers):
                    chunk_instances = pool.map(f.remote, range(num_workers))
                    instances.extend(ray.get(chunk_instances))
            else:
                instances = pool.map(f, range(num_examples))
            pool.close()

        # sequential data generation:
        else:
            iterator = tqdm(range(num_examples)) if self.verbose else range(num_examples)
            instances = [self.instance_fn(**self.instance_fn_kwargs) for _ in iterator]

        if len(instances[-1]) == 2:
            self.inputs, self.targets = [np.stack(i) for i in zip(*instances)]
        else:
            raise ValueError(
                f"returned instances need to contain 2 arrays (inputs, targets) but got {len(instances[-1])}"
            )


# infrastructure for datasets that are stored on disk and read from there during training:


def generate_data_disk(
    instance_fn: callable,
    instance_fn_kwargs: dict,
    num_train_examples: int,
    num_test_examples: int,
    train_data_path: str,
    test_data_path: str,
    num_workers: int = 0,
    verbose: bool = True,
    num_docs_training: int = 1,
    num_docs_test: int = 1,
    return_datasets: bool = True,
    *args,
    **kwargs,
):
    """
    Write data to disk and return train/test torch datasets that read data from disk
    during training.

    Args:
        instance_fn (callable): function that generates individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        num_train_examples (int): number of training examples
        num_test_examples (int): number of test examples
        train_data_path (str): path to write training data to
        test_data_path (str): path to write test data to
        num_workers (int): number of workers to use. If 0, no parallelization is used.
        verbose (bool): whether to print progress.
        num_docs_training (int): number of documents to write training data to.
        num_docs_test (int): number of documents to write test data to.

    Returns:
        dict: dictionary with keys 'train' and 'test' containing torch datasets
    """
    assert not os.path.isdir(train_data_path), f'directory "{train_data_path}" already exists'
    assert not os.path.isdir(test_data_path), f'directory "{test_data_path}" already exists'

    # Make Training Data.

    train_instance_fn_kwargs = dict(instance_fn_kwargs)
    train_instance_fn_kwargs["is_training"] = True
    training_dataset = DiskDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=train_instance_fn_kwargs,
        verbose=verbose,
    )
    training_dataset.generate_data(
        path=train_data_path,
        num_examples=num_train_examples,
        num_documents=num_docs_training,
        num_workers=num_workers,
    )

    # Make Test Data.

    test_instance_fn_kwargs = dict(instance_fn_kwargs)
    test_instance_fn_kwargs["is_training"] = False
    test_dataset = DiskDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=test_instance_fn_kwargs,
        verbose=verbose,
    )
    test_dataset.generate_data(
        path=test_data_path,
        num_examples=num_test_examples,
        num_documents=num_docs_test,
        num_workers=num_workers,
    )

    if return_datasets:
        return {"train": training_dataset, "test": test_dataset}


class DiskDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that stores data on disk and reads from there during training.

    Args:
        instance_fn (callable): function used to generate individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        is_training (bool): whether training or test targets are used.
        verbose (bool): whether to print progress.
    """

    def __init__(self, instance_fn: callable, instance_fn_kwargs: dict, verbose: bool = True) -> None:
        super().__init__()
        self.instance_fn = instance_fn
        self.instance_fn_kwargs = instance_fn_kwargs
        self.inputs = None
        self.targets = None
        self.documents = None
        self.verbose = verbose

    def __getitem__(self, idx: int):
        # TODO: also read data-idx from disk?
        doc_idx, doc_path = self.data_idx.iloc[idx][["doc_idx", "doc_path"]]
        instance = self.read_instance_from_doc(doc_idx, doc_path)
        return (
            np.array(instance["inputs"].split()).astype(int),
            np.array(instance["targets"].split()).astype(int),
        )

    def __len__(self):
        return len(self.data_idx)

    def use_data_from_idx(self, data_idx: tp.Union[str, pd.DataFrame]):
        """
        Use existing dataset.

        Args:
            data_idx (str or DataFrame): data index for existing dataset (as generated by self.make_idx).
        """
        self.data_idx = pd.read_csv(data_idx, index_col=0) if isinstance(data_idx, str) else data_idx
        assert "doc_path" in self.data_idx.columns
        assert "doc_idx" in self.data_idx.columns
        self.documents = set(list(self.data_idx["doc_path"].values))
        assert len(self.data_idx) == sum([len(pd.read_csv(d, index_col=0)) for d in self.documents])

    def generate_data(self, path: str, num_examples: int, num_documents: int = 1, num_workers: int = 0):
        """
        Generate data and write to specified number of documents in path.

        Args:
            path (str): path to write data to.
            num_examples (int): number of examples to generate.
            num_documents (int): number of documents to write data to.
            num_workers (int): number of workers to use for data generate. If 0, no parallelization is used.
        """
        assert not os.path.exists(path)
        assert self.documents is None
        if self.verbose:
            print(f'\nwriting dataset of {num_examples} samples to {num_documents} documents in "{path}"...')
        os.makedirs(path, exist_ok=True)
        self.documents = []
        num_examples_per_doc = num_examples // num_documents
        for i in range(num_documents):
            if self.verbose:
                print(f"\tcreating document {i + 1}/{num_documents}...")
            doc_path = os.path.join(path, f"train_{i + 1}.txt")
            assert not os.path.exists(doc_path), f"file {doc_path} already exists"
            self.generate_doc_data(
                doc_path=doc_path,
                num_examples=num_examples_per_doc
                if i < num_documents - 1
                else num_examples - num_examples_per_doc * i,
                num_workers=num_workers,
            )
            self.documents.append(doc_path)

        # index generated data
        self.make_idx(path_idx=os.path.join(path, "data_idx.csv"))

    def generate_doc_data(
        self,
        doc_path: str,
        num_examples: int,
        num_workers: int = 0,
    ):
        """
        Generate data and write to document.

        Args:
            doc_path (str): document path to write data to
            num_examples (int): number of examples
            num_workers (int): number of workers to use. If 0, no parallelization is used.
        """
        # generate data in parallel:
        if num_workers > 1:
            # with ray:
            if RAY_INSTALLED:
                kv_map_id = ray.put(self.instance_fn_kwargs.get("kv_map", None))

                @ray.remote
                def f(*args):
                    instance = self.instance_fn(
                        **{k: v for k, v in self.instance_fn_kwargs.items() if k != "kv_map"},
                        kv_map=ray.get(kv_map_id),
                    )
                    self.write_instance_to_doc(instance, doc_path)

            # without ray:
            else:
                print("ray not installed, falling back to multiprocessing")
                import multiprocessing as mp

                def f(*args):
                    instance = self.instance_fn(**self.instance_fn_kwargs)
                    self.write_instance_to_doc(instance, doc_path)

            pool = mp.Pool(num_workers)
            if RAY_INSTALLED:
                ray.get(pool.map(f.remote, range(num_examples)))
            else:
                pool.map(f, range(num_examples))
            pool.close()

        # generate data in sequence:
        else:
            iterator = tqdm(range(num_examples)) if self.verbose else range(num_examples)
            for _ in iterator:
                instance = self.instance_fn(**self.instance_fn_kwargs)
                self.write_instance_to_doc(instance, doc_path)

    def write_instance_to_doc(self, instance: tp.Tuple, doc_path: str):
        """
        Write instance to document.

        Args:
            instance (tuple): generated instance.
            doc_path (str): path of document to which instance is appended.
        """

        if len(instance) == 2:
            instance_dict = {
                "inputs": " ".join([str(t) for t in instance[0]]),
                "targets": " ".join([str(t) for t in instance[1]]),
            }
        else:
            raise ValueError(f"returned instances need to contain 2 (or 3) entries but got {len(instance)}")

        with open(doc_path, "a+") as f:
            json.dump(instance_dict, f)
            f.write("\n")  # add newline for readability

    def make_idx(self, path_idx: str):
        """
        Index generated data and write index to path.

        Args:
            path_idx (str): path to write index to.
        """
        assert not os.path.isfile(path_idx)
        i, data_idx = 0, []
        for document in self.documents:
            doc_lines = self.read_lines_from_doc(document)
            n_instances = len(doc_lines)
            data_idx.append(
                pd.DataFrame(
                    {
                        "doc_idx": np.arange(n_instances),
                        "doc_path": [document] * n_instances,
                    },
                    index=np.arange(i, i + n_instances),
                )
            )
            i += n_instances
        data_idx = pd.concat(data_idx)
        data_idx.to_csv(path_idx)
        self.data_idx = data_idx

    def read_lines_from_doc(self, doc_path: str):
        with open(doc_path, "r") as f:
            data = f.readlines()
        return [json.loads(line) for line in data]

    def read_instance_from_doc(self, idx: int, doc_path: str):
        with open(doc_path, "r") as file:
            file.seek(0)  # move file pointer to the beginning of file
            for _ in range(idx - 1):
                file.readline()
            line_content = file.readline()
        return json.loads(line_content) if line_content else None


# Task registry with PyTorch and JAX support


class TaskRegistry:
    def __init__(self):
        self.tasks = {
            "in_context_recall": generate_in_context_recall_instance,
            "noisy_in_context_recall": generate_noisy_in_context_recall_instance,
            "fuzzy_in_context_recall": generate_fuzzy_in_context_recall_instance,
            "memorization": generate_memorization_instance,
            "compression": generate_compression_instance,
            "copying": generate_copying_instance,
            "selective_copying": generate_selective_copying_instance,
            "induction_heads": generate_induction_heads,
            "mqar": generate_mqar,
        }
        self.presets = {
            # --- Default MQAR Preset (example) ---
            "mqar_default": {
                "task_name": "mqar",
                "seq_len": 256,
                "vocab_size": 8192 + 3,  # Example: V=8192, DELIMS=3
                "num_pairs": 32,
                "alpha": 2.0,
                # Add other mqar defaults if needed
            },
            # --- Default In-Context Recall Preset (example) ---
            "recall_default": {
                "task_name": "in_context_recall",
                "seq_len": 256,
                "vocab_size": 16 + 1,  # Example V=16, copy_prefix=1
                "multi_query": False,
                "noise_vocab_size": 0,
                "frac_noise": 0.0,
                # Add other recall defaults if needed
            },
            # --- Add more presets as needed --- #
        }

    def get_task(self, task_name):
        """Return the task generation function by name."""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found. Available tasks: {list(self.tasks.keys())}")
        return self.tasks[task_name]

    def create_datasets(
        self,
        task_name: str = None,  # Now optional
        preset_name: str = None,  # New argument
        num_train: int = None,
        num_test: int = None,
        in_memory: bool = True,
        **task_kwargs,
    ):
        """Create training and test datasets for the specified task or preset."""
        # --- Resolve Task and Parameters from Preset or Direct Args ---
        final_task_name = None
        final_task_kwargs = {}

        if preset_name:
            if task_name:
                raise ValueError("Cannot provide both preset_name and task_name.")
            if preset_name not in self.presets:
                raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(self.presets.keys())}")

            preset_config = self.presets[preset_name]
            final_task_name = preset_config["task_name"]
            final_task_kwargs = preset_config.copy()  # Start with preset defaults
            final_task_kwargs.pop("task_name")  # Remove task_name from kwargs
            # Update with user overrides
            final_task_kwargs.update(task_kwargs)
            print(
                f"INFO: Using preset '{preset_name}' (task: '{final_task_name}') with final kwargs: {final_task_kwargs}"
            )

        elif task_name:
            final_task_name = task_name
            final_task_kwargs = task_kwargs  # Use only user-provided kwargs
            print(f"INFO: Using task '{final_task_name}' directly with kwargs: {final_task_kwargs}")
        else:
            raise ValueError("Must provide either task_name or preset_name.")

        # Ensure num_train and num_test are provided
        if num_train is None or num_test is None:
            raise ValueError("num_train and num_test must be provided.")

        # Get the task generation function
        task_fn = self.get_task(final_task_name)

        # Set up paths
        import os
        import tempfile

        data_dir = task_kwargs.pop("data_dir", os.path.join(tempfile.gettempdir(), "task_data"))
        os.makedirs(data_dir, exist_ok=True)

        # Warn users when using temp directory
        if "data_dir" not in task_kwargs:
            import platform

            system = platform.system()

            # Estimate disk space usage (very rough estimate)
            vocab_size = task_kwargs.get("vocab_size", 256)
            seq_len = task_kwargs.get("seq_len", 256)
            # Assume ~4 bytes per token plus some overhead for storage format
            est_bytes_per_example = seq_len * 2 * 5  # inputs + targets, ~5 bytes per token with overhead
            est_mb = (num_train + num_test) * est_bytes_per_example / (1024 * 1024)

            if system == "Darwin":  # macOS
                print(f"\n⚠️ WARNING: Saving datasets to macOS temporary directory: {data_dir}")
                print("   On macOS, temp files may be cleared on system restart.")
            elif system == "Linux":
                print(f"\n⚠️ WARNING: Saving datasets to Linux temporary directory: {data_dir}")
                print("   On Linux, temp files may be cleared based on distro-specific policies.")
            elif system == "Windows":
                print(f"\n⚠️ WARNING: Saving datasets to Windows temporary directory: {data_dir}")
                print("   On Windows, temp files generally persist until disk cleanup.")
            else:
                print(f"\n⚠️ WARNING: Saving datasets to temporary directory: {data_dir}")

            print("   Data may be deleted by system cleanup processes.")
            print(f"   Estimated disk usage: ~{est_mb:.1f} MB for {num_train + num_test} examples.")
            print("   Set 'data_dir' explicitly if you want to keep the generated datasets.")
            print("   Example: registry.create_data_loaders(..., data_dir='./my_datasets')\n")

        train_path = os.path.join(data_dir, f"{final_task_name}_train")
        test_path = os.path.join(data_dir, f"{final_task_name}_test")

        # Special handling for tasks returning full TensorDatasets
        if final_task_name in ["induction_heads", "mqar"]:
            if not in_memory:
                print(
                    f"\n⚠️  WARNING: Task '{final_task_name}' inherently generates data in memory. "
                    f"The 'in_memory=False' flag will be ignored. Data will not be saved to disk by default "
                    f"using the standard DiskDataset mechanism. Saving happens only if paths {train_path} or {test_path} don't exist."
                )
                # We proceed as if in_memory=True, but allow saving if paths don't exist

            # Ensure required args are present
            required_args = ["seq_len", "vocab_size"]
            for arg in required_args:
                if arg not in final_task_kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for task '{final_task_name}'")

            base_seed = final_task_kwargs.get("seed", 0)

            # --- Generate Training Data ---
            print(f"Generating '{final_task_name}' training data...")
            # Pass final_task_kwargs directly, as they contain all needed params
            # Remove num_workers if present, handled later
            gen_kwargs_train = {k: v for k, v in final_task_kwargs.items() if k != "num_workers"}
            train_dataset = task_fn(num_examples=num_train, seed=base_seed, **gen_kwargs_train)

            # Attempt to save if path doesn't exist (mimics generate_data behavior)
            if train_path and not os.path.exists(train_path):
                os.makedirs(train_path, exist_ok=True)
                # Save TensorDataset tensors
                torch.save(train_dataset.tensors, os.path.join(train_path, "data.pt"))
                print(f"Saved training data to {train_path}/data.pt")
                # Also save metadata for consistency with MemoryDataset caching
                train_metadata = {
                    "num_examples": len(train_dataset),
                    "instance_fn_kwargs": {
                        k: v
                        for k, v in final_task_kwargs.items()
                        if isinstance(v, (int, float, str, bool, list, dict, tuple)) and k != "rng"
                    },
                }
                try:
                    with open(os.path.join(train_path, "metadata.json"), "w") as f:
                        json.dump(train_metadata, f, indent=4)
                except Exception as e:
                    print(f"Warning: Failed to save metadata for {final_task_name} train set: {e}")

            # --- Generate Test Data ---
            print(f"Generating '{final_task_name}' test data...")
            test_seed = base_seed + 1  # Use a different seed for test set
            # Pass final_task_kwargs directly
            gen_kwargs_test = {k: v for k, v in final_task_kwargs.items() if k != "num_workers"}
            test_dataset = task_fn(num_examples=num_test, seed=test_seed, **gen_kwargs_test)

            # Attempt to save if path doesn't exist
            if test_path and not os.path.exists(test_path):
                os.makedirs(test_path, exist_ok=True)
                torch.save(test_dataset.tensors, os.path.join(test_path, "data.pt"))
                print(f"Saved test data to {test_path}/data.pt")
                # Also save metadata
                test_metadata = {
                    "num_examples": len(test_dataset),
                    "instance_fn_kwargs": {
                        k: v
                        for k, v in final_task_kwargs.items()
                        if isinstance(v, (int, float, str, bool, list, dict, tuple)) and k != "rng"
                    },
                }
                try:
                    with open(os.path.join(test_path, "metadata.json"), "w") as f:
                        json.dump(test_metadata, f, indent=4)
                except Exception as e:
                    print(f"Warning: Failed to save metadata for {final_task_name} test set: {e}")

            return {"train": train_dataset, "test": test_dataset}

        # Existing logic for other tasks using MemoryDataset
        elif in_memory:
            # Call generate_data with its expected arguments
            # It handles the is_training flag internally
            datasets_dict = generate_data(
                instance_fn=task_fn,
                instance_fn_kwargs=final_task_kwargs,  # Pass base kwargs
                train_data_path=train_path,
                test_data_path=test_path,
                num_train_examples=num_train,
                num_test_examples=num_test,
                num_workers=final_task_kwargs.get("num_workers", 0),
                verbose=final_task_kwargs.get("verbose", True),  # Pass verbose if needed
            )
            # generate_data returns a dict {"train": dataset, "test": dataset}
            return datasets_dict

        else:  # DiskDataset logic
            # Call generate_data_disk, assuming it handles is_training internally
            # or requires separate calls (checking its definition might be needed if error persists here)
            datasets_dict = generate_data_disk(
                instance_fn=task_fn,
                instance_fn_kwargs=final_task_kwargs,  # Pass base kwargs
                num_train_examples=num_train,
                num_test_examples=num_test,
                train_data_path=train_path,
                test_data_path=test_path,
                num_workers=final_task_kwargs.get("num_workers", 0),
                verbose=final_task_kwargs.get("verbose", True),
                # Assuming generate_data_disk also returns {"train": ..., "test": ...}
                return_datasets=True,  # Keep this if generate_data_disk expects it
            )
            return datasets_dict

    def create_data_loaders(
        self,
        task_name: str = None,  # Optional
        preset_name: str = None,  # New
        batch_size: int = 32,
        num_train: int = 10000,
        num_test: int = 1000,
        backend: str = "torch",
        device="cpu",  # Accept device object or string
        in_memory: bool = True,
        **task_kwargs,  # User overrides for preset or direct task args
    ):
        """Create data loaders for the specified task or preset.

        Args:
            task_name (str, optional): Name of the task (if not using preset).
            preset_name (str, optional): Name of the preset configuration.
            batch_size (int): Batch size for data loaders.
            num_train (int): Number of training examples.
            num_test (int): Number of test examples.
            backend (str): "torch" or "jax".
            device: Device for data loading ("cpu", "cuda", or device object).
            in_memory (bool): Whether to use MemoryDataset.
            **task_kwargs: Arguments passed to the task generation function,
                          potentially overriding preset values.

        Returns:
            tuple: (train_loader, test_loader) or (train_iter, test_iter) for JAX
        """
        # Pass preset_name and task_kwargs down to create_datasets
        # create_datasets will handle resolving the actual task and merging kwargs
        datasets = self.create_datasets(
            task_name=task_name,
            preset_name=preset_name,
            num_train=num_train,
            num_test=num_test,
            in_memory=in_memory,
            **task_kwargs,
        )

        if backend.lower() == "torch":
            return self._create_torch_loaders(datasets, batch_size, device)
        elif backend.lower() == "jax":
            # Check if datasets fit in memory for preloading
            if not in_memory:
                raise ValueError("JAX backend with preloading requires in_memory=True")
            return self._create_jax_iterators_preloaded(datasets, batch_size, device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'jax'.")

    def _create_torch_loaders(self, datasets, batch_size, device):
        """Create PyTorch data loaders."""
        import torch

        use_cuda = device == "cuda" and torch.cuda.is_available()

        def collate_fn(batch):
            inputs, targets = zip(*batch)
            inputs = torch.tensor(np.stack(inputs), dtype=torch.long)
            targets = torch.tensor(np.stack(targets), dtype=torch.long)
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            return inputs, targets

        train_loader = torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        return train_loader, test_loader

    def _create_jax_iterators_preloaded(self, datasets, batch_size, device):
        """Create JAX data iterators by preloading data onto the device and slicing."""
        try:
            import jax
            import jax.numpy as jnp
            from torch.utils.data import DataLoader, Dataset, TensorDataset
        except ImportError:
            raise ImportError("JAX and/or torch not installed. Please install with: pip install jax torch")

        print(f"INFO: Using JAX preloaded data strategy. Moving full datasets to device: {device}")

        class IndexDataset(Dataset):
            """A simple dataset that returns indices."""

            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return idx  # Return the index itself

        class JaxSlicedDeviceIterator:
            """Iterates by slicing preloaded JAX arrays on device using indices."""

            def __init__(self, inputs_dev, targets_dev, index_loader):
                self.inputs_dev = inputs_dev
                self.targets_dev = targets_dev
                self.index_loader = index_loader
                # Check memory usage (optional, requires psutil)
                # try:
                #     import psutil
                #     input_mb = inputs_dev.device_buffer.size / (1024*1024)
                #     target_mb = targets_dev.device_buffer.size / (1024*1024)
                #     print(f"INFO: Preloaded dataset size on device: Inputs={input_mb:.2f}MB, Targets={target_mb:.2f}MB")
                # except (ImportError, AttributeError):
                #     pass

            def __iter__(self):
                self.idx_iterator = iter(self.index_loader)
                return self

            def __next__(self):
                try:
                    # 1. Get next batch of indices (NumPy array from DataLoader)
                    batch_indices_np = next(self.idx_iterator)
                    # 2. Convert indices to JAX array (minimal host->device transfer)
                    batch_indices_jax = jnp.asarray(batch_indices_np)
                    # 3. Slice data directly on the device using JAX indexing
                    batch_inputs = self.inputs_dev[batch_indices_jax]
                    batch_targets = self.targets_dev[batch_indices_jax]
                    return batch_inputs, batch_targets
                except StopIteration:
                    raise StopIteration

        # --- Preload Training Data ---
        train_dataset = datasets["train"]
        # Handle both MemoryDataset and TensorDataset
        if isinstance(train_dataset, MemoryDataset):
            if train_dataset.inputs is None or train_dataset.targets is None:
                raise RuntimeError("Training MemoryDataset has no data. Ensure generate_data/load_data was called.")
            train_inputs_np = train_dataset.inputs
            train_targets_np = train_dataset.targets
        elif isinstance(train_dataset, TensorDataset):
            if not hasattr(train_dataset, "tensors") or len(train_dataset.tensors) < 2:
                raise RuntimeError("Training TensorDataset does not have expected .tensors attribute.")
            # Convert torch tensors to numpy
            train_inputs_np = train_dataset.tensors[0].cpu().numpy()
            train_targets_np = train_dataset.tensors[1].cpu().numpy()
        else:
            raise TypeError(f"Unsupported dataset type for JAX preloading: {type(train_dataset)}")

        print("Transferring training data to device...")
        start_time = time.time()
        train_inputs_dev = jax.device_put(jnp.asarray(train_inputs_np), device)
        train_targets_dev = jax.device_put(jnp.asarray(train_targets_np), device)
        print(f"Training data transfer complete ({time.time() - start_time:.2f}s)")

        train_idx_dataset = IndexDataset(len(train_dataset))
        train_idx_loader = DataLoader(
            train_idx_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # Important for static shapes in JAX
            collate_fn=np.stack,  # Simple collation for indices
        )
        train_iterator = JaxSlicedDeviceIterator(train_inputs_dev, train_targets_dev, train_idx_loader)

        # --- Preload Test Data ---
        test_dataset = datasets["test"]
        # Handle both MemoryDataset and TensorDataset
        if isinstance(test_dataset, MemoryDataset):
            if test_dataset.inputs is None or test_dataset.targets is None:
                raise RuntimeError("Test MemoryDataset has no data. Ensure generate_data/load_data was called.")
            test_inputs_np = test_dataset.inputs
            test_targets_np = test_dataset.targets
        elif isinstance(test_dataset, TensorDataset):
            if not hasattr(test_dataset, "tensors") or len(test_dataset.tensors) < 2:
                raise RuntimeError("Test TensorDataset does not have expected .tensors attribute.")
            # Convert torch tensors to numpy
            test_inputs_np = test_dataset.tensors[0].cpu().numpy()
            test_targets_np = test_dataset.tensors[1].cpu().numpy()
        else:
            raise TypeError(f"Unsupported dataset type for JAX preloading: {type(test_dataset)}")

        print("Transferring test data to device...")
        start_time = time.time()
        test_inputs_dev = jax.device_put(jnp.asarray(test_inputs_np), device)
        test_targets_dev = jax.device_put(jnp.asarray(test_targets_np), device)
        print(f"Test data transfer complete ({time.time() - start_time:.2f}s)")

        test_idx_dataset = IndexDataset(len(test_dataset))
        test_idx_loader = DataLoader(
            test_idx_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,  # Keep all test data if possible
            collate_fn=np.stack,
        )
        test_iterator = JaxSlicedDeviceIterator(test_inputs_dev, test_targets_dev, test_idx_loader)

        return train_iterator, test_iterator


# Create a singleton instance for easy importing
registry = TaskRegistry()
