import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import Tuple

def exists(obj):
    return obj is not None and obj != ""

def generate_copying(
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    selective: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.array, np.array]:
    """
    Generate an instance of the copying task.

    In this formulation, we reserve the two highest indices:
      - copy_token = vocab_size - 1
      - blank_token = vocab_size - 2
    Regular tokens are 0 ... (vocab_size - 3).

    For a given seq_len, we assume:
      - The first num_tokens_to_copy tokens are the content to be copied.
      - Then there is a delay period of (seq_len - (num_tokens_to_copy*2) - 1) blank tokens.
      - Then we append a single copy_token (acting as a delimiter).
      - Finally, we append num_tokens_to_copy blanks.

    The target is constructed as:
      - The first (num_tokens_to_copy + num_blank_tokens + 1) entries are set to target_ignore_idx.
      - The last num_tokens_to_copy entries are the content to be copied.

    In the selective case, the content tokens are interleaved with blanks randomly.
    """
    if not exists(rng):
        rng = np.random.default_rng()

    # Define special tokens:
    copy_token = vocab_size - 1  # highest index
    blank_token = vocab_size - 2  # second-highest index
    # Regular tokens: 0 to (vocab_size - 3)
    non_special_vocab_size = vocab_size - 2
    vocab = np.arange(non_special_vocab_size)

    # Ensure sequence length is sufficient:
    assert seq_len > (num_tokens_to_copy * 2) + 1, "seq_len must be > (num_tokens_to_copy*2)+1"
    num_blank_tokens = seq_len - (num_tokens_to_copy * 2) - 1

    # Sample the content tokens:
    to_copy = rng.choice(vocab, size=(num_tokens_to_copy,), replace=True).reshape(-1)

    if not selective:
        # Non-selective: simply concatenate the content tokens then blanks.
        inputs = list(to_copy)
        inputs += [blank_token] * num_blank_tokens
    else:
        # Selective: interleave the content tokens with blanks.
        inputs = np.array(to_copy)
        # Randomly choose positions to insert blank tokens:
        insert_indices = rng.integers(0, len(inputs), num_blank_tokens)
        inputs = np.insert(inputs, insert_indices, [blank_token] * num_blank_tokens).tolist()

    # Append the delimiter (copy_token) and then additional blanks:
    inputs += [copy_token]
    inputs += [blank_token] * num_tokens_to_copy
    inputs = np.array(inputs)

    # Build targets:
    # For the waiting period, we fill with target_ignore_idx.
    waiting_length = num_tokens_to_copy + num_blank_tokens + 1
    targets = [target_ignore_idx] * waiting_length
    targets += list(to_copy)
    targets = np.array(targets)

    return inputs, targets


def generate_copy_dataset(
    num_examples: int = 10,
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    target_ignore_idx: int = -100,
    selective: bool = False,
    rng: np.random.Generator = None,
) -> TensorDataset:
    """
    Generate a dataset (TensorDataset) for the copy task using v1.
    """
    if not exists(rng):
        rng = np.random.default_rng()
    inputs_list = []
    targets_list = []
    for _ in range(num_examples):
        ins, tar = generate_copying(
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_tokens_to_copy=num_tokens_to_copy,
            rng=rng,
            target_ignore_idx=target_ignore_idx,
            selective=selective,
        )
        inputs_list.append(ins)
        targets_list.append(tar)
    inputs_np = np.stack(inputs_list)  # shape: (num_examples, seq_len)
    targets_np = np.stack(targets_list)  # shape: (num_examples, waiting_length + num_tokens_to_copy)
    inputs_tensor = torch.from_numpy(inputs_np).to(torch.long)
    targets_tensor = torch.from_numpy(targets_np).to(torch.long)
    return TensorDataset(inputs_tensor, targets_tensor)
