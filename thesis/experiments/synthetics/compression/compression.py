import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Union, Optional


def generate_compression_instance(
    num_examples: int = 1000,
    vocab_size: int = 16,
    seq_len: int = 32,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: Optional[Union[np.random.Generator, int]] = None,
    target_ignore_idx: int = -100,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
) -> TensorDataset:
    """
    Generate an instance of the compression task.

    Args:
        num_examples (int, optional): Number of examples to generate.
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (Union[np.random.Generator, int], optional): The random number generator or seed to use.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        device (torch.device, optional): Device to place tensors on.
        dtype (torch.dtype, optional): Data type for tensors.

    Returns:
        TensorDataset: Dataset containing inputs and targets tensors.
    """

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    # generate inputs/targets:
    compression_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    vocab = np.arange(non_special_vocab_size)

    all_inputs = []
    all_targets = []

    for _ in range(num_examples):
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

        all_inputs.append(inputs)
        all_targets.append(targets)

    # convert to tensors
    inputs_tensor = torch.tensor(np.stack(all_inputs), dtype=torch.long)
    targets_tensor = torch.tensor(np.stack(all_targets), dtype=torch.long)

    if device is not None:
        inputs_tensor = inputs_tensor.to(device)
        targets_tensor = targets_tensor.to(device)

    return TensorDataset(inputs_tensor, targets_tensor)
