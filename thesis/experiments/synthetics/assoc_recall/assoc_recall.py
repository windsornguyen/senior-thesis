import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import Tuple, Union, Literal
import jax.numpy as jnp


def exists(obj):
    return obj is not None and obj != ""


def generate_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.0,
    return_type: Literal["numpy", "torch", "jax"] = "numpy",
    *args,
    **kwargs,
) -> Union[Tuple[np.array, np.array], Tuple[torch.Tensor, torch.Tensor], Tuple[jnp.array, jnp.array]]:
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
        return_type (Literal["numpy", "torch", "jax"], optional): The type of return format.

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
        targets.append(-100)  # copy prefix
    targets.append(-100)  # k_probe
    targets.append(v_probe)

    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    if is_training:
        # autoregressive shift
        return inputs[:-1], inputs[1:]  # use shifted inputs as targets for training
    else:
        return inputs[:-1], targets[1:]

    if return_type == "torch":
        inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
    elif return_type == "jax":
        inputs, targets = jnp.array(inputs), jnp.array(targets)

    return inputs, targets
