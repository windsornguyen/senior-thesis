import torch
from torch.utils.data import TensorDataset


def generate_adding(
    num_examples: int = 10,
    sequence_len: int = 128,
    p: int = None,
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generate an adding task dataset with an optional modulo operation.

    The adding task is adapted from Arjovsky, Shah, and Bengio (2016). Each example consists of two sequences:
    1. The first sequence contains random numbers sampled uniformly from [0, 1).
    2. The second sequence is an indicator sequence with exactly two entries set to 1:
       - One in the first half of the sequence.
       - One in the second half of the sequence.
    
    The target output for each example is the sum of the two numbers from the first sequence corresponding to the positions
    of the 1s in the second sequence. If a modulo value `p` is provided, the sum is computed modulo `p`.

    NOTE: A naive strategy of predicting 1 as the output regardless of the input sequence yields an expected mean squared error
    of 0.167, which is the variance of the sum of two independent uniform distributions.

    Args:
        num_examples (int): Number of examples to generate.
        sequence_len (int): Length of each input sequence. Must be at least 2 to accommodate two '1's.
        p (int, optional): If provided, the target sum is computed modulo `p`. Must be a positive integer if specified.
                             Defaults to `None` (no modulo operation).
        seed (int): Random seed for reproducibility.

    Returns:
        TensorDataset:
            - Inputs: (num_examples, 2 * sequence_len)
            - Targets: (num_examples,)
    """
    torch.manual_seed(seed)
    
    if sequence_len < 2:
        raise ValueError("sequence_len must be at least 2 to accommodate two '1's in seq_2.")
    if p is not None:
        if not isinstance(p, int):
            raise TypeError("p must be an integer if provided.")
        if p <= 0:
            raise ValueError("p must be a positive integer.")

    # Construct the first sequence.
    seq_1 = torch.rand((num_examples, sequence_len), dtype=torch.float)

    # Construct the second sequence: indicator sequence with exactly two '1's
    seq_2 = torch.zeros((num_examples, sequence_len), dtype=torch.float)
    half_seq = sequence_len // 2

    # Randomly choose one index in the first half for each example
    idx_1 = torch.randint(0, half_seq, (num_examples,))

    # Randomly choose one index in the second half for each example
    idx_2 = torch.randint(half_seq, sequence_len, (num_examples,))

    # Assign '1's at the chosen indices
    seq_2[torch.arange(num_examples), idx_1] = 1
    seq_2[torch.arange(num_examples), idx_2] = 1

    # Compute the target outputs: sum of the two selected numbers in seq_1
    outputs = torch.sum(seq_1 * seq_2, dim=1)

    # Apply modulo operation if p is provided
    if p is not None:
        outputs = outputs % p

    # Concatenate the two sequences along the sequence dimension
    inputs = torch.cat((seq_1, seq_2), dim=1)

    return TensorDataset(inputs, outputs)
