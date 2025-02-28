import torch
from torch.utils.data import TensorDataset


def generate_mode(
    num_examples: int = 10,
    sequence_len: int = 128,
    num_classes: int = 5,
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generate a mode tracking task dataset.

    In this task, each example consists of a sequence of integers. The target at each time step is the mode
    (most frequent element) of all elements seen so far in the sequence. In the case of a tie, the smallest
    integer is chosen as the mode.

    Args:
        num_examples (int): Number of examples (sequences) to generate.
        sequence_len (int): Length of each sequence.
        num_classes (int): Number of possible integer classes (ranging from `0` to `num_classes - 1`).
                           Must be at least `1`.
        seed (int): Random seed for reproducibility.

    Returns:
        TensorDataset:
            - Inputs: (num_examples, sequence_len)
            - Targets: (num_examples, sequence_len)
    """
    torch.manual_seed(seed)

    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")
    if sequence_len < 1:
        raise ValueError("sequence_len must be at least 1.")

    # Generate input sequences: integers sampled uniformly from [0, num_classes)
    inputs = torch.randint(0, num_classes, (num_examples, sequence_len), dtype=torch.long)

    # Initialize the targets tensor
    targets = torch.zeros_like(inputs)

    # Initialize counts tensor
    counts = torch.zeros((num_examples, num_classes), dtype=torch.float)

    # Iterate over each time step
    for j in range(sequence_len):
        current_elements = inputs[:, j]
        counts[torch.arange(num_examples), current_elements] += 1
        # Determine the mode for each example
        targets[:, j] = torch.argmax(counts, dim=1)

    return TensorDataset(inputs, targets)
