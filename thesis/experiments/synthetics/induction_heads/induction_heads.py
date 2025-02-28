import torch

from torch.utils.data import TensorDataset


def generate_induction_heads(
    num_examples: int = 1000,
    sequence_len: int = 512,
    vocab_size: int = 64,
    min_prefix_len: int = 2,
    max_prefix_len: int = 5,
    min_pattern_len: int = 2,
    max_pattern_len: int = 5,
    num_patterns: int = 1,
    seed: int = 1746,
) -> TensorDataset:
    """
    Generates synthetic sequences for the Induction Heads Task.

    Args:
        num_examples: Number of sequences to generate.
        sequence_len: Length of each sequence.
        vocab_size: Size of the vocabulary (excluding special tokens).
        min_prefix_len: Minimum length of the prefix.
        max_prefix_len: Maximum length of the prefix.
        min_pattern_len: Minimum length of the pattern.
        max_pattern_len: Maximum length of the pattern.
        num_patterns: Number of patterns in each sequence.
        seed: Random seed for reproducibility.

    Returns:
        A PyTorch TensorDataset with input sequences and corresponding targets.

        Inputs shape: (num_examples, sequence_len)
        Targets shape: (num_examples, sequence_len)
    """
    torch.manual_seed(seed)

    # Special tokens
    START, END, PAD = vocab_size, vocab_size + 1, vocab_size + 2

    # Initialize inputs and targets
    inputs = torch.full((num_examples, sequence_len), PAD, dtype=torch.long)
    targets = torch.full((num_examples, sequence_len), -1, dtype=torch.long)

    for i in range(num_examples):
        inputs[i, 0] = START  # Start token
        idx = 1

        for pattern_idx in range(num_patterns):
            # Randomly determine prefix and pattern lengths
            prefix_len = torch.randint(min_prefix_len, max_prefix_len + 1, (1,)).item()
            pattern_len = torch.randint(min_pattern_len, max_pattern_len + 1, (1,)).item()
            total_len = prefix_len + pattern_len

            # Check if there's enough space for two occurrences and a gap
            remaining_space = sequence_len - idx - (total_len * 2 + 1)
            if remaining_space < 0:
                print(f"Insufficient space for pattern {pattern_idx + 1}, stopping further pattern insertion.")
                break

            # Generate random prefix and pattern
            prefix = torch.randint(0, vocab_size, (prefix_len,), dtype=torch.long)
            pattern = torch.randint(0, vocab_size, (pattern_len,), dtype=torch.long)

            # First occurrence: prefix + pattern
            inputs[i, idx : idx + prefix_len] = prefix
            inputs[i, idx + prefix_len : idx + total_len] = pattern
            idx += total_len

            # Random gap (at least 1 token)
            if pattern_idx < num_patterns - 1:
                max_gap = (sequence_len - idx - total_len - 1) // (num_patterns - pattern_idx - 1)
                gap = torch.randint(1, max_gap + 1, (1,)).item()
            else:
                gap = 1  # Minimal gap for the last pattern

            idx += gap

            # Second occurrence: same prefix + same pattern
            inputs[i, idx : idx + prefix_len] = prefix
            inputs[i, idx + prefix_len : idx + total_len] = pattern

            # Assign targets for pattern tokens only (not the prefix)
            targets[i, idx + prefix_len : idx + total_len] = pattern
            idx += total_len

        # Fill remaining positions with random tokens
        while idx < sequence_len - 1:
            inputs[i, idx] = torch.randint(0, vocab_size, (1,)).item()
            idx += 1

        inputs[i, -1] = END  # End token

    return TensorDataset(inputs, targets)
