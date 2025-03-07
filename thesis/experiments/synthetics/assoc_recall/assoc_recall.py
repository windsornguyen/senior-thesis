import torch
from torch.utils.data import TensorDataset


# def generate_assoc_recall(
#     num_examples: int = 1000,
#     sequence_len: int = 512,
#     vocab_size: int = 64,
#     num_pairs: int = 2,
#     seed: int = 1746,
# ) -> TensorDataset:
#     """
#     Generate synthetic sequences for testing a model's ability to perform long-range associative recall
#     of key-value pairs in the presence of intervening information.

#     Task Description:
#         - Each sequence contains unique key-value pairs presented sequentially
#         - After presenting all pairs, one key is randomly selected and presented again
#         - The model must recall the correct value associated with the repeated key
#         - All other positions are filled with random tokens to test robustness to interference
#         - Unlike in-context recall, this tests pure associative memory rather than pattern learning

#     Sequence Structure:
#         - Format: [START, k1,v1, k2,v2, ..., kn,vn, k_probe, random_tokens..., END]
#         - Special tokens: START, END, PAD
#         - Each key appears exactly once in the key-value section
#         - One key (k_probe) is repeated later to test recall
#         - Random tokens fill remaining positions to test robustness

#     Target Structure:
#         - All positions are masked with -1 (ignore_index) except:
#         - At the probe position, the target is the value originally paired with the probe key
#         - Example target: [-1, -1, -1, -1, v2, -1, -1] if k2 was the probe key

#     Args:
#         num_examples (int): Number of sequences to generate
#         sequence_len (int): Length of each sequence
#         vocab_size (int): Size of vocabulary for keys and values
#                          (excluding special tokens START, END, PAD)
#         num_pairs (int): Number of key-value pairs to include in each sequence
#                         If this exceeds available space, places as many as possible
#         seed (int): Random seed for reproducibility

#     Returns:
#         TensorDataset: Contains:
#             - inputs: (num_examples, sequence_len) tensor of input tokens
#             - targets: (num_examples, sequence_len) tensor where all positions
#                       are -1 except the position after the probe key

#     Example:
#         A typical sequence might look like:
#         Inputs:  [START, k1,v1, k2,v2, k1, rand,rand,rand, END]
#         Targets: [-1,   -1,-1, -1,-1, -1, v1,  -1, -1,  -1]
#         where k1 is the probe key and v1 is its associated value
#     """
#     torch.manual_seed(seed)

#     START, END, PAD = vocab_size, vocab_size + 1, vocab_size + 2

#     inputs = torch.full((num_examples, sequence_len), PAD, dtype=torch.long)
#     targets = torch.full((num_examples, sequence_len), -1, dtype=torch.long)

#     for i in range(num_examples):
#         inputs[i, 0] = START
#         idx = 1

#         # Ensure unique keys by sampling from a random permutation of [0, vocab_size)
#         # If num_pairs > vocab_size, we won't be able to pick unique keys. Just place what we can.
#         # We'll take the first 'num_pairs' from a permutation for keys
#         perm = torch.randperm(vocab_size)
#         chosen_keys = perm[:num_pairs]
#         chosen_values = torch.randint(0, vocab_size, (num_pairs,))

#         pairs = list(zip(chosen_keys.tolist(), chosen_values.tolist()))

#         placed_pairs = []
#         for key, value in pairs:
#             if idx + 1 >= sequence_len - 1:
#                 # Not enough space for this pair
#                 break
#             inputs[i, idx] = key
#             inputs[i, idx + 1] = value
#             placed_pairs.append((key, value))
#             idx += 2

#         if placed_pairs and idx < sequence_len - 1:
#             # Choose a random placed pair to recall
#             recall_idx = torch.randint(0, len(placed_pairs), (1,)).item()
#             recall_key, recall_value = placed_pairs[recall_idx]
#             inputs[i, idx] = recall_key
#             targets[i, idx] = recall_value
#             idx += 1
#         elif num_pairs == 0:
#             # no pairs expected, do nothing special
#             pass
#         # else no pairs placed, so no recall key

#         # Fill the rest with random tokens
#         while idx < sequence_len - 1:
#             inputs[i, idx] = torch.randint(0, vocab_size, (1,)).item()
#             idx += 1

#         inputs[i, -1] = END

#     return TensorDataset(inputs, targets)


def generate_assoc_recall(
    num_examples: int = 100000,
    sequence_len: int = 128,
    vocab_size: int = 8192,
    num_pairs: int = 4,
    random_non_queries: bool = True,
    num_queries: int = 3,
    seed: int = 1746,
):
    """
    Generates synthetic data for the associative recall task.

    In this task the model is given a sequence that starts with key–value pairs,
    followed by additional tokens where one or more query keys are inserted at random
    positions. The model must recall the corresponding value for each queried key.
    All positions in the target are set to -100 (to be ignored by the loss) except
    at the query positions, where the target is the value associated with the query key.
    The final inputs and targets are obtained by shifting the initial sequences by one token.

    Args:
        num_examples (int): Number of examples to generate.
        sequence_len (int): The length of each input sequence. Must be even and at least 2*num_pairs + num_queries.
        vocab_size (int): Size of the vocabulary. Must be greater than sequence_len.
        num_pairs (int): Number of key–value pairs per example.
        random_non_queries (bool): If True, filler (zero) tokens in non-query positions are replaced with random tokens.
        num_queries (int): Number of queries per example (i.e. number of key–value pairs to recall). A good default is 1.
        seed (int): Seed for reproducibility.

    Returns:
        TensorDataset: Contains:
            - inputs: Tensor of shape [num_examples, sequence_len] containing the input sequence.
            - targets: Tensor of shape [num_examples, sequence_len] containing the target sequence,
              where non-query positions are filled with -100.
    """
    # Basic sanity checks.
    assert sequence_len % 2 == 0, "sequence_len must be even"
    assert vocab_size > sequence_len, "vocab_size must be greater than sequence_len"
    assert num_pairs * 2 + num_queries <= sequence_len, "sequence_len must be >= 2*num_pairs + num_queries"

    torch.manual_seed(seed)

    # The first part of the sequence is reserved for key–value pairs.
    context_size = num_pairs * 2

    # Create unique keys for each example.
    key_vocab_size = vocab_size // 2  # keys come from the first half (skipping 0 for clarity)
    possible_keys = torch.arange(1, key_vocab_size, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_keys = torch.rand(num_examples, key_vocab_size - 1)
    _, key_perm = rand_keys.sort(dim=1)
    keys = possible_keys.gather(1, key_perm[:, :num_pairs])

    # Create corresponding values from the second half of the vocabulary.
    possible_values = torch.arange(key_vocab_size, vocab_size, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_values = torch.rand(num_examples, vocab_size - key_vocab_size)
    _, value_perm = rand_values.sort(dim=1)
    values = possible_values.gather(1, value_perm[:, :num_pairs])

    # Build the key–value sequence (keys in even positions, values in odd positions).
    kvs = torch.empty(num_examples, context_size, dtype=torch.long)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Initialize inputs and targets with length sequence_len + 1 to account for shifting.
    inputs = torch.zeros(num_examples, sequence_len + 1, dtype=torch.long)
    targets = torch.full((num_examples, sequence_len + 1), -100, dtype=torch.long)

    # Insert the key–value pairs at the beginning.
    inputs[:, :context_size] = kvs

    # Prepare advanced indexing: rows for each example.
    rows = torch.arange(num_examples, dtype=torch.long).unsqueeze(1).expand(-1, num_queries)

    # Sample key–value pair indices (without replacement) for the queries.
    possible_idx = torch.arange(num_pairs, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_idx = torch.rand(num_examples, num_pairs)
    _, idx_perm = rand_idx.sort(dim=1)
    chosen_idxs = possible_idx.gather(1, idx_perm[:, :num_queries])

    # Select query keys and corresponding target values.
    queries = keys.gather(1, chosen_idxs)
    query_labels = values.gather(1, chosen_idxs)

    # Randomly choose positions up to sequence_len - 1
    pos_choices = torch.arange(context_size, sequence_len, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_pos = torch.rand(num_examples, sequence_len - context_size)
    _, pos_perm = rand_pos.sort(dim=1)
    query_pos = pos_choices.gather(1, pos_perm[:, :num_queries])

    # Insert queries into inputs and their labels into targets at query_pos + 1
    inputs[rows, query_pos] = queries
    targets[rows, query_pos + 1] = query_labels  # Shift target to next position

    # Shift inputs/targets by one to get final shapes
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]

    # Optionally replace filler zeros (which are not queries) with random tokens.
    if random_non_queries:
        mask = inputs == 0
        if mask.any():
            inputs[mask] = torch.randint(0, vocab_size, (mask.sum().item(),), dtype=torch.long)

    return TensorDataset(inputs, targets)
