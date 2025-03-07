import random
import numpy as np
import torch
from torch.utils.data import TensorDataset


# def generate_mqar(
#     num_examples: int = 1000,
#     sequence_len: int = 512,
#     vocab_size: int = 50000,  # Typical LM vocab sizes (must be even)
#     num_pairs: int = 16,  # D: number of key–value pairs per example
#     alpha: float = 2.0,  # Power-law exponent for sampling query positions (α > 1)
#     seed: int = 1746,
# ) -> TensorDataset:
#     """
#     Generates synthetic sequences for the Multi-Query Associative Recall (MQAR) Task, as described
#     in "Zoology: Measuring and Improving Recall in Efficient Language Models" (Arora et al.). This task
#     is designed to test a model's ability to recall multiple associations over long contexts by forcing
#     non-linear (discrete) recall of key–value pairs. The key challenges include handling a power-law
#     distribution of token-interaction distances and a large vocabulary size relative to model dimension.

#     The procedure is as follows (see Procedure 1 in the paper):

#     1. Partition the vocabulary C into two halves:
#        - Keys: tokens 0 to (vocab_size/2 - 1)
#        - Values: tokens (vocab_size/2) to (vocab_size - 1)
#        (Directly from the paper.)

#     2. Generate a random key–value mapping by pairing each key with a random value, then sub-select D = num_pairs
#        unique key–value pairs.
#        (Procedure 1, steps 1–3.)

#     3. Reserve the first 2D tokens (positions 0 to 2*num_pairs – 1) to write the D key–value pairs consecutively,
#        where for each pair j, the key is placed at position 2*j and the value at position 2*j+1.
#        (Procedure 1, step 4.)

#     4. Define the candidate region for query placement as the positions from 2D to sequence_len – 1 (the final
#        token is reserved for the END marker). For candidate offsets j = 0,...,L–1, where L = sequence_len – 1 – 2D,
#        assign weights proportional to (j + 1)^(-alpha) (i.e. a discrete power-law distribution).
#        (Procedure 1, step 5.)

#     5. Sample D distinct query positions (without replacement) from the candidate region using the computed
#        power-law probabilities. At each sampled position, reinsert the key from the corresponding key–value pair,
#        and set the target at that position to be the associated value. All other target positions are set to -1.

#     6. Fill any remaining positions in the sequence with a filler token.

#     7. Reserve special tokens:
#        - START token is placed at position 0.
#        - END token is placed at the final position.

#     Note: To allow at least one candidate token for query placement (and to avoid an empty candidate region),
#           the sequence length must be at least 2*num_pairs + 2.

#     Args:
#         num_examples (int): Number of sequences to generate.
#         sequence_len (int): Total length of each sequence (must be >= 2*num_pairs + 2).
#         vocab_size (int): Total vocabulary size; keys are drawn from [0, vocab_size/2) and values from [vocab_size/2, vocab_size).
#                           Must be even.
#         num_pairs (int): Number of key–value pairs (D) per example.
#         alpha (float): Power-law exponent for sampling query positions (α > 1).
#         seed (int): Random seed for reproducibility.

#     Returns:
#         TensorDataset: A dataset containing:
#           - inputs: Tensor of shape (num_examples, sequence_len) with token IDs.
#           - targets: Tensor of shape (num_examples, sequence_len) with the target value at query positions and -1 elsewhere.
#     """
#     # Set seeds for reproducibility.
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     # Check minimal sequence length.
#     if sequence_len < 2 * num_pairs + 2:
#         raise ValueError("sequence_len must be at least 2 * num_pairs + 2")

#     # --- Special tokens ---
#     # Define special tokens outside the key/value ranges.
#     START = vocab_size  # Start token (placed at index 0)
#     END = vocab_size + 1  # End token (placed at the final index)
#     FILLER = vocab_size + 2  # Filler token for unused positions
#     IGNORE = -1  # Target ignore index

#     # --- Step 1: Partition vocabulary into keys and values ---
#     if vocab_size % 2 != 0:
#         raise ValueError("vocab_size must be even.")
#     half_vocab = vocab_size // 2
#     keys_pool = list(range(0, half_vocab))
#     values_pool = list(range(half_vocab, vocab_size))

#     # --- Step 2: Generate key-value mapping and select D pairs ---
#     kv_dict = {k: random.choice(values_pool) for k in keys_pool}  # Pair each key with a random value.
#     D = min(num_pairs, len(keys_pool))
#     selected_keys = random.sample(keys_pool, D)
#     pairs = [(k, kv_dict[k]) for k in selected_keys]

#     # --- Initialize input and target tensors ---
#     inputs = torch.full((num_examples, sequence_len), FILLER, dtype=torch.long)
#     targets = torch.full((num_examples, sequence_len), IGNORE, dtype=torch.long)

#     # --- Step 7: Insert special tokens ---
#     inputs[:, 0] = START
#     inputs[:, -1] = END

#     # --- Step 3: Insert key-value pairs in the first 2D positions ---
#     for i in range(num_examples):
#         for j, (k, v) in enumerate(pairs):
#             pos = 2 * j
#             inputs[i, pos] = k  # Insert key
#             inputs[i, pos + 1] = v  # Insert value

#         # --- Step 4: Define candidate region for query placement ---
#         candidate_start = 2 * D
#         candidate_end = sequence_len - 1  # Reserve final position for END.
#         num_candidates = candidate_end - candidate_start  # Number of candidate positions (>= 1 by our check).

#         # --- Step 4 (continued): Compute power-law weights over candidate offsets ---
#         candidate_offsets = np.arange(num_candidates)  # Offsets: 0,1,...,num_candidates-1
#         weights = (candidate_offsets + 1) ** (-alpha)  # Weight ∝ (j+1)^(-alpha)
#         weights = weights / weights.sum()  # Normalize

#         # --- Step 5: Sample D distinct query positions from the candidate region ---
#         sampled_offsets = np.random.choice(candidate_offsets, size=D, replace=False, p=weights)
#         # Map offsets to absolute positions in the sequence.
#         query_positions = candidate_start + sampled_offsets
#         query_positions.sort()  # Optional: sort to impose natural order.

#         # --- Step 5 (continued): Insert second occurrences (queries) and set targets ---
#         for j, (k, v) in enumerate(pairs):
#             pos = int(query_positions[j])
#             inputs[i, pos] = k  # Insert the query (repeated key)
#             targets[i, pos] = v  # Set the target to the associated value

#     return TensorDataset(inputs, targets)

def generate_mqar(
    num_examples: int = 100000,
    sequence_len: int = 512,
    vocab_size: int = 8192,  # Must be even.
    num_pairs: int = 64,
    alpha: float = 2.0,
    seed: int = 1746,
) -> TensorDataset:
    assert sequence_len % 2 == 0, "sequence_len must be even"
    assert vocab_size > sequence_len, "vocab_size must be greater than sequence_len"
    assert num_pairs * 4 <= sequence_len, "sequence_len must be >= 4 * num_pairs"

    torch.manual_seed(seed)

    # Context: key-value pairs occupy the first context_size tokens.
    context_size = num_pairs * 2

    # Define key and value vocab ranges.
    key_vocab_size = vocab_size // 2  # keys in [1, key_vocab_size-1]
    
    # Vectorized sampling without replacement for keys.
    possible_keys = torch.arange(1, key_vocab_size, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_keys = torch.rand(num_examples, key_vocab_size - 1)
    _, key_indices = rand_keys.sort(dim=1)
    keys = possible_keys.gather(1, key_indices[:, :num_pairs])

    # Similarly for values: sample from [key_vocab_size, vocab_size-1]
    possible_values = torch.arange(key_vocab_size, vocab_size, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_values = torch.rand(num_examples, vocab_size - key_vocab_size)
    _, value_indices = rand_values.sort(dim=1)
    values = possible_values.gather(1, value_indices[:, :num_pairs])

    # Create the key-value sequence tensor (shape: [num_examples, context_size]).
    kvs = torch.empty(num_examples, context_size, dtype=torch.long)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Compute power-law distribution for query gaps.
    space = (sequence_len - context_size) // 2  # available gap positions
    r = torch.arange(1, space + 1, dtype=torch.float)
    p = alpha * (r ** (alpha - 1))
    p /= p.sum()  # normalize

    # Repeat probability vector for each example.
    p_batch = p.unsqueeze(0).expand(num_examples, -1)  # shape: [num_examples, space]
    # For each example, sample num_pairs unique gap positions using torch.multinomial.
    gaps = torch.multinomial(p_batch, num_samples=num_pairs, replacement=False)  # shape: [num_examples, num_pairs]
    
    # Build queries tensor: length is (sequence_len - context_size + 1)
    queries_len = sequence_len - context_size + 1
    queries = torch.zeros(num_examples, queries_len, dtype=torch.long)
    # Compute target indices: gaps are multiplied by 2.
    target_indices = gaps * 2
    # Scatter the keys into the queries tensor at the specified indices.
    queries.scatter_(1, target_indices, keys)

    # Concatenate kvs and queries to get a sequence of length (context_size + queries_len) = sequence_len + 1.
    examples = torch.cat([kvs, queries], dim=1)

    # Prepare labels: initialize with ignore value (-100).
    labels_full = torch.full((num_examples, sequence_len + 1), -100, dtype=torch.long)
    # Determine positions for placing the values: shift indices by context_size + 1.
    label_indices = target_indices + context_size + 1
    labels_full.scatter_(1, label_indices, values)

    # Shift inputs/labels by one to get final shapes.
    inputs = examples[:, :-1]
    labels = labels_full[:, 1:]

    # Replace filler zeros in inputs with random tokens from [0, vocab_size).
    mask = inputs == 0
    if mask.any():
        inputs[mask] = torch.randint(0, vocab_size, (mask.sum().item(),), dtype=torch.long)

    return TensorDataset(inputs, labels)
