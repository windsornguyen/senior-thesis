"""
HashHop task adapted from https://github.com/magicproduct/hash-hop/
Read their blog post here: https://magic.dev/blog/100m-token-context-windows
"""

import math
import random
import copy
import string
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import TensorDataset


# --- Simple Character-level Tokenizer for Synthetic Experiments ---
def build_char_tokenizer() -> Dict[str, int]:
    # Define a fixed vocabulary: letters, digits, punctuation, space and newline.
    vocab = string.ascii_letters + string.digits + string.punctuation + " \n"
    # Reserve index 0 for padding.
    return {ch: i + 1 for i, ch in enumerate(vocab)}


def encode_text(text: str, token2idx: Dict[str, int]) -> List[int]:
    return [token2idx.get(ch, 0) for ch in text]


def pad_sequences(seq_list: List[List[int]], pad_value: int = 0) -> torch.Tensor:
    max_len = max(len(seq) for seq in seq_list)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in seq_list]
    return torch.tensor(padded, dtype=torch.long)


# --- Prompt Templates ---
TASK_PROMPT_TEMPLATE = "{VARIABLE_LIST}"
# For synthetic experiments, we simplify the completion template.
COMPLETION_TEMPLATE = "HOPS={HOPS}\nCoT={COT}\n{COMPLETION}"


# --- Data Structure for a Single Multi-Hop Sample ---
@dataclass
class MultiHopSample:
    prompt: str
    """The prompt including shuffled hash pairs."""
    completion: str
    """The expected output chain (either full chain-of-thought or compressed)."""
    targets: Dict[str, str]
    """Mapping from query hash to ground-truth target (for evaluation)."""


# --- Multi-Hop Evaluation Generator ---
class MultiHopEval:
    @staticmethod
    def make_one(
        n_chars_problem: int,
        num_queries: int,
        hops: int,
        hash_pair_str_length: int,
        chain_of_thought: bool,
    ) -> MultiHopSample:
        """
        Generate one multi-hop sample.

        Args:
            n_chars_problem: Approximate target prompt size (in characters).
            num_queries: Number of query pairs to include in the completion.
            hops: Number of hops (levels) in the chain.
            hash_pair_str_length: Number of characters for each hash string.
            chain_of_thought: If True, include full intermediate hops; else only first and final hash.

        Returns:
            A MultiHopSample with a prompt, a completion string, and a dictionary of targets.
        """
        # Estimate how many hash pairs (chains) to produce.
        chars_per_hash_pair = (hash_pair_str_length * 2 + 3) * hops
        n_chains = math.ceil(n_chars_problem / chars_per_hash_pair)

        levels = MultiHopEval._make_levels(n=n_chains, hops=hops, string_length=hash_pair_str_length)

        # Build prompt lines: each level produces a list of assignments.
        lines = []
        for i, level in enumerate(levels):
            if i == len(levels) - 1:
                # For the last level, quote the values.
                lines.extend([f"{k} = '{v}'" for k, v in level.items()])
            else:
                lines.extend([f"{k} = {v}" for k, v in level.items()])

        # Build the target chain.
        all_query_pairs = copy.deepcopy(levels[0])
        all_query_strings = {k: "" for k in all_query_pairs.keys()}
        if hops > 1:
            for i, level in enumerate(levels[1:]):
                if chain_of_thought:
                    if i == 0:
                        all_query_strings = {k: f"{v}" for k, v in all_query_pairs.items()}
                    else:
                        all_query_strings = {
                            k: f"{all_query_strings[k]} = {v}" if all_query_strings[k] != "" else v
                            for k, v in all_query_pairs.items()
                        }
                all_query_pairs = {k: level[v] for k, v in all_query_pairs.items()}
        if chain_of_thought:
            all_query_strings = {
                k: f"{all_query_strings[k]} = {v}" if all_query_strings[k] != "" else v
                for k, v in all_query_pairs.items()
            }
        else:
            all_query_strings = all_query_pairs

        # Shuffle the prompt lines and query pairs.
        random.shuffle(lines)
        all_query_strings = shuffle_dict(all_query_strings)

        assert num_queries <= len(
            all_query_strings
        ), f"Requested {num_queries} queries, but only {len(all_query_strings)} available."

        completion = COMPLETION_TEMPLATE.format(
            COMPLETION="\n".join([f"{k} = '{v}'" for k, v in list(all_query_strings.items())[:num_queries]]),
            HOPS=hops,
            COT=chain_of_thought,
        )
        prompt = TASK_PROMPT_TEMPLATE.format(VARIABLE_LIST="\n".join(lines))

        return MultiHopSample(prompt=prompt, completion=completion, targets=all_query_strings)

    @staticmethod
    def _make_levels(n: int, hops: int, string_length: int) -> List[Dict[str, str]]:
        """
        Generate levels of hash mappings.

        Each level is a dictionary mapping a random hash string to another random hash string.
        """
        levels = [
            {make_random_string(length=string_length): make_random_string(length=string_length) for _ in range(n)}
        ]
        for _ in range(hops - 1):
            levels.append({v: make_random_string(length=string_length) for v in levels[-1].values()})
        return levels


# --- Dataset Generation ---
def generate_hashhop_dataset(
    num_examples: int,
    n_chars_problem: int,
    num_queries: int,
    hops: int,
    hash_pair_str_length: int,
    chain_of_thought: bool,
    seed: int = 1746,
) -> TensorDataset:
    """
    Generate a HashHop dataset for synthetic long-context experiments.

    Each example consists of a prompt (a shuffled list of hash assignments) and a target completion
    (a chain representing the multi-hop mapping). The strings are encoded at the character level.

    Args:
        num_examples: Number of examples to generate.
        n_chars_problem: Approximate prompt size (in characters).
        num_queries: Number of query pairs to include in the completion.
        hops: Number of hops in the chain.
        hash_pair_str_length: Length of each hash string.
        chain_of_thought: Whether to include full chain-of-thought (all intermediate hops).
        seed: Random seed for reproducibility.

    Returns:
        TensorDataset containing:
            inputs: Tensor of shape (num_examples, max_input_length) of token IDs.
            targets: Tensor of shape (num_examples, max_target_length) of token IDs.
    """
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build our simple tokenizer.
    token2idx = build_char_tokenizer()

    # Generate samples.
    samples = [
        MultiHopEval.make_one(
            n_chars_problem=n_chars_problem,
            num_queries=num_queries,
            hops=hops,
            hash_pair_str_length=hash_pair_str_length,
            chain_of_thought=chain_of_thought,
        )
        for _ in range(num_examples)
    ]

    # Encode prompts and completions.
    input_ids_list = [encode_text(sample.prompt, token2idx) for sample in samples]
    target_ids_list = [encode_text(sample.completion, token2idx) for sample in samples]

    # Pad sequences.
    inputs_tensor = pad_sequences(input_ids_list, pad_value=0)
    targets_tensor = pad_sequences(target_ids_list, pad_value=0)

    return TensorDataset(inputs_tensor, targets_tensor)


# --- Helper Functions ---
def make_random_string(length: int) -> str:
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return "".join(random.choices(alphabet, k=length))


def shuffle_dict(to_shuffle: dict) -> dict:
    items = list(to_shuffle.items())
    random.shuffle(items)
    return dict(items)
