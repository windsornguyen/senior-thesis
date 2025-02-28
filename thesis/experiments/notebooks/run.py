import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from itertools import chain
import math
import time
from tqdm import tqdm
import random
import numpy as np
from torch.distributed._tensor import DTensor
import matplotlib.pyplot as plt

# Set high-precision matmul for float32 (if supported)
torch.set_float32_matmul_precision("high")


##############################################
# Data Generation Functions
##############################################


def generate_copy(
    num_examples: int = 10,
    num_categories: int = 10,
    copy_len: int = 10,
    blank_len: int = 10,
    selective: bool = False,
    one_hot: bool = True,
    seed: int = 1_337,
    dtype: torch.dtype = torch.bfloat16,
) -> TensorDataset:
    """
    Generate a copy task dataset inspired by Arjovsky, Shah, and Bengio (2016).

    Task Description:
      - Input sequence: [copy_sequence][pre_delim_blanks][delimiter][post_delim_blanks]
      - Output sequence: [blank_tokens][copy_sequence]

    The task requires remembering a categorical sequence for a variable number of time steps.

    Args:
      num_examples: Number of examples to generate.
      num_categories: Number of token categories.
        - Categories 0 to num_categories-3: Tokens to be copied.
        - Category num_categories-2: Blank token.
        - Category num_categories-1: Delimiter token.
      copy_len: Length of the sequence to be copied.
      blank_len: Number of blank tokens after the delimiter in the input sequence.
      selective: If True, inserts blank tokens between the tokens in the copied sequence (pre-delimiter).
      one_hot: Whether to one-hot encode the inputs and outputs.
      seed: Random seed for reproducibility.
      dtype: Data type for one-hot encoded tensors.

    Returns:
      A TensorDataset with:
        - inputs: Shape (num_examples, seq_len)
          where seq_len = copy_len + (num_categories-1) + 1 + blank_len
        - targets: Shape (num_examples, num_categories + blank_len + copy_len)
          consisting of blank tokens followed by the copied sequence.
    """
    torch.manual_seed(seed)

    # Define special tokens
    blank_char = num_categories - 2  # Blank token
    delim_char = num_categories - 1  # Delimiter token

    # Generate the sequence to be copied
    to_copy = torch.randint(0, blank_char, (num_examples, copy_len))
    pre_delim_blanks = torch.full((num_examples, num_categories - 1), blank_char)
    delim = torch.full((num_examples, 1), delim_char)
    post_delim_blanks = torch.full((num_examples, blank_len), blank_char)

    if selective:
        # In selective mode, insert blanks randomly within the copied sequence.
        def insert_pre_delim_blanks(row: torch.Tensor) -> torch.Tensor:
            pre_delim_len = copy_len + num_categories - 1
            insert_positions = torch.randperm(pre_delim_len)[: num_categories - 1]
            inserted_row = torch.full((pre_delim_len,), blank_char)
            mask = torch.ones(pre_delim_len, dtype=torch.bool)
            mask[insert_positions] = False  # Mark positions where the copy sequence will be inserted
            inserted_row[mask] = row  # Insert copied tokens
            return inserted_row

        inputs = torch.stack([insert_pre_delim_blanks(row) for row in to_copy])
    else:
        # Simply concatenate the copy sequence with the pre-delimiter blanks.
        inputs = torch.cat((to_copy, pre_delim_blanks), dim=1)

    # Append delimiter and post-delimiter blanks
    inputs = torch.cat((inputs, delim, post_delim_blanks), dim=1)

    # Construct the target: a block of blank tokens followed by the copy sequence.
    blank_output = torch.full((num_examples, num_categories + blank_len), blank_char)
    outputs = torch.cat((blank_output, to_copy), dim=1)

    if one_hot:
        inputs = F.one_hot(inputs, num_classes=num_categories).to(dtype)
        outputs = F.one_hot(outputs, num_classes=num_categories).to(dtype)

    return TensorDataset(inputs, outputs)


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
      A TensorDataset with:
        - Inputs shape: (num_examples, sequence_len)
        - Targets shape: (num_examples, sequence_len)
          (Targets are set to -1 for positions that are not part of a pattern.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define special tokens
    START, END, PAD = vocab_size, vocab_size + 1, vocab_size + 2

    # Initialize inputs and targets
    inputs = torch.full((num_examples, sequence_len), PAD, dtype=torch.long)
    targets = torch.full((num_examples, sequence_len), -1, dtype=torch.long)

    for i in range(num_examples):
        inputs[i, 0] = START  # Set start token
        idx = 1

        for pattern_idx in range(num_patterns):
            prefix_len = torch.randint(min_prefix_len, max_prefix_len + 1, (1,)).item()
            pattern_len = torch.randint(min_pattern_len, max_pattern_len + 1, (1,)).item()
            total_len = prefix_len + pattern_len

            # Check if there's enough space for two occurrences and a gap
            remaining_space = sequence_len - idx - (total_len * 2 + 1)
            if remaining_space < 0:
                print(f"Insufficient space for pattern {pattern_idx + 1}, stopping further pattern insertion.")
                break

            # Generate prefix and pattern
            prefix = torch.randint(0, vocab_size, (prefix_len,), dtype=torch.long)
            pattern = torch.randint(0, vocab_size, (pattern_len,), dtype=torch.long)

            # First occurrence: prefix + pattern
            inputs[i, idx : idx + prefix_len] = prefix
            inputs[i, idx + prefix_len : idx + total_len] = pattern
            idx += total_len

            # Compute random gap safely (if possible)
            if pattern_idx < num_patterns - 1:
                max_gap = (sequence_len - idx - total_len - 1) // (num_patterns - pattern_idx - 1)
                gap = torch.randint(1, max_gap + 1, (1,)).item() if max_gap >= 1 else 1
            else:
                gap = 1
            idx += gap

            # Second occurrence: same prefix + same pattern
            inputs[i, idx : idx + prefix_len] = prefix
            inputs[i, idx + prefix_len : idx + total_len] = pattern

            # Set targets only for the pattern (not the prefix)
            targets[i, idx + prefix_len : idx + total_len] = pattern
            idx += total_len

        # Fill remaining positions with random tokens (except last position)
        while idx < sequence_len - 1:
            inputs[i, idx] = torch.randint(0, vocab_size, (1,)).item()
            idx += 1
        inputs[i, -1] = END  # Set end token

    return TensorDataset(inputs, targets)


def generate_document_similarity(
    num_examples: int = 10,
    num_documents: int = 10,
    num_elements: int = 10,
    top_k: int = 2,
    seed: int = 1_337,
    dtype: torch.dtype = torch.bfloat16,
) -> TensorDataset:
    """
    Generate a dataset for the cosine similarity task.
    The goal is to identify the pair(s) of documents with the highest cosine similarity.

    Args:
      num_examples: Number of examples (sets of documents) to generate.
      num_documents: Number of documents per example.
      num_elements: Number of elements in each document.
      top_k: Number of top similar document pairs to identify.
      seed: Random seed for reproducibility.
      dtype: Data type for the tensors.

    Returns:
      A TensorDataset with:
        - Inputs: Shape (num_examples, num_documents, num_elements)
        - Targets: Shape (num_examples, top_k, 2) representing the indices of the document pairs with highest cosine similarity.
    """
    torch.manual_seed(seed)

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if num_documents < 2:
        raise ValueError("num_documents must be at least 2 to form pairs.")
    max_topk = num_documents * (num_documents - 1) // 2
    if top_k > max_topk:
        raise ValueError(f"top_k={top_k} exceeds the maximum number of unique document pairs ({max_topk}).")

    # Generate random documents and normalize them
    inputs = torch.randn((num_examples, num_documents, num_elements), dtype=dtype)
    normalized_inputs = F.normalize(inputs, p=2, dim=2)
    cosine_similarity = normalized_inputs @ normalized_inputs.transpose(1, 2)

    triu_indices = torch.triu_indices(num_documents, num_documents, offset=1)
    sim_pairs = cosine_similarity[:, triu_indices[0], triu_indices[1]]  # (num_examples, num_pairs)
    _, topk_indices = torch.topk(sim_pairs, top_k, dim=1, largest=True, sorted=True)
    topk_pairs = triu_indices[:, topk_indices]  # (2, num_examples, top_k)
    targets = topk_pairs.permute(1, 2, 0)  # (num_examples, top_k, 2)

    return TensorDataset(inputs, targets)


##############################################
# Spectral Filtering Utilities
##############################################


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    """
    Compute a Hankel matrix with a specified formula.

    Args:
      seq_len: Length of the sequence (determines matrix size).
      use_hankel_L: Whether to use an alternative Hankel formulation.

    Returns:
      A tensor of shape (seq_len, seq_len) representing the Hankel matrix.
    """
    entries = torch.arange(1, seq_len + 1, dtype=torch.float32)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k *= sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


def compute_dimensions(n: int) -> tuple[int, int, int]:
    if n <= 2:
        raise ValueError("n must be greater than 2")
    T_prime = (math.ceil(math.sqrt(n - 2))) ** 2 + 2
    sqrt_T_prime = math.ceil(math.sqrt(T_prime - 2))
    k_max = sqrt_T_prime
    return T_prime, sqrt_T_prime, k_max


def get_tensorized_spectral_filters(
    n: int = 8192,
    k: int = 24,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute tensorized spectral filters for a given sequence length and filter count.

    Args:
      n: Sequence length.
      k: Number of filters.
      use_hankel_L: Whether to use an alternative Hankel formulation.
      device: Computation device.
      dtype: Computation dtype.

    Returns:
      A tensor representing the tensorized spectral filters.
    """
    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    Z = get_hankel(sqrt_T_prime)
    sigma, phi = torch.linalg.eigh(Z)
    phi_i = phi[:, -k:] * sigma[-k:] ** 0.25

    if use_hankel_L:
        Z_L = get_hankel(sqrt_T_prime, True)
        sigma_L, phi_L = torch.linalg.eigh(Z_L)
        phi_j = phi_L[:, -k:] * sigma_L[-k:] ** 0.25
    else:
        phi_j = phi_i

    filters = torch.kron(phi_i, phi_j)
    return filters.to(device=device, dtype=dtype)


##############################################
# Model Definitions
##############################################


class Attention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        assert n_embd % n_heads == 0, f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)
        self.o_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        # Compute Q, K, V and reshape for multi-head attention
        Q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)  # (B, heads, S, D)
        K = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Create a causal mask
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()

        # Compute scaled dot-product attention using PyTorch's built-in function
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.hidden_dim = 4 * dim
        self.gate_proj = nn.Linear(dim, self.hidden_dim)
        self.up_proj = nn.Linear(dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        modulated_gate = F.silu(gate)
        up = self.up_proj(x)
        fuse = modulated_gate * up
        outputs = self.down_proj(fuse)
        return self.dropout(outputs)


class AttentionLayer(nn.Module):
    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        self.attn = Attention(n_embd, n_heads)
        self.mlp = MLP(n_embd)
        self.attn_norm = nn.RMSNorm(n_embd)
        self.mlp_norm = nn.RMSNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([AttentionLayer(config.n_embd, config.n_heads) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = x.size()
        positions = torch.arange(seqlen, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


class LearnableSpectralFilters(nn.Module):
    def __init__(
        self, seq_len: int, k: int, use_hankel_L: bool = False, device: torch.device = None, dtype=torch.float32
    ):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        self.use_hankel_L = use_hankel_L
        self.device = device

        # Use the tensorized spectral filters (fixed) as the Q/K basis.
        filters = get_tensorized_spectral_filters(
            n=seq_len,
            k=k,
            use_hankel_L=use_hankel_L,
            device=device,
            dtype=dtype,
        )
        self.filters = nn.Parameter(filters)  # Fixed basis; gradients will not change the spectral design.

    def forward(self) -> torch.Tensor:
        return self.filters


class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, n_embd: int, k: int, use_hankel_L: bool = False, device: torch.device = None):
        super().__init__()
        self.seq_len = seq_len
        self.k = k

        # Fixed spectral filters for Q and K.
        self.Q_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters
        self.K_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters

        # Learnable projections: i_proj projects the input into the spectral basis dimensions;
        # v_proj produces the values that will be combined with K.
        self.i_proj = nn.Linear(n_embd, self.Q_filt.shape[0])
        self.v_proj = nn.Linear(n_embd, self.Q_filt.shape[1])
        self.o_proj = nn.Linear(self.Q_filt.shape[1], n_embd)

        # Learnable decay parameter over the sequence positions.
        self.decay = nn.Parameter(torch.ones(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, dim = x.shape
        assert T == self.seq_len, "Input length must match seq_len"

        # Project input to spectral basis space.
        x_proj = self.i_proj(x)  # [B, T, H] where H = self.Q_filt.shape[0]

        # Compute Q and K using fixed spectral filters.
        Q = torch.einsum("bth,hk->btk", x_proj, self.Q_filt)  # [B, T, F]
        K = torch.einsum("bth,hk->btk", x_proj, self.K_filt)  # [B, T, F]
        V = self.v_proj(x)  # [B, T, P] where P = self.v_proj.out_features

        # Compute an outer product between V and K, then apply a decay.
        Z = torch.einsum("bsp,bsn->bspn", V, K)  # [B, T, P, F]
        decay = self.decay.view(1, T, 1, 1)
        Z = Z * decay

        # Cumulatively sum over time to aggregate information.
        H = torch.cumsum(Z, dim=1)  # [B, T, P, F]
        Y = torch.einsum("btn,btpn->btp", Q, H)  # [B, T, P]

        # Final output projection.
        return self.o_proj(Y)  # [B, T, n_embd]


class SpectralAttentionLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_embd: int,
        k: int,
        dropout: float = 0.0,
        use_hankel_L: bool = False,
        device: torch.device = None,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(n_embd)
        self.mlp_norm = nn.RMSNorm(n_embd)
        self.attn = SpectralAttention(seq_len, n_embd, k, use_hankel_L, device)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Spectron(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                SpectralAttentionLayer(
                    config.seq_len,
                    config.n_embd,
                    config.k,
                    config.dropout,
                    use_hankel_L=config.use_hankel_L,
                    device=config.device,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.size()
        x = self.dropout(self.tok_emb(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


##############################################
# Utility Functions
##############################################


@torch.no_grad()
def check_model_value_range(model: nn.Module, range_val: float = 1e3, std_val: float = 1e3):
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        # If using distributed tensors, convert to local
        if isinstance(param, DTensor):
            param = param.to_local()

        if param.numel() == 0:
            print(f"Model parameter {name} is empty (possibly due to FSDP sharding).")
            continue

        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Model parameter {name} contains NaN or Inf.")

        param_range = param.max() - param.min()
        param_std = param.std()

        if param_range > range_val:
            print(
                f"Model parameter {name} has a suspiciously large range ({param_range}): "
                "please check initialization."
            )

        if param_std > std_val:
            print(
                f"Model parameter {name} has a suspiciously large standard deviation ({param_std}): "
                "please check initialization."
            )

        if (param == 0).all() and "bias" not in name:
            print(f"Model parameter {name} is all zeros; check for missing initialization.")


def format_num(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1000:
        return f"{num / 1000:.2f}K"
    else:
        return str(num)


def count_non_embedding_params(model: nn.Module):
    total_params = 0
    non_embedding_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            if "emb" not in name:
                non_embedding_params += num_params

    print(f"Total parameters: {format_num(total_params)}")
    print(f"Parameters (excluding embeddings): {format_num(non_embedding_params)}")


def set_seed(seed: int = 1746):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set the seed for reproducibility
set_seed(1746)


##############################################
# Configuration
##############################################


class Config:
    def __init__(self, task: str = "induction"):
        self.task = task

        # Common model/training parameters
        self.num_layers = 2
        self.n_heads = 8
        self.n_embd = 128
        self.k = 8
        self.use_hankel_L = False
        self.dropout = 0.0
        self.batch_size = 64
        self.lr = 3e-4
        self.num_epochs = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 1746
        self.torch_compile = True

        # Task-specific parameters
        if self.task == "copy":
            self.num_examples = 1000000
            self.num_categories = 10
            self.copy_len = 10
            self.blank_len = 10
            self.selective = True
            self.vocab_size = self.num_categories
            self.seq_len = self.copy_len + (self.num_categories - 1) + 1 + self.blank_len

        elif self.task == "induction":
            self.num_examples = 10000
            self.induction_vocab_size = 64
            self.min_prefix_len = 2
            self.max_prefix_len = 5
            self.min_pattern_len = 2
            self.max_pattern_len = 5
            self.num_patterns = 1
            self.vocab_size = self.induction_vocab_size
            self.seq_len = 512

        elif self.task == "doc_sim":
            self.num_examples = 10000
            self.num_documents = 10
            self.num_elements = 10
            self.top_k = 2
            self.vocab_size = 128
            self.seq_len = self.num_elements
        else:
            raise ValueError(f"Unknown task: {self.task}")


# -----------------------------
# Dataset & DataLoader Setup
# -----------------------------

config = Config(task="copy")
print(f"Using device: {config.device}")

if config.task == "copy":
    dataset = generate_copy(
        num_examples=config.num_examples,
        num_categories=config.num_categories,
        copy_len=config.copy_len,
        blank_len=config.blank_len,
        selective=config.selective,
        one_hot=False,
        seed=config.seed,
        dtype=torch.long,
    )
elif config.task == "induction":
    dataset = generate_induction_heads(
        num_examples=config.num_examples,
        sequence_len=config.seq_len,
        vocab_size=config.vocab_size,
        min_prefix_len=config.min_prefix_len,
        max_prefix_len=config.max_prefix_len,
        min_pattern_len=config.min_pattern_len,
        max_pattern_len=config.max_pattern_len,
        num_patterns=config.num_patterns,
        seed=config.seed,
    )
elif config.task == "doc_sim":
    dataset = generate_document_similarity(
        num_examples=config.num_examples,
        num_documents=config.num_documents,
        num_elements=config.num_elements,
        top_k=config.top_k,
        seed=config.seed,
        dtype=torch.bfloat16,
    )

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

pin_memory = config.device.type == "cuda"
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, pin_memory=pin_memory)


##############################################
# Model Initialization
##############################################

transformer = Transformer(config).to(config.device)
spectral_transformer = Spectron(config).to(config.device)
check_model_value_range(transformer, range_val=10.0, std_val=1.0)
check_model_value_range(spectral_transformer, range_val=10.0, std_val=1.0)

if config.torch_compile:
    transformer = torch.compile(transformer)
    spectral_transformer = torch.compile(spectral_transformer)

count_non_embedding_params(transformer)
count_non_embedding_params(spectral_transformer)


##############################################
# Loss, Optimizer, Training & Evaluation
##############################################

criterion = nn.CrossEntropyLoss()
optimizer_transformer = torch.optim.AdamW(transformer.parameters(), lr=config.lr)
optimizer_spectral = torch.optim.AdamW(spectral_transformer.parameters(), lr=config.lr)
grad_clip = None  # Optional gradient clipping


def train(model: nn.Module, optimizer, loader: DataLoader, device: torch.device, desc="Training") -> float:
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=desc, leave=False)
    for _, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Flatten outputs and targets for loss computation
        outputs = outputs.view(-1, config.vocab_size)
        targets = targets.view(-1)
        mask = targets != -1  # Only compute loss for valid targets

        loss = criterion(outputs[mask], targets[mask])
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item(), "LR": optimizer.param_groups[0]["lr"]})
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc="Evaluating") -> tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=desc, leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)

        outputs = outputs.view(-1, config.vocab_size)
        targets = targets.view(-1)
        mask = targets != -1

        loss = criterion(outputs[mask], targets[mask])
        total_loss += loss.item()

        _, predicted = torch.max(outputs[mask], dim=1)
        correct += (predicted == targets[mask]).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


##############################################
# Training Execution
##############################################

best_test_acc_t = 0.0
best_test_acc_s = 0.0

transformer_train_losses = []
transformer_test_losses = []
transformer_test_accuracies = []

spectral_train_losses = []
spectral_test_losses = []
spectral_test_accuracies = []

for epoch in range(1, config.num_epochs + 1):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch}/{config.num_epochs}")
    print("-" * 30)

    loss_train_s = train(
        spectral_transformer, optimizer_spectral, train_loader, config.device, desc="Training Spectron"
    )
    loss_test_s, acc_test_s = evaluate(spectral_transformer, test_loader, config.device, desc="Evaluating Spectron")

    loss_train_t = train(transformer, optimizer_transformer, train_loader, config.device, desc="Training Transformer")
    loss_test_t, acc_test_t = evaluate(transformer, test_loader, config.device, desc="Evaluating Transformer")

    transformer_train_losses.append(loss_train_t)
    transformer_test_losses.append(loss_test_t)
    transformer_test_accuracies.append(acc_test_t)

    spectral_train_losses.append(loss_train_s)
    spectral_test_losses.append(loss_test_s)
    spectral_test_accuracies.append(acc_test_s)

    epoch_elapsed = time.time() - epoch_start_time
    best_test_acc_t = max(best_test_acc_t, acc_test_t)
    best_test_acc_s = max(best_test_acc_s, acc_test_s)

    print(f"Epoch {epoch} completed in {epoch_elapsed:.2f}s")
    print(
        f"  Transformer    | Train Loss: {loss_train_t:.4f} | Test Loss: {loss_test_t:.4f} | Test Acc: {acc_test_t:.4f} (Best: {best_test_acc_t:.4f})"
    )
    print(
        f"  Spectron       | Train Loss: {loss_train_s:.4f} | Test Loss: {loss_test_s:.4f} | Test Acc: {acc_test_s:.4f} (Best: {best_test_acc_s:.4f})"
    )

print("\nFinal Evaluation on Test Set:")
loss_test_t, acc_test_t = evaluate(transformer, test_loader, config.device, desc="Final Evaluation Transformer")
loss_test_s, acc_test_s = evaluate(spectral_transformer, test_loader, config.device, desc="Final Evaluation Spectron")

print(f"Transformer    | Test Loss: {loss_test_t:.4f} | Test Accuracy: {acc_test_t:.4f}")
print(f"Spectron       | Test Loss: {loss_test_s:.4f} | Test Accuracy: {acc_test_s:.4f}")

# Plot accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, config.num_epochs + 1), transformer_test_accuracies, label="Transformer Test Accuracy")
plt.plot(range(1, config.num_epochs + 1), spectral_test_accuracies, label="Spectron Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy over Epochs")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.close()

# Plot learned decay factors from a chosen layer of Spectron
layer_idx = 1  # Adjust if you want a different layer
decay_params = spectral_transformer.layers[layer_idx].attn.decay.detach().cpu()
decay_factors = torch.sigmoid(decay_params).numpy()

plt.figure(figsize=(10, 4))
plt.plot(decay_factors, marker="o")
plt.title(f"Learned Decay Factors (Layer {layer_idx})")
plt.xlabel("Sequence Position")
plt.ylabel("Decay Factor (0-1)")
plt.grid(True)
plt.savefig("decay_factors_plot_1.png")
plt.show()
