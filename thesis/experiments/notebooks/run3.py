#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
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
# (OLD) Data Generation Functions (not used in this experiment)
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

    Input sequence: [copy_sequence][pre_delim_blanks][delimiter][post_delim_blanks]
    Output sequence: [blank_tokens][copy_sequence]
    """
    torch.manual_seed(seed)
    blank_char = num_categories - 2  # Blank token
    delim_char = num_categories - 1  # Delimiter token
    to_copy = torch.randint(0, blank_char, (num_examples, copy_len))
    pre_delim_blanks = torch.full((num_examples, num_categories - 1), blank_char)
    delim = torch.full((num_examples, 1), delim_char)
    post_delim_blanks = torch.full((num_examples, blank_len), blank_char)
    if selective:

        def insert_pre_delim_blanks(row: torch.Tensor) -> torch.Tensor:
            pre_delim_len = copy_len + num_categories - 1
            insert_positions = torch.randperm(pre_delim_len)[: num_categories - 1]
            inserted_row = torch.full((pre_delim_len,), blank_char)
            mask = torch.ones(pre_delim_len, dtype=torch.bool)
            mask[insert_positions] = False
            inserted_row[mask] = row
            return inserted_row

        inputs = torch.stack([insert_pre_delim_blanks(row) for row in to_copy])
    else:
        inputs = torch.cat((to_copy, pre_delim_blanks), dim=1)
    inputs = torch.cat((inputs, delim, post_delim_blanks), dim=1)
    blank_output = torch.full((num_examples, num_categories + blank_len), blank_char)
    outputs = torch.cat((blank_output, to_copy), dim=1)
    if one_hot:
        inputs = F.one_hot(inputs, num_classes=num_categories).to(dtype)
        outputs = F.one_hot(outputs, num_classes=num_categories).to(dtype)
    return TensorDataset(inputs, outputs)


##############################################
# Spectral Filtering Utilities (unchanged)
##############################################


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
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
# Model Definitions (unchanged)
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
        Q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
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
        filters = get_tensorized_spectral_filters(
            n=seq_len,
            k=k,
            use_hankel_L=use_hankel_L,
            device=device,
            dtype=dtype,
        )
        self.filters = nn.Parameter(filters)

    def forward(self) -> torch.Tensor:
        return self.filters


class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, n_embd: int, k: int, use_hankel_L: bool = False, device: torch.device = None):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        self.Q_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters
        self.K_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters
        self.i_proj = nn.Linear(n_embd, self.Q_filt.shape[0])
        self.v_proj = nn.Linear(n_embd, self.Q_filt.shape[1])
        self.o_proj = nn.Linear(self.Q_filt.shape[1], n_embd)
        self.decay = nn.Parameter(torch.ones(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, dim = x.shape
        assert T == self.seq_len, "Input length must match seq_len"
        x_proj = self.i_proj(x)
        Q = torch.einsum("bth,hk->btk", x_proj, self.Q_filt)
        K = torch.einsum("bth,hk->btk", x_proj, self.K_filt)
        V = self.v_proj(x)
        Z = torch.einsum("bsp,bsn->bspn", V, K)
        decay = self.decay.view(1, T, 1, 1)
        Z = Z * decay
        H = torch.cumsum(Z, dim=1)
        Y = torch.einsum("btn,btpn->btp", Q, H)
        return self.o_proj(Y)


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
# Utility Functions (unchanged)
##############################################


@torch.no_grad()
def check_model_value_range(model: nn.Module, range_val: float = 1e3, std_val: float = 1e3):
    for name, param in chain(model.named_parameters(), model.named_buffers()):
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


set_seed(1746)


##############################################
# S4-Style Copying Data Generator and Training Loop
##############################################
def torch_copying_data(L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
    """
    Generate a dataset for a sequence copying task.
    Adopted from the S4 repo.

    Args:
      L (int): Number of padding/noise tokens.
      M (int): Number of tokens to memorize.
      A (int): Alphabet size.
      variable (bool): If True, use selective (random) copying.
      variable_length (bool): If True, randomize the number of tokens to memorize.
      batch_shape (tuple): Shape of the batch.
      one_hot (bool): If True, one-hot encode the input.
      reverse (bool): If True, reverse the target sequence.

    Returns:
      tuple: (input tensor, target tensor)
    """
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A - 1, size=batch_shape + (M,))
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([torch.randperm(L + M)[:M] for _ in range(total_batch)], 0)
        inds = inds.reshape(batch_shape + (M,))
        inds, _ = inds.sort(dim=-1)
    else:
        inds = torch.arange(M).repeat(batch_shape + (1,))
    zeros_x = torch.zeros(batch_shape + (M + L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A - 1) * torch.ones(batch_shape + (M,), dtype=torch.long)
    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = tokens.clone()
    if reverse:
        y_ = y_.flip(-1)
    if one_hot:
        x = F.one_hot(x_, num_classes=A).float()
    else:
        x = x_
    y = y_
    return x, y


def generate_dataset(dataset_config, training_config):
    """
    Generate a dataset based on provided configuration dictionaries.
    """
    x, y = torch_copying_data(
        L=dataset_config["l_noise"],
        M=dataset_config["l_memorize"],
        A=dataset_config["n_tokens"],
        variable=dataset_config["variable"],
        variable_length=dataset_config["variable_length"],
        batch_shape=(training_config["batch_size"],),
        one_hot=dataset_config["one_hot"],
        reverse=dataset_config["reverse"],
    )
    return x, y


# Updated S4 dataset/training configuration for longer training (mimicking the paper)
dataset_config = {
    "l_noise": 100,  # Number of padding/noise tokens
    "l_memorize": 50,  # Number of tokens to memorize
    "n_tokens": 30,  # Alphabet size
    "variable": False,  # Fixed positions (non-selective copying)
    "variable_length": False,
    "one_hot": False,
    "reverse": False,
}

training_config = {
    "batch_size": 64,
    "num_steps": 20000,  # Extended number of training steps
    "lr": 3e-4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For the S4 copying task, the input sequence length is (l_noise + l_memorize) + l_memorize.
s4_seq_len = dataset_config["l_noise"] + 2 * dataset_config["l_memorize"]  # 100 + 2*50 = 200 tokens

# Build a simple configuration for the models.
s4_config = type("S4Config", (), {})()
s4_config.num_layers = 2
s4_config.n_heads = 8
s4_config.n_embd = 512  # Increased embedding dimension
s4_config.k = 8
s4_config.use_hankel_L = False
s4_config.dropout = 0.0
s4_config.vocab_size = dataset_config["n_tokens"]
s4_config.seq_len = s4_seq_len
s4_config.device = device
s4_config.lr = training_config["lr"]
s4_config.torch_compile = False

# Instantiate both models:
transformer_model = Transformer(s4_config).to(device)
spectron_model = Spectron(s4_config).to(device)

print("Transformer model parameters:")
count_non_embedding_params(transformer_model)
check_model_value_range(transformer_model, range_val=10.0, std_val=1.0)

print("\nSpectron model parameters:")
count_non_embedding_params(spectron_model)
check_model_value_range(spectron_model, range_val=10.0, std_val=1.0)

# Create separate optimizers for each
optimizer_transformer = torch.optim.AdamW(transformer_model.parameters(), lr=training_config["lr"])
optimizer_spectron = torch.optim.AdamW(spectron_model.parameters(), lr=training_config["lr"])

criterion = nn.CrossEntropyLoss()


def train_model(model: nn.Module, optimizer: torch.optim.Optimizer, model_name: str):
    """
    Train the given model on the S4-style copying task using tqdm.
    Only the final l_memorize tokens are scored.
    """
    model.train()
    start_time = time.time()
    pbar = tqdm(range(training_config["num_steps"]), desc=f"Training {model_name}")
    for step in pbar:
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # Full sequence output: shape (batch, seq_len, vocab_size)
        outputs = model(inputs)
        # Score only the last l_memorize tokens (i.e. the copied part)
        outputs = outputs[:, -dataset_config["l_memorize"] :, :]
        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = outputs.argmax(dim=2)  # shape: (batch, M)
        correct = (preds == targets).sum().item()
        total = targets.numel()
        accuracy = 100 * correct / total

        # Update progress bar
        pbar.set_postfix(loss=loss.item(), acc=f"{accuracy:.2f}%")
    end_time = time.time()
    print(f"\nTraining {model_name} completed in {(end_time - start_time)/60:.2f} minutes.")


def validate_model(model: nn.Module, model_name: str):
    """
    Validate the given model on a single batch.
    """
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs[:, -dataset_config["l_memorize"] :, :]
        loss = criterion(outputs.transpose(1, 2), targets)
        preds = outputs.argmax(dim=2)
        correct = (preds == targets).sum().item()
        total = targets.numel()
        accuracy = 100 * correct / total
        print(f"{model_name} Validation Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


##############################################
# Main Execution: Train and Compare Both Models
##############################################

if __name__ == "__main__":
    print("Starting training for Transformer model...")
    train_model(transformer_model, optimizer_transformer, "Transformer")
    print("Starting validation for Transformer model...")
    validate_model(transformer_model, "Transformer")

    print("\nStarting training for Spectron model...")
    train_model(spectron_model, optimizer_spectron, "Spectron")
    print("Starting validation for Spectron model...")
    validate_model(spectron_model, "Spectron")
