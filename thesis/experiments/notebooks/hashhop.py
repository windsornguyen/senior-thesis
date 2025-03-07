import random
import numpy as np
import math
import copy
import string
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# === Simple Character-level Tokenizer and Helpers ===
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


# === Prompt Templates and Data Structures ===
TASK_PROMPT_TEMPLATE = "{VARIABLE_LIST}"
COMPLETION_TEMPLATE = "HOPS={HOPS}\nCoT={COT}\n{COMPLETION}"


@dataclass
class MultiHopSample:
    prompt: str
    completion: str
    targets: Dict[str, str]


# === Multi-Hop Evaluation Generator (unchanged) ===
class MultiHopEval:
    @staticmethod
    def make_one(
        n_chars_problem: int,
        num_queries: int,
        hops: int,
        hash_pair_str_length: int,
        chain_of_thought: bool,
    ) -> MultiHopSample:
        # Estimate how many hash pairs (chains) to produce.
        chars_per_hash_pair = (hash_pair_str_length * 2 + 3) * hops
        n_chains = math.ceil(n_chars_problem / chars_per_hash_pair)

        levels = MultiHopEval._make_levels(n=n_chains, hops=hops, string_length=hash_pair_str_length)

        lines = []
        for i, level in enumerate(levels):
            if i == len(levels) - 1:
                lines.extend([f"{k} = '{v}'" for k, v in level.items()])
            else:
                lines.extend([f"{k} = {v}" for k, v in level.items()])

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
        levels = [
            {make_random_string(length=string_length): make_random_string(length=string_length) for _ in range(n)}
        ]
        for _ in range(hops - 1):
            levels.append({v: make_random_string(length=string_length) for v in levels[-1].values()})
        return levels


def make_random_string(length: int) -> str:
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return "".join(random.choices(alphabet, k=length))


def shuffle_dict(to_shuffle: dict) -> dict:
    items = list(to_shuffle.items())
    random.shuffle(items)
    return dict(items)


# === HashHop Dataset Generation ===
def generate_hashhop_dataset(
    num_examples: int,
    n_chars_problem: int,
    num_queries: int,
    hops: int,
    hash_pair_str_length: int,
    chain_of_thought: bool,
    seed: int = 1746,
) -> TensorDataset:
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    token2idx = build_char_tokenizer()

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

    input_ids_list = [encode_text(sample.prompt, token2idx) for sample in samples]
    target_ids_list = [encode_text(sample.completion, token2idx) for sample in samples]

    inputs_tensor = pad_sequences(input_ids_list, pad_value=0)
    targets_tensor = pad_sequences(target_ids_list, pad_value=0)

    return TensorDataset(inputs_tensor, targets_tensor), token2idx


# === Build a Seq2Seq Dataset by Concatenating Prompt and Completion ===
def generate_hashhop_seq2seq_dataset(
    num_examples: int,
    n_chars_problem: int,
    num_queries: int,
    hops: int,
    hash_pair_str_length: int,
    chain_of_thought: bool,
    seed: int = 1746,
) -> (TensorDataset, Dict[str, int]):
    # Generate the separate prompt and completion tensors.
    ds, token2idx = generate_hashhop_dataset(
        num_examples, n_chars_problem, num_queries, hops, hash_pair_str_length, chain_of_thought, seed
    )
    new_inputs = []
    new_targets = []
    # We'll insert a newline between prompt and completion.
    newline = "\n"
    newline_token = encode_text(newline, token2idx)  # typically a one-token list
    for prompt_tokens, completion_tokens in zip(ds.tensors[0].tolist(), ds.tensors[1].tolist()):
        # Remove trailing padding (zeros)
        prompt_tokens = [t for t in prompt_tokens if t != 0]
        completion_tokens = [t for t in completion_tokens if t != 0]
        combined = prompt_tokens + newline_token + completion_tokens
        # For the target, we mask the prompt and newline (set to -1) and keep the completion.
        target = [-1] * (len(prompt_tokens) + len(newline_token)) + completion_tokens
        new_inputs.append(combined)
        new_targets.append(target)
    inputs_tensor = pad_sequences(new_inputs, pad_value=0)
    targets_tensor = pad_sequences(new_targets, pad_value=0)
    return TensorDataset(inputs_tensor, targets_tensor), token2idx


# === Helper: Build Causal Mask (for decoder) ===
def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


# === Custom Transformer Components (as before) ===
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        B, T, _ = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        output = self.out_proj(attn_output)
        return output, attn_weights


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dim_feedforward: int = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.self_attn = CustomMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        attn_out, _ = self.self_attn(src, src, src, mask=mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CustomTransformerEncoderLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            src = layer(src, mask=mask)
        return src


# === Spectral Filters and Spectral Attention (DO NOT ALTER) ===
def get_monic_chebyshev_coeffs(n: int) -> torch.Tensor:
    def chebyshev_t_int(n: int) -> list[int]:
        if n == 0:
            return [1]
        elif n == 1:
            return [1, 0]
        T0 = [1]
        T1 = [1, 0]
        for _ in range(2, n + 1):
            T2 = [2 * c for c in T1] + [0]
            d = len(T2) - len(T0)
            padded_T0 = [0] * d + T0
            T2 = [a - b for a, b in zip(T2, padded_T0)]
            T0, T1 = T1, T2
        return T2

    coeffs = torch.tensor(chebyshev_t_int(n), dtype=torch.complex128)
    if n > 0:
        coeffs = coeffs / (2.0 ** (n - 1))
    return coeffs


def get_hankel(seq_len: int, use_hankel_L: bool = False, device=None) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False, device: torch.device = None, dtype=torch.float32
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L, device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k = phi_k * sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


class LearnableSpectralFilters(nn.Module):
    def __init__(self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        self.p_coeffs = get_monic_chebyshev_coeffs(seq_len - 1)
        self.Q_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.K_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.v_proj = nn.Linear(d_model, k).to(device)
        self.o_proj = nn.Linear(k, d_model).to(device)
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)
        Q = torch.einsum("bti,ik->btk", x_proj, self.Q_filt())
        K = torch.einsum("bti,ik->btk", x_proj, self.K_filt())
        V = self.v_proj(x)
        Z = torch.einsum("btp,btn->btpn", V, K)
        decay = self.decay.view(1, T, 1, 1)
        Z = Z * decay
        H = torch.cumsum(Z, dim=1)
        Y = torch.einsum("btk,btkn->btn", Q, H)
        return self.o_proj(Y)


# === Seq2Seq Models for HashHop Task ===
class TransformerHashHopModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Note: vocab_size here should equal len(token2idx)+1.
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T)
        x_embed = self.embedding(x)  # (B, T, d_model)
        enc_out = self.encoder(x_embed, mask=mask)  # (B, T, d_model)
        logits = self.out_proj(enc_out)  # (B, T, vocab_size)
        return logits


class SpectronHashHop(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, vocab_size: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.spec_attn = SpectralAttention(seq_len, d_model, k, use_hankel_L, device)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        out = x_emb + self.spec_attn(x_emb)
        logits = self.out_proj(out)
        return logits


# === Training Loop for Seq2Seq HashHop Task with Loss Recording ===
def train_model(model, loader, attn_mask=None, epochs: int = 5):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs, mask=attn_mask) if attn_mask is not None else model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")
    return loss_history


# === Helper: Decode tokens back to text ===
def decode_tokens(token_ids: List[int], idx2token: Dict[int, str]) -> str:
    # Skip pad (0) and ignore (-1)
    return "".join([idx2token.get(t, "") for t in token_ids if t > 0])


# === Build the Seq2Seq HashHop Dataset ===
dataset, token2idx = generate_hashhop_seq2seq_dataset(
    num_examples=100,
    n_chars_problem=300,
    num_queries=3,
    hops=3,
    hash_pair_str_length=4,
    chain_of_thought=True,
    seed=1746,
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Build inverse mapping.
idx2token = {idx: ch for ch, idx in token2idx.items()}
idx2token[0] = "␣"  # pad

# Build causal mask for the full sequence length.
seq_len = dataset.tensors[0].shape[1]
causal_mask = build_causal_mask(seq_len).to(device)

# === Instantiate & Train Models ===
print("\nTraining TransformerHashHopModel...")
vocab_size = len(token2idx) + 1  # tokens from 0 to len(token2idx)
trans_model = TransformerHashHopModel(
    seq_len=seq_len, d_model=64, vocab_size=vocab_size, num_layers=2, num_heads=8, dropout=0.1
).to(device)
loss_history_trans = train_model(trans_model, loader, attn_mask=causal_mask, epochs=10)

print("\nTraining SpectronHashHop...")
spectron_model = SpectronHashHop(
    seq_len=seq_len, d_model=64, k=16, vocab_size=vocab_size, use_hankel_L=False, device=device
).to(device)
loss_history_spectron = train_model(spectron_model, loader, epochs=10)

# === Plot Training Loss for the HashHop Task ===
plt.figure(figsize=(10, 5))
plt.plot(loss_history_trans, label="TransformerHashHopModel", marker="o", linestyle="-", linewidth=1)
plt.plot(loss_history_spectron, label="Mystery STU", marker="s", linestyle="--", linewidth=1)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Cross-Entropy Loss", fontsize=12)
plt.title("Training Loss for HashHop Task", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()


# === Display Example Predictions ===
def show_example_predictions(model, model_name: str, attn_mask=None):
    model.eval()
    with torch.no_grad():
        # Pick the first example.
        input_seq, target_seq = dataset[0]
        input_seq = input_seq.unsqueeze(0).to(device)
        logits = model(input_seq, mask=attn_mask) if attn_mask is not None else model(input_seq)
        pred_seq = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        input_seq = input_seq.squeeze(0).cpu().tolist()
        target_seq = target_seq.cpu().tolist()
        print(f"\n{model_name} - Example")
        print("Input:    ", decode_tokens(input_seq, idx2token))
        target_decoded = decode_tokens([t for t in target_seq if t != -1], idx2token)
        print("Target:   ", target_decoded)
        print("Predicted:", decode_tokens(pred_seq, idx2token))


print("\n--- Predictions from TransformerHashHopModel ---")
show_example_predictions(trans_model, "TransformerHashHopModel", attn_mask=causal_mask)

print("\n--- Predictions from SpectronHashHop ---")
show_example_predictions(spectron_model, "SpectronHashHop")
