import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Device Setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")


# -----------------------------------------------------------------------------
# v1: Dataset Generation for the Copy Task (NumPy-based)
# -----------------------------------------------------------------------------
def exists(val):
    return val is not None


def generate_copying_instance(
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    selective: bool = False,
    *args,
    **kwargs,
) -> Tuple[np.array, np.array]:
    """
    Generate an instance of the copying task.

    In this formulation, we reserve the two highest indices:
      - copy_token = vocab_size - 1
      - blank_token = vocab_size - 2
    Regular tokens are 0 ... (vocab_size - 3).

    For a given seq_len, we assume:
      - The first num_tokens_to_copy tokens are the content to be copied.
      - Then there is a delay period of (seq_len - (num_tokens_to_copy*2) - 1) blank tokens.
      - Then we append a single copy_token (acting as a delimiter).
      - Finally, we append num_tokens_to_copy blanks.

    The target is constructed as:
      - The first (num_tokens_to_copy + num_blank_tokens + 1) entries are set to target_ignore_idx.
      - The last num_tokens_to_copy entries are the content to be copied.

    In the selective case, the content tokens are interleaved with blanks randomly.
    """
    if not exists(rng):
        rng = np.random.default_rng()

    # Define special tokens:
    copy_token = vocab_size - 1  # highest index
    blank_token = vocab_size - 2  # second-highest index
    # Regular tokens: 0 to (vocab_size - 3)
    non_special_vocab_size = vocab_size - 2
    vocab = np.arange(non_special_vocab_size)

    # Ensure sequence length is sufficient:
    assert seq_len > (num_tokens_to_copy * 2) + 1, "seq_len must be > (num_tokens_to_copy*2)+1"
    num_blank_tokens = seq_len - (num_tokens_to_copy * 2) - 1

    # Sample the content tokens:
    to_copy = rng.choice(vocab, size=(num_tokens_to_copy,), replace=True).reshape(-1)

    if not selective:
        # Non-selective: simply concatenate the content tokens then blanks.
        inputs = list(to_copy)
        inputs += [blank_token] * num_blank_tokens
    else:
        # Selective: interleave the content tokens with blanks.
        inputs = np.array(to_copy)
        # Randomly choose positions to insert blank tokens:
        insert_indices = rng.integers(0, len(inputs), num_blank_tokens)
        inputs = np.insert(inputs, insert_indices, [blank_token] * num_blank_tokens).tolist()

    # Append the delimiter (copy_token) and then additional blanks:
    inputs += [copy_token]
    inputs += [blank_token] * num_tokens_to_copy
    inputs = np.array(inputs)

    # Build targets:
    # For the waiting period, we fill with target_ignore_idx.
    waiting_length = num_tokens_to_copy + num_blank_tokens + 1
    targets = [target_ignore_idx] * waiting_length
    targets += list(to_copy)
    targets = np.array(targets)

    return inputs, targets


def generate_copy_dataset(
    num_examples: int = 10,
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    target_ignore_idx: int = -100,
    selective: bool = False,
    rng: np.random.Generator = None,
) -> TensorDataset:
    """
    Generate a dataset (TensorDataset) for the copy task using v1.
    """
    if not exists(rng):
        rng = np.random.default_rng()
    inputs_list = []
    targets_list = []
    for _ in range(num_examples):
        ins, tar = generate_copying_instance(
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_tokens_to_copy=num_tokens_to_copy,
            rng=rng,
            target_ignore_idx=target_ignore_idx,
            selective=selective,
        )
        inputs_list.append(ins)
        targets_list.append(tar)
    inputs_np = np.stack(inputs_list)  # shape: (num_examples, seq_len)
    targets_np = np.stack(targets_list)  # shape: (num_examples, waiting_length + num_tokens_to_copy)
    inputs_tensor = torch.from_numpy(inputs_np).to(torch.long)
    targets_tensor = torch.from_numpy(targets_np).to(torch.long)
    return TensorDataset(inputs_tensor, targets_tensor)


# -----------------------------------------------------------------------------
# (Rest of model components remain unchanged)
# -----------------------------------------------------------------------------
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

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, mask=mask)
        return src


class TransformerCopyModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,  # This is the total vocab size including special tokens
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        # Create sinusoidal positional embeddings
        self.d_model = d_model
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Regular token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)  # Remove the +4
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)  # Remove the +4

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Add positional embeddings to help with the copying task
        x_embed = self.embedding(x) + self.pe.unsqueeze(0)
        enc_out = self.encoder(x_embed, mask=mask)
        logits = self.out_proj(enc_out)
        return logits


# For brevity, SpectralAttention and Spectron remain unchanged.
# (They work the same way as in our previous version.)
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


import torch
import torch.nn as nn
import math


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
    seq_len: int, K: int, use_hankel_L: bool = False, device: torch.device = None, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L, device=device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k = phi_k * sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


class LearnableSpectralFilters(nn.Module):
    def __init__(
        self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


class FixedSpectralFilters(nn.Module):
    def __init__(
        self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        self.register_buffer("filters", filters)

    def forward(self):
        return self.filters


def symmetric_outer(V: torch.Tensor) -> torch.Tensor:
    """
    Given V of shape (B, T, k), compute for each vector its outer product v v^T,
    then extract the upper-triangular entries and scale off-diagonals by sqrt(2).
    This yields a compressed vector of size cp = k*(k+1)//2.
    """
    B, T, k = V.shape
    outer = V.unsqueeze(-1) * V.unsqueeze(-2)
    indices = torch.triu_indices(k, k, device=V.device)
    Z = outer[..., indices[0], indices[1]]  # shape: (B, T, cp)
    scale = torch.where(
        indices[0] == indices[1],
        torch.tensor(1.0, device=V.device, dtype=V.dtype),
        torch.tensor(math.sqrt(2.0), device=V.device, dtype=V.dtype)
    )
    Z = Z * scale
    return Z


class SpectralAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        use_hankel_L: bool = False,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pre_proj = nn.Linear(d_model, seq_len, dtype=dtype) if d_model != seq_len else nn.Identity()
        # Learnable Q filters
        self.Q_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype)
        # Fixed K filters for spectral structure
        self.K_filt = FixedSpectralFilters(seq_len, k, use_hankel_L, device, dtype)
        # Compressed dimension after symmetric outer product
        self.cp = k * (k + 1) // 2
        # Map Q (from dimension k) to the compressed space (cp)
        self.q_proj = nn.Linear(k, self.cp, dtype=dtype).to(device)
        # We drop v_proj by tying V to K for symmetry.
        self.o_proj = nn.Linear(self.cp, d_model, dtype=dtype).to(device)
        self.decay = nn.Parameter(torch.ones(seq_len, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (B, T, d_model)
        B, T, d = x.shape
        x_proj = self.pre_proj(x)  # (B, T, seq_len)

        # Compute Q: learnable filters
        Q_filters = self.Q_filt()  # (seq_len, k)
        Q = torch.einsum("bti,ik->btk", x_proj, Q_filters)  # (B, T, k)
        Q = self.q_proj(Q)  # (B, T, cp)

        # Compute K from fixed filters
        K_filters = self.K_filt()  # (seq_len, k)
        K = torch.einsum("bti,ik->btk", x_proj, K_filters)  # (B, T, k)

        # Tie V to K to enforce symmetry
        V = K  # (B, T, k)

        # Compute compressed symmetric outer product per token
        Z = symmetric_outer(V)  # (B, T, cp)
        decay = self.decay.view(1, T, 1)  # (1, T, 1)
        Z = Z * decay

        # Cumulative sum along time to aggregate past information
        H = torch.cumsum(Z, dim=1)  # (B, T, cp)

        # Combine Q and aggregated H (elementwise multiplication)
        Y = Q * H  # (B, T, cp)
        return self.o_proj(Y)  # (B, T, d_model)


class Spectron(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        vocab_size: int,
        d_out: int = None,  # Defaults to vocab_size if None
        use_hankel_L: bool = False,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if d_out is None:
            d_out = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.spec_attn = SpectralAttention(seq_len, d_model, k, use_hankel_L, device, dtype)
        self.out_proj = nn.Linear(d_model, d_out, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        out = x_emb + self.spec_attn(x_emb)
        return self.out_proj(out)


# -----------------------------------------------------------------------------
# Helper: Build Causal Mask
# -----------------------------------------------------------------------------
def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


# -----------------------------------------------------------------------------
# Token-Level Accuracy Helper
# -----------------------------------------------------------------------------
def compute_token_level_accuracy(model, loader, attn_mask=None, device=device):
    """
    Compute token-level accuracy while ignoring special tokens and target_ignore_idx.
    """
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCopyModel) else model(inputs)
            predictions = logits.argmax(dim=-1)

            # Create mask for valid tokens (not target_ignore_idx)
            valid_mask = targets != -100

            # Calculate accuracy only on valid tokens
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().item()
            total_tokens += valid_mask.sum().item()

            if batch_idx == 0:
                print("\nDEBUG INFO (Token-level):")
                for i in range(min(2, len(inputs))):
                    print(f"\nExample {i}:")
                    print("Input:     ", inputs[i].cpu().tolist())
                    print("Target:    ", targets[i].cpu().tolist())
                    print("Predicted: ", predictions[i].cpu().tolist())
                    print("Valid Mask:", valid_mask[i].cpu().tolist())
                print()

    token_acc = 100.0 * correct_tokens / (total_tokens if total_tokens > 0 else 1)
    print(f"Token-Level Accuracy (ignoring padding): {token_acc:.2f}%")
    return token_acc


# -----------------------------------------------------------------------------
# Training Loop (step-based)
# -----------------------------------------------------------------------------
def train_model(model, loader, val_loader, attn_mask=None, max_steps: int = 10000, eval_interval: int = 50):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    loss_history = []
    accuracy_history = []
    eval_steps = []
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for inputs, targets in loader:
            if step >= max_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCopyModel) else model(inputs)
            # No need to shift logits - our targets are already aligned
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            loss_history.append(total_loss)
            step += 1

            if step % eval_interval == 0:
                acc = compute_token_level_accuracy(model, val_loader, attn_mask=attn_mask)
                accuracy_history.append(acc)
                eval_steps.append(step)
                print(f"Step {step}/{max_steps} Loss: {total_loss:.4f} | Token-Level Accuracy: {acc:.2f}%")
            else:
                print(f"Step {step}/{max_steps} Loss: {total_loss:.4f}")
    return loss_history, accuracy_history, eval_steps


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For v1, typical parameters might be:
    #   vocab_size (including special tokens) = 16 (so regular tokens: 0...13, blank=14, copy=15)
    #   seq_len = 256
    #   num_tokens_to_copy = e.g. 16
    selective = False  # Change as desired (selective task)
    vocab_size = 16  # Here vocab_size includes regular tokens plus our two reserved indices
    seq_len = 256
    num_tokens_to_copy = 16
    target_ignore_idx = -100
    num_examples = 10000
    rng = np.random.default_rng(1746)

    # Generate training dataset using v1:
    train_dataset = generate_copy_dataset(
        num_examples=num_examples,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_tokens_to_copy=num_tokens_to_copy,
        target_ignore_idx=target_ignore_idx,
        selective=selective,
        rng=rng,
    )
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # For validation, generate a separate dataset:
    val_dataset = generate_copy_dataset(
        num_examples=num_examples // 10,  # smaller validation set
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_tokens_to_copy=num_tokens_to_copy,
        target_ignore_idx=target_ignore_idx,
        selective=selective,
        rng=np.random.default_rng(1747),  # different seed for validation
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Create and compile TransformerCopyModel
    # trans_copy_model = TransformerCopyModel(
    #     seq_len=seq_len,
    #     d_model=128,  # Increased model capacity
    #     vocab_size=vocab_size,
    #     num_layers=3,  # Increased depth
    #     num_heads=4,
    #     dropout=0.1,
    # ).to(device)
    # compiled_trans_model = torch.compile(trans_copy_model, fullgraph=True)
    # print("\nTraining TransformerCopyModel (v1)...")
    # loss_history_trans, acc_history_trans, eval_steps_trans = train_model(
    #     compiled_trans_model, loader, val_loader, attn_mask=None, max_steps=1000, eval_interval=50
    # )

    # Create and compile Spectron
    spectron = Spectron(
        seq_len=seq_len,
        d_model=128,  # Increased model capacity
        k=16,  # Increased number of spectral components
        vocab_size=vocab_size,
        d_out=vocab_size,  # Match vocab size
        use_hankel_L=False,
        device=device,
        dtype=torch.bfloat16,
    ).to(device=device, dtype=torch.bfloat16)
    compiled_spectron = torch.compile(spectron, fullgraph=True)
    print("\nTraining Spectron (v1)...")
    loss_history_spectron, acc_history_spectron, eval_steps_spectron = train_model(
        compiled_spectron, loader, val_loader, max_steps=1000, eval_interval=50
    )

    # Plot results.
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(
        loss_history_trans,
        label="TransformerCopyModel",
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1,
        color="blue",
    )
    ax1.plot(
        loss_history_spectron,
        label="Mystery STU",
        marker="s",
        markersize=4,
        linestyle="--",
        linewidth=1,
        color="green",
    )
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Training Loss Comparison", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    ax2.plot(
        eval_steps_trans,
        acc_history_trans,
        label="TransformerCopyModel",
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1,
        color="blue",
    )
    ax2.plot(
        eval_steps_spectron,
        acc_history_spectron,
        label="Mystery STU",
        marker="s",
        markersize=4,
        linestyle="--",
        linewidth=1,
        color="green",
    )
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Token-Level Accuracy (%)", fontsize=12)
    ax2.set_title("Validation Accuracy Over Time (Token-Level)", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    final_acc_text = (
        f"Final Accuracies:\nTransformer: {acc_history_trans[-1]:.2f}%\nMystery STU: {acc_history_spectron[-1]:.2f}%"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax2.text(0.05, 0.95, final_acc_text, transform=ax2.transAxes, fontsize=12, verticalalignment="top", bbox=props)
    plt.tight_layout()
    plt.show()

    # Example predictions.
    def show_example_predictions(model, model_name: str, attn_mask=None):
        model.eval()
        with torch.no_grad():
            first_input, first_target = train_dataset[0]
            last_input, last_target = train_dataset[-1]
            first_input = first_input.unsqueeze(0).to(device)
            last_input = last_input.unsqueeze(0).to(device)
            if isinstance(model, TransformerCopyModel):
                first_logits = model(first_input, mask=attn_mask)
                last_logits = model(last_input, mask=attn_mask)
            elif isinstance(model, Spectron):
                first_logits = model(first_input)
                last_logits = model(last_input)
            else:
                first_logits, _ = model(first_input, attn_mask)
                last_logits, _ = model(last_input, attn_mask)
            target_seq_len = first_target.size(0)
            # Shift logits by 1 to drop BOS so that predictions align with target.
            first_logits = first_logits[:, 1 : 1 + target_seq_len, :]
            last_logits = last_logits[:, 1 : 1 + target_seq_len, :]
            first_pred = first_logits.argmax(dim=-1).squeeze(0).cpu()
            last_pred = last_logits.argmax(dim=-1).squeeze(0).cpu()
            print(f"\n{model_name} - First Example")
            print("Input:     ", train_dataset[0][0].cpu().tolist())
            print("Target:    ", first_target.cpu().tolist())
            print("Predicted: ", first_pred.tolist())
            print(f"\n{model_name} - Last Example")
            print("Input:     ", train_dataset[-1][0].cpu().tolist())
            print("Target:    ", last_target.cpu().tolist())
            print("Predicted: ", last_pred.tolist())

    print("\n--- Predictions from TransformerCopyModel (v1) ---")
    show_example_predictions(trans_copy_model, "TransformerCopyModel", attn_mask=None)
    print("\n--- Predictions from Spectron (v1) ---")
    show_example_predictions(spectron, "Spectron")
