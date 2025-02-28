import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from thesis.experiments.synthetics.mqar import generate_mqar


# -----------------------------------------------------------------------------
# Helper: Build Causal Mask (for transformer models)
# -----------------------------------------------------------------------------
def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


# -----------------------------------------------------------------------------
# Custom Transformer Components (unchanged)
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


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
    """
    SpectralAttention implements a spectral-based attention mechanism.
    It projects inputs into a (B, T, seq_len) space, applies learned spectral filters
    to obtain queries (Q) and keys (K), and applies a linear projection to obtain values (V).
    The attention output is produced via a sequence of einsum operations:
      - Outer products between V and K are scaled by a decay parameter.
      - A lower-triangular Hankel mask aggregates the results.
      - Finally, the query (Q) is applied to the aggregated tensor.
    The output is then projected back to the original model dimension.

    Args:
        seq_len (int): Sequence length (T).
        d_model (int): Input model dimension.
        k (int): Projection dimension for the spectral filters and linear layer.
        use_hankel_L (bool): Whether to use a Hankel matrix in computing attention.
        device: Torch device.
    """

    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        dtype = torch.bfloat16

        # Optionally project input from d_model -> seq_len if needed.
        self.pre_proj = (
            nn.Linear(d_model, seq_len, device=device, dtype=dtype)
            if d_model != seq_len
            else nn.Identity()
        )

        # Q and K are learned spectral filters.
        self.Q = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype=dtype)
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype=dtype)
        # V is a linear layer.
        self.v_proj = nn.Linear(seq_len, k, device=device, dtype=dtype)

        # Final projection from k back to d_model.
        self.o_proj = nn.Linear(k, d_model, device=device, dtype=dtype)

        # Decay parameter: one per time step.
        self.decay = nn.Parameter(torch.ones(seq_len, device=device, dtype=dtype))

        # Precompute and store the lower-triangular Hankel matrix L (shape [T, T]).
        L = get_hankel(seq_len, use_hankel_L, device=device).to(dtype)
        self.L = nn.Parameter(torch.tril(L))

    def forward(self, x: torch.Tensor, chunk_len: int = 128) -> torch.Tensor:
        """
        Forward pass of SpectralAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).
            chunk_len (int): Chunk size for potential block-sum approaches (unused here).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape

        # Pre-project input to match internal sequence length.
        x_proj = self.pre_proj(x)  # Shape: (B, T, seq_len)

        # Compute Q and K via learned spectral filters.
        Q_out = torch.einsum("bti,ik->btk", x_proj, self.Q())
        K_out = torch.einsum("bti,ik->btk", x_proj, self.K())

        # Compute V via linear projection.
        V_out = self.v_proj(x_proj)  # Shape: (B, T, k)

        # Compute outer product between V and K, scaling by decay.
        # V_out: (B, T, k), K_out: (B, T, k), decay: (T,)
        Z = torch.einsum("btp,btn,t->btpn", V_out, K_out, self.decay)

        # Expand Hankel matrix L for the batch.
        L_batched = self.L.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, T, T)

        # Aggregate the outer products using the Hankel mask.
        H = torch.einsum("bts,bspn->btpn", L_batched, Z)

        # Apply the query to the aggregated tensor.
        Y = torch.einsum("btk,btkn->btn", Q_out, H)

        # Final projection to output the model dimension.
        return self.o_proj(Y)


class SpectralAttentionLayer(nn.Module):
    """
    A single layer that applies SpectralAttention, followed by an MLP,
    each of which is added (residual) to the input, then normalized.

    Args:
        seq_len (int): Sequence length (T).
        d_model (int): Model dimension.
        k (int): Projection dimension for the spectral filters.
        use_hankel_L (bool): Whether to use a Hankel matrix.
        device: Torch device.
    """

    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.spectral_attention = SpectralAttention(seq_len, d_model, k, use_hankel_L, device)
        self.mlp = MLP(d_model, 4 * d_model)
        self.spec_attn_norm = nn.RMSNorm(d_model)
        self.mlp_norm = nn.RMSNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpectralAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        x = x + self.spectral_attention(self.spec_attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SpectronMQAR(nn.Module):
    """
    A stacked spectral-transformer-like model. Uses an embedding, a dropout,
    multiple SpectralAttention layers in sequence, and an output projection.

    Args:
        seq_len (int): Sequence length.
        d_model (int): Model dimension.
        k (int): Projection dimension for the spectral filters.
        vocab_size (int): Vocabulary size.
        d_out (int): Output dimension (defaults to vocab_size).
        num_layers (int): Number of SpectralAttention layers.
        dropout (float): Dropout probability.
        use_hankel_L (bool): Whether to use a Hankel matrix in the attention.
        device: Torch device.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        vocab_size: int,
        d_out: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_hankel_L: bool = False,
        device=None,
    ):
        super().__init__()

        if d_out is None:
            d_out = vocab_size

        # Embedding and dropout
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Stack of SpectralAttention layers
        self.layers = nn.ModuleList(
            [SpectralAttentionLayer(seq_len, d_model, k, use_hankel_L, device=device) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Spectron model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T) containing token indices.

        Returns:
            torch.Tensor: Output logits of shape (B, T, d_out).
        """
        # Embed and apply dropout
        x_emb = self.in_dropout(self.embedding(x))

        # Pass through stacked layers
        out = x_emb
        for layer in self.layers:
            out = layer(out)

        # Normalize and project
        out = self.norm(out)
        out = self.out_proj(out)
        return self.out_dropout(out)


# -----------------------------------------------------------------------------
# MQAR Task Models
# -----------------------------------------------------------------------------
class TransformerMQARModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        """
        The transformer-based model for MQAR.
        The input embedding is over a vocabulary of size (vocab_size + 3) to account for special tokens,
        but the output prediction is over the base vocab (of size vocab_size).
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Regular token embeddings
        self.embedding = nn.Embedding(vocab_size + 3, d_model)

        # Create sinusoidal positional embeddings
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Dropout layers
        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)

        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Create and register causal mask buffer
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Add positional embeddings and apply dropout
        x_embed = self.in_dropout(self.embedding(x) + self.pe.unsqueeze(0))

        # Use the model's causal mask if no mask is provided
        mask = mask if mask is not None else self.causal_mask

        # Pass through encoder
        enc_out = self.encoder(x_embed, mask=mask)

        # Normalize and project
        out = self.norm(enc_out)
        logits = self.out_proj(out)
        return self.out_dropout(logits)


# -----------------------------------------------------------------------------
# Token-Level Accuracy Helper
# -----------------------------------------------------------------------------
def compute_token_level_accuracy(model, loader, attn_mask=None, device=None):
    """
    Compute token-level accuracy while ignoring special tokens and target_ignore_idx.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerMQARModel) else model(inputs)
            predictions = logits.argmax(dim=-1)

            # Create mask for valid tokens (not target_ignore_idx)
            valid_mask = targets != -100

            # Calculate accuracy only on valid tokens
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().item()
            total_tokens += valid_mask.sum().item()

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
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerMQARModel) else model(inputs)
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
# Device Setup & Dataset Creation
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 64
seq_len = 64
d_model = 64
num_examples = 100000
vocab_size = 8192
num_pairs = 4
num_layers = 2
num_heads = 4
dropout = 0.0
k = 16
alpha = 0.01


# Create causal mask
causal_mask = build_causal_mask(seq_len).to(device)

# Create training dataset
train_dataset = generate_mqar(
    num_examples=num_examples,
    sequence_len=seq_len,
    vocab_size=vocab_size,
    num_pairs=num_pairs,
    alpha=alpha,
    seed=1746,
)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create validation dataset
val_dataset = generate_mqar(
    num_examples=num_examples // 20,
    sequence_len=seq_len,
    vocab_size=vocab_size,
    num_pairs=num_pairs,
    alpha=alpha,
    seed=1747,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Train Models
# -----------------------------------------------------------------------------

# print("\nTraining TransformerMQARModel...")
# trans_mqar_model = TransformerMQARModel(
#     seq_len=seq_len, d_model=d_model, vocab_size=vocab_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout
# ).to(device)

# loss_history_trans, acc_history_trans, eval_steps_trans = train_model(
#     trans_mqar_model, loader, val_loader, attn_mask=causal_mask, max_steps=10000, eval_interval=250
# )

print("\nTraining SpectronMQAR...")
spectron_mqar = SpectronMQAR(
    seq_len=seq_len, d_model=d_model, k=k, vocab_size=vocab_size, use_hankel_L=False, device=device
).to(device, dtype=torch.bfloat16)

loss_history_spectron, acc_history_spectron, eval_steps_spectron = train_model(
    spectron_mqar, loader, val_loader, max_steps=10000, eval_interval=250
)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot training loss for both models
ax1.plot(
    loss_history_trans,
    label="Transformer",
    marker="o",
    markersize=4,
    linestyle="-",
    linewidth=1,
    color="blue",
)
ax1.plot(
    loss_history_spectron,
    label="SpectronMQAR",
    marker="s",
    markersize=4,
    linestyle="-",
    linewidth=1,
    color="green",
)
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax1.set_title("Training Loss", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True)

# Plot validation accuracy for both models
ax2.plot(
    eval_steps_trans,
    acc_history_trans,
    label="Transformer",
    marker="o",
    markersize=4,
    linestyle="-",
    linewidth=1,
    color="blue",
)
ax2.plot(
    eval_steps_spectron,
    acc_history_spectron,
    label="SpectronMQAR",
    marker="s",
    markersize=4,
    linestyle="-",
    linewidth=1,
    color="green",
)
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Token-Level Accuracy (%)", fontsize=12)
ax2.set_title("Validation Accuracy Over Time (Token-Level)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True)

# Add final accuracy text
final_acc_text = (
    f"Final Accuracy:\nTransformer: {acc_history_trans[-1]:.2f}%\nSpectronMQAR: {acc_history_spectron[-1]:.2f}%"
)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax2.text(0.05, 0.95, final_acc_text, transform=ax2.transAxes, fontsize=12, verticalalignment="top", bbox=props)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Example Predictions
# -----------------------------------------------------------------------------
def show_example_predictions(model, model_name: str, attn_mask=None):
    model.eval()
    with torch.no_grad():
        first_input, first_target = train_dataset[0]
        last_input, last_target = train_dataset[-1]
        first_input = first_input.unsqueeze(0).to(device)
        last_input = last_input.unsqueeze(0).to(device)

        if isinstance(model, TransformerMQARModel):
            first_logits = model(first_input, mask=attn_mask)
            last_logits = model(last_input, mask=attn_mask)
        else:
            first_logits = model(first_input)
            last_logits = model(last_input)

        first_pred = first_logits.argmax(dim=-1).squeeze(0).cpu()
        last_pred = last_logits.argmax(dim=-1).squeeze(0).cpu()

        print(f"\n{model_name} - First Example")
        print("Input:     ", first_input.squeeze().cpu().tolist())
        print("Target:    ", first_target.cpu().tolist())
        print("Predicted: ", first_pred.tolist())
        print(f"\n{model_name} - Last Example")
        print("Input:     ", last_input.squeeze().cpu().tolist())
        print("Target:    ", last_target.cpu().tolist())
        print("Predicted: ", last_pred.tolist())


print("\n--- Predictions from TransformerMQARModel ---")
show_example_predictions(trans_mqar_model, "TransformerMQARModel", attn_mask=causal_mask)

print("\n--- Predictions from SpectronMQAR ---")
show_example_predictions(spectron_mqar, "SpectronMQAR")
