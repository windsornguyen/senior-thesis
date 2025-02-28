import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from thesis.experiments.synthetics.in_context_recall.fuzzy import generate_fuzzy_in_context_recall_instance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")


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
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
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


class TransformerRecallModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        # Create sinusoidal positional embeddings
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Create and register causal mask buffer
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_embed = self.embedding(x) + self.pe.unsqueeze(0)
        mask = mask if mask is not None else self.causal_mask
        enc_out = self.encoder(x_embed, mask=mask)
        logits = self.out_proj(enc_out)
        return logits


# -----------------------------------------------------------------------------
# Spectral Components
# -----------------------------------------------------------------------------
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
    Z = get_hankel(seq_len, use_hankel_L).to(device)
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


class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        self.q_proj = nn.Linear(seq_len, k).to(device)
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.V = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.o_proj = nn.Linear(k, d_model).to(device)
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))
        self.L = nn.Parameter(get_hankel(seq_len, use_hankel_L, device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)  # (B, T, seq_len)

        # Compute Q: (B, T, k)
        Q = self.q_proj(x_proj)

        # Compute K and V: (B, T, k)
        K = torch.einsum("bti,ik->btk", x_proj, self.K())
        V = torch.einsum("bti,ik->btk", x_proj, self.V())

        Z = torch.einsum("btp,btn,t->btpn", V, K, self.decay)

        L_masked = torch.tril(self.L).unsqueeze(0)  # (1, T, T)

        H = torch.einsum("bts,bspn->btpn", L_masked, Z)
        Y = torch.einsum("btk,btkn->btn", Q, H)

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


class Spectron(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        vocab_size: int,
        d_out: int = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_hankel_L: bool = False,
        device=None,
    ):
        super().__init__()
        if d_out is None:
            d_out = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [SpectralAttentionLayer(seq_len, d_model, k, use_hankel_L, device) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.in_dropout(self.embedding(x))
        out = x_emb
        for layer in self.layers:
            out = out + layer(out)
        out = self.norm(out)
        return self.out_dropout(self.out_proj(out))


# -----------------------------------------------------------------------------
# Training and Evaluation Functions
# -----------------------------------------------------------------------------
def compute_token_level_accuracy(model, loader, attn_mask=None, device=device):
    """
    Compute token-level accuracy while ignoring special tokens and target_ignore_idx.
    This is a general metric that considers all valid positions.
    """
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerRecallModel) else model(inputs)
            predictions = logits.argmax(dim=-1)
            valid_mask = targets != -100
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().item()
            total_tokens += valid_mask.sum().item()

    token_acc = 100.0 * correct_tokens / (total_tokens if total_tokens > 0 else 1)
    print(f"Overall Token-Level Accuracy: {token_acc:.2f}%")
    return token_acc


def compute_recall_accuracy(model, loader, v_motif_size: int, attn_mask=None, device=device):
    """
    Compute accuracy specifically on positions where we test key-value recall.
    For fuzzy recall, we need to consider that values are motifs of size v_motif_size.
    """
    model.eval()
    correct_recalls = 0
    total_recalls = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerRecallModel) else model(inputs)
            predictions = logits.argmax(dim=-1)

            # For fuzzy recall, we need to check if the entire value motif is correct
            B = inputs.size(0)
            for b in range(B):
                for t in range(0, targets.size(1) - v_motif_size + 1):
                    # Check if this position starts a recall value motif
                    if all(targets[b, t : t + v_motif_size] != -100):
                        total_recalls += 1
                        if all(predictions[b, t : t + v_motif_size] == targets[b, t : t + v_motif_size]):
                            correct_recalls += 1

    if total_recalls == 0:
        print("Warning: No recall positions found in batch!")
        return 0.0

    recall_acc = 100.0 * correct_recalls / total_recalls
    print(f"Key-Value Recall Accuracy: {recall_acc:.2f}% ({correct_recalls}/{total_recalls} motifs)")
    return recall_acc


def train_model(
    model, loader, val_loader, v_motif_size: int, attn_mask=None, max_steps: int = 10000, eval_interval: int = 50
):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    loss_history = []
    overall_accuracy_history = []
    recall_accuracy_history = []
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
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerRecallModel) else model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            loss_history.append(total_loss)
            step += 1

            if step % eval_interval == 0:
                overall_acc = compute_token_level_accuracy(model, val_loader, attn_mask=attn_mask)
                recall_acc = compute_recall_accuracy(model, val_loader, v_motif_size, attn_mask=attn_mask)
                overall_accuracy_history.append(overall_acc)
                recall_accuracy_history.append(recall_acc)
                eval_steps.append(step)
                print(f"Step {step}/{max_steps}")
                print(f"Loss: {total_loss:.4f}")
                print(f"Overall Accuracy: {overall_acc:.2f}%")
                print(f"Recall Accuracy: {recall_acc:.2f}%")
                print("-" * 50)
            else:
                print(f"Step {step}/{max_steps} Loss: {total_loss:.4f}")

    return loss_history, overall_accuracy_history, recall_accuracy_history, eval_steps


def show_example_predictions(model, model_name: str, dataset, v_motif_size: int, attn_mask=None):
    model.eval()
    with torch.no_grad():
        first_input, first_target = dataset[0]
        last_input, last_target = dataset[-1]
        first_input = first_input.unsqueeze(0).to(device)
        last_input = last_input.unsqueeze(0).to(device)

        if isinstance(model, TransformerRecallModel):
            first_logits = model(first_input, mask=attn_mask)
            last_logits = model(last_input, mask=attn_mask)
        else:
            first_logits = model(first_input)
            last_logits = model(last_input)

        first_pred = first_logits.argmax(dim=-1).squeeze(0).cpu()
        last_pred = last_logits.argmax(dim=-1).squeeze(0).cpu()

        print(f"\n{model_name} - First Example")
        print("Input:     ", dataset[0][0].cpu().tolist())
        print("Target:    ", first_target.cpu().tolist())
        print("Predicted: ", first_pred.tolist())

        # Find and highlight recall positions
        for t in range(len(first_target) - v_motif_size + 1):
            if all(first_target[t : t + v_motif_size] != -100):
                print(f"Recall at position {t}:")
                print(f"  Expected: {first_target[t:t+v_motif_size].tolist()}")
                print(f"  Got:      {first_pred[t:t+v_motif_size].tolist()}")
                print()

        print(f"\n{model_name} - Last Example")
        print("Input:     ", dataset[-1][0].cpu().tolist())
        print("Target:    ", last_target.cpu().tolist())
        print("Predicted: ", last_pred.tolist())

        # Find and highlight recall positions
        for t in range(len(last_target) - v_motif_size + 1):
            if all(last_target[t : t + v_motif_size] != -100):
                print(f"Recall at position {t}:")
                print(f"  Expected: {last_target[t:t+v_motif_size].tolist()}")
                print(f"  Got:      {last_pred[t:t+v_motif_size].tolist()}")
                print()


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset parameters
    vocab_size = 16
    seq_len = 128
    num_examples = 10000
    k_motif_size = 3  # size of key motifs
    v_motif_size = 2  # size of value motifs
    noise_vocab_size = 8
    frac_noise = 0.0

    # Generate training dataset
    train_dataset = generate_fuzzy_in_context_recall_instance(
        num_examples=num_examples,
        vocab_size=vocab_size,
        seq_len=seq_len,
        k_motif_size=k_motif_size,
        v_motif_size=v_motif_size,
        is_training=True,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
        device=device,
    )
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Generate validation dataset
    val_dataset = generate_fuzzy_in_context_recall_instance(
        num_examples=num_examples // 20,
        vocab_size=vocab_size,
        seq_len=seq_len,
        k_motif_size=k_motif_size,
        v_motif_size=v_motif_size,
        is_training=False,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
        device=device,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create and train TransformerRecallModel
    trans_recall_model = TransformerRecallModel(
        seq_len=seq_len,
        d_model=64,
        vocab_size=vocab_size,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    # compiled_trans_model = torch.compile(trans_recall_model, fullgraph=True)
    print("\nTraining TransformerRecallModel...")
    loss_history_trans, overall_accuracy_history_trans, recall_accuracy_history_trans, eval_steps_trans = train_model(
        trans_recall_model, loader, val_loader, v_motif_size, max_steps=10000, eval_interval=500
    )

    # Create and train Spectron
    spectron = Spectron(
        seq_len=seq_len,
        d_model=64,
        k=math.ceil(math.log(seq_len)),
        vocab_size=vocab_size,
        d_out=vocab_size,
        num_layers=2,
        dropout=0.1,
        use_hankel_L=False,
        device=device,
    ).to(device=device, dtype=torch.bfloat16)

    # compiled_spectron = torch.compile(spectron, fullgraph=True)
    print("\nTraining Spectron...")
    loss_history_spectron, overall_accuracy_history_spectron, recall_accuracy_history_spectron, eval_steps_spectron = (
        train_model(spectron, loader, val_loader, v_motif_size, max_steps=10000, eval_interval=500)
    )

    # Plot results
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot loss histories
    ax1.plot(loss_history_trans, label="Transformer", color="blue", alpha=0.7)
    ax1.plot(loss_history_spectron, label="Spectron", color="green", alpha=0.7)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot overall accuracy histories
    ax2.plot(eval_steps_trans, overall_accuracy_history_trans, label="Transformer", color="blue", marker="o")
    ax2.plot(eval_steps_spectron, overall_accuracy_history_spectron, label="Spectron", color="green", marker="s")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Overall Token-Level Accuracy (%)")
    ax2.set_title("Overall Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Plot recall accuracy histories
    ax3.plot(eval_steps_trans, recall_accuracy_history_trans, label="Transformer", color="blue", marker="o")
    ax3.plot(eval_steps_spectron, recall_accuracy_history_spectron, label="Spectron", color="green", marker="s")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Key-Value Recall Accuracy (%)")
    ax3.set_title("Recall Validation Accuracy")
    ax3.legend()
    ax3.grid(True)

    # Add final accuracy text box
    final_acc_text = (
        f"Final Accuracies:\n"
        f"Transformer:\n"
        f"  Overall: {overall_accuracy_history_trans[-1]:.2f}%\n"
        f"  Recall:  {recall_accuracy_history_trans[-1]:.2f}%\n"
        f"Spectron:\n"
        f"  Overall: {overall_accuracy_history_spectron[-1]:.2f}%\n"
        f"  Recall:  {recall_accuracy_history_spectron[-1]:.2f}%"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax3.text(1.05, 0.95, final_acc_text, transform=ax3.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    plt.tight_layout()
    plt.show()

    # Show example predictions
    print("\n--- Example Predictions ---")
    show_example_predictions(trans_recall_model, "Transformer", train_dataset, v_motif_size)
    show_example_predictions(spectron, "Spectron", train_dataset, v_motif_size)
