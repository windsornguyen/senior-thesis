import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Device Setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")


# -----------------------------------------------------------------------------
# Dataset Generation for COPY Task (supports selective and non-selective)
# -----------------------------------------------------------------------------
def generate_copy(
    num_examples: int = 10,
    vocab_size: int = 26,  # number of regular tokens (e.g., letters)
    copy_len: int = 10,
    blank_len: int = 10,
    selective: bool = False,
    one_hot: bool = True,
    seed: int = 1746,
    dtype: str | torch.dtype = torch.bfloat16,
) -> TensorDataset:
    """
    Generate a copy task dataset using explicit special tokens.

    Special tokens:
      - '$': BOS token
      - '|': Delimiter token
      - '.': EOS token
      - '*': BLANK token (selective mode only)

    Non-selective:
      Input: [BOS] + copy_tokens + [Delimiter, EOS]
      Target: copy_tokens

    Selective:
      Input: [BOS] + inserted_sequence (mix of blanks and copy tokens) + [Delimiter, EOS]
      Target: [BLANK repeated blank_len] + copy_tokens
    """
    torch.manual_seed(seed)
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    bos_token = vocab_size  # '$'
    delimiter_token = vocab_size + 1  # '|'
    eos_token = vocab_size + 2  # '.'
    blank_token = vocab_size + 3  # '*'
    total_vocab_size = vocab_size + 4

    if not selective:
        # Non-selective mode
        to_copy = torch.randint(low=0, high=vocab_size, size=(num_examples, copy_len))
        bos_column = torch.full((num_examples, 1), bos_token, dtype=torch.long)
        delimiter_column = torch.full((num_examples, 1), delimiter_token, dtype=torch.long)
        eos_column = torch.full((num_examples, 1), eos_token, dtype=torch.long)
        inputs = torch.cat((bos_column, to_copy, delimiter_column, eos_column), dim=1)
        targets = to_copy.clone()
    else:
        # Selective mode
        to_copy = torch.randint(low=0, high=vocab_size, size=(num_examples, copy_len))

        def insert_selective(row: torch.Tensor) -> torch.Tensor:
            seq_len = copy_len + blank_len
            inserted = torch.full((seq_len,), blank_token, dtype=row.dtype)
            positions = torch.randperm(seq_len)[:copy_len]
            positions, _ = torch.sort(positions)
            inserted[positions] = row
            return inserted

        inserted_rows = torch.stack([insert_selective(row) for row in to_copy])
        bos_column = torch.full((num_examples, 1), bos_token, dtype=torch.long)
        delimiter_column = torch.full((num_examples, 1), delimiter_token, dtype=torch.long)
        eos_column = torch.full((num_examples, 1), eos_token, dtype=torch.long)
        inputs = torch.cat((bos_column, inserted_rows, delimiter_column, eos_column), dim=1)
        target_blanks = torch.full((num_examples, blank_len), blank_token, dtype=torch.long)
        targets = torch.cat((target_blanks, to_copy), dim=1)

    if one_hot:
        inputs = F.one_hot(inputs, num_classes=total_vocab_size).to(dtype)
        targets = F.one_hot(targets, num_classes=total_vocab_size).to(dtype)

    return TensorDataset(inputs, targets)


# -----------------------------------------------------------------------------
# Custom Transformer Components
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


# -----------------------------------------------------------------------------
# Transformer-based Model for Copy Task
# -----------------------------------------------------------------------------
class TransformerCopyModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Valid token indices: 0..(vocab_size+3)
        self.seq_len = seq_len
        self.d_model = d_model

        # Create sinusoidal positional embeddings
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.embedding = nn.Embedding(vocab_size + 4, d_model)
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size + 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_embed = self.embedding(x)
        # Add positional embeddings
        x_embed = x_embed + self.pe.unsqueeze(0)
        enc_out = self.encoder(x_embed, mask=mask)
        logits = self.out_proj(enc_out)
        return logits


# -----------------------------------------------------------------------------
# Spectral Attention Components
# -----------------------------------------------------------------------------
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
    def __init__(self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters

class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        # Pre-project x to shape (B, T, seq_len) if needed.
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        # Q projection: maps pre-projected input to k dimensions.
        self.q_proj = nn.Linear(seq_len, k).to(device)
        # Learnable spectral filters for K and V.
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.V = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        # Final projection from k back to d_model.
        self.o_proj = nn.Linear(k, d_model).to(device)
        # Decay parameter: one per time step.
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))
        # Hankel matrix L (shape: [T, T]); we'll mask it to be lower-triangular.
        self.L = nn.Parameter(get_hankel(seq_len, use_hankel_L, device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)  # (B, T, seq_len)

        # Compute Q: (B, T, k)
        Q = self.q_proj(x_proj)
        
        # Compute K and V: (B, T, k)
        K = torch.einsum("bti,ik->btk", x_proj, self.K())
        V = torch.einsum("bti,ik->btk", x_proj, self.V())
        
        # Compute Z as the outer product between V and K per time step,
        # scaled by the decay factor. In our notation:
        #   V: (B, T, k) with indices "btp"
        #   K: (B, T, k) with indices "btn"
        #   decay: (T,) with index "t"
        # Result Z: (B, T, k, k)
        Z = torch.einsum("btp,btn,t->btpn", V, K, self.decay)
        
        # Prepare the Hankel mask: force L to be lower-triangular and reshape to (1, T, T)
        # so it broadcasts over batch.
        L_masked = torch.tril(self.L).unsqueeze(0)  # (1, T, T)
        
        # Aggregate Z over time using L_masked.
        # Here we interpret L_masked as weighting contributions from all time steps s to output time t.
        # That is, for each b, t, p, n:
        #    H[b,t,p,n] = sum_s L_masked[0,t,s] * Z[b,s,p,n]
        H = torch.einsum("bts,bspn->btpn", L_masked, Z)
        
        # Query the aggregated result with Q.
        # Q: (B, T, k), H: (B, T, k, k) → Y: (B, T, k)
        Y = torch.einsum("btk,btkn->btn", Q, H)
        
        return self.o_proj(Y)


class Spectron(nn.Module):
    def __init__(
        self, seq_len: int, d_model: int, k: int, vocab_size: int, d_out: int, use_hankel_L: bool = False, device=None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 4, d_model)
        self.spec_attn = SpectralAttention(seq_len, d_model, k, use_hankel_L, device)
        self.out_proj = nn.Linear(d_model, d_out)

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
    Compute token-level accuracy while ignoring delimiter (65), eos (66) and blank (67) tokens.
    Note: During evaluation we shift the model output to remove the BOS token.
    """
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # simplified logic for model forward pass
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCopyModel) else model(inputs)
            # Shift logits by 1 to drop BOS token:
            target_seq_len = targets.size(1)
            logits = logits[:, 1 : 1 + target_seq_len, :]
            predictions = logits.argmax(dim=-1)
            # Create a mask to ignore delimiter (65), eos (66) and blank (67) tokens:
            ignore_mask = (targets != 65) & (targets != 66) & (targets != 67)
            match = (predictions == targets) & ignore_mask
            correct_tokens += match.sum().item()
            total_tokens += ignore_mask.sum().item()
            # if batch_idx == 0:
            #     print("\nDEBUG INFO (Token-level):")
            #     for i in range(min(2, len(inputs))):
            #         print(f"\nExample {i}:")
            #         print("Input:     ", inputs[i].cpu().tolist())
            #         print("Target:    ", targets[i].cpu().tolist())
            #         print("Predicted: ", predictions[i].cpu().tolist())
            #         print("Match Mask:", match[i].cpu().tolist())
            #     print()
    token_acc = 100.0 * correct_tokens / (total_tokens if total_tokens > 0 else 1)
    print(f"Token-Level Accuracy (ignoring tokens 65,66,67): {token_acc:.2f}%")
    return token_acc


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_model(model, loader, val_loader, attn_mask=None, max_steps: int = 10000, eval_interval: int = 50):
    # During training, we include the full sequence except we ignore blank tokens (67)
    criterion = nn.CrossEntropyLoss(ignore_index=29)
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
            # simplified logic for model forward pass
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCopyModel) else model(inputs)
            target_seq_len = targets.size(1)
            # Shift logits by 1 (drop BOS) so that predictions align with the target.
            logits = logits[:, 1 : 1 + target_seq_len, :]
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
    # Parameter setup from OmegaConf resolvers:
    # For non-selective mode:
    #   copy_len = 509, so sequence length = 509 + 3 = 512
    #   total vocab = 26 + 4 = 30
    selective = True  # non-selective mode
    copy_len = 32
    blank_len = 32  # not used in non-selective mode
    vocab_size = 26
    one_hot = False
    seed = 1746
    dtype = torch.int64
    num_examples = 400000

    dataset = generate_copy(
        num_examples=num_examples,
        vocab_size=vocab_size,
        copy_len=copy_len,
        blank_len=blank_len,
        selective=selective,
        one_hot=one_hot,
        seed=seed,
        dtype=dtype,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_seq_len = dataset[0][0].size(0)  # should be 512 for non-selective mode
    causal_mask = build_causal_mask(input_seq_len).to(device)

    val_dataset = generate_copy(
        num_examples=20000,  # smaller validation set
        vocab_size=vocab_size,
        copy_len=copy_len,
        blank_len=blank_len,
        selective=selective,
        one_hot=one_hot,
        seed=seed + 1,  # different seed for validation
        dtype=dtype,
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # TransformerCopyModel
    trans_copy_model = TransformerCopyModel(
        seq_len=input_seq_len, d_model=64, vocab_size=vocab_size, num_layers=2, num_heads=4, dropout=0.1
    ).to(device)
    compiled_trans_model = torch.compile(trans_copy_model, fullgraph=True)
    print("\nTraining TransformerCopyModel...")
    loss_history_trans, acc_history_trans, eval_steps_trans = train_model(
        compiled_trans_model, loader, val_loader, attn_mask=causal_mask, max_steps=10000, eval_interval=10000
    )

    # Spectron
    print("\nTraining Spectron...")
    spectron = Spectron(
        seq_len=input_seq_len,
        d_model=64,
        k=16,
        vocab_size=vocab_size,
        d_out=(vocab_size + 4),
        use_hankel_L=False,
        device=device,
    ).to(device=device, dtype=torch.bfloat16)
    compiled_spectron = torch.compile(spectron, fullgraph=True)
    loss_history_spectron, acc_history_spectron, eval_steps_spectron = train_model(
        compiled_spectron, loader, val_loader, max_steps=10000, eval_interval=10000
    )

    # Plot results
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

    # Example predictions
    def show_example_predictions(model, model_name: str, attn_mask=None):
        model.eval()
        with torch.no_grad():
            first_input, first_target = dataset[0]
            last_input, last_target = dataset[-1]
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
            first_logits = first_logits[:, 1 : 1 + target_seq_len, :]
            last_logits = last_logits[:, 1 : 1 + target_seq_len, :]
            first_pred = first_logits.argmax(dim=-1).squeeze(0).cpu()
            last_pred = last_logits.argmax(dim=-1).squeeze(0).cpu()
            print(f"\n{model_name} - First Example")
            print("Input:     ", dataset[0][0])
            print("Target:    ", first_target)
            print("Predicted: ", first_pred)
            print(f"\n{model_name} - Last Example")
            print("Input:     ", dataset[-1][0])
            print("Target:    ", last_target)
            print("Predicted: ", last_pred)

    print("\n--- Predictions from TransformerCopyModel (token-level) ---")
    show_example_predictions(trans_copy_model, "TransformerCopyModel", attn_mask=causal_mask)
    print("\n--- Predictions from Spectron (token-level) ---")
    show_example_predictions(spectron, "Spectron")
