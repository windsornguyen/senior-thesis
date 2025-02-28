import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Mode Task Dataset Generation
# -----------------------------------------------------------------------------
def generate_mode(
    num_examples: int = 10,
    sequence_len: int = 128,
    num_classes: int = 5,
    seed: int = 1337,
) -> TensorDataset:
    """
    Generate a mode tracking task dataset.

    In this task, each example consists of a sequence of integers. The target at each time step is the mode
    (most frequent element) of all elements seen so far in the sequence. In the case of a tie, the smallest
    integer is chosen as the mode.
    """
    torch.manual_seed(seed)

    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")
    if sequence_len < 1:
        raise ValueError("sequence_len must be at least 1.")

    # Generate input sequences: integers sampled uniformly from [0, num_classes)
    inputs = torch.randint(0, num_classes, (num_examples, sequence_len), dtype=torch.long)

    # Initialize the targets tensor
    targets = torch.zeros_like(inputs)

    # Initialize counts tensor
    counts = torch.zeros((num_examples, num_classes), dtype=torch.float)

    # Iterate over each time step
    for j in range(sequence_len):
        current_elements = inputs[:, j]
        counts[torch.arange(num_examples), current_elements] += 1
        # Determine the mode for each example
        targets[:, j] = torch.argmax(counts, dim=1)

    return TensorDataset(inputs, targets)


# -----------------------------------------------------------------------------
# Helper: Build Causal Mask (for transformer)
# -----------------------------------------------------------------------------
def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


# -----------------------------------------------------------------------------
# Custom Transformer Components (same as before)
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


# -----------------------------------------------------------------------------
# Spectral Filters and Spectral Attention (DO NOT ALTER)
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
    seq_len: int, K: int, use_hankel_L: bool = False, device: torch.device = None, dtype=torch.float32
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z, UPLO="U")
    tol = 1e-9
    if not (sigma > -tol).all():
        raise ValueError(f"Expected positive eigenvalues, got min={sigma.min()}")
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


# -----------------------------------------------------------------------------
# Mode Task Models
# -----------------------------------------------------------------------------
class TransformerModeModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(num_classes, d_model)

        # Create positional embeddings
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(seq_len, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_embedding", pos_embedding)

        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_embed = self.embedding(x)  # (B, T, d_model)
        # Add positional embeddings
        x_embed = x_embed + self.pos_embedding.unsqueeze(0)
        enc_out = self.encoder(x_embed, mask=mask)  # (B, T, d_model)
        logits = self.out_proj(enc_out)  # (B, T, num_classes)
        return logits


class SpectronMode(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, num_classes: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(num_classes, d_model)
        self.spec_attn = SpectralAttention(seq_len, d_model, k, use_hankel_L, device)
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        out = x_emb + self.spec_attn(x_emb)
        logits = self.out_proj(out)
        return logits


# -----------------------------------------------------------------------------
# Training Loop for the Mode Task (Classification)
# -----------------------------------------------------------------------------
def train_model(model, loader, attn_mask=None, epochs: int = 20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if isinstance(model, TransformerModeModel):
                logits = model(inputs, mask=attn_mask)
            elif isinstance(model, SpectronMode):
                logits = model(inputs)
            else:
                logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")


# -----------------------------------------------------------------------------
# Device Setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------------------------------------------
# Dataset & DataLoader for the Mode Task
# -----------------------------------------------------------------------------
dataset = generate_mode(num_examples=256, sequence_len=64, num_classes=5)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
causal_mask = build_causal_mask(64).to(device)

# -----------------------------------------------------------------------------
# Instantiate & Train the Models
# -----------------------------------------------------------------------------
print("\nTraining TransformerModeModel...")
trans_mode_model = TransformerModeModel(
    seq_len=64, d_model=32, num_classes=5, num_layers=2, num_heads=1, dropout=0.1
).to(device)
train_model(trans_mode_model, loader, attn_mask=causal_mask, epochs=50)

print("\nTraining SpectronMode...")
spectron_mode = SpectronMode(seq_len=64, d_model=32, k=16, num_classes=5, use_hankel_L=False, device=device).to(device)
train_model(spectron_mode, loader, epochs=50)


# -----------------------------------------------------------------------------
# Display Example Predictions
# -----------------------------------------------------------------------------
def show_example_predictions(model, model_name: str, attn_mask=None):
    model.eval()
    with torch.no_grad():
        first_input, first_target = dataset[0]
        last_input, last_target = dataset[-1]
        first_input = first_input.unsqueeze(0).to(device)
        last_input = last_input.unsqueeze(0).to(device)
        if isinstance(model, TransformerModeModel):
            first_logits = model(first_input, mask=attn_mask)
            last_logits = model(last_input, mask=attn_mask)
        elif isinstance(model, SpectronMode):
            first_logits = model(first_input)
            last_logits = model(last_input)
        else:
            first_logits = model(first_input)
            last_logits = model(last_input)
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


print("\n--- Predictions from TransformerModeModel ---")
show_example_predictions(trans_mode_model, "TransformerModeModel", attn_mask=causal_mask)

print("\n--- Predictions from SpectronMode ---")
show_example_predictions(spectron_mode, "SpectronMode")
