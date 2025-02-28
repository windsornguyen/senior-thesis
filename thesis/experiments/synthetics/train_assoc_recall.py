import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from thesis.experiments.synthetics.assoc_recall import generate_assoc_recall


def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


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
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, T, T)

        # Apply causal mask if provided
        if mask is not None:
            # Expand mask for multiple heads: (1, 1, T, T) -> (B, num_heads, T, T)
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


class Transformer(nn.Module):
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
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Create and register causal mask buffer
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Add positional embeddings to help with the copying task
        x_embed = self.embedding(x) + self.pe.unsqueeze(0)
        # Use the model's causal mask if no mask is provided
        mask = mask if mask is not None else self.causal_mask
        enc_out = self.encoder(x_embed, mask=mask)
        logits = self.out_proj(enc_out)
        return logits


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
            T2 = [a - b for a, b in zip(T2, padded_T0, strict=True)]
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


def poly_mul_x(poly):
    # Multiply polynomial by x: shift coefficients right by one index.
    return [0] + poly


def poly_scale(poly, factor):
    # Scale polynomial coefficients by factor.
    return [coef * factor for coef in poly]


def poly_sub(poly1, poly2):
    # Subtract poly2 from poly1; extend with zeros if necessary.
    length = max(len(poly1), len(poly2))
    result = []
    for i in range(length):
        coef1 = poly1[i] if i < len(poly1) else 0
        coef2 = poly2[i] if i < len(poly2) else 0
        result.append(coef1 - coef2)
    return result


def chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x)
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    T_nm2 = [1]  # T_0(x)
    T_nm1 = [0, 1]  # T_1(x)
    for _ in range(2, n + 1):
        # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        term = poly_mul_x(T_nm1)
        term = poly_scale(term, 2)
        T_n = poly_sub(term, T_nm2)
        T_nm2, T_nm1 = T_nm1, T_n
    return T_n


def normalized_chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x) normalized by 2**(n-1).
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    coeff = chebyshev_coeff(n)
    leading_term = coeff[-1]
    return [c / leading_term for c in coeff]


def integrate_polar_monomial(a, b, beta):
    """
    Compute the integral of z^a * z̄^b over the polar wedge:
      r ∈ [0, 1], θ ∈ [-beta, beta],
    in closed form:
      if a==b: 2*beta/(a+b+2)
      else:   2*sin((a-b)*beta)/((a-b)*(a+b+2))
    Here a and b are tensors (floats).
    """
    diff = a - b
    denom = a + b + 2
    return torch.where(
        condition=diff == 0,
        input=2 * beta / denom,
        other=2 * torch.sin(diff * beta) / (diff * denom),
    )


def get_polynomial_hankel(n, beta, t, chunk_size=2048, device="cuda", dtype=torch.bfloat16):
    """ """
    matrix_size = t - n

    # Compute Chebyshev coefficients
    poly_coeff = normalized_chebyshev_coeff(n)
    poly = torch.tensor(poly_coeff, device=device)  # (n+1,)

    # Outer product of polynomial coefficients
    P = torch.outer(poly, poly).unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

    # Precompute the index arrays for the summation indices (ii, jj)
    ii = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    jj = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    ii, jj = torch.meshgrid(ii, jj, indexing="ij")  # (n+1, n+1)
    ii = ii.unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)
    jj = jj.unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

    # Allocate the result matrix
    Z = torch.empty((matrix_size, matrix_size), dtype=torch.complex64, device=device)

    # Process in chunks to save memory.
    for i_start in range(0, matrix_size, chunk_size):
        # Create i indices
        i_end = min(i_start + chunk_size, matrix_size)
        i_vals = torch.arange(i_start, i_end, device=device, dtype=torch.float32).view(-1, 1, 1, 1)

        for j_start in range(0, matrix_size, chunk_size):
            # Create j indices
            j_end = min(j_start + chunk_size, matrix_size)
            j_vals = torch.arange(j_start, j_end, device=device, dtype=torch.float32).view(1, -1, 1, 1)

            # Compute A and B for the chunk: shape (chunk_i, chunk_j, n+1, n+1)
            A = i_vals + ii  # Compute i + ii for each chunk element
            B = j_vals + jj

            # Compute the closed-form integral for each (i+ii, j+jj)
            int_vals = integrate_polar_monomial(A, B, beta)

            # Multiply by P and sum over the polynomial indices to yield the (i,j) entry
            chunk_Z = torch.sum(int_vals * P, dim=(2, 3))

            # Write the computed chunk to the result matrix.
            Z[i_start:i_end, j_start:j_end] = chunk_Z.to(torch.complex64)

    return Z


def get_opt_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7 / 6) * math.log2(seq_len)))


def get_polynomial_spectral_filters(
    seq_len: int,
    k: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    n = get_opt_degree(seq_len)
    beta = 1.0 / (64.0 * n**2)
    Z = get_polynomial_hankel(n, beta, seq_len + n, device=device)
    _, phi = torch.linalg.eigh(Z, UPLO="U")
    phi_k = phi[:, -k:] / math.sqrt(seq_len)

    # Validate that the eigenvectors are real since Z is Hermitian
    if torch.abs(phi_k.imag).max() > 1e-7:
        raise ValueError("Unexpectedly large imaginary components in eigenvectors")

    # Take real part only (imaginary part is due to floating point imprecision)
    return phi_k.real.to(dtype)


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
    It projects inputs into a (B, T, seq_len) space, applies linear transformations
    to obtain queries (Q), and obtains spectral filters (K, V) for keys and values.
    The attention output is produced via a sequence of einsum operations,
    finally projected back to the model dimension.

    Args:
        seq_len (int): Sequence length (T).
        d_model (int): Model dimension.
        k (int): The projection dimension for Q, K, and V filters.
        use_hankel_L (bool): Whether to use a Hankel matrix in computing the attention.
        device: Torch device.
    """

    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        dtype = torch.bfloat16

        # Optionally project input from d_model -> seq_len
        self.pre_proj = (
            nn.Linear(d_model, seq_len, device=device, dtype=dtype) if d_model != seq_len else nn.Identity()
        )

        # Query, Key, Value transforms
        self.q_proj = nn.Linear(seq_len, k, device=device, dtype=dtype)
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype=dtype)
        self.V = LearnableSpectralFilters(seq_len, k, use_hankel_L, device, dtype=dtype)
        self.o_proj = nn.Linear(k, d_model, device=device, dtype=dtype)

        # Normalization
        self.norm = nn.RMSNorm(d_model)

        # Hankel matrix L (shape [T, T]); masked to be lower-triangular
        L = get_hankel(seq_len, use_hankel_L, device=device).to(dtype)
        self.L = nn.Parameter(torch.tril(L))

    def forward(self, x: torch.Tensor, chunk_len: int = 128) -> torch.Tensor:
        """
        Forward pass of SpectralAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).
            chunk_len (int): Chunk size for potential block-sum approach (unused).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        B, T, d = x.shape

        # Optional pre-projection to match internal shape
        x_proj = self.pre_proj(x)

        # Compute query
        Q = self.q_proj(x_proj)

        # Compute Key and Value spectral filters
        K_ = torch.einsum("bti,ik->btk", x_proj, self.K())
        V_ = torch.einsum("bti,ik->btk", x_proj, self.V())

        # Outer product of K_ and V_ to get (B, T, k, k) but we keep dim naming
        Z = torch.einsum("btp,btn->btpn", V_, K_)

        # Expand Hankel matrix L for each batch
        L_batched = self.L.unsqueeze(0).expand(B, -1, -1)

        # Multiply L * Z => shape (B, T, p, n)
        H = torch.einsum("bts,bspn->btpn", L_batched, Z)

        # Multiply Q * H => shape (B, T, n) => final
        Y = torch.einsum("btk,btkn->btn", Q, H)

        return self.norm(self.o_proj(Y))


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
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpectralAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        x = x + self.spectral_attention(x)
        x = x + self.mlp(x)
        return self.norm(x)


class Spectron(nn.Module):
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
            [SpectralAttention(seq_len, d_model, k, use_hankel_L, device=device) for _ in range(num_layers)]
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
            logits = model(inputs, mask=attn_mask) if isinstance(model, Transformer) else model(inputs)
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
            logits = model(inputs, mask=attn_mask) if isinstance(model, Transformer) else model(inputs)
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
random_non_queries = True
num_queries = 1
num_layers = 2
num_heads = 4
dropout = 0.0
k = 16

# Create causal mask
causal_mask = build_causal_mask(seq_len).to(device)

# Create training dataset
train_dataset = generate_assoc_recall(
    num_examples=num_examples,
    sequence_len=seq_len,
    vocab_size=vocab_size,
    num_pairs=num_pairs,
    random_non_queries=random_non_queries,
    num_queries=num_queries,
    seed=1746,
)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create validation dataset
val_dataset = generate_assoc_recall(
    num_examples=num_examples // 20,
    sequence_len=seq_len,
    vocab_size=vocab_size,
    num_pairs=num_pairs,
    random_non_queries=random_non_queries,
    num_queries=num_queries,
    seed=1747,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Train Models
# -----------------------------------------------------------------------------

print("\nTraining Transformer...")
transformer = Transformer(
    seq_len=seq_len,
    d_model=d_model,
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout,
).to(device)

compiled_transformer = torch.compile(transformer)
loss_history_trans, acc_history_trans, eval_steps_trans = train_model(
    compiled_transformer, loader, val_loader, attn_mask=causal_mask, max_steps=20000, eval_interval=250
)

print("\nTraining Spectron...")
spectron = Spectron(
    seq_len=seq_len,
    d_model=d_model,
    k=k,
    vocab_size=vocab_size,
    num_layers=num_layers,
    dropout=dropout,
    use_hankel_L=False,
    device=device,
).to(device, dtype=torch.bfloat16)

compiled_spectron = torch.compile(spectron)
loss_history_spectron, acc_history_spectron, eval_steps_spectron = train_model(
    compiled_spectron, loader, val_loader, max_steps=20000, eval_interval=250
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
    label="Spectron",
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
    label="Spectron",
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
    f"Final Accuracy:\nTransformer: {acc_history_trans[-1]:.2f}%\nSpectron: {acc_history_spectron[-1]:.2f}%"
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

        if isinstance(model, Transformer):
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


print("\n--- Predictions from Transformer ---")
show_example_predictions(transformer, "Transformer", attn_mask=causal_mask)

print("\n--- Predictions from Spectron ---")
show_example_predictions(spectron, "Spectron")
