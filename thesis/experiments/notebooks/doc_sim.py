import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import Mamba from thesis.models.mamba
from thesis.models.mamba import MambaConfig, BaseMamba

# -----------------------------------------------------------------------------
# Device Setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------------------------
# Document Similarity Dataset Generation
# -----------------------------------------------------------------------------
def generate_document_similarity(
    num_examples: int = 256,
    num_documents: int = 10,
    num_elements: int = 10,
    top_k: int = 2,
    seed: int = 1_337,
    dtype: torch.dtype = torch.bfloat16,
) -> TensorDataset:
    """
    Generate a dataset for the cosine similarity task. The goal is to find the
    pair of documents (tensors) with the highest cosine similarity.
    """
    torch.manual_seed(seed)

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if num_documents < 2:
        raise ValueError("num_documents must be at least 2 to form pairs.")
    max_topk = num_documents * (num_documents - 1) // 2
    if top_k > max_topk:
        raise ValueError(f"top_k={top_k} exceeds the maximum number of unique document pairs ({max_topk}).")

    inputs = torch.randn((num_examples, num_documents, num_elements), dtype=dtype)
    normalized_inputs = F.normalize(inputs, p=2, dim=2)
    cosine_similarity = normalized_inputs @ normalized_inputs.transpose(1, 2)
    triu_indices = torch.triu_indices(num_documents, num_documents, offset=1)
    sim_pairs = cosine_similarity[:, triu_indices[0], triu_indices[1]]  # (num_examples, num_pairs)
    topk_values, topk_indices = torch.topk(sim_pairs, top_k, dim=1, largest=True, sorted=True)
    topk_pairs = triu_indices[:, topk_indices]  # (2, num_examples, top_k)
    targets = topk_pairs.permute(1, 2, 0)  # (num_examples, top_k, 2)

    return TensorDataset(inputs, targets)


# -----------------------------------------------------------------------------
# Helper Functions for Spectral Attention
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
    phi_j = phi_i  # for simplicity, use the same for both dimensions
    filters = torch.kron(phi_i, phi_j)
    return filters.to(device=device, dtype=dtype)


class LearnableSpectralFilters(nn.Module):
    """
    Learnable spectral filters.
    For simplicity, we initialize these with a tensorized spectral filter.
    """

    def __init__(self, d_model: int, k: int, use_hankel_L: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        filters = get_spectral_filters(d_model, k, use_hankel_L, device, dtype)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


# -----------------------------------------------------------------------------
# Spectral Attention Components (Spectron)
# -----------------------------------------------------------------------------
class SpectralAttention(nn.Module):
    """
    Simplified spectral attention operating in a fixed embedding space.
    Assumes the input is already projected to d_model.
    """

    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.Q_filt = LearnableSpectralFilters(d_model, k, use_hankel_L, device)
        self.K_filt = LearnableSpectralFilters(d_model, k, use_hankel_L, device)
        self.v_proj = nn.Linear(d_model, k).to(device)
        self.o_proj = nn.Linear(k, d_model).to(device)
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, d = x.shape
        Q = torch.einsum("btd,dk->btk", x, self.Q_filt())  # (B, T, k)
        K = torch.einsum("btd,dk->btk", x, self.K_filt())  # (B, T, k)
        V = self.v_proj(x)  # (B, T, k)
        Z = torch.einsum("btp,btn->btpn", V, K)  # (B, T, k, k)
        decay = self.decay.view(1, T, 1, 1)
        Z = Z * decay
        H = torch.cumsum(Z, dim=1)  # (B, T, k, k)
        Y = torch.einsum("btk,btkn->btn", Q, H)  # (B, T, k)
        return self.o_proj(Y)  # (B, T, d_model)


# -----------------------------------------------------------------------------
# Spectron Document Similarity Model (SpectronDoc)
# -----------------------------------------------------------------------------
class SpectronDoc(nn.Module):
    """
    Spectral attention model (SpectronDoc) for the Document Similarity Task.
    """

    def __init__(
        self,
        num_documents: int,
        num_elements: int,
        d_model: int,
        k: int,
        d_out: int,
        use_hankel_L: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_proj = nn.Linear(num_elements, d_model)
        self.spec_attn = SpectralAttention(num_documents, d_model, k, use_hankel_L, device)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_documents, num_elements)
        x_proj = self.in_proj(x)
        out = x_proj + self.spec_attn(x_proj)
        return self.out_proj(out)


# -----------------------------------------------------------------------------
# Transformer Document Similarity Model (TransformerDoc)
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
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
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

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(src, src, src)
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

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src)
        return src


class TransformerDoc(nn.Module):
    """
    Transformer-based model for the Document Similarity Task.
    """

    def __init__(
        self,
        num_documents: int,
        num_elements: int,
        d_model: int,
        d_out: int,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(num_elements, d_model)

        # Create positional embeddings for document positions
        position = torch.arange(num_documents).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(num_documents, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_embedding", pos_embedding)

        self.encoder = CustomTransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.in_proj(x)
        # Add positional embeddings to help model understand document ordering
        x_proj = x_proj + self.pos_embedding.unsqueeze(0)
        enc_out = x_proj + self.encoder(x_proj)
        return self.out_proj(enc_out)


# -----------------------------------------------------------------------------
# Training Loop for Document Similarity Models
# -----------------------------------------------------------------------------
def train_model_doc(model, loader, epochs: int = 20):
    """
    Trains the model by regressing the predicted cosine similarity matrix
    (computed from the document embeddings) to the ground-truth cosine similarity
    matrix (computed from the raw normalized inputs).
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, _ in loader:
            inputs = inputs.to(device).float()
            optimizer.zero_grad()
            pred = model(inputs)
            pred_norm = F.normalize(pred, p=2, dim=2)
            pred_cos = pred_norm @ pred_norm.transpose(1, 2)
            inputs_norm = F.normalize(inputs, p=2, dim=2)
            gt_cos = inputs_norm @ inputs_norm.transpose(1, 2)
            loss = criterion(pred_cos, gt_cos)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")


# -----------------------------------------------------------------------------
# Dataset & DataLoader for Document Similarity Models
# -----------------------------------------------------------------------------
dataset = generate_document_similarity(
    num_examples=256, num_documents=10, num_elements=10, top_k=2, seed=1337, dtype=torch.bfloat16
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------------------------------------------------------
# Instantiate & Train the Document Similarity Models
# -----------------------------------------------------------------------------
print("\nTraining TransformerDoc on Document Similarity Task...")
transformer_doc = TransformerDoc(
    num_documents=10, num_elements=10, d_model=32, d_out=32, num_heads=2, num_layers=2, dropout=0.1
).to(device)
train_model_doc(transformer_doc, loader, epochs=20)

print("\nTraining SpectronDoc on Document Similarity Task...")
spectron_doc = SpectronDoc(
    num_documents=10, num_elements=10, d_model=32, k=16, d_out=32, use_hankel_L=False, device=device
).to(device)
train_model_doc(spectron_doc, loader, epochs=20)


# -----------------------------------------------------------------------------
# Mamba Document Similarity Model (MambaDoc)
# -----------------------------------------------------------------------------
class MambaDoc(nn.Module):
    """
    Mamba-based model for the Document Similarity Task.
    Using fixed dimensions known to work well with Mamba's kernels.
    """

    def __init__(self, num_documents: int, num_elements: int, d_model: int, d_out: int, num_layers: int = 2):
        super().__init__()
        # Use fixed dimensions that work well with Mamba
        d_inner = 128  # known to work well

        self.in_proj = nn.Linear(num_elements, d_inner)

        # Create Mamba config with fixed dimensions
        mamba_config = MambaConfig(
            dim=d_inner,
            num_layers=num_layers,
            num_heads=2,
            vocab_size=1,  # dummy value
            state_dim=128,  # fixed state dimension
            num_groups=1,
            conv_size=4,
            use_mem_eff_path=False,
            bias=False,
            torch_dtype=torch.float32,
            multiple_of=8,
            ffn_dim_multiplier=1.0,
        )

        # Initialize Mamba backbone
        self.backbone = BaseMamba(mamba_config)
        self.out_proj = nn.Linear(d_inner, d_out)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_documents, num_elements)
        x_proj = self.in_proj(x)
        # Ensure contiguous memory layout
        x_proj = x_proj.contiguous()
        out = x_proj + self.backbone(x_proj, tok_idx=None, cu_seqlens=None)
        return self.out_proj(out)


# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
print("\nTraining MambaDoc on Document Similarity Task...")
mamba_doc = MambaDoc(
    num_documents=10,
    num_elements=10,
    d_model=32,  # this will be mapped to d_inner=128 internally
    d_out=32,
).to(device)
train_model_doc(mamba_doc, loader, epochs=20)
