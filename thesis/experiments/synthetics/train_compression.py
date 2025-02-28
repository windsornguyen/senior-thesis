import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from thesis.experiments.synthetics.compression.compression import generate_compression_instance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")


# Reuse model architectures from fuzzy_in_context_recall.py
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


class TransformerCompressionModel(nn.Module):
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
        self.k = k
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        
        self.Q = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.v_proj = nn.Linear(seq_len, k).to(device)

        self.o_proj = nn.Linear(k, d_model).to(device)
        self.L = nn.Parameter(get_hankel(seq_len, use_hankel_L, device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)  # (B, T, seq_len)
        
        # Q computed as a spectral filter: (B, T, k)
        Q = torch.einsum("bti,ik->btk", x_proj, self.Q())
        
        # K remains a spectral filter: (B, T, k)
        K = torch.einsum("bti,ik->btk", x_proj, self.K())
        
        # V is now a linear projection: (B, T, k)
        V = self.v_proj(x_proj)
        
        # Compute Z as the outer product between V and K per time step,
        Z = torch.einsum("btp,btn->btpn", V, K)
        
        # Prepare the Hankel mask: force L to be lower-triangular and add batch dim.
        # L_masked = torch.tril(self.L).unsqueeze(0)  # (1, T, T)
        L_raw = torch.einsum("btk,bsk->bts", Q, K) / math.sqrt(self.k)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)  # (1, T, T)
        
        H = torch.einsum("bts,bspn->btpn", L_raw, Z)

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
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCompressionModel) else model(inputs)
            predictions = logits.argmax(dim=-1)
            valid_mask = targets != -100
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().item()
            total_tokens += valid_mask.sum().item()

    token_acc = 100.0 * correct_tokens / (total_tokens if total_tokens > 0 else 1)
    print(f"Overall Token-Level Accuracy: {token_acc:.2f}%")
    return token_acc


def compute_compression_accuracy(model, loader, attn_mask=None, device=device):
    """
    Compute accuracy specifically on positions after the compression token.
    This measures how well the model can reconstruct the original sequence.
    """
    model.eval()
    correct_compressions = 0
    total_compressions = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCompressionModel) else model(inputs)
            predictions = logits.argmax(dim=-1)

            # Find positions where we're testing compression (targets != -100)
            compression_positions = targets != -100

            # Calculate accuracy only on compression positions
            match = (predictions == targets) & compression_positions
            correct_compressions += match.sum().item()
            total_compressions += compression_positions.sum().item()

    if total_compressions == 0:
        print("Warning: No compression positions found in batch!")
        return 0.0

    compression_acc = 100.0 * correct_compressions / total_compressions
    print(f"Compression Accuracy: {compression_acc:.2f}% ({correct_compressions}/{total_compressions} tokens)")
    return compression_acc


def train_model(model, loader, val_loader, attn_mask=None, max_steps: int = 10000, eval_interval: int = 50):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    loss_history = []
    overall_accuracy_history = []
    compression_accuracy_history = []
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
            logits = model(inputs, mask=attn_mask) if isinstance(model, TransformerCompressionModel) else model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            loss_history.append(total_loss)
            step += 1

            if step % eval_interval == 0:
                overall_acc = compute_token_level_accuracy(model, val_loader, attn_mask=attn_mask)
                compression_acc = compute_compression_accuracy(model, val_loader, attn_mask=attn_mask)
                overall_accuracy_history.append(overall_acc)
                compression_accuracy_history.append(compression_acc)
                eval_steps.append(step)
                print(f"Step {step}/{max_steps}")
                print(f"Loss: {total_loss:.4f}")
                print(f"Overall Accuracy: {overall_acc:.2f}%")
                print(f"Compression Accuracy: {compression_acc:.2f}%")
                print("-" * 50)
            else:
                print(f"Step {step}/{max_steps} Loss: {total_loss:.4f}")

    return loss_history, overall_accuracy_history, compression_accuracy_history, eval_steps


def show_example_predictions(model, model_name: str, dataset, attn_mask=None):
    model.eval()
    with torch.no_grad():
        first_input, first_target = dataset[0]
        last_input, last_target = dataset[-1]
        first_input = first_input.unsqueeze(0).to(device)
        last_input = last_input.unsqueeze(0).to(device)

        if isinstance(model, TransformerCompressionModel):
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

        # Find and highlight compression positions
        compression_positions = first_target != -100
        if compression_positions.any():
            print("\nCompression Results:")
            print("  Expected:", first_target[compression_positions].tolist())
            print("  Got:     ", first_pred[compression_positions].tolist())

        print(f"\n{model_name} - Last Example")
        print("Input:     ", dataset[-1][0].cpu().tolist())
        print("Target:    ", last_target.cpu().tolist())
        print("Predicted: ", last_pred.tolist())

        # Find and highlight compression positions
        compression_positions = last_target != -100
        if compression_positions.any():
            print("\nCompression Results:")
            print("  Expected:", last_target[compression_positions].tolist())
            print("  Got:     ", last_pred[compression_positions].tolist())


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset parameters
    vocab_size = 16
    seq_len = 32
    num_examples = 10000
    noise_vocab_size = 4
    frac_noise = 0.1

    # Generate training dataset
    train_dataset = generate_compression_instance(
        num_examples=num_examples,
        vocab_size=vocab_size,
        seq_len=seq_len,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
        device=device,
    )
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Generate validation dataset
    val_dataset = generate_compression_instance(
        num_examples=num_examples // 10,
        vocab_size=vocab_size,
        seq_len=seq_len,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
        device=device,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create and train TransformerCompressionModel
    # trans_compression_model = TransformerCompressionModel(
    #     seq_len=seq_len,
    #     d_model=64,
    #     vocab_size=vocab_size,
    #     num_layers=2,
    #     num_heads=4,
    #     dropout=0.1,
    # ).to(device)

    # print("\nTraining TransformerCompressionModel...")
    # loss_history_trans, overall_accuracy_history_trans, compression_accuracy_history_trans, eval_steps_trans = (
    #     train_model(trans_compression_model, loader, val_loader, max_steps=5000, eval_interval=500)
    # )

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

    print("\nTraining Spectron...")
    (
        loss_history_spectron,
        overall_accuracy_history_spectron,
        compression_accuracy_history_spectron,
        eval_steps_spectron,
    ) = train_model(spectron, loader, val_loader, max_steps=5000, eval_interval=500)

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

    # Plot compression accuracy histories
    ax3.plot(eval_steps_trans, compression_accuracy_history_trans, label="Transformer", color="blue", marker="o")
    ax3.plot(eval_steps_spectron, compression_accuracy_history_spectron, label="Spectron", color="green", marker="s")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Compression Accuracy (%)")
    ax3.set_title("Compression Validation Accuracy")
    ax3.legend()
    ax3.grid(True)

    # Add final accuracy text box
    final_acc_text = (
        f"Final Accuracies:\n"
        f"Transformer:\n"
        f"  Overall: {overall_accuracy_history_trans[-1]:.2f}%\n"
        f"  Compression: {compression_accuracy_history_trans[-1]:.2f}%\n"
        f"Spectron:\n"
        f"  Overall: {overall_accuracy_history_spectron[-1]:.2f}%\n"
        f"  Compression: {compression_accuracy_history_spectron[-1]:.2f}%"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax3.text(1.05, 0.95, final_acc_text, transform=ax3.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    plt.tight_layout()
    plt.show()

    # Show example predictions
    print("\n--- Example Predictions ---")
    show_example_predictions(trans_compression_model, "Transformer", train_dataset)
    show_example_predictions(spectron, "Spectron", train_dataset)
