import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------
# 1. Generate the Induction Heads Dataset
# ---------------------------------------------
def generate_induction_heads(
    num_examples: int = 1000,
    sequence_len: int = 64,  # choose a perfect square for Kron model (e.g. 64=8*8)
    vocab_size: int = 64,
    min_prefix_len: int = 2,
    max_prefix_len: int = 5,
    min_pattern_len: int = 2,
    max_pattern_len: int = 5,
    num_patterns: int = 1,
    seed: int = 1746,
) -> TensorDataset:
    """
    Generates synthetic sequences for the Induction Heads Task.

    Each sequence begins with a START token and ends with an END token.
    For each pattern, a random prefix and a random pattern are inserted twice
    (with a gap), and only the pattern tokens in the second occurrence are the target.
    
    Args:
        num_examples: Number of sequences to generate.
        sequence_len: Length of each sequence.
        vocab_size: Size of the vocabulary (excluding special tokens).
        min_prefix_len, max_prefix_len: Range for the prefix length.
        min_pattern_len, max_pattern_len: Range for the pattern length.
        num_patterns: Number of patterns in each sequence.
        seed: Random seed for reproducibility.
    
    Returns:
        A TensorDataset of (inputs, targets) where:
          - inputs: (num_examples, sequence_len) LongTensor
          - targets: (num_examples, sequence_len) LongTensor, with -1 at non-target positions.
    """
    torch.manual_seed(seed)

    # Special tokens: add three tokens (START, END, PAD)
    START, END, PAD = vocab_size, vocab_size + 1, vocab_size + 2

    inputs = torch.full((num_examples, sequence_len), PAD, dtype=torch.long)
    targets = torch.full((num_examples, sequence_len), -1, dtype=torch.long)

    for i in range(num_examples):
        inputs[i, 0] = START  # start token at beginning
        idx = 1

        for pattern_idx in range(num_patterns):
            prefix_len = torch.randint(min_prefix_len, max_prefix_len + 1, (1,)).item()
            pattern_len = torch.randint(min_pattern_len, max_pattern_len + 1, (1,)).item()
            total_len = prefix_len + pattern_len

            # Check if there's enough space for two occurrences + a gap
            remaining_space = sequence_len - idx - (total_len * 2 + 1)
            if remaining_space < 0:
                break

            # Sample random prefix and pattern tokens
            prefix = torch.randint(0, vocab_size, (prefix_len,), dtype=torch.long)
            pattern = torch.randint(0, vocab_size, (pattern_len,), dtype=torch.long)

            # First occurrence: prefix + pattern
            inputs[i, idx : idx + prefix_len] = prefix
            inputs[i, idx + prefix_len : idx + total_len] = pattern
            idx += total_len

            # Random gap (at least 1 token)
            if pattern_idx < num_patterns - 1:
                max_gap = (sequence_len - idx - total_len - 1) // (num_patterns - pattern_idx - 1)
                gap = torch.randint(1, max_gap + 1, (1,)).item()
            else:
                gap = 1
            idx += gap

            # Second occurrence: same prefix + same pattern
            inputs[i, idx : idx + prefix_len] = prefix
            inputs[i, idx + prefix_len : idx + total_len] = pattern

            # Set targets only for the pattern part (skip prefix)
            targets[i, idx + prefix_len : idx + total_len] = pattern
            idx += total_len

        # Fill remaining positions with random tokens (except last token)
        while idx < sequence_len - 1:
            inputs[i, idx] = torch.randint(0, vocab_size, (1,)).item()
            idx += 1

        inputs[i, -1] = END  # end token at final position

    return TensorDataset(inputs, targets)

# ---------------------------------------------
# 2. Define Models for the Induction Heads Task
# ---------------------------------------------
# Constants
VOCAB_SIZE = 64      # base vocabulary size (special tokens will be added)
SEQ_LEN = 64         # must be a perfect square for Kron model (e.g. 8*8)
D_MODEL = 32         # hidden dimension for embeddings & models
FACTOR_DIM = 8       # because 8*8 = 64

def build_causal_mask(seq_len: int) -> torch.Tensor:
    # Causal mask: positions j > i are masked (set to -inf)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask.bool(), float('-inf'))

class AttnModelInduction(nn.Module):
    """
    Self-attention model for the induction heads task.
    It embeds token indices, applies single-head self-attention (with a causal mask),
    and projects back to logits over the vocabulary (including special tokens).
    """
    def __init__(self, seq_len: int, d_model: int, vocab_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size + 3, d_model)  # +3 for START, END, PAD
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.out_proj = nn.Linear(d_model, vocab_size + 3)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len) of token indices
        x_embed = self.embedding(x)  # (batch, seq_len, d_model)
        attn_output, attn_weights = self.attn(x_embed, x_embed, x_embed, attn_mask=attn_mask)
        logits = self.out_proj(attn_output)  # (batch, seq_len, vocab_size+3)
        return logits, attn_weights

class KronModelInduction(nn.Module):
    """
    Kronecker-based model for the induction heads task.
    It embeds token indices and then mixes the sequence dimension via a learned Kronecker product.
    The idea is to learn a global mixing matrix over positions (with causal masking).
    """
    def __init__(self, seq_len: int, d_model: int, vocab_size: int, factor_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size + 3, d_model)
        self.factor_dim = factor_dim  # such that factor_dim**2 == seq_len
        self.u = nn.Parameter(torch.randn(factor_dim, factor_dim))
        self.v = nn.Parameter(torch.randn(factor_dim, factor_dim))
        self.out_proj = nn.Linear(d_model, vocab_size + 3)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len) of token indices
        x_embed = self.embedding(x)  # (batch, seq_len, d_model)
        # Transpose to mix over the sequence dimension: (batch, d_model, seq_len)
        x_embed_t = x_embed.transpose(1, 2)
        # Construct the Kronecker mixing matrix (seq_len x seq_len)
        kron_matrix = torch.kron(self.u, self.v)
        # Apply causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device=x.device))
        kron_matrix = kron_matrix * causal_mask
        # Mix along sequence dimension: (batch, d_model, seq_len)
        mixed = torch.matmul(x_embed_t, kron_matrix)
        # Transpose back to (batch, seq_len, d_model)
        mixed = mixed.transpose(1, 2)
        logits = self.out_proj(mixed)  # (batch, seq_len, vocab_size+3)
        return logits, kron_matrix

# ---------------------------------------------
# 3. Training Setup
# ---------------------------------------------
def train_model(model, loader, attn_mask=None, epochs=5):
    # Use CrossEntropyLoss; ignore positions where target == -1
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            if isinstance(model, AttnModelInduction):
                logits, _ = model(inputs, attn_mask)
            else:
                logits, _ = model(inputs)
            # Reshape: (batch*seq_len, vocab_size+3)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

# ---------------------------------------------
# 4. Generate Dataset, Instantiate, and Train Models
# ---------------------------------------------
dataset = generate_induction_heads(num_examples=256, sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Build causal mask for the attention model
causal_mask = build_causal_mask(SEQ_LEN)

print("Training AttnModelInduction on the Induction Heads Task...")
attn_model = AttnModelInduction(seq_len=SEQ_LEN, d_model=D_MODEL, vocab_size=VOCAB_SIZE)
train_model(attn_model, loader, attn_mask=causal_mask, epochs=20)

print("\nTraining KronModelInduction on the Induction Heads Task...")
kron_model = KronModelInduction(seq_len=SEQ_LEN, d_model=D_MODEL, vocab_size=VOCAB_SIZE, factor_dim=FACTOR_DIM)
train_model(kron_model, loader, epochs=20)

# ---------------------------------------------
# 5. Show Example Inputs and Model Predictions
# ---------------------------------------------
# We'll pick the first and last examples from the dataset.
print("\n--- AttnModelInduction Predictions ---")
attn_model.eval()
with torch.no_grad():
    # Get the first and last examples (each is a tuple: (input, target))
    first_input, _ = dataset[0]
    last_input, _  = dataset[-1]
    
    # Add batch dimension
    first_input_batch = first_input.unsqueeze(0)  # shape: (1, SEQ_LEN)
    last_input_batch  = last_input.unsqueeze(0)
    
    # Get predictions
    first_logits, _ = attn_model(first_input_batch, causal_mask)
    last_logits, _  = attn_model(last_input_batch, causal_mask)
    
    # Convert logits to predicted token indices (argmax over vocab dimension)
    first_preds = first_logits.argmax(dim=-1).squeeze(0)
    last_preds  = last_logits.argmax(dim=-1).squeeze(0)
    
    print("AttnModelInduction - First Example:")
    print("Input:     ", first_input)
    print("Predicted: ", first_preds)
    
    print("\nAttnModelInduction - Last Example:")
    print("Input:     ", last_input)
    print("Predicted: ", last_preds)

print("\n--- KronModelInduction Predictions ---")
kron_model.eval()
with torch.no_grad():
    # Using the same first and last inputs
    first_logits, _ = kron_model(first_input_batch)
    last_logits, _  = kron_model(last_input_batch)
    
    first_preds = first_logits.argmax(dim=-1).squeeze(0)
    last_preds  = last_logits.argmax(dim=-1).squeeze(0)
    
    print("KronModelInduction - First Example:")
    print("Input:     ", first_input)
    print("Predicted: ", first_preds)
    
    print("\nKronModelInduction - Last Example:")
    print("Input:     ", last_input)
    print("Predicted: ", last_preds)
