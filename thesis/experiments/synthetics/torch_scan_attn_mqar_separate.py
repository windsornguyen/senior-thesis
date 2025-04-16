import sys
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from thesis.experiments.synthetics.mqar import generate_mqar


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""

    # Model architecture
    vocab_size: int = 1024
    dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    dropout_rate: float = 0.0
    max_seq_len: int = 512

    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100

    # MQAR task parameters
    seq_len: int = 512
    num_pairs: int = 4
    alpha: float = 0.01

    # Data parameters
    train_size: int = 131072
    val_size: int = 4096
    batch_size: int = 64
    num_epochs: int = 16
    seed: int = 1746


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float = 0.1
):
    """Creates a schedule with linear warmup and cosine decay."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[: x.size(1)]


_DEFAULT_ROPE_BASE_FREQUENCY = 10_000


def apply_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    base_frequency: int = _DEFAULT_ROPE_BASE_FREQUENCY,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Applies RoPE to input tensors.

    Args:
        inputs: Tensor of shape [B, L, N, H]
        positions: Tensor of shape [B, L]
        base_frequency: Base frequency for rotations
        scale_factor: Scale factor for positional interpolation
    """
    head_dim = inputs.size(-1)
    fraction = 2 * torch.arange(0, head_dim // 2, device=inputs.device) / head_dim
    timescale = base_frequency**fraction

    sinusoid_inp = positions.unsqueeze(-1) / timescale.unsqueeze(0).unsqueeze(0)
    sinusoid_inp = sinusoid_inp.unsqueeze(-1)

    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp = sinusoid_inp / scale_factor

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    first_half, second_half = torch.split(inputs, inputs.size(-1) // 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin

    return torch.cat([first_part, second_part], dim=-1)


class VanillaAttention(nn.Module):
    """Standard scaled dot-product attention with causal mask."""

    def __init__(self, dim: int, num_heads: int, seq_len: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)

        # Create causal mask once
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))  # [1, 1, L, L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        positions = torch.arange(seq_len, device=x.device)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)  # [B, H, L, D]
        v = v.transpose(1, 2)  # [B, H, L, D]

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        # Apply softmax and compute context
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # [B, H, L, D]

        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.wo(context)


class FeedForward(nn.Module):
    """MLP feed-forward layer with GELU activation."""

    def __init__(self, dim: int, expansion_factor: int = 4, dropout_rate: float = 0.0):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """A single transformer layer with attention and feed-forward."""

    def __init__(self, dim: int, num_heads: int, seq_len: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = VanillaAttention(dim, num_heads, seq_len)
        self.ff = FeedForward(dim, dropout_rate=dropout_rate)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual

        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual

        return x


class Transformer(nn.Module):
    """Complete transformer model for the MQAR task."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.dim, config.max_seq_len)

        # Dropout for embeddings
        self.dropout = nn.Dropout(config.dropout_rate)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len,
                    dropout_rate=config.dropout_rate,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Output projection
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings and apply positional encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Apply output head
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def count_params(self) -> int:
        """Count the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    pbar: tqdm,
) -> list:
    """Train for one epoch."""
    model.train()
    train_metrics = []

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = targets != -100
            correct = ((preds == targets) * mask).sum()
            total = mask.sum()
            accuracy = correct.float() / total.float()

            metrics = {"loss": loss.item(), "accuracy": accuracy.item(), "lr": scheduler.get_last_lr()[0]}
            train_metrics.append(metrics)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']*100:.2f}%",
                    "lr": f"{metrics['lr']:.2e}",
                }
            )
            pbar.update()

    return train_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, max_batches: int = 10
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for inputs, targets in val_loader:
        if num_batches >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Compute metrics
        preds = logits.argmax(dim=-1)
        mask = targets != -100
        correct = ((preds == targets) * mask).sum()
        tokens = mask.sum()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += tokens
        num_batches += 1

    return {"loss": total_loss / num_batches, "accuracy": (total_correct / total_tokens).item()}


def train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: TransformerConfig, device: torch.device
) -> tuple:
    """Train the model."""
    # Setup criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8
    )

    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_training_steps
    )

    # Training loop
    train_metrics = []
    val_metrics = []
    best_val_loss = float("inf")
    step = 0

    with tqdm(total=num_training_steps, desc="Training") as pbar:
        for epoch in range(config.num_epochs):
            # Training
            epoch_metrics = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, pbar)
            train_metrics.extend(epoch_metrics)

            # Validation (every 50 steps)
            if step % 50 == 0:
                val_metric = evaluate(model, val_loader, criterion, device)
                val_metrics.append({"step": step, **val_metric})

                # Update progress bar
                pbar.set_postfix(
                    {
                        "train_loss": f"{epoch_metrics[-1]['loss']:.4f}",
                        "train_acc": f"{epoch_metrics[-1]['accuracy']*100:.2f}%",
                        "val_loss": f"{val_metric['loss']:.4f}",
                        "val_acc": f"{val_metric['accuracy']*100:.2f}%",
                    }
                )

                # Save best model
                if val_metric["loss"] < best_val_loss:
                    best_val_loss = val_metric["loss"]
                    # Optionally save checkpoint here

            step += len(train_loader)

            # Print epoch summary
            print(
                f"\nEpoch {epoch + 1}/{config.num_epochs} | "
                f"Train Loss: {epoch_metrics[-1]['loss']:.4f} | "
                f"Train Acc: {epoch_metrics[-1]['accuracy']*100:.2f}% | "
                f"Val Loss: {val_metrics[-1]['loss']:.4f} | "
                f"Val Acc: {val_metrics[-1]['accuracy']*100:.2f}%"
            )

    return train_metrics, val_metrics


def plot_results(config: TransformerConfig, train_metrics: list, val_metrics: list):
    """Plot training curves."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract metrics
    train_steps = range(len(train_metrics))
    train_loss = [m["loss"] for m in train_metrics]
    train_acc = [m["accuracy"] * 100 for m in train_metrics]

    val_steps = [m["step"] for m in val_metrics]
    val_loss = [m["loss"] for m in val_metrics]
    val_acc = [m["accuracy"] * 100 for m in val_metrics]

    # Plot losses
    ax1.plot(train_steps, train_loss, "b-", alpha=0.3, label="Train")
    ax1.plot(val_steps, val_loss, "r-", label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_steps, train_acc, "b-", alpha=0.3, label="Train")
    ax2.plot(val_steps, val_acc, "r-", label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Transformer Training on MQAR Task", fontsize=14)
    plt.tight_layout()
    plt.savefig("transformer_mqar_training.png")
    print("Training plot saved to transformer_mqar_training.png")
    plt.show()


def run_experiment(config_override=None):
    """Run the MQAR experiment with vanilla Transformer."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base configuration
    config = TransformerConfig(
        dim=128,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.0,
        max_seq_len=256,
        vocab_size=8192 + 3,  # Add 3 for special tokens
        learning_rate=3e-4,
        weight_decay=1e-1,
        warmup_steps=2000,
        seq_len=256,
        num_pairs=32,
        alpha=2.0,
        train_size=100000,
        val_size=3000,
        batch_size=64,
        num_epochs=64,
        seed=1746,
    )

    # Apply overrides if any
    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)

    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Generate data
    print("Generating training data...")
    train_dataset = generate_mqar(
        num_examples=config.train_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_pairs=config.num_pairs,
        alpha=config.alpha,
        seed=config.seed,
    )

    print("Generating validation data...")
    val_dataset = generate_mqar(
        num_examples=config.val_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_pairs=config.num_pairs,
        alpha=config.alpha,
        seed=config.seed + 1,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    print("\nInitializing Transformer...")
    model = Transformer(config).to(device)
    print(f"Number of parameters: {model.count_params():,}")

    # Train model
    print("\nStarting training...")
    train_metrics, val_metrics = train_model(model, train_loader, val_loader, config, device)

    # Plot results
    plot_results(config, train_metrics, val_metrics)

    return {"model": model, "train_metrics": train_metrics, "val_metrics": val_metrics, "config": config}


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    results = run_experiment()
