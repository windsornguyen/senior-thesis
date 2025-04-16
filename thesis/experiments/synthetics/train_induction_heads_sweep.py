# -*- coding: utf-8 -*-
"""induction_heads.ipynb

# Effects of Convolutions on Inducing Induction Heads

A study of how local convolutions can help induce the formation of induction heads, potentially without the need of a second attention layer.
"""

# Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1746)
torch.set_float32_matmul_precision("high")

IGNORE_IDX = -1
SEED = 1746

# Data Generation
def generate_induction_heads(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 20,
    seed: int = 0,
) -> TensorDataset:
    torch.manual_seed(seed)
    special = vocab_size - 1
    inputs = torch.randint(0, vocab_size - 1, (num_examples, sequence_len), dtype=torch.long)
    idx = torch.randint(0, sequence_len - 2, (num_examples,), dtype=torch.long)
    inputs[torch.arange(num_examples), idx] = special
    inputs[torch.arange(num_examples), -1] = special
    targets = inputs[torch.arange(num_examples), idx + 1]
    return TensorDataset(inputs, targets)

# Model Definitions
class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.w2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, seq_len, use_conv=True, kernel_size=3) -> None:
        super().__init__()
        self.dim = dim
        self.H = num_heads
        self.h = dim // num_heads
        self.seq_len = seq_len
        self.scale = self.h ** -0.5
        self.use_conv = use_conv
        self.kernel_size = kernel_size

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len))

        # Linear projections for Q, K, V
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Depthwise 1D convolutions for Q, K, V (optional)
        if self.use_conv:
            padding = (kernel_size - 1) // 2  # Ensure same output length
            self.conv_q = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding, groups=dim)
            self.conv_k = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding, groups=dim)
            self.conv_v = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding, groups=dim)

    def forward(self, x):
        B, L, D = x.shape

        # Compute Q, K, V
        q, k, v = self.wq(x), self.wk(x) * self.scale, self.wv(x)

        # Apply depthwise convolution if enabled
        if self.use_conv:
            q = q.permute(0, 2, 1)  # [B, D, L]
            k = k.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
            q = self.conv_q(q).permute(0, 2, 1)  # [B, L, D]
            k = self.conv_k(k).permute(0, 2, 1)
            v = self.conv_v(v).permute(0, 2, 1)

        # Reshape for multi-head attention
        q = q.view(B, L, self.H, self.h).transpose(1, 2)  # [B, H, L, h]
        k = k.view(B, L, self.H, self.h).transpose(1, 2)
        v = v.view(B, L, self.H, self.h).transpose(1, 2)

        # Compute attention scores
        sim = q @ k.transpose(-2, -1)  # [B, H, L, L]
        sim = sim.masked_fill(self.mask, float("-inf"))
        attn = F.softmax(sim, dim=-1)
        ctxt = attn @ v  # [B, H, L, h]
        out = ctxt.transpose(1, 2).reshape(B, L, -1)  # [B, L, D]
        out = self.wo(out)
        return out, attn

class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, seq_len, use_conv=True, kernel_size=3) -> None:
        super().__init__()
        self.attn = Attention(dim, num_heads, seq_len, use_conv, kernel_size)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 4 * dim)
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm(x))
        x = x + attn_out
        # x = x + self.mlp(self.mlp_norm(x))  # Commented out as in original
        return x, attn_weights

class Transformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, seq_len, vocab_size, use_conv=True, kernel_size=3) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn_layer = AttentionLayer(dim, num_heads, seq_len, use_conv, kernel_size)
            self.layers.append(attn_layer)
        self.norm_f = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.size()
        pos = torch.arange(0, L, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)
        tok_emb = self.tok_emb(x)
        x = pos_emb + tok_emb

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_list.append(attn_weights)

        x = self.norm_f(x)
        out = self.out(x)
        return out, attn_weights_list

# Utility Functions
def create_lr_lambda(warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * step / warmup_steps
            return lr / max_lr
        lr = max_lr - (max_lr - min_lr) * (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return lr / max_lr
    return lr_lambda

def compute_acc(model, loader, device=None):
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            predictions = logits[:, -1, :].argmax(dim=-1)
            correct_tokens += (predictions == targets).sum().item()
            total_tokens += targets.size(0)
    model.train()
    return 100.0 * correct_tokens / total_tokens

def plot_attention_heatmap(attn_weights_list, layer_idx, head_idx, seq_len, example_idx=0, inputs=None, target=None):
    attn_weights = attn_weights_list[layer_idx][example_idx, head_idx].cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_weights, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key Positions")
    ax.set_ylabel("Query Positions")
    plt.gcf().colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1}", fontsize=12)
    if inputs is not None and target is not None:
        input_str = " ".join(map(str, inputs[example_idx].cpu().numpy()))
        target_str = str(target[example_idx].cpu().item())
        info_text = f"Input: [{input_str}]\nTarget: {target_str}"
        plt.gcf().text(
            0.5, 0.01, info_text, fontsize=10, ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="gray")
        )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def train_model(model, train_loader, val_loader, max_steps, device, max_lr=1e-3, min_lr=1e-4, warmup_steps=None, loss_fn=None):
    if warmup_steps is None:
        warmup_steps = max_steps // 10
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    lr_lambda_fn = create_lr_lambda(warmup_steps, max_steps, max_lr, min_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

    loss_history = []
    acc_history = []
    eval_steps = []
    curr_step = 0
    running_acc = 0.0
    examples_seen = 0
    epochs_completed = 0
    reached_90 = False
    eval_period = max_steps // 50

    pbar = tqdm(total=max_steps, desc="Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    model.train()
    train_iter = iter(train_loader)

    while curr_step < max_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
            epochs_completed += 1

        batch_examples = inputs.size(0)
        examples_seen += batch_examples
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, attn_weights_list = model(inputs)
        last_logits = logits[:, -1, :]
        loss = loss_fn(last_logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        curr_loss = loss.item()
        loss_history.append(curr_loss)
        curr_step += 1

        if curr_step % eval_period == 0 or curr_step == max_steps:
            acc = compute_acc(model, val_loader, device=device)
            acc_history.append(acc)
            eval_steps.append(curr_step)
            running_acc = acc
            if not reached_90 and acc >= 90.0:
                print(f"Reached 90% accuracy at step {curr_step}, examples seen: {examples_seen}, epochs: {epochs_completed}")
                reached_90 = True

        pbar.set_postfix(
            loss=f"{curr_loss:.3f}",
            acc=f"{running_acc:.1f}%",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
            ex=f"{examples_seen//1000}k",
            ep=f"{epochs_completed}"
        )
        pbar.update(1)

    pbar.close()
    return loss_history, acc_history, eval_steps

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base Parameters
E = 1000
L = 32
V = 16
dim = 4
num_heads = 2
num_layers = 1
DELIMS = 3
batch_size = 32
max_steps = 10000

# Experiment 1: Base Model (Single Layer with Convolution)
print("Experiment 1: Base Model (Single Layer with Convolution)")
train_dataset = generate_induction_heads(num_examples=E, sequence_len=L, vocab_size=V, seed=SEED)
val_dataset = generate_induction_heads(num_examples=E // 10, sequence_len=L, vocab_size=V, seed=SEED + 1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Transformer(dim=dim, num_heads=num_heads, num_layers=num_layers, seq_len=L, vocab_size=V + DELIMS, use_conv=True, kernel_size=3)
model.to(device)
total_vocab = V + DELIMS
baseline_loss = math.log(total_vocab)

loss_history, acc_history, eval_steps = train_model(model, train_loader, val_loader, max_steps, device)

print("\nTraining complete!")
print(f"Final loss: {loss_history[-1]:.4f} (Baseline: {baseline_loss:.4f})")
print(f"Final accuracy: {acc_history[-1]:.2f}%")

# Visualize attention heatmaps
model.eval()
with torch.no_grad():
    val_inputs, val_targets = next(iter(val_loader))
    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
    logits, attn_weights_list = model(val_inputs)
    for layer_idx in range(len(attn_weights_list)):
        for head_idx in range(num_heads):
            plot_attention_heatmap(
                attn_weights_list, layer_idx=layer_idx, head_idx=head_idx, seq_len=L,
                inputs=val_inputs, target=val_targets
            )

# Plot training metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Training Loss")
plt.axhline(y=baseline_loss, color="r", linestyle="--", label="Baseline Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(eval_steps, acc_history, marker="o", color="orange", label="Validation Accuracy")
plt.xlabel("Training Steps")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Steps")
plt.legend()
plt.tight_layout()
plt.show()

# Experiment 2: Role of Depthwise Convolution (Ablation)
print("\nExperiment 2: Role of Depthwise Convolution (Ablation)")
model_no_conv = Transformer(dim=dim, num_heads=num_heads, num_layers=num_layers, seq_len=L, vocab_size=V + DELIMS, use_conv=False)
model_no_conv.to(device)
loss_history_no_conv, acc_history_no_conv, eval_steps_no_conv = train_model(model_no_conv, train_loader, val_loader, max_steps, device)
print("\nTraining complete (No Convolution)!")
print(f"Final loss: {loss_history_no_conv[-1]:.4f} (Baseline: {baseline_loss:.4f})")
print(f"Final accuracy: {acc_history_no_conv[-1]:.2f}%")

# Experiment 3: Vary Kernel Size
print("\nExperiment 3: Vary Kernel Size")
for kernel_size in [5, 7]:
    print(f"Kernel Size: {kernel_size}")
    model_kernel = Transformer(dim=dim, num_heads=num_heads, num_layers=num_layers, seq_len=L, vocab_size=V + DELIMS, use_conv=True, kernel_size=kernel_size)
    model_kernel.to(device)
    loss_history_kernel, acc_history_kernel, eval_steps_kernel = train_model(model_kernel, train_loader, val_loader, max_steps, device)
    print(f"\nTraining complete (Kernel Size {kernel_size})!")
    print(f"Final loss: {loss_history_kernel[-1]:.4f} (Baseline: {baseline_loss:.4f})")
    print(f"Final accuracy: {acc_history_kernel[-1]:.2f}%")

# Experiment 4: Task Complexity (Vary V and L)
print("\nExperiment 4: Task Complexity (Vary V and L)")
for vocab_size in [64, 128]:
    for seq_len in [64, 128]:
        print(f"Vocab Size: {vocab_size}, Sequence Length: {seq_len}")
        train_dataset = generate_induction_heads(num_examples=E, sequence_len=seq_len, vocab_size=vocab_size, seed=SEED)
        val_dataset = generate_induction_heads(num_examples=E // 10, sequence_len=seq_len, vocab_size=vocab_size, seed=SEED + 1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model_complex = Transformer(dim=dim, num_heads=num_heads, num_layers=num_layers, seq_len=seq_len, vocab_size=vocab_size + DELIMS, use_conv=True, kernel_size=3)
        model_complex.to(device)
        loss_history_complex, acc_history_complex, eval_steps_complex = train_model(model_complex, train_loader, val_loader, max_steps, device)
        print(f"\nTraining complete (V={vocab_size}, L={seq_len})!")
        print(f"Final loss: {loss_history_complex[-1]:.4f}")
        print(f"Final accuracy: {acc_history_complex[-1]:.2f}%")

# Experiment 5: Compare with Two-Layer Model
print("\nExperiment 5: Compare with Two-Layer Model")
model_two_layer = Transformer(dim=dim, num_heads=num_heads, num_layers=2, seq_len=L, vocab_size=V + DELIMS, use_conv=False)
model_two_layer.to(device)
loss_history_two, acc_history_two, eval_steps_two = train_model(model_two_layer, train_loader, val_loader, max_steps, device)
print("\nTraining complete (Two Layers, No Convolution)!")
print(f"Final loss: {loss_history_two[-1]:.4f} (Baseline: {baseline_loss:.4f})")
print(f"Final accuracy: {acc_history_two[-1]:.2f}%")

# Scaling Law Sweep
print("\nScaling Law Sweep: Vary dim and vocab_size")
results = []
for vocab_size in [16, 32, 64, 128]:
    for d in [2, 4, 8, 16, 32, 64]:
        print(f"Vocab Size: {vocab_size}, dim: {d}")
        train_dataset = generate_induction_heads(num_examples=E, sequence_len=L, vocab_size=vocab_size, seed=SEED)
        val_dataset = generate_induction_heads(num_examples=E // 10, sequence_len=L, vocab_size=vocab_size, seed=SEED + 1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = Transformer(dim=d, num_heads=2, num_layers=1, seq_len=L, vocab_size=vocab_size + DELIMS, use_conv=True, kernel_size=3)
        model.to(device)
        loss_history, acc_history, eval_steps = train_model(model, train_loader, val_loader, max_steps, device)
        acc = acc_history[-1]
        results.append({'vocab_size': vocab_size, 'dim': d, 'accuracy': acc})
        print(f"Final accuracy: {acc:.2f}%")

# Plot scaling law
df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
for v in df['vocab_size'].unique():
    subset = df[df['vocab_size'] == v]
    plt.plot(subset['dim'], subset['accuracy'], marker='o', label=f'V={v}')
plt.xlabel('Embedding Dimension (dim)')
plt.ylabel('Validation Accuracy (%)')
plt.title('Scaling Law: Accuracy vs. Embedding Dimension for Different Vocab Sizes')
plt.legend()
plt.grid(True)
plt.show()