# -*- coding: utf-8 -*-
"""
Training Spectron model on associative recall task.
"""

import math
import torch
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax
from jax.tree_util import tree_leaves
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Iterator
import optax
from functools import partial


from tqdm import tqdm

# from thesis.experiments.synthetics.assoc_recall import generate_assoc_recall
from mad.data.instances import instances

# Configure JAX to use GPU if available
# jax.config.update('jax_platform_name', 'cpu')  # Use this if running on CPU
jax.config.update("jax_platform_name", "gpu")  # Use this to force using GPU
# Print device information
print(f"JAX devices: {jax.devices()}")
print(f"Using device: {jax.devices()[0]}")

IGNORE_IDX = -1
SEED = 1746

# Sample data to understand the structure
sample_loader = generate_assoc_recall(num_examples=3, sequence_len=24, vocab_size=30)
for batch_ndx, sample in enumerate(sample_loader):
    print(f"Batch {batch_ndx+1}:")
    print(f"Inputs: {sample[0]}")
    print(f"Targets: {sample[1]}")


def convert_dataset_to_jax(dataset):
    inputs = []
    targets = []
    for X, y in dataset:
        inputs.append(jnp.array(X.numpy()))
        targets.append(jnp.array(y.numpy()))
    return jnp.stack(inputs), jnp.stack(targets)


def jax_loader(ds, batch_size, shuffle=True, rng=None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    inputs, targets = ds
    num_samples = inputs.shape[0]

    if shuffle:
        assert rng is not None, "You must provide a rng key to shuffle"
        perm = jax.random.permutation(rng, num_samples)
        inputs = inputs[perm]
        targets = targets[perm]

    for i in range(0, num_samples, batch_size):
        yield inputs[i : i + batch_size], targets[i : i + batch_size]


# Load and convert datasets
rng = jax.random.PRNGKey(SEED)
train_ds = list(generate_assoc_recall(num_examples=10000, sequence_len=24, vocab_size=30))
train_ds_jax = convert_dataset_to_jax(train_ds)
batch_size = 128
train_loader = jax_loader(train_ds_jax, batch_size=batch_size, shuffle=True, rng=rng)

val_ds = list(generate_assoc_recall(num_examples=1000, sequence_len=24, vocab_size=30))
val_ds_jax = convert_dataset_to_jax(val_ds)
val_loader = jax_loader(val_ds_jax, batch_size=batch_size, shuffle=False, rng=rng)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@partial(jax.jit, static_argnames=("p", "dim", "eps"))
def normalize(
    input: jnp.ndarray, p: float = 2.0, dim: Union[int, Tuple[int, ...]] = 1, eps: float = 1e-12
) -> jnp.ndarray:
    norm = jnp.linalg.norm(input, ord=p, axis=dim, keepdims=True)
    return input / jnp.maximum(norm, eps)


def get_num_params(params) -> int:
    """
    Counts the total number of parameters in a JAX/Flax model.
    `params` is a nested dictionary (pytree) containing all model parameters.
    """
    return sum(p.size for p in tree_leaves(params))


@jax.jit
def tensordot_conv(f: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Perform a causal 1D convolution via FFT along the sequence dimension
    for multiheaded inputs.

    Args:
        f: jnp.ndarray of shape [L, D]
            The projected spectral filters (to be reshaped to [H, L, h]).
        u: jnp.ndarray of shape [B, H, L, h]
            The input sequences for each head.

    Returns:
        jnp.ndarray of shape [B, H, L, h]
            The convolved sequences per head.
    """
    reshaped = lambda x: x.reshape(u.shape[2], u.shape[1], u.shape[3]).transpose(1, 0, 2)
    tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method="fft")[: x.shape[0]]
    cconv = jax.vmap(tr_conv, in_axes=(0, 0), out_axes=0)
    hconv = lambda u1, f1: cconv(jnp.moveaxis(u1, -1, 0), jnp.moveaxis(f1, -1, 0)).T
    hmap = jax.vmap(hconv, in_axes=(0, 0), out_axes=0)
    bmap = jax.vmap(hmap, in_axes=(0, None), out_axes=0)
    return bmap(u, reshaped(f))


@jax.jit
def stu_conv(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Compute FFT-based convolution for multiheaded inputs.

    Args:
    v: [L, K] spectral filters (shared across heads).
    u: [B, H, L, h] inputs for each head (h is per-head feature dim).

    Returns:
    [B, H, L, K, h] output of convolving each head's input with each of the K filters.
    """
    tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method="fft")[: y.shape[0]]
    mvconv = jax.vmap(tr_conv, in_axes=(1, None), out_axes=1)
    mmconv = jax.vmap(mvconv, in_axes=(None, 1), out_axes=-1)
    conv_one = lambda u1: mmconv(v, u1)
    conv_heads = jax.vmap(conv_one, in_axes=0, out_axes=0)
    conv_batch = jax.vmap(conv_heads, in_axes=0, out_axes=0)
    return conv_batch(u)


@dataclass(frozen=True)
class SpectronConfig:
    dim: int = 1024
    head_dim: int = 256
    num_eigh: int = 32
    num_heads: int = 4
    num_local_heads: Optional[int] = -1
    num_layers: int = 12
    seq_len: int = 4096
    vocab_size: int = 200064
    inter_dim: Optional[int] = 4096
    mlp_scale: float = 12.0
    weight_tying: bool = True
    use_tensordot: bool = False
    bias: bool = False
    eps: float = 1e-5

    def __post_init__(self):
        if self.num_local_heads == -1:
            self.num_local_heads = self.num_heads
        if self.inter_dim is None:
            hidden_dim = self.mlp_scale * self.dim
            num_hidden = int(2 * hidden_dim / 3)
            self.inter_dim = find_multiple(num_hidden, 256)
            self.head_dim = self.dim // self.num_heads


# MLP
class MLP(nn.Module):
    config: SpectronConfig

    def setup(self):
        self.w1 = nn.Dense(self.config.inter_dim)
        self.w2 = nn.Dense(self.config.dim)

    def __call__(self, x):
        return self.w2(nn.gelu(self.w1(x)))


@jax.jit
def attention_combine_fn(x, y):
    """
    Combines two leaves for gated causal attention with online softmax.
    Each leaf is a tuple:
    (m, s, n, Z, g)
    where:
    m: the score/logit for numerical stability
    s: running sum of exp(score-m)
    n: running sum of exp(score-m)*value
    Z: the outer product interaction matrix
    g: the gate value
    """
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y

    # Compute new maximum
    m_new = lax.max(m_x, m_y)

    # Scale factors
    exp_x = lax.exp(m_x - m_new)
    exp_y = lax.exp(m_y - m_new)

    # Update softmax components
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x[..., None] + n_y * exp_y[..., None]

    # Update gated Z and gate accumulation
    Z_new = Z_x + Z_y
    g_new = g_x + g_y

    return (m_new, s_new, n_new, Z_new, g_new)


@jax.jit
def scan_fn(qk_slice, v_slice, Z_slice, g_slice):
    """Process a single (batch, head) slice"""
    # Set up leaf elements
    # qk_slice: [L], v_slice: [L, h], Z_slice: [L, h, h], g_slice: [L, 1, 1]
    leaves_m = qk_slice  # [L]
    leaves_s = jnp.ones_like(qk_slice)  # [L]
    leaves_n = v_slice  # [L, h] (from original v)
    leaves_Z = Z_slice  # [L, h, h] (from transformed k,v)
    leaves_g = g_slice  # [L, 1, 1]

    leaves = (leaves_m, leaves_s, leaves_n, leaves_Z, leaves_g)
    return jax.lax.associative_scan(attention_combine_fn, leaves, axis=0)


# AssociativeAttention
class AssociativeAttention(nn.Module):
    config: SpectronConfig
    spectral_filters: jnp.ndarray
    use_tensordot: bool

    def setup(self):
        self.head_dim = self.config.dim // self.config.num_heads
        self.wq = nn.Dense(self.config.dim)
        self.wk = nn.Dense(self.config.dim)
        self.wv = nn.Dense(self.config.dim)
        self.wo = nn.Dense(self.config.dim)
        self.wg = nn.Dense(1)
        self.eps = 1e-5

        if self.use_tensordot:
            self.tensordot_proj = nn.Dense(self.config.dim)

        self.qk_norm_scale = self.param(
            "qk_norm_scale", lambda rng: jnp.full((1, self.config.num_heads, 1), 1 / jnp.sqrt(self.head_dim))
        )
        self.kv_norm_scale = self.param(
            "kv_norm_scale", lambda rng: jnp.ones((1, self.config.num_heads, 1, self.head_dim, self.head_dim))
        )

    def __call__(self, x, training: bool = False):
        B, L, D = x.shape
        H, h = self.config.num_heads, self.head_dim

        # QKV projections (original)
        q = self.wq(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)  # [B, H, L, h]
        k_orig = self.wk(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)
        v_orig = self.wv(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)

        # Similarity (using original q, k)
        sim = jnp.einsum("bhld,bhld->bhl", q, k_orig) * self.qk_norm_scale

        # Normalize original k and v for transformation
        k_norm = normalize(k_orig, dim=-1)
        v_norm = normalize(v_orig, dim=-1)

        # Apply spectral filters to normalized k, v
        if self.use_tensordot:
            spectral_filters_proj = self.tensordot_proj(self.spectral_filters)
            k_transformed = tensordot_conv(spectral_filters_proj, k_norm)
            v_transformed = tensordot_conv(spectral_filters_proj, v_norm)
        else:
            k_transformed = stu_conv(self.spectral_filters, k_norm)
            v_transformed = stu_conv(self.spectral_filters, v_norm)

        # Pairwise interactions using transformed k, v
        if self.use_tensordot:
            # v_transformed, k_transformed shape: [B, H, L, h]
            Z = jnp.einsum("bhld,bhle->bhlde", v_transformed, k_transformed)
        else:
            # v_transformed, k_transformed shape: [B, H, L, K, h]
            Z = jnp.einsum("bhlkd,bhlke->bhlde", v_transformed, k_transformed)
        Z = Z * self.kv_norm_scale  # Z shape: [B, H, L, h, h]

        # Gating
        gate_inputs = Z.reshape(B, H, L, -1)
        gates_logits = self.wg(gate_inputs)
        gates = (nn.relu(gates_logits) ** 2 + self.eps)[..., None]
        gated_Z = gates * Z  # Shape: [B, H, L, h, h]

        # Vmap over the head dimension (for a single batch)
        batch_scan_fn = jax.vmap(scan_fn, in_axes=(0, 0, 0, 0))

        # Vmap over the batch dimension
        batched_scan_fn = jax.vmap(batch_scan_fn, in_axes=(0, 0, 0, 0))

        # Run the scan over all dimensions at once, passing ORIGINAL v
        m_scan, s_scan, n_scan, Z_scan, g_scan = batched_scan_fn(sim, v_orig, gated_Z, gates)

        # --- Compute the final attention outputs ---

        # 1. Compute the normalized online softmax weights
        # sim & m_scan shape: [B, H, L], s_scan shape: [B, H, L]
        softmax_weights = lax.exp(sim - m_scan)[..., None, None] / (
            s_scan[..., None, None] + self.eps
        )  # Shape: [B, H, L, 1, 1]

        # 2. Gated accumulation normalization
        # Z_scan shape: [B, H, L, h, h], g_scan shape: [B, H, L, 1, 1]
        gated_weights = Z_scan / (g_scan + self.eps)  # Shape: [B, H, L, h, h]

        # Multiplicatively modulate the gated weights with the softmax weights
        attn_weights = gated_weights * (1.0 + jax.nn.silu(softmax_weights))  # Shape: [B, H, L, h, h]

        # Query from the attention weights (using original q)
        ctxt = jnp.einsum("bhld,bhlde->bhle", q, attn_weights)  # Shape: [B, H, L, h]

        # Reshape and project
        output_normalized = ctxt.transpose(0, 2, 1, 3).reshape(B, L, D)
        output = self.wo(output_normalized)

        return output


# SpectronBlock
class SpectronBlock(nn.Module):
    config: SpectronConfig
    spectral_filters: jnp.ndarray

    def setup(self):
        self.aa_norm = nn.LayerNorm(epsilon=self.config.eps)
        self.mlp_norm = nn.LayerNorm(epsilon=self.config.eps)
        self.aa = AssociativeAttention(
            config=self.config, spectral_filters=self.spectral_filters, use_tensordot=self.config.use_tensordot
        )
        self.mlp = MLP(config=self.config)

    def __call__(self, x, training: bool = False):
        x = x + self.aa(self.aa_norm(x), training=training)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# Spectron
class Spectron(nn.Module):
    config: SpectronConfig
    spectral_filters: jnp.ndarray

    def setup(self):
        self.tok_emb = nn.Embed(self.config.vocab_size, self.config.dim)
        self.layers = [SpectronBlock(self.config, self.spectral_filters) for _ in range(self.config.num_layers)]
        self.norm_f = nn.LayerNorm(epsilon=self.config.eps)
        # If we're tying weights, we want lm_head to project to the embedding dimension,
        # and then later use a dot product with the tied embeddings.
        if self.config.weight_tying:
            self.lm_head = nn.Dense(self.config.dim, use_bias=False)
        else:
            self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids, labels=None, training: bool = False):
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.norm_f(x)

        logits = self.lm_head(x)
        if self.config.weight_tying:
            # Tie logits to the input embedding by multiplying with the transpose.
            logits = logits @ self.tok_emb.embedding.T

        if labels is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, self.config.vocab_size), labels.reshape(-1)
            ).mean()
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def get_hankel_matrix(n: int, use_hankel_L: bool = False) -> jnp.ndarray:
    ij = jnp.arange(1, n + 1, dtype=jnp.float32)
    i_plus_j = ij[:, None] + ij[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(n: int, k: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    eig_vals, eig_vecs = jnp.linalg.eigh(get_hankel_matrix(n))
    eig_vals_k, eig_vecs_k = eig_vals[-k:], eig_vecs[:, -k:]
    eig_vecs_k *= eig_vals_k**0.25
    return eig_vals_k, eig_vecs_k


def init_model(config: SpectronConfig, rng):
    spectral_filters = get_spectral_filters(config.seq_len, config.num_eigh)[1]
    model = Spectron(config, spectral_filters)
    params = model.init(rng, jnp.ones((1, config.seq_len), dtype=jnp.int32))["params"]
    return model, params


# Training function
def train_step(params, optimizer_state, batch, config, spectral_filters):
    inputs, targets = batch

    def loss_fn(params):
        model = Spectron(config, spectral_filters)
        outputs = model.apply({"params": params}, inputs, labels=targets, training=True)
        return outputs["loss"]

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, optimizer_state = tx.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss


# Evaluation function
def eval_step(params, batch, config, spectral_filters):
    inputs, targets = batch
    model = Spectron(config, spectral_filters)
    outputs = model.apply({"params": params}, inputs, labels=targets, training=False)
    loss = outputs["loss"]

    # Calculate accuracy (only considering the last position)
    logits = outputs["logits"]

    # Get the shape information
    batch_size, seq_len, vocab_size = logits.shape

    # Extract last token predictions for each sequence
    last_logits = logits[:, -1, :]
    predictions = jnp.argmax(last_logits, axis=-1)  # Shape: [batch_size]

    # Ensure targets has the right shape - take the last token from each sequence
    last_targets = targets[:, -1]  # Shape: [batch_size]

    # Compare predictions with targets
    correct = jnp.sum(predictions == last_targets)
    total = predictions.shape[0]

    return loss, correct, total


# JIT-compiled versions
train_step_jit = jax.jit(train_step, static_argnames=["config"])
eval_step_jit = jax.jit(eval_step, static_argnames=["config"])

# Parameters for the model
E = 10000  # Number of examples
L = 24  # Sequence length
V = 30  # Vocabulary size
dim = 128  # Model dimension
num_heads = 1
num_layers = 1
num_eigh = 16  # Number of eigenvectors to use

# Create config
config = SpectronConfig(
    dim=dim,
    num_heads=num_heads,
    num_local_heads=num_heads,
    num_layers=num_layers,
    seq_len=L,
    vocab_size=V + 1,  # +1 for padding
    num_eigh=num_eigh,
    weight_tying=True,
    use_tensordot=True,
    bias=False,
)

# Initialize model
rng = jax.random.PRNGKey(SEED)
rng, init_rng = jax.random.split(rng)

# Get spectral filters
spectral_filters = get_spectral_filters(L, num_eigh)[1]

# Create model for initialization only
model = Spectron(config, spectral_filters)
params = model.init(init_rng, jnp.ones((1, L), dtype=jnp.int32))["params"]

# Print model info
num_params = get_num_params(params)
print(f"Model initialized with {num_params:,} parameters")

# Set up training components
max_lr = 1e-3
min_lr = 1e-4
max_steps = 10000
warmup_steps = max_steps // 10
eval_period = max_steps // 50

# Create optimizer with learning rate schedule
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=max_lr, warmup_steps=warmup_steps, decay_steps=max_steps, end_value=min_lr
)
tx = optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
optimizer_state = tx.init(params)

# Compute baseline loss (random guessing)
baseline_loss = math.log(config.vocab_size)

# Training loop
loss_history = []
acc_history = []
eval_steps = []
curr_step = 0
running_acc = 0.0
examples_seen = 0
epochs_completed = 0
reached_90 = False

pbar = tqdm(
    total=max_steps,
    desc="Training",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
)

# Create iterators
train_ds_size = train_ds_jax[0].shape[0]
steps_per_epoch = train_ds_size // batch_size

for epoch in range(max_steps // steps_per_epoch + 1):
    # Shuffle data at the beginning of each epoch
    rng, data_rng = jax.random.split(rng)
    train_loader = jax_loader(train_ds_jax, batch_size=batch_size, shuffle=True, rng=data_rng)

    for batch in train_loader:
        if curr_step >= max_steps:
            break

        params, optimizer_state, loss = train_step_jit(params, optimizer_state, batch, config, spectral_filters)

        curr_loss = loss.item()
        loss_history.append(curr_loss)
        curr_step += 1
        examples_seen += batch[0].shape[0]

        if curr_step % eval_period == 0 or curr_step == max_steps:
            # Evaluate on validation set
            val_loss = 0
            correct = 0
            total = 0

            for val_batch in jax_loader(val_ds_jax, batch_size=batch_size, shuffle=False, rng=None):
                batch_loss, batch_correct, batch_total = eval_step_jit(params, val_batch, config, spectral_filters)
                val_loss += batch_loss
                correct += batch_correct
                total += batch_total

            val_loss /= val_ds_jax[0].shape[0] // batch_size
            acc = 100.0 * (correct / total)
            acc_history.append(acc)
            eval_steps.append(curr_step)
            running_acc = acc

            if not reached_90 and acc >= 90.0:
                print(
                    f"Reached 90% accuracy at step {curr_step}, examples seen: {examples_seen}, epochs: {epochs_completed}"
                )
                reached_90 = True

        # Update progress bar
        pbar.set_postfix(
            loss=f"{curr_loss:.3f}",
            base=f"{baseline_loss:.3f}",
            acc=f"{running_acc:.1f}%",
            lr=f"{max_lr:.1e}",
            ex=f"{examples_seen//1000}k",
            ep=f"{epoch}",
        )
        pbar.update(1)

    epochs_completed += 1

pbar.close()
print("\nTraining complete!")
print(f"Final loss: {loss_history[-1]:.4f} (Baseline: {baseline_loss:.4f})")
print(f"Final accuracy: {acc_history[-1]:.2f}%")
print(f"Total examples seen: {examples_seen}")
print(f"Epochs completed: {epochs_completed}")

# Plotting
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
