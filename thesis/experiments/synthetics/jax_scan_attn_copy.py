import inspect
import dataclasses
from functools import wraps
from typing import Tuple, Any
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# Import copying task functions directly
from thesis.experiments.synthetics.copying.copying import generate_copying, generate_copy_dataset

# Set platform-specific defaults
jax.config.update("jax_default_matmul_precision", "float32")

matplotlib.use('TkAgg')

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------


@dataclasses.dataclass
class TransformerConfig:
    """Configuration class for the Transformer model."""

    vocab_size: int = 16
    dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    dropout_rate: float = 0.0
    max_seq_len: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    attention_type: str = "vanilla"  # "vanilla" or "scan"
    # Dataset configuration
    seq_len: int = 256
    num_tokens_to_copy: int = 16
    train_size: int = 50000
    val_size: int = 3000
    batch_size: int = 64
    num_epochs: int = 32
    seed: int = 1746


# ------------------------------------------------------------------------
# Data Generation for Selective Copying Task
# ------------------------------------------------------------------------


def generate_jax_copy_data(
    rng: jax.random.PRNGKey,
    train_size: int = 10000,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    vocab_size: int = 16,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Wrapper that uses the imported copying functions and converts to JAX arrays.
    """
    # Extract seed from key
    seed = jax.random.randint(rng, (), 0, 2**31 - 1).item()

    # Use NumPy random generator for compatibility with copying.py
    np_rng = np.random.default_rng(seed)

    # Create dataset using imported function
    dataset = generate_copy_dataset(
        num_examples=train_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_tokens_to_copy=num_tokens_to_copy,
        selective=True,  # We want selective copying task
        rng=np_rng,
    )

    # Extract inputs and targets from PyTorch dataset
    inputs_tensor, targets_tensor = dataset.tensors

    # Convert to numpy then JAX arrays
    inputs_np = inputs_tensor.numpy()
    targets_np = targets_tensor.numpy()

    inputs_jax = jnp.array(inputs_np, dtype=jnp.int32)
    targets_jax = jnp.array(targets_np, dtype=jnp.int32)

    # Update key
    new_key = jax.random.split(rng)[0]

    return inputs_jax, targets_jax, new_key

# ------------------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    d_model: int
    max_len: int = 2048

    def setup(self):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))

        pos_enc = jnp.zeros((self.max_len, self.d_model))
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pos_enc = pos_enc

    def __call__(self, x):
        seq_len = x.shape[1]
        return x + self.pos_enc[:seq_len][None, :, :]


# ------------------------------------------------------------------------
# Attention Mechanisms
# ------------------------------------------------------------------------


class VanillaAttention(nn.Module):
    """Standard scaled dot-product attention with causal mask."""

    dim: int
    num_heads: int

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads

        self.wq = nn.Dense(self.dim)
        self.wk = nn.Dense(self.dim)
        self.wv = nn.Dense(self.dim)
        self.wo = nn.Dense(self.dim)

    def __call__(self, x, training=False):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.wq(x)  # (batch_size, seq_len, dim)
        k = self.wk(x)  # (batch_size, seq_len, dim)
        v = self.wv(x)  # (batch_size, seq_len, dim)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for batched matrix multiplication
        q = q.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scale = jnp.sqrt(self.head_dim)
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale  # (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask == 0, jnp.finfo(scores.dtype).min, scores)

        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply attention weights
        context = jnp.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape output
        context = context.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.dim)  # (batch_size, seq_len, dim)

        # Output projection
        out = self.wo(context)  # (batch_size, seq_len, dim)
        return out


def enforce_associativity(op=None, *, check_flag="check_associativity", sample_key="z_sample", tol=1e-4, eq_fn=None):
    """
    Decorator to enforce the associativity property of a binary operation.

    When the decorated function is called with check_flag=True in its kwargs,
    it computes:
        left  = op(op(x, y, ...), z, ...)
        right = op(x, op(y, z, ...), ...)
    where z is taken from kwargs[sample_key] if provided, else defaults to y.

    The decorator supports both standalone functions and methods (i.e., functions with a 'self' parameter).
    """
    if eq_fn is None:
        eq_fn = lambda a, b: jnp.allclose(a, b, atol=tol)

    def decorator(func):
        # Determine if we're dealing with a method (i.e. first parameter is "self")
        try:
            is_method = inspect.getfullargspec(func).args[0] == "self"
        except Exception:
            is_method = False

        @wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs.get(check_flag, False):
                if is_method:
                    if len(args) < 3:
                        raise ValueError("Expected at least three arguments (self, x, y) for method.")
                    self_obj, x, y, *rest = args
                    z = kwargs.get(sample_key, y)
                    # Recursively call the method with self preserved.
                    left = func(self_obj, func(self_obj, x, y, *rest, **kwargs), z, *rest, **kwargs)
                    right = func(self_obj, x, func(self_obj, y, z, *rest, **kwargs), *rest, **kwargs)
                else:
                    if len(args) < 2:
                        raise ValueError("Expected at least two arguments (x, y) for function.")
                    x, y, *rest = args
                    z = kwargs.get(sample_key, y)
                    left = func(func(x, y, *rest, **kwargs), z, *rest, **kwargs)
                    right = func(x, func(y, z, *rest, **kwargs), *rest, **kwargs)
                if not eq_fn(left, right):
                    raise ValueError(f"Associativity check failed: {left} != {right}")
            return func(*args, **kwargs)

        return wrapper

    if op is not None:
        return decorator(op)
    return decorator


# class ScanAttention(nn.Module):
#     dim: int
#     num_heads: int
#     seq_len: int
#     spectral_basis: jnp.ndarray

#     def setup(self):
#         assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
#         assert (
#             self.spectral_basis.shape == (self.seq_len, self.num_heads)
#         ), "Spectral basis shape must match seq_len and num_heads"
#         self.head_dim = self.dim // self.num_heads

#         self.wq = nn.Dense(self.dim)
#         self.wk = nn.Dense(self.dim)
#         self.wv = nn.Dense(self.dim)
#         self.wo = nn.Dense(self.dim)

#     def __call__(self, x, training: bool = False):
#         batch_size, seq_len, _ = x.shape
#         assert seq_len == self.seq_len, f"Input seq_len {seq_len} must match initialized seq_len {self.seq_len}"

#         # Linear projections and split into heads.
#         q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         q = q.transpose(0, 2, 1, 3)  # [B, H, L, head_dim]

#         k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.transpose(0, 2, 1, 3) / jnp.sqrt(self.head_dim)  # [B, H, L, head_dim]

#         v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.transpose(0, 2, 1, 3)  # [B, H, L, head_dim]

#         # Convolve each head's key and value with its corresponding spectral filter.
#         k_conv = jax.vmap(
#             jax.vmap(conv_single, in_axes=(0, 0), out_axes=0),
#             in_axes=(None, 0),
#             out_axes=0,
#         )(self.spectral_basis.T, k)

#         # Classical scaled dot-product attention.
#         scale = jnp.sqrt(self.head_dim)
#         scores = jnp.matmul(q, jnp.swapaxes(k_conv, -2, -1)) / scale  # [B, H, L, L]

#         # Apply a causal mask.
#         mask = jnp.tril(jnp.ones((seq_len, seq_len)))
#         scores = jnp.where(mask == 0, jnp.finfo(scores.dtype).min, scores)

#         attn_weights = jax.nn.softmax(scores, axis=-1)  # [B, H, L, L]

#         context = jnp.matmul(attn_weights, v)  # [B, H, L, head_dim]
#         context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
#         out = self.wo(context)
#         return out

def get_hankel_matrix(n: int) -> np.ndarray:
    z = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            z[i - 1, j - 1] = 2 / ((i + j) ** 3 - (i + j))
    return z


def get_top_hankel_eigh(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    eig_vals, eig_vecs = np.linalg.eigh(get_hankel_matrix(n))
    return eig_vals[-k:], eig_vecs[:, -k:]

def get_precomputed_spectral_basis(seq_len: int, num_heads: int) -> jnp.ndarray:
    """Precompute the spectral basis for ScanAttention.

    Args:
        seq_len: Length of the sequence
        num_heads: Number of attention heads

    Returns:
        jnp.ndarray: Precomputed spectral basis of shape [seq_len, num_heads]
    """
    _, spectral_basis = get_top_hankel_eigh(seq_len, num_heads)
    return jnp.array(spectral_basis)

@jax.jit
def conv(filters: jnp.ndarray, keys: jnp.ndarray) -> jnp.ndarray:
    """
    Compute convolution to project input sequences into the spectral basis.

    Args:
      filters: jnp.ndarray of shape [seq_len, num_heads]
        Each column is the spectral filter for a head.
      keys: jnp.ndarray of shape [batch_size, num_heads, seq_len, head_dim]
        Input sequences for each head and feature.

    Returns:
      jnp.ndarray of shape [batch_size, num_heads, seq_len, head_dim]
        The result of convolving each head's filter with the corresponding input sequences.
    """

    # 1. Basic 1D convolution that truncates the output to the input length.
    def conv1d(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.signal.convolve(x, y, method="fft")[: x.shape[0]]

    # 2. Apply conv1d over the feature dimension.
    # Given a filter (shape [seq_len]) and a key matrix (shape [seq_len, head_dim]),
    # apply conv1d independently for each feature.
    conv_over_features = jax.vmap(conv1d, in_axes=(None, 1), out_axes=1)

    # 3. For one head: given a filter (shape [seq_len]) and keys for that head
    # (shape [seq_len, head_dim]), convolve each feature dimension.
    def conv_head(filter_seq: jnp.ndarray, key_seq: jnp.ndarray) -> jnp.ndarray:
        return conv_over_features(filter_seq, key_seq)

    # 4. Vectorize conv_head over the heads.
    # filters_T will have shape [num_heads, seq_len],
    # and for each head, we pair the filter with the corresponding key sequence (shape [seq_len, head_dim]).
    conv_over_heads = jax.vmap(conv_head, in_axes=(0, 0), out_axes=0)

    # 5. Transpose filters so that each head's filter is a row: [num_heads, seq_len].
    filters_T = filters.T

    # 6. Finally, vectorize over the batch dimension.
    # For each element in the batch (of shape [num_heads, seq_len, head_dim]),
    # apply conv_over_heads.
    conv_over_batch = jax.vmap(lambda keys_batch: conv_over_heads(filters_T, keys_batch), in_axes=0, out_axes=0)

    return conv_over_batch(keys)

# class ScanAttention(nn.Module):
#     dim: int
#     num_heads: int
#     seq_len: int
#     spectral_basis: jnp.ndarray
#     eps: float = 1e-5

#     def setup(self):
#         self.head_dim = self.dim // self.num_heads
#         self.wq = nn.Dense(self.dim)
#         self.wk = nn.Dense(self.dim)
#         self.wv = nn.Dense(self.dim)
#         self.wo = nn.Dense(self.dim)
#         # Gating projection: maps [D/H * D/H] to [1]
#         self.gate_proj = nn.Dense(1)
#         self.beta = self.param("beta", lambda rng: jnp.array(1.0))

#     def softplus(self, x, beta=1.0):
#         return (1.0 / beta) * jnp.log1p(jnp.exp(beta * x))


#     def __call__(self, x, training=False):
#         batch_size, seq_len, _ = x.shape

#         # Compute QKV projections
#         # q: [B, H, S, D/H]
#         q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
#         k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) / jnp.sqrt(self.head_dim)
#         v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

#         # Apply spectral filtering to v (assuming conv is defined elsewhere)
#         # q = conv(self.spectral_basis, q)
#         k = conv(self.spectral_basis, k)
#         v = conv(self.spectral_basis, v)

#         # Compute outer product Z = V ⊗ K for each position
#         # Z: [B, H, S, D/H, D/H]
#         Z = jnp.einsum('bhsd,bhse->bhsde', v, k)

#         # Compute gates
#         # gate_input: [B*H*S, D/H*D/H]
#         gate_input = Z.reshape(batch_size * self.num_heads * seq_len, self.head_dim * self.head_dim)
#         gates_logits = self.gate_proj(gate_input)
#         # gates = self.softplus(gates_logits, 1.0) + self.eps  # [B*H*S, 1]
#         gates = jnp.exp(gates_logits) + self.eps
#         gates = gates.reshape(batch_size, self.num_heads, seq_len, 1, 1)  # [B, H, S, 1, 1]

#         # Prepare scan inputs
#         gated_Z = gates * Z
#         gated_Z_scan = jnp.moveaxis(gated_Z, 2, 0)  # [S, B, H, D/H, D/H]
#         gates_scan = jnp.moveaxis(gates, 2, 0)        # [S, B, H, 1, 1]

#         # Define associative combination function
#         def combine_fn(carry, next_val):
#             # carry and next_val are tuples: (sum_gated_Z, sum_gates)
#             # sum_gated_Z: [B, H, D/H, D/H]
#             # sum_gates: [B, H, 1, 1]
#             carry_gated_Z, carry_gates = carry
#             next_gated_Z, next_gates = next_val

#             # Associative operation: addition
#             new_gated_Z = carry_gated_Z + next_gated_Z
#             new_gates = carry_gates + next_gates

#             return (new_gated_Z, new_gates)

#         # Run associative scan
#         # scanned: [S, B, H, D/H, D/H], [S, B, H, 1, 1]
#         scanned_gated_Z, scanned_gates = jax.lax.associative_scan(
#             combine_fn,
#             (gated_Z_scan, gates_scan),
#             axis=0
#         )

#         # Move sequence dimension back
#         # cumulative_gated_Z: [B, H, S, D/H, D/H]
#         cumulative_gated_Z = jnp.moveaxis(scanned_gated_Z, 0, 2)
#         # cumulative_gates: [B, H, S, 1, 1]
#         cumulative_gates = jnp.moveaxis(scanned_gates, 0, 2)

#         # Normalize causally
#         # attn_weights: [B, H, S, D/H, D/H]
#         attn_weights = cumulative_gated_Z / (cumulative_gates + self.eps)

#         # Compute output
#         # output_raw: [B, H, S, D/H]
#         output_raw = jnp.einsum('bhsd,bhsde->bhse', q, attn_weights)

#         # L2 normalization for stability
#         output_norm = jnp.linalg.norm(output_raw, axis=3, keepdims=True)
#         output_norm = jnp.maximum(output_norm, self.eps)
#         output_normalized = output_raw / output_norm

#         # Reshape and project
#         # output_normalized: [B, S, H, D/H] -> [B, S, D]
#         output_normalized = output_normalized.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
#         output = self.wo(output_normalized)

#         return output

class ScanAttention(nn.Module):
    dim: int
    num_heads: int
    seq_len: int
    spectral_basis: jnp.ndarray
    eps: float = 1e-5

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.wq = nn.Dense(self.dim)
        self.wk = nn.Dense(self.dim)
        self.wv = nn.Dense(self.dim)
        self.wo = nn.Dense(self.dim)
        # Gating projection: maps [D/H * D/H] to [1]
        self.gate_proj = nn.Dense(1)
        self.beta = self.param("beta", lambda rng: jnp.array(1.0))

    def softplus(self, x, beta=1.0):
        return (1.0 / beta) * jnp.log1p(jnp.exp(beta * x))

    def __call__(self, x, training=False):
        batch_size, seq_len, _ = x.shape

        # Compute QKV projections
        q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for bmm
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3) / jnp.sqrt(self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        # Spectral basis
        k = conv(self.spectral_basis, k)
        v = conv(self.spectral_basis, v)

        # Compute pairwise interactions via outer product
        # Z: [B, H, S, D/H, D/H]
        Z = jnp.einsum("bhsd,bhse->bhsde", v, k)

        # Compute gates
        # Z: [B, H, S, D/H, D/H]
        gate_input = Z.reshape(*Z.shape[:3], -1)  # [B, H, S, (D/H)^2]
        gates_logits = self.gate_proj(gate_input)  # [B, H, S, 1]
        gates = jax.nn.relu(gates_logits) ** 2 + self.eps  # [B, H, S, 1]
        gates = gates[..., None]  # [B, H, S, 1, 1]

        # Prepare inputs for associative scan by gating Z
        gated_Z = gates * Z  # [B, H, S, D/H, D/H]

        # Define the associative addition operation
        def combine_fn(carry, next_val):
            sum_gated_Z, sum_gates = carry
            next_gated_Z, next_gates = next_val
            return (sum_gated_Z + next_gated_Z, sum_gates + next_gates)

        # Build a cumulative memory across the sequence dimension
        cumulative_gated_Z, cumulative_gates = jax.lax.associative_scan(combine_fn, (gated_Z, gates), axis=2)

        # Normalize to obtain causal attention weights.
        attn_weights = cumulative_gated_Z / (cumulative_gates + self.eps)

        # Compute raw output by applying the attention weights to the query.
        output_raw = jnp.einsum("bhsd,bhsde->bhse", q, attn_weights)

        # Normalize the raw output onto the unit sphere for stability.
        output_norm = jnp.linalg.norm(output_raw, axis=3, keepdims=True)
        output_normalized = output_raw / jnp.maximum(output_norm, self.eps)

        # Rearrange dimensions and apply the final projection.
        output_normalized = output_normalized.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        output = self.wo(output_normalized)

        return output

# ------------------------------------------------------------------------
# Transformer Architecture
# ------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """Transformer layer with configurable attention mechanism."""

    dim: int
    num_heads: int
    mlp_dim: int
    seq_len: int
    dropout_rate: float = 0.1
    attention_type: str = "vanilla"  # 'vanilla' or 'scan'
    spectral_basis: jnp.ndarray | None = None

    def setup(self):
        # Attention layer
        if self.attention_type == "vanilla":
            self.attention = VanillaAttention(self.dim, self.num_heads)
        elif self.attention_type == "scan":
            self.attention = ScanAttention(self.dim, self.num_heads, self.seq_len, self.spectral_basis)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        # Normalization and MLP layers
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

        # Define MLP components separately instead of using Sequential
        self.mlp_dense1 = nn.Dense(self.mlp_dim)
        self.mlp_dense2 = nn.Dense(self.dim)

        # Dropout layers
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=False):
        # Apply attention with residual connection
        attn_output = self.attention(self.ln1(x), training=training)
        x = x + self.dropout(attn_output, deterministic=not training)

        # Apply MLP with residual connection
        h = self.ln2(x)
        h = self.mlp_dense1(h)
        h = nn.gelu(h)
        h = self.mlp_dense2(h)
        h = self.dropout(h, deterministic=not training)

        x = x + h

        return x


class Transformer(nn.Module):
    """Transformer model for the selective copying task."""

    config: TransformerConfig
    spectral_basis: jnp.ndarray | None = None

    def setup(self):
        # Token embedding
        self.token_embedding = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)

        # Positional encoding
        self.position_encoding = PositionalEncoding(d_model=self.config.dim, max_len=self.config.max_seq_len)

        # Transformer layers
        self.layers = [
            TransformerLayer(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                seq_len=self.config.seq_len,
                mlp_dim=self.config.dim * 4,
                dropout_rate=self.config.dropout_rate,
                attention_type=self.config.attention_type,
                spectral_basis=self.spectral_basis,
            )
            for _ in range(self.config.num_layers)
        ]

        # Output head
        self.ln_f = nn.LayerNorm()
        self.output_head = nn.Dense(self.config.vocab_size)

    def __call__(self, x, training=False):
        # Get token embeddings
        x = self.token_embedding(x)

        # Add positional encodings
        x = self.position_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, training=training)

        # Apply final layer norm
        x = self.ln_f(x)

        # Generate logits
        logits = self.output_head(x)

        return logits


# ------------------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------------------


def cross_entropy_loss(logits, targets, ignore_index=-100):
    """Compute cross entropy loss with optional label smoothing."""
    vocab_size = logits.shape[-1]

    # Create one-hot encodings for targets, ignoring padding tokens
    mask = targets != ignore_index
    targets_one_hot = jax.nn.one_hot(targets * mask, vocab_size)

    # Compute softmax cross entropy
    loss = optax.softmax_cross_entropy(logits, targets_one_hot)

    # Apply mask to ignore specified tokens
    loss = loss * mask

    # Compute mean loss over non-ignored tokens
    mean_loss = jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1)

    return mean_loss


def compute_accuracy(logits, targets, ignore_index=-100):
    """Compute classification accuracy, ignoring specified tokens."""
    predictions = jnp.argmax(logits, axis=-1)
    mask = targets != ignore_index
    correct = (predictions == targets) * mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)
    return accuracy  # Return as decimal [0,1] - conversion to percentage happens elsewhere


@jax.jit
def train_step(state, batch, dropout_rng):
    """Single training step."""
    inputs, targets = batch
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs, training=True, rngs={"dropout": dropout_rng})
        loss = cross_entropy_loss(logits, targets)
        accuracy = compute_accuracy(logits, targets)
        return loss, accuracy  # Return accuracy directly
    (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, accuracy

@jax.jit
def eval_step(state, batch):
    """Single evaluation step."""
    inputs, targets = batch
    logits = state.apply_fn({"params": state.params}, inputs, training=False)
    loss = cross_entropy_loss(logits, targets)
    accuracy = compute_accuracy(logits, targets)
    return loss, accuracy

def create_train_state(config, rng):
    """Initialize the training state."""
    # Split the random key
    rng, init_rng = jax.random.split(rng)

    # Create model instance
    spectral_basis = get_precomputed_spectral_basis(config.seq_len, config.num_heads)
    model = Transformer(config, spectral_basis=spectral_basis)

    # Initialize model with dummy input
    dummy_input = jnp.ones((1, config.seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)["params"]

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay),
    )

    # Create training state
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    return state, rng

def train_model(config, state, train_data, val_data, rng):
    """Train the model and evaluate periodically, emphasizing validation accuracy."""

    # Create batches
    num_train_batches = len(train_data[0]) // config.batch_size
    num_val_batches = len(val_data[0]) // config.batch_size

    train_batch_size = num_train_batches * config.batch_size
    val_batch_size = num_val_batches * config.batch_size

    train_inputs = train_data[0][:train_batch_size].reshape(num_train_batches, config.batch_size, -1)
    train_targets = train_data[1][:train_batch_size].reshape(num_train_batches, config.batch_size, -1)
    val_inputs = val_data[0][:val_batch_size].reshape(num_val_batches, config.batch_size, -1)
    val_targets = val_data[1][:val_batch_size].reshape(num_val_batches, config.batch_size, -1)

    # Metrics tracking
    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Progress bar
    num_steps = config.num_epochs * num_train_batches
    progress_bar = tqdm(range(num_steps), desc="Training")

    for epoch in range(config.num_epochs):
        perm = jax.random.permutation(rng, num_train_batches)
        rng, _ = jax.random.split(rng)

        # Training phase
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0

        for i in range(num_train_batches):
            batch_idx = perm[i]
            inputs = train_inputs[batch_idx]
            targets = train_targets[batch_idx]
            rng, dropout_rng = jax.random.split(rng)
            state, loss, accuracy = train_step(state, (inputs, targets), dropout_rng)
            epoch_train_loss += loss / num_train_batches
            epoch_train_accuracy += accuracy / num_train_batches

            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{epoch_train_loss:.4f}",
                    "train_acc": f"{epoch_train_accuracy * 100:.2f}%",
                }
            )

        # Record training metrics
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy * 100)  # Store as percentage

        # Validation phase
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        for i in range(num_val_batches):
            inputs = val_inputs[i]
            targets = val_targets[i]
            loss, accuracy = eval_step(state, (inputs, targets))
            epoch_val_loss += loss / num_val_batches
            epoch_val_accuracy += accuracy / num_val_batches

        # Record validation metrics
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy * 100)  # Store as percentage

        # Update progress bar with both training and validation metrics
        progress_bar.set_postfix(
            {
                "epoch": epoch + 1,
                "train_loss": f"{epoch_train_loss:.4f}",
                "train_acc": f"{epoch_train_accuracy * 100:.2f}%",
                "val_loss": f"{epoch_val_loss:.4f}",
                "val_acc": f"{epoch_val_accuracy * 100:.2f}%",
            }
        )

        # Save best model based on validation accuracy
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_params = jax.tree.map(lambda x: x.copy(), state.params)
            print(f"New best validation accuracy at epoch {epoch + 1}: {best_val_accuracy * 100:.2f}%")

    progress_bar.close()

    # Return best model and metrics
    best_state = train_state.TrainState.create(apply_fn=state.apply_fn, params=best_params, tx=state.tx)
    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_accuracy * 100,  # Stored as percentage
    }

    return best_state, metrics


# ------------------------------------------------------------------------
# Evaluation and Analysis Functions
# ------------------------------------------------------------------------


def visualize_attention_params(state, config):
    """Extract and visualize the learned attention parameters."""
    params = state.params

    if config.attention_type == "scan":
        # Extract learnable parameters from ScanAttention layers
        params_data = []
        layer_names = []

        for i in range(config.num_layers):
            layer_params = params["layers_{}".format(i)]["attention"]

            # Get beta, temp, and sharp values
            beta = jnp.exp(layer_params["log_beta"])
            temp = jnp.exp(layer_params["log_temp"])
            sharp = jnp.exp(layer_params["log_sharp"])

            params_data.append({"beta": beta, "temp": temp, "sharp": sharp})
            layer_names.append(f"Layer {i+1}")

        # Create visualizations
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot beta values
        for i, (name, data) in enumerate(zip(layer_names, params_data)):
            axs[0].bar(np.arange(config.num_heads) + i * (config.num_heads + 1), data["beta"], label=name)
        axs[0].set_title("Beta Values (Sharpness)")
        axs[0].set_xlabel("Head Index")
        axs[0].set_ylabel("Value")

        # Plot temperature values
        for i, (name, data) in enumerate(zip(layer_names, params_data)):
            axs[1].bar(np.arange(config.num_heads) + i * (config.num_heads + 1), data["temp"], label=name)
        axs[1].set_title("Temperature Values")
        axs[1].set_xlabel("Head Index")
        axs[1].set_ylabel("Value")

        # Plot sharpening values
        for i, (name, data) in enumerate(zip(layer_names, params_data)):
            axs[2].bar(np.arange(config.num_heads) + i * (config.num_heads + 1), data["sharp"], label=name)
        axs[2].set_title("Sharpening Values")
        axs[2].set_xlabel("Head Index")
        axs[2].set_ylabel("Value")

        # Add legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=config.num_layers)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig("scan_attn_params.png")
        plt.close()

        # Print parameters
        print("Learned Scan Attention Parameters:")
        for i, (name, data) in enumerate(zip(layer_names, params_data)):
            print(f"{name}:")
            for h in range(config.num_heads):
                print(
                    f"  Head {h+1}: beta={data['beta'][h]:.4f}, temp={data['temp'][h]:.4f}, sharp={data['sharp'][h]:.4f}"
                )

def compare_models(vanilla_metrics, scan_metrics):
    """Compare performance between vanilla and scan attention models."""
    # Extract metrics
    epochs = range(1, len(vanilla_metrics["train_losses"]) + 1)

    # Plot training curves - accuracies already stored as percentages
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss
    axs[0, 0].plot(epochs, vanilla_metrics["train_losses"], label="Vanilla Attention")
    axs[0, 0].plot(epochs, scan_metrics["train_losses"], label="Scan Attention")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Validation loss
    axs[0, 1].plot(epochs, vanilla_metrics["val_losses"], label="Vanilla Attention")
    axs[0, 1].plot(epochs, scan_metrics["val_losses"], label="Scan Attention")
    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Training accuracy
    axs[1, 0].plot(epochs, vanilla_metrics["train_accuracies"], label="Vanilla Attention")
    axs[1, 0].plot(epochs, scan_metrics["train_accuracies"], label="Scan Attention")
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Accuracy (%)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Validation accuracy
    axs[1, 1].plot(epochs, vanilla_metrics["val_accuracies"], label="Vanilla Attention")
    axs[1, 1].plot(epochs, scan_metrics["val_accuracies"], label="Scan Attention")
    axs[1, 1].set_title("Validation Accuracy")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Accuracy (%)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

    # Print final metrics - values already stored as percentages
    print("=== Model Comparison ===")
    print(f"Vanilla Attention - Best Validation Accuracy: {vanilla_metrics['best_val_accuracy']:.2f}%")
    print(f"Scan Attention - Best Validation Accuracy: {scan_metrics['best_val_accuracy']:.2f}%")

    improvement = scan_metrics["best_val_accuracy"] - vanilla_metrics["best_val_accuracy"]
    print(f"Absolute Improvement: {improvement:.2f} percentage points")


# ------------------------------------------------------------------------
# Main Experiment Functions
# ------------------------------------------------------------------------


def train_and_evaluate(config):
    """Train and evaluate a model with the given configuration."""
    print(f"=== Training model with {config.attention_type} attention ===")

    # Set random seed
    seed = config.seed
    rng = jax.random.PRNGKey(seed)

    # Generate datasets
    print("Generating datasets...")
    train_inputs, train_targets, rng = generate_jax_copy_data(
        rng=rng,
        train_size=config.train_size,
        seq_len=config.seq_len,
        num_tokens_to_copy=config.num_tokens_to_copy,
        vocab_size=config.vocab_size,
    )
    train_data = (train_inputs, train_targets)

    val_inputs, val_targets, rng = generate_jax_copy_data(
        rng=rng,
        train_size=config.val_size,
        seq_len=config.seq_len,
        num_tokens_to_copy=config.num_tokens_to_copy,
        vocab_size=config.vocab_size,
    )
    val_data = (val_inputs, val_targets)

    # Create model and initialize training state
    print("Initializing model...")
    state, rng = create_train_state(config, rng)

    # Train model
    print("Training model...")
    best_state, metrics = train_model(config, state, train_data, val_data, rng)

    # # Visualize attention parameters (for scan attention)
    # if config.attention_type == "scan":
    #     visualize_attention_params(best_state, config)

    return best_state, metrics


def run_experiments():
    """Run experiments to compare vanilla and scan attention."""
    # Base configuration with values suitable for the copying task
    config_base = TransformerConfig(
        vocab_size=32,  # Increased vocab size to accommodate special tokens
        dim=128,  # Embedding dimension
        num_heads=8,  # Number of attention heads
        num_layers=2,  # Number of transformer layers
        dropout_rate=0.0,  # Dropout rate for regularization
        max_seq_len=256,  # Maximum sequence length
        learning_rate=3e-4,  # Learning rate
        weight_decay=1e-2,  # Weight decay for regularization
        attention_type="vanilla",  # Default attention type
        seq_len=64,  # Sequence length for the copying task
        num_tokens_to_copy=16,  # Number of tokens to copy
        train_size=98304,  # Training set size
        val_size=2048,  # Validation set size
        batch_size=64,  # Batch size
        num_epochs=10,  # Number of training epochs
        seed=1746,  # Random seed for reproducibility
    )

    # Print experiment configuration
    print("Running selective copying experiment with the following configuration:")
    print(f"- Vocabulary size: {config_base.vocab_size}")
    print(f"- Sequence length: {config_base.seq_len}")
    print(f"- Tokens to copy: {config_base.num_tokens_to_copy}")
    print(f"- Model dimensions: {config_base.dim}")
    print(f"- Attention heads: {config_base.num_heads}")
    print(f"- Model layers: {config_base.num_layers}")

    # Configuration for vanilla attention
    config_vanilla = config_base

    # Configuration for scan attention
    config_scan = dataclasses.replace(config_base, attention_type="scan")

    # Train and evaluate scan attention model
    print("\n====== Training Scan Attention Model ======")
    scan_state, scan_metrics = train_and_evaluate(config_scan)

    # Train and evaluate vanilla attention model
    print("\n====== Training Vanilla Attention Model ======")
    vanilla_state, vanilla_metrics = train_and_evaluate(config_vanilla)

    # Compare models
    print("\n====== Comparing Models ======")
    compare_models(vanilla_metrics, scan_metrics)

    # Print summary of learned parameters for scan attention
    if hasattr(scan_state, "params"):
        print("\n====== Learned Scan Attention Parameters ======")
        for i in range(config_scan.num_layers):
            if f"layers_{i}" in scan_state.params:
                layer_params = scan_state.params[f"layers_{i}"]["attention"]
                print(f"\nLayer {i+1}:")

                # Beta (softplus sharpness)
                if "log_beta" in layer_params:
                    beta = jnp.exp(layer_params["log_beta"])
                    print(f"- Beta (softplus sharpness): {beta}")

                # Temperature
                if "log_temp" in layer_params:
                    temp = jnp.exp(layer_params["log_temp"])
                    print(f"- Temperature: {temp}")

                # Output sharpening
                if "log_sharp" in layer_params:
                    sharp = jnp.exp(layer_params["log_sharp"])
                    print(f"- Output sharpening: {sharp}")

    print("\nExperiment complete! Results saved to model_comparison.png")


if __name__ == "__main__":
    run_experiments()
