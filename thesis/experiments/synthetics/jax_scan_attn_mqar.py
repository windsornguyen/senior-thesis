import sys
import os
from functools import wraps, partial
import inspect
from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import flax
import flax.linen as nn
from flax.training import train_state

import optax
from tqdm.auto import tqdm

from thesis.experiments.synthetics.mqar import generate_mqar


class TransformerConfig:
    """Configuration class for the Transformer model, similar to PyTorch's approach."""

    def __init__(
        self,
        # Model architecture
        vocab_size: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.0,
        max_seq_len: int = 512,
        attention_type: str = "scan",  # 'vanilla' or 'scan'
        # Training parameters
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        # MQAR task parameters
        seq_len: int = 512,
        num_pairs: int = 4,
        alpha: float = 0.01,
        # Data parameters
        train_size: int = 131072,
        val_size: int = 4096,
        batch_size: int = 64,
        num_epochs: int = 16,
        seed: int = 1746,
        warmup_steps: int = 100,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.attention_type = attention_type

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.alpha = alpha

        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        self.warmup_steps = warmup_steps


# ------------------------------------------------------------------------
# Data Generation for MQAR Task
# ------------------------------------------------------------------------


def generate_jax_mqar_data(
    rng: jax.random.PRNGKey,
    num_examples: int = 10000,
    sequence_len: int = 512,
    vocab_size: int = 8192,
    num_pairs: int = 64,
    alpha: float = 2.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Wrapper that uses the imported MQAR function and converts to JAX arrays.
    """
    # Extract seed from key
    seed = jax.random.randint(rng, (), 0, 2**31 - 1).item()

    # Create dataset using imported function
    dataset = generate_mqar(
        num_examples=num_examples,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        num_pairs=num_pairs,
        alpha=alpha,
        seed=seed,
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


def generate_test_cases(rng, vocab_size, num_pairs, sequence_len, num_examples=10, alpha=0.01):
    """Generate test cases for evaluation."""
    # Extract seed from key
    seed = jax.random.randint(rng, (), 0, 2**31 - 1).item()

    # Create smaller test dataset
    dataset = generate_mqar(
        num_examples=num_examples,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        num_pairs=num_pairs,
        alpha=alpha,
        seed=seed,
    )

    # Extract inputs and targets
    inputs_tensor, targets_tensor = dataset.tensors

    # Convert to JAX arrays and create test cases
    test_cases = []
    for i in range(num_examples):
        inputs = jnp.array(inputs_tensor[i].numpy(), dtype=jnp.int32)
        targets = jnp.array(targets_tensor[i].numpy(), dtype=jnp.int32)
        test_cases.append((inputs, targets))

    return test_cases


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


_DEFAULT_ROPE_BASE_FREQUENCY = 10_000


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jax.Array:
    """Applies RoPE.

    Let B denote batch size, L denote sequence length, N denote number of heads,
    and H denote head dimension. Note that H must be divisible by 2.

    Args:
      inputs: Array of shape [B, L, N, H].
      positions:  Array of shape [B, L].
      base_frequency: Base frequency used to compute rotations.
      scale_factor: The scale factor used for positional interpolation, allowing
        an expansion of sequence length beyond the pre-trained context length.

    Returns:
      Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base_frequency**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


# ------------------------------------------------------------------------
# Attention Mechanisms
# ------------------------------------------------------------------------


class VanillaAttention(nn.Module):
    """Standard scaled dot-product attention with causal mask."""

    dim: int
    num_heads: int
    seq_len: int

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

        # Apply RoPE
        q = apply_rope(q, jnp.arange(seq_len), _DEFAULT_ROPE_BASE_FREQUENCY)
        k = apply_rope(k, jnp.arange(seq_len), _DEFAULT_ROPE_BASE_FREQUENCY)

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


def get_hankel_matrix(n: int) -> jnp.ndarray:
    i = jnp.arange(1, n + 1)
    j = jnp.arange(1, n + 1)
    I, J = jnp.meshgrid(i, j, indexing="ij")
    return 2 / ((I + J) ** 3 - (I + J))


def get_spectral_filters(n: int, k: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    eig_vals, eig_vecs = jnp.linalg.eigh(get_hankel_matrix(n))
    return eig_vals[-k:], eig_vecs[:, -k:]


def compute_dimensions(n: int) -> tuple[int, int, int]:
    # T_prime = (ceil(sqrt(n-2)))^2 + 2, sqrt_T_prime = ceil(sqrt(T_prime-2))
    T_prime = int((jnp.ceil(jnp.sqrt(n - 2))) ** 2 + 2)
    sqrt_T_prime = int(jnp.ceil(jnp.sqrt(T_prime - 2)))
    k_max = sqrt_T_prime
    return T_prime, sqrt_T_prime, k_max


def get_tensorized_spectral_filters(
    n: int,
    k: int,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Compute tensorized spectral filters for a given sequence length and filter count.

    Args:
        n: Sequence length.
        k: Number of filters.
        dtype: Desired computation dtype.

    Returns:
        A JAX array representing the tensorized spectral filters.
    """
    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    # Compute eigen-decomposition on a Hankel matrix of size sqrt_T_prime.
    Z = get_hankel_matrix(sqrt_T_prime)
    sigma, phi = jnp.linalg.eigh(Z)

    # Select the last k eigenvectors and weight them by sigma**0.25.
    phi_i = phi[:, -k:] * (sigma[-k:] ** 0.25)
    # Here phi_j is identical.
    phi_j = phi_i

    # Cast to desired dtype and compute the Kronecker product.
    phi_i = phi_i.astype(dtype)
    phi_j = phi_j.astype(dtype)
    filters = jnp.kron(phi_i, phi_j)
    return filters, filters


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


def get_precomputed_spectral_basis(seq_len: int, num_heads: int) -> jnp.ndarray:
    """Precompute the spectral basis for ScanAttention.

    Args:
        seq_len: Length of the sequence
        num_heads: Number of attention heads

    Returns:
        jnp.ndarray: Precomputed spectral basis of shape [seq_len, num_heads]
    """
    _, spectral_basis = get_spectral_filters(seq_len, num_heads)
    return jnp.array(spectral_basis)


# class ScanAttention(nn.Module):
#     dim: int
#     num_heads: int
#     seq_len: int
#     spectral_basis: jnp.ndarray

#     def setup(self):
#         assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
#         assert self.spectral_basis.shape == (
#             self.seq_len,
#             self.num_heads,
#         ), "Spectral basis shape must match seq_len and num_heads"
#         self.head_dim = self.dim // self.num_heads

#         self.wq = nn.Dense(self.dim)
#         self.wk = nn.Dense(self.dim)
#         self.wv = nn.Dense(self.dim)
#         self.wo = nn.Dense(self.dim)

#     def __call__(self, x, training: bool = False, num_chunks: int = 16):
#         batch_size, seq_len, _ = x.shape
#         assert seq_len == self.seq_len, f"Input seq_len {seq_len} must match initialized seq_len {self.seq_len}"

#         # What if we just apply the filters here? Impossible to learn? Only on K and V for example?
#         q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Apply positional embeddings
#         q = apply_rope(q, jnp.arange(seq_len), _DEFAULT_ROPE_BASE_FREQUENCY)
#         k = apply_rope(k, jnp.arange(seq_len), _DEFAULT_ROPE_BASE_FREQUENCY)

#         # Transpose for bmm
#         q = q.transpose(0, 2, 1, 3)
#         k = k.transpose(0, 2, 1, 3) / jnp.sqrt(self.head_dim)
#         v = v.transpose(0, 2, 1, 3)

#         # q_conv = conv(self.spectral_basis, q)
#         # k_conv = conv(self.spectral_basis, k)
#         # v_conv = conv(self.spectral_basis, v)

#         scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1))

#         mask = jnp.tril(jnp.ones((seq_len, seq_len)))
#         scores = jnp.where(mask == 0, jnp.finfo(scores.dtype).min, scores)

#         attn_weights = jax.nn.softmax(scores, axis=-1)

#         context = jnp.matmul(attn_weights, v)
#         context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
#         out = self.wo(context)
#         return out


def cdiv(seq_len: int, num_chunks: int) -> int:
    return (seq_len + num_chunks - 1) // num_chunks


@partial(jax.jit, static_argnames=("axis",))
def _causal_softmax(inputs: jax.Array, axis: int) -> jax.Array:
    running_max = jax.lax.associative_scan(jnp.maximum, inputs, axis=axis)
    num = jnp.exp(inputs - running_max)
    denom = jax.lax.associative_scan(jnp.add, num, axis=axis)
    return num / (denom + 1e-9)


def causal_softmax(inputs: jax.Array, axis: int = -1) -> jax.Array:
    if not isinstance(axis, int):
        raise ValueError("axis must be a Python int")
    if not isinstance(inputs.shape, tuple):
        raise ValueError("inputs must be a valid jax.Array with shape")
    ndim = len(inputs.shape)
    assert 0 <= axis < ndim, f"Axis {axis} out of bounds for ndim {ndim}"
    return _causal_softmax(inputs, axis)


def safe_exp(x: jnp.ndarray, beta: float = 1.0, axis: int = None) -> jnp.ndarray:
    scaled_x = beta * x
    if axis is not None:
        # Compute the maximum along the specified axis and subtract it from scaled_x.
        max_val = jnp.max(scaled_x, axis=axis, keepdims=True)
        return jnp.exp(scaled_x - max_val)
    else:
        return jnp.exp(scaled_x)


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
# Transformer Layers and Model
# ------------------------------------------------------------------------


class FeedForward(nn.Module):
    """MLP feed-forward layer with GELU activation, similar to PyTorch's MLP."""

    dim: int
    expansion_factor: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        hidden_dim = self.dim * self.expansion_factor
        x = nn.Dense(hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


class TransformerLayer(nn.Module):
    """A single transformer layer with attention and feed-forward, including dropout."""

    dim: int
    num_heads: int
    seq_len: int
    attention_class: Any  # VanillaAttention or ScanAttention
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=False):
        # Attention block
        residual = x
        x = nn.LayerNorm()(x)
        x = self.attention_class(dim=self.dim, num_heads=self.num_heads, seq_len=self.seq_len)(x, training=training)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = x + residual

        # Feed-forward block
        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForward(dim=self.dim, dropout_rate=self.dropout_rate)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    """Complete transformer model for the MQAR task."""

    config: TransformerConfig
    spectral_basis: Optional[jnp.ndarray] = None

    def setup(self):
        # Select attention class based on config
        if self.config.attention_type == "vanilla":
            attention_class = VanillaAttention
        elif self.config.attention_type == "scan":
            if self.spectral_basis is None:
                raise ValueError("spectral_basis must be provided for scan attention")
            attention_class = lambda *args, **kwargs: ScanAttention(
                *args, **kwargs, spectral_basis=self.spectral_basis
            )
        else:
            raise ValueError(f"Unknown attention type: {self.config.attention_type}")

        # Token embedding
        self.token_embedding = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=self.config.dim, max_len=self.config.max_seq_len)

        # Dropout for embeddings
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

        # Transformer layers
        self.layers = [
            TransformerLayer(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                seq_len=self.config.seq_len,
                attention_class=attention_class,
                dropout_rate=self.config.dropout_rate,
            )
            for _ in range(self.config.num_layers)
        ]

        # Output projection
        self.norm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size)

    def __call__(self, x, training=False):
        # Get embeddings and apply positional encoding
        x = self.token_embedding(x)
        # x = self.pos_encoding(x)

        # Apply dropout to embeddings
        x = self.dropout(x, deterministic=not training)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, training=training)

        # Apply output head
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def count_params(self):
        """Count the total number of parameters in the model."""
        # Create dummy input with correct sequence length
        dummy_input = jnp.ones((1, self.config.seq_len), dtype=jnp.int32)
        init_key = jax.random.PRNGKey(0)
        variables = self.init(init_key, dummy_input)
        return sum(x.size for x in jax.tree_util.tree_leaves(variables))


# ------------------------------------------------------------------------
# Training and Evaluation Logic
# ------------------------------------------------------------------------


@jax.jit
def cross_entropy_loss(logits, targets, ignore_index=-100):
    """Cross entropy loss that handles ignored indices without using boolean indexing."""
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    # Create a mask for valid positions (not ignored)
    valid_mask = jnp.where(targets != ignore_index, 1.0, 0.0)

    # Compute cross entropy for all positions
    one_hot_targets = jax.nn.one_hot(jnp.maximum(targets, 0), vocab_size)
    loss_all = -jnp.sum(one_hot_targets * jax.nn.log_softmax(logits), axis=-1)

    # Zero out the loss for ignored positions
    loss_masked = loss_all * valid_mask

    # Compute mean over valid positions only
    valid_count = jnp.maximum(jnp.sum(valid_mask), 1.0)  # Avoid division by zero
    loss = jnp.sum(loss_masked) / valid_count

    return loss


@jax.jit
def compute_metrics(logits, targets):
    """Compute loss and accuracy metrics."""
    loss = cross_entropy_loss(logits, targets)

    # Compute accuracy on non-padding tokens
    preds = jnp.argmax(logits, axis=-1)
    mask = targets != -100
    correct = (preds == targets) * mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)

    return {"loss": loss, "accuracy": accuracy}


@jax.jit
def train_step(state, inputs, targets, dropout_rng):
    """Single training step with improved JAX semantics and dropout RNG handling."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs, training=True, rngs={"dropout": dropout_rng})
        loss = cross_entropy_loss(logits, targets)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, targets)
    return new_state, metrics, new_dropout_rng


@jax.jit
def eval_step(state, inputs, targets):
    """Single evaluation step with improved JAX semantics."""
    logits = state.apply_fn({"params": state.params}, inputs, training=False)
    metrics = compute_metrics(logits, targets)
    return metrics


def create_train_state(config, model, key):
    """Create initial training state with AdamW and warmup cosine decay scheduler."""
    # Create a dummy input for model initialization
    dummy_input = jnp.ones((config.batch_size, config.seq_len), dtype=jnp.int32)

    # Initialize model
    variables = model.init(key, dummy_input)
    params = variables["params"]

    # Create learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_epochs * (config.train_size // config.batch_size),
        end_value=config.learning_rate * 0.1,
    )

    # Create optimizer with AdamW configuration
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay, b1=0.9, b2=0.999, eps=1e-8),
    )

    # Create training state
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_batch_iterator(inputs, targets, batch_size, shuffle=True):
    """Create batched dataset iterator with JAX RNG handling."""
    num_examples = inputs.shape[0]
    num_batches = num_examples // batch_size

    # This function isn't JIT-compiled, so regular Python control flow is fine
    def shuffle_indices(key):
        indices = jnp.arange(num_examples)
        if shuffle:
            indices = jax.random.permutation(key, indices)
        return indices

    # This function doesn't need JIT since it's just slicing
    def get_batch(indices, batch_idx):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_examples)
        batch_indices = indices[start_idx:end_idx]
        return inputs[batch_indices], targets[batch_indices]

    def iterator_fn(key):
        # Get shuffled indices for this iteration
        key, subkey = jax.random.split(key)
        all_indices = shuffle_indices(subkey)

        for i in range(num_batches):
            key, subkey = jax.random.split(key)
            batch_inputs, batch_targets = get_batch(all_indices, i)
            yield batch_inputs, batch_targets, key

    return iterator_fn


def train_model(config, model, train_inputs, train_targets, val_inputs=None, val_targets=None):
    """Train model with a step-based approach similar to PyTorch implementation."""
    # Initialize random keys
    rng = jax.random.PRNGKey(config.seed)
    rng, init_key, train_key, dropout_key = jax.random.split(rng, 4)

    # Create train state (model + optimizer)
    state = create_train_state(config, model, init_key)

    # Create batch iterators
    train_iter = get_batch_iterator(train_inputs, train_targets, config.batch_size, shuffle=True)
    val_iter = None
    if val_inputs is not None and val_targets is not None:
        val_iter = get_batch_iterator(val_inputs, val_targets, config.batch_size, shuffle=False)

    # Training metrics
    train_metrics = []
    val_metrics = []
    step = 0
    total_steps = config.num_epochs * (train_inputs.shape[0] // config.batch_size)

    # Initialize validation metrics
    val_loss = float("inf")
    val_acc = 0.0

    # Main training loop with tqdm progress bar
    with tqdm(total=total_steps, desc=f"Training {config.attention_type.capitalize()} Attention") as pbar:
        for epoch in range(config.num_epochs):
            # Training
            train_batch_metrics = []
            for batch_inputs, batch_targets, key in train_iter(train_key):
                state, metrics, dropout_key = train_step(state, batch_inputs, batch_targets, dropout_key)
                train_batch_metrics.append(metrics)
                train_key = key

                # Update progress bar
                step += 1
                pbar.update(1)

                # Always show both training and validation metrics
                pbar.set_postfix(
                    {
                        "train_loss": f"{metrics['loss']:.4f}",
                        "train_acc": f"{metrics['accuracy']*100:.2f}%",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc*100:.2f}%",
                    }
                )

                # Evaluate periodically
                if step % 50 == 0 and val_iter is not None:
                    val_batch_metrics = []
                    for val_inputs, val_targets, key in list(val_iter(rng))[:10]:  # Limit validation batches for speed
                        metrics = eval_step(state, val_inputs, val_targets)
                        val_batch_metrics.append(metrics)

                    # Compute average validation metrics
                    val_loss = jnp.mean(jnp.array([m["loss"] for m in val_batch_metrics]))
                    val_acc = jnp.mean(jnp.array([m["accuracy"] for m in val_batch_metrics]))

                    # Update progress bar with the new validation metrics
                    pbar.set_postfix(
                        {
                            "train_loss": f"{train_batch_metrics[-1]['loss']:.4f}",
                            "train_acc": f"{train_batch_metrics[-1]['accuracy']*100:.2f}%",
                            "val_loss": f"{val_loss:.4f}",
                            "val_acc": f"{val_acc*100:.2f}%",
                        }
                    )

                    # Store metrics
                    val_metrics.append({"step": step, "loss": val_loss, "accuracy": val_acc})

            # Compute average training metrics for this epoch
            train_loss = jnp.mean(jnp.array([m["loss"] for m in train_batch_metrics]))
            train_acc = jnp.mean(jnp.array([m["accuracy"] for m in train_batch_metrics]))
            train_metrics.append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc})

            # Print epoch summary
            print(
                f"\nEpoch {epoch+1}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

    return state, train_metrics, val_metrics


def compute_attention_stats(state, test_cases):
    """Compute attention-related statistics on test cases."""
    # Initialize empty lists to store results
    results = []

    # For each test case
    for test_idx, (inputs, targets) in enumerate(test_cases):
        # Get predictions
        logits = state.apply_fn({"params": state.params}, inputs[None, ...], training=False)
        predictions = jnp.argmax(logits, axis=-1)[0]  # Remove batch dimension

        # Calculate accuracy without boolean indexing
        mask = (targets != -100).astype(jnp.float32)
        correct = ((predictions == targets) * mask).astype(jnp.float32)
        accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)

        # Store the result
        results.append(
            {"test_idx": test_idx, "accuracy": float(accuracy), "predictions": predictions, "targets": targets}
        )

    return results


def visualize_attention_params(state, save_path=None):
    """Visualize the learned attention parameters."""
    import matplotlib.pyplot as plt

    # Extract ScanAttention parameters
    scan_attn_params = {}

    # In Flax, layers are stored with keys like 'layers_0', 'layers_1', etc.
    for key in state.params:
        # Check if the key is a layer key
        if isinstance(key, str) and key.startswith("layers_"):
            try:
                layer_idx = int(key.split("_")[1])
                layer_params = state.params[key]

                # Look for attention parameters in the layer
                if "log_beta" in layer_params:
                    # Extract and convert log parameters to their actual values
                    scan_attn_params[f"layer_{layer_idx}"] = {
                        "beta": jnp.exp(layer_params["log_beta"]),
                        "temp": jnp.exp(layer_params["log_temp"]),
                        "sharp": jnp.exp(layer_params["log_sharp"]),
                    }
            except (IndexError, ValueError):
                continue  # Skip if we can't parse the layer index

    if not scan_attn_params:
        print("No ScanAttention parameters found. Make sure your model uses ScanAttention.")
        # Try to print the available parameter structure to help debugging
        print("Parameter structure:", jax.tree_util.tree_map(lambda x: x.shape, state.params))
        return

    # Create a figure to visualize parameters
    num_layers = len(scan_attn_params)
    fig, axs = plt.subplots(num_layers, 3, figsize=(15, 5 * num_layers))

    # For a single layer, reshape axs
    if num_layers == 1:
        axs = axs.reshape(1, -1)

    # Plot parameters for each layer
    for i, (layer_name, params) in enumerate(scan_attn_params.items()):
        # Plot beta values
        axs[i, 0].bar(range(len(params["beta"])), params["beta"])
        axs[i, 0].set_title(f"{layer_name} - Beta (Sharpness)")
        axs[i, 0].set_xlabel("Head")
        axs[i, 0].set_ylabel("Value")

        # Plot temperature values
        axs[i, 1].bar(range(len(params["temp"])), params["temp"])
        axs[i, 1].set_title(f"{layer_name} - Temperature")
        axs[i, 1].set_xlabel("Head")
        axs[i, 1].set_ylabel("Value")

        # Plot sharpening values
        axs[i, 2].bar(range(len(params["sharp"])), params["sharp"])
        axs[i, 2].set_title(f"{layer_name} - Output Sharpening")
        axs[i, 2].set_xlabel("Head")
        axs[i, 2].set_ylabel("Value")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.savefig("attention_params.png")
        print("Figure saved to attention_params.png")

    plt.show()


def plot_results(config, train_metrics, val_metrics):
    """Plot training curves for a single model."""
    import matplotlib.pyplot as plt

    # Extract metrics
    epochs = [m["epoch"] for m in train_metrics]
    train_loss = [m["loss"] for m in train_metrics]
    train_acc = [m["accuracy"] * 100 for m in train_metrics]  # Convert to percentage

    val_steps = [m["step"] for m in val_metrics]
    val_loss = [m["loss"] for m in val_metrics]
    val_acc = [m["accuracy"] * 100 for m in val_metrics]  # Convert to percentage

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    ax1.plot(epochs, train_loss, "b-", label="Train")
    ax1.plot(
        [s / (len(epochs) * len(train_metrics) / len(val_metrics)) for s in val_steps],
        val_loss,
        "r-",
        label="Validation",
    )
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, train_acc, "b-", label="Train")
    ax2.plot(
        [s / (len(epochs) * len(train_metrics) / len(val_metrics)) for s in val_steps],
        val_acc,
        "r-",
        label="Validation",
    )
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Set title
    plt.suptitle(f"{config.attention_type.capitalize()} Attention on MQAR", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{config.attention_type}_mqar_training.png")
    plt.show()


def compare_models(
    vanilla_config, scan_config, vanilla_train_metrics, vanilla_val_metrics, scan_train_metrics, scan_val_metrics
):
    """Plot comparison of training curves for both models."""
    import matplotlib.pyplot as plt

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract validation metrics for both models
    vanilla_steps = [m["step"] for m in vanilla_val_metrics]
    vanilla_loss = [m["loss"] for m in vanilla_val_metrics]
    vanilla_acc = [m["accuracy"] * 100 for m in vanilla_val_metrics]  # Convert to percentage

    scan_steps = [m["step"] for m in scan_val_metrics]
    scan_loss = [m["loss"] for m in scan_val_metrics]
    scan_acc = [m["accuracy"] * 100 for m in scan_val_metrics]  # Convert to percentage

    # Plot loss comparison
    ax1.plot(vanilla_steps, vanilla_loss, "b-", label="Vanilla")
    ax1.plot(scan_steps, scan_loss, "r-", label="Scan")
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy comparison
    ax2.plot(vanilla_steps, vanilla_acc, "b-", label="Vanilla")
    ax2.plot(scan_steps, scan_acc, "r-", label="Scan")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Set title
    plt.suptitle("Vanilla vs Scan Attention on MQAR Task", fontsize=14)

    plt.tight_layout()
    plt.savefig("mqar_comparison.png")

    print("Comparison plot saved to mqar_comparison.png")
    plt.show()

    # Also create individual plots for each model
    plot_results(vanilla_config, vanilla_train_metrics, vanilla_val_metrics)
    plot_results(scan_config, scan_train_metrics, scan_val_metrics)


def run_experiment(config_override=None):
    """Run the Multi-Query Associative Recall experiment with both vanilla and scan attention."""
    # Default configuration
    base_config = TransformerConfig(
        dim=64,
        num_heads=1,
        num_layers=2,
        dropout_rate=0.0,
        max_seq_len=256,
        vocab_size=8192,
        learning_rate=3e-4,
        weight_decay=1e-1,
        warmup_steps=2000,
        # MQAR-specific parameters
        seq_len=256,  # Sequence length for MQAR
        num_pairs=32,  # Number of key-value pairs
        alpha=0.01,
        train_size=100000,
        val_size=3000,
        batch_size=64,
        num_epochs=64,
        seed=1746,
    )

    # Update base config with any overrides (except attention_type which we'll set for each model)
    if config_override:
        for key, value in config_override.items():
            if key != "attention_type":  # Skip attention_type as we'll set it specifically
                setattr(base_config, key, value)

    # Create configs for both vanilla and scan attention
    vanilla_config = TransformerConfig(**{k: getattr(base_config, k) for k in base_config.__dict__})
    scan_config = TransformerConfig(**{k: getattr(base_config, k) for k in base_config.__dict__})

    # Set attention types
    vanilla_config.attention_type = "vanilla"
    scan_config.attention_type = "scan"

    # Set up JAX random key
    rng = jax.random.PRNGKey(base_config.seed)

    # Generate training and validation data
    print("Generating training data...")
    rng, data_rng = jax.random.split(rng)
    train_inputs, train_targets, rng = generate_jax_mqar_data(
        data_rng, base_config.train_size, base_config.seq_len, base_config.vocab_size, base_config.num_pairs
    )

    print("Generating validation data...")
    rng, data_rng = jax.random.split(rng)
    val_inputs, val_targets, rng = generate_jax_mqar_data(
        data_rng, base_config.val_size, base_config.seq_len, base_config.vocab_size, base_config.num_pairs
    )

    # Generate test cases for evaluation
    print("Generating test cases...")
    rng, test_rng = jax.random.split(rng)
    test_cases = generate_test_cases(
        test_rng,
        base_config.vocab_size,
        base_config.num_pairs,
        base_config.seq_len,
        base_config.val_size,
        base_config.alpha,
    )

    # Train Scan Attention model
    print("\nTraining with Scan Attention...")
    # Add spectral basis computation here
    spectral_basis = get_precomputed_spectral_basis(scan_config.seq_len, scan_config.num_heads)
    scan_model = Transformer(scan_config, spectral_basis=spectral_basis)  # Pass spectral_basis here
    print(f"\nScan Attention Model:")
    print(f"- Parameters: ~{scan_model.count_params() / 1e6:.2f}M")

    scan_state, scan_train_metrics, scan_val_metrics = train_model(
        scan_config, scan_model, train_inputs, train_targets, val_inputs, val_targets
    )

    # Evaluate scan model
    # print("\nEvaluating Scan Attention model...")
    # scan_results = compute_attention_stats(scan_state, test_cases)
    # scan_avg_acc = sum(res["accuracy"] for res in scan_results) / len(scan_results)
    # print(f"Scan Attention average test accuracy: {scan_avg_acc:.4f}")

    # Train Vanilla Attention model
    print("\nTraining with Vanilla Attention...")
    vanilla_model = Transformer(vanilla_config)
    print(f"\nVanilla Attention Model:")
    print(f"- Parameters: ~{vanilla_model.count_params() / 1e6:.2f}M")

    vanilla_state, vanilla_train_metrics, vanilla_val_metrics = train_model(
        vanilla_config, vanilla_model, train_inputs, train_targets, val_inputs, val_targets
    )

    # Evaluate vanilla model
    # print("\nEvaluating Vanilla Attention model...")
    # vanilla_results = compute_attention_stats(vanilla_state, test_cases)
    # vanilla_avg_acc = sum(res["accuracy"] for res in vanilla_results) / len(vanilla_results)
    # print(f"Vanilla Attention average test accuracy: {vanilla_avg_acc:.4f}")

    # Compare results
    # print("\nComparison:")
    # print(f"- Vanilla Attention: {vanilla_avg_acc:.4f}")
    # print(f"- Scan Attention: {scan_avg_acc:.4f}")
    # print(f"- Improvement: {(scan_avg_acc - vanilla_avg_acc) * 100:.2f}%")

    # Visualize attention parameters if requested
    print("\nVisualizing Scan Attention parameters...")
    visualize_attention_params(scan_state, save_path="scan_attention_params.png")

    # Plot training curves for both models
    compare_models(
        vanilla_config, scan_config, vanilla_train_metrics, vanilla_val_metrics, scan_train_metrics, scan_val_metrics
    )

    return {
        "vanilla": {
            "state": vanilla_state,
            # "results": vanilla_results,
            "train_metrics": vanilla_train_metrics,
            "val_metrics": vanilla_val_metrics,
        },
        "scan": {
            "state": scan_state,
            # "results": scan_results,
            "train_metrics": scan_train_metrics,
            "val_metrics": scan_val_metrics,
        },
    }


if __name__ == "__main__":
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"JAX version: {jax.__version__}")
    print(f"Flax version: {flax.__version__}")
    print(f"Optax version: {optax.__version__}")

    # Run experiment for both models and get results
    results = run_experiment()

    # Results are already plotted inside run_experiment
