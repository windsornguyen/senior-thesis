# ===----------------------------------------------------------------------=== #
# File: jax_scan_attn_mqar_separate.py
# Description: Comparison of Transformer and Spectron models on MQAR task
# ===----------------------------------------------------------------------=== #

import sys
from functools import wraps, partial
import inspect
from typing import Tuple, Any, Optional, Union
import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import flax
import flax.linen as lnn
from flax.training import train_state

import optax
from tqdm.auto import tqdm

from thesis.experiments.synthetics.mqar import generate_mqar

# ------------------------------------------------------------------------
# Model Configurations
# ------------------------------------------------------------------------


class BaseConfig:
    """Base configuration shared between models."""

    def __init__(
        self,
        # Model architecture
        vocab_size: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        max_seq_len: int = 512,
        # Training parameters
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        weight_tying: bool = False,
        bias: bool = False,
        eps: float = 1e-5,
        # MQAR task parameters
        seq_len: int = 512,
        num_pairs: int = 4,
        alpha: float = 0.1,
        # Data parameters
        train_size: int = 131072,
        val_size: int = 4096,
        bsz: int = 64,
        num_epochs: int = 16,
        seed: int = 1746,
        warmup_steps: int = 1000,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_tying = weight_tying
        self.bias = bias
        self.eps = eps
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.alpha = alpha
        self.train_size = train_size
        self.val_size = val_size
        self.bsz = bsz
        self.num_epochs = num_epochs
        self.seed = seed
        self.warmup_steps = warmup_steps
        self.dtype = dtype


class TransformerConfig(BaseConfig):
    """Configuration for vanilla Transformer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "transformer"


class SpectronConfig(BaseConfig):
    """Configuration for Spectron model."""

    def __init__(
        self,
        num_eigh: int = 32,
        use_tensordot: bool = True,
        inter_dim: Optional[int] = None,
        weight_tying: bool = False,
        bias: bool = False,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "spectron"
        self.num_eigh = num_eigh
        self.use_tensordot = use_tensordot
        self.inter_dim = inter_dim if inter_dim is not None else 4 * self.dim
        self.weight_tying = weight_tying
        self.bias = bias
        self.eps = eps

        # Generate spectral filters
        self.spectral_filters = get_spectral_filters(
            seq_len=self.seq_len, K=self.num_eigh, use_hankel_L=False, dtype=self.dtype
        )


# ------------------------------------------------------------------------
# Attention Mechanisms
# ------------------------------------------------------------------------


class VanillaAttention(lnn.Module):
    """Standard scaled dot-product attention with causal mask and RoPE."""

    dim: int
    num_heads: int
    seq_len: int

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads
        self.wq = lnn.Dense(self.dim)
        self.wk = lnn.Dense(self.dim)
        self.wv = lnn.Dense(self.dim)
        self.wo = lnn.Dense(self.dim)

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
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale

        # Apply causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask == 0, jnp.finfo(scores.dtype).min, scores)

        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention weights
        context = jnp.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape output
        context = context.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.dim)  # (batch_size, seq_len, dim)

        # Output projection
        out = self.wo(context)  # (batch_size, seq_len, dim)
        return out


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> jnp.ndarray:
    """Generate a Hankel matrix for spectral filtering.

    Args:
        seq_len: Sequence length.
        use_hankel_L: If True, use Hankel-L variant.

    Returns:
        Hankel matrix of shape [seq_len, seq_len].
    """
    entries = jnp.arange(1, seq_len + 1, dtype=jnp.float32)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        return sgn * (8.0 / denom)
    return 2.0 / (i_plus_j**3 - i_plus_j)


def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Compute spectral filters via eigen-decomposition of Hankel matrix.

    Args:
        seq_len: Sequence length.
        K: Number of top eigenvalues to retain.
        use_hankel_L: If True, use Hankel-L variant.
        dtype: Data type for the output.

    Returns:
        Spectral filters of shape [seq_len, K].
    """
    eig_vals, eig_vecs = jnp.linalg.eigh(get_hankel(seq_len))
    eig_vals, eig_vecs = eig_vals[-K:], eig_vecs[:, -K:]
    eig_vecs = eig_vecs * eig_vals ** 0.25
    return eig_vecs


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


def normalize(
    input: jnp.ndarray, p: float = 2.0, axis: Union[int, Tuple[int, ...]] = 1, eps: float = 1e-12
) -> jnp.ndarray:
    """Normalize input along specified dimension.

    Args:
        input: Input array.
        p: Norm order (default: 2.0 for L2 norm).
        dim: Dimension(s) to normalize over.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized array.
    """
    norm = jnp.linalg.norm(input, ord=p, axis=axis, keepdims=True)
    return input / jnp.maximum(norm, eps)


class AssociativeAttention(lnn.Module):
    dim: int
    num_heads: int
    seq_len: int
    spectral_basis: jnp.ndarray
    use_tensordot: bool = False
    eps: float = 1e-5

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.wq = lnn.Dense(self.dim)
        self.wk = lnn.Dense(self.dim)
        self.wv = lnn.Dense(self.dim)
        self.wo = lnn.Dense(self.dim)

        if self.use_tensordot:
            self.tensordot_proj = lnn.Dense(self.dim)  # Project filters: [L, K] @ [K, dim]

        # Gating projection: maps [D/H * D/H] to [1]
        self.gate_proj = lnn.Dense(1)
        self.kv_norm_scale = self.param(
            "kv_norm_scale", lambda rng: jnp.ones((1, self.num_heads, 1, self.head_dim, self.head_dim))
        )
        self.qk_norm_scale = self.param(
            "qk_norm_scale", lambda rng: jnp.full((1, self.num_heads, 1), 1 / jnp.sqrt(self.head_dim))
        )

    def combine_fn(self, x, y):
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
        m_new = jnp.maximum(m_x, m_y)

        # Scale factors
        exp_x = jnp.exp(m_x - m_new)
        exp_y = jnp.exp(m_y - m_new)

        # Update softmax components
        s_new = s_x * exp_x + s_y * exp_y
        n_new = n_x * exp_x[..., None] + n_y * exp_y[..., None]

        # Update gated Z and gate accumulation
        Z_new = Z_x + Z_y
        g_new = g_x + g_y

        return (m_new, s_new, n_new, Z_new, g_new)

    def scan_fn(self, qk_slice, v_slice, Z_slice, g_slice):
        """Process a single (batch, head) slice"""
        # Set up leaf elements
        # qk_slice: [L], v_slice: [L, h], Z_slice: [L, h, h], g_slice: [L, 1, 1]
        leaves_m = qk_slice  # [L]
        leaves_s = jnp.ones_like(qk_slice)  # [L]
        leaves_n = v_slice  # [L, h]
        leaves_Z = Z_slice  # [L, h, h]
        leaves_g = g_slice  # [L, 1, 1]

        leaves = (leaves_m, leaves_s, leaves_n, leaves_Z, leaves_g)
        return jax.lax.associative_scan(self.combine_fn, leaves, axis=0)

    def __call__(self, x, training=False):
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim

        # Compute QKV projections
        q = self.wq(x).reshape(B, L, H, h)
        k = self.wk(x).reshape(B, L, H, h)
        v = self.wv(x).reshape(B, L, H, h)

        # Transpose for better memory layout
        q = q.transpose(0, 2, 1, 3)  # [B, H, L, h]
        k = k.transpose(0, 2, 1, 3)  # [B, H, L, h]
        v = v.transpose(0, 2, 1, 3)  # [B, H, L, h]

        # [B, H, L]
        sim = jnp.einsum("bhld,bhld->bhl", q, k) * self.qk_norm_scale

        # KV norm (QK norm analog)
        k = normalize(k, p=2.0, axis=-1, eps=self.eps)
        v = normalize(v, p=2.0, axis=-1, eps=self.eps)

        # Apply spectral basis
        if self.use_tensordot:
            filters = self.tensordot_proj(self.spectral_basis)  # [L, K] @ [K, D] -> [L, D]
            k = tensordot_conv(filters, k)
            v = tensordot_conv(filters, v)
        else:
            k = stu_conv(self.spectral_basis, k)
            v = stu_conv(self.spectral_basis, v)

        # Compute pairwise interactions via outer product
        if self.use_tensordot:
            Z = jnp.einsum("bhld,bhle->bhlde", v, k)
        else:
            Z = jnp.einsum("bhlkd,bhlke->bhlde", v, k)
        Z = Z * self.kv_norm_scale

        # Compute gates
        gate_input = Z.reshape(*Z.shape[:3], -1)  # [B, H, L, h²]
        gates_logits = self.gate_proj(gate_input)  # [B, H, L, 1]
        gates = jax.nn.relu(gates_logits) ** 2 + self.eps  # [B, H, L, 1]
        gates = gates[..., None]  # [B, H, L, 1, 1]

        # Apply gating to Z
        gated_Z = gates * Z  # [B, H, L, h, h]

        # Vmap over the head dimension (for a single batch)
        batch_scan_fn = jax.vmap(self.scan_fn, in_axes=(0, 0, 0, 0))

        # Vmap over the batch dimension
        batched_scan_fn = jax.vmap(batch_scan_fn, in_axes=(0, 0, 0, 0))

        # Run the scan over all dimensions at once
        m_scan, s_scan, n_scan, Z_scan, g_scan = batched_scan_fn(sim, v, gated_Z, gates)

        # --- Compute the final attention outputs ---

        # 1. Compute the normalized online softmax weights
        # [B, H, L, 1, 1]
        softmax_weights = jnp.exp(sim - m_scan)[..., None, None] / (s_scan[..., None, None] + self.eps)

        # 2. Gated accumulation normalization
        gated_weights = Z_scan / (g_scan + self.eps)

        # Multiplicatively modulate the gated weights with the softmax weights
        attn_weights = gated_weights * (1.0 + jax.nn.silu(softmax_weights))

        # Query from the attention weights
        ctxt = jnp.einsum("bhld,bhlde->bhle", q, attn_weights)  # [B, H, L, h]

        # Reshape and project
        ctxt_norm = normalize(ctxt, axis=-1)
        output = ctxt_norm.transpose(0, 2, 1, 3).reshape(B, L, D)
        output = self.wo(output)

        return output


# ------------------------------------------------------------------------
# Model Components
# ------------------------------------------------------------------------


class FeedForward(lnn.Module):
    """MLP feed-forward layer with GELU activation."""

    dim: int
    expansion_factor: int = 4

    @lnn.compact
    def __call__(self, x, training=False):
        hidden_dim = self.dim * self.expansion_factor
        x = lnn.Dense(hidden_dim)(x)
        x = lnn.gelu(x)
        x = lnn.Dense(self.dim)(x)
        return x


class TransformerLayer(lnn.Module):
    """A single transformer layer with attention and feed-forward."""

    dim: int
    num_heads: int
    seq_len: int
    attention_class: Any

    @lnn.compact
    def __call__(self, x, training=False):
        # Attention block
        residual = x
        x = lnn.LayerNorm()(x)
        x = self.attention_class(dim=self.dim, num_heads=self.num_heads, seq_len=self.seq_len)(x, training=training)
        x = x + residual

        # Feed-forward block
        residual = x
        x = lnn.LayerNorm()(x)
        x = FeedForward(dim=self.dim)(x, training=training)
        x = x + residual
        return x


class SpectronBlock(lnn.Module):
    """A single Spectron block with associative attention and feed-forward network."""

    dim: int
    inter_dim: int
    num_heads: int
    seq_len: int
    spectral_basis: jnp.ndarray
    use_tensordot: bool = False
    eps: float = 1e-5

    @lnn.compact
    def __call__(self, x, training=False):
        # Attention block
        residual = x
        x = lnn.LayerNorm()(x)
        x = AssociativeAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            spectral_basis=self.spectral_basis,
            use_tensordot=self.use_tensordot,
            eps=self.eps,
        )(x, training=training)
        x = x + residual

        # Feed-forward block
        residual = x
        x = lnn.LayerNorm()(x)
        x = FeedForward(dim=self.dim)(x, training=training)
        x = x + residual
        return x


# ------------------------------------------------------------------------
# Main Models
# ------------------------------------------------------------------------


class Transformer(lnn.Module):
    """Vanilla Transformer with RoPE positional embeddings."""

    config: TransformerConfig

    def setup(self):
        self.tok_emb = lnn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)
        self.pos_enc = PositionalEncoding(d_model=self.config.dim, max_len=self.config.max_seq_len)
        self.layers = [
            TransformerLayer(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                seq_len=self.config.seq_len,
                attention_class=VanillaAttention,
            )
            for _ in range(self.config.num_layers)
        ]
        self.norm = lnn.LayerNorm()
        if self.config.weight_tying:
            self.lm_head = lambda x: jnp.dot(x, self.tok_emb.embedding.T)
        else:
            self.lm_head = lnn.Dense(self.config.vocab_size, use_bias=self.config.bias)

    def __call__(self, x, training=False):
        x = self.tok_emb(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits.astype(self.config.dtype)

    def count_params(self):
        """Count the total number of parameters in the model."""
        dummy_input = jnp.ones((1, self.config.seq_len), dtype=jnp.int32)
        variables = self.init(jax.random.PRNGKey(0), dummy_input)
        return sum(x.size for x in jax.tree_util.tree_leaves(variables))


class Spectron(lnn.Module):
    """Spectron transformer model for causal language modeling."""

    config: SpectronConfig

    def setup(self):
        self.tok_emb = lnn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)
        self.layers = [
            SpectronBlock(
                dim=self.config.dim,
                inter_dim=self.config.inter_dim,
                num_heads=self.config.num_heads,
                seq_len=self.config.seq_len,
                spectral_basis=self.config.spectral_filters,
                use_tensordot=self.config.use_tensordot,
                eps=self.config.eps,
            )
            for _ in range(self.config.num_layers)
        ]
        self.norm = lnn.LayerNorm()
        if self.config.weight_tying:
            self.lm_head = lambda x: jnp.dot(x, self.tok_emb.embedding.T)
        else:
            self.lm_head = lnn.Dense(self.config.vocab_size, use_bias=self.config.bias)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = self.tok_emb(x)
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits.astype(self.config.dtype)

    def count_params(self) -> int:
        """Count the total number of parameters in the model."""
        dummy_input = jnp.ones((1, self.config.seq_len), dtype=jnp.int32)
        variables = self.init(jax.random.PRNGKey(0), dummy_input)
        return sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))


# ------------------------------------------------------------------------
# Data Generation for MQAR Task
# ------------------------------------------------------------------------


def generate_jax_mqar_data(
    rng: jax.random.PRNGKey,
    num_examples: int = 10000,
    sequence_len: int = 512,
    vocab_size: int = 8192,
    num_pairs: int = 64,
    alpha: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Wrapper that uses the imported MQAR function and converts to JAX arrays.
    """
    # Extract seed from key
    seed = jax.random.randint(rng, (), 0, 2**31 - 1).item()

    # Create dataset using imported function
    dataset = generate_mqar(
        num_examples=num_examples,
        seq_len=sequence_len,
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


def generate_test_cases(rng, vocab_size, num_pairs, sequence_len, num_examples=10, alpha=0.1):
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


class PositionalEncoding(lnn.Module):
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
def train_step(state, inputs, targets):
    """Single training step with improved JAX semantics and RNG handling."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs, training=True)
        loss = cross_entropy_loss(logits, targets)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, targets)
    return new_state, metrics


@jax.jit
def eval_step(state, inputs, targets):
    """Single evaluation step with improved JAX semantics."""
    logits = state.apply_fn({"params": state.params}, inputs, training=False)
    metrics = compute_metrics(logits, targets)
    return metrics


def create_train_state(config, model, key):
    """Create initial training state with AdamW and warmup cosine decay scheduler."""
    # Create a dummy input for model initialization
    dummy_input = jnp.ones((config.bsz, config.seq_len), dtype=jnp.int32)

    # Initialize model
    variables = model.init(key, dummy_input)
    params = variables["params"]

    # Create learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_epochs * (config.train_size // config.bsz),
        end_value=config.learning_rate * 0.1,
    )

    # Create optimizer with AdamW configuration
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay, b1=0.9, b2=0.95, eps=1e-8),
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


def train_model(model_name, config, model, train_inputs, train_targets, val_inputs=None, val_targets=None):
    """Train model with a step-based approach similar to PyTorch implementation."""
    # Initialize random keys
    rng = jax.random.PRNGKey(config.seed)
    rng, init_key, train_key = jax.random.split(rng, 3)

    # Create train state (model + optimizer)
    state = create_train_state(config, model, init_key)

    # Create batch iterators
    train_iter = get_batch_iterator(train_inputs, train_targets, config.bsz, shuffle=True)
    val_iter = None
    if val_inputs is not None and val_targets is not None:
        val_iter = get_batch_iterator(val_inputs, val_targets, config.bsz, shuffle=False)

    # Training metrics
    train_metrics = []
    val_metrics = []
    step = 0
    total_steps = config.num_epochs * (train_inputs.shape[0] // config.bsz)

    # Initialize validation metrics
    val_loss = float("inf")
    val_acc = 0.0

    # Main training loop with tqdm progress bar
    with tqdm(total=total_steps, desc=f"Training {config.model_type.capitalize()} Attention") as pbar:
        for epoch in range(config.num_epochs):
            # Training
            train_batch_metrics = []
            for batch_inputs, batch_targets, key in train_iter(train_key):
                state, metrics = train_step(state, batch_inputs, batch_targets)
                train_batch_metrics.append(metrics)
                train_key = key

                # Update progress bar
                step += 1
                pbar.update(1)

                # Always show both training and validation metrics
                pbar.set_postfix(
                    {
                        "train_loss": f"{metrics['loss']:.4f}",
                        "train_acc": f"{metrics['accuracy'] * 100:.2f}%",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc * 100:.2f}%",
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
                            "train_acc": f"{train_batch_metrics[-1]['accuracy'] * 100:.2f}%",
                            "val_loss": f"{val_loss:.4f}",
                            "val_acc": f"{val_acc * 100:.2f}%",
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
                f"\nEpoch {epoch + 1}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}%"
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
    plt.suptitle(f"{config.model_type.capitalize()} Attention on MQAR", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{config.model_type}_mqar_training.png")
    plt.show()


def compare_models(
    tf_config,
    sp_config,  # Use specific config names
    tf_train_metrics,
    tf_val_metrics,
    sp_train_metrics,
    sp_val_metrics,
):
    """Plot comparison: Transformer vs Spectron."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    tf_steps = [m["step"] for m in tf_val_metrics]
    tf_loss = [m["loss"] for m in tf_val_metrics]
    tf_acc = [m["accuracy"] * 100 for m in tf_val_metrics]
    sp_steps = [m["step"] for m in sp_val_metrics]
    sp_loss = [m["loss"] for m in sp_val_metrics]
    sp_acc = [m["accuracy"] * 100 for m in sp_val_metrics]
    ax1.plot(tf_steps, tf_loss, "b-", label="Transformer")
    ax1.plot(sp_steps, sp_loss, "r-", label="Spectron")
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(tf_steps, tf_acc, "b-", label="Transformer")
    ax2.plot(sp_steps, sp_acc, "r-", label="Spectron")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    plt.suptitle("Transformer vs Spectron on MQAR Task", fontsize=14)
    plt.tight_layout()
    plt.savefig("transformer_vs_spectron_mqar.png")
    print("Comparison plot saved to transformer_vs_spectron_mqar.png")
    plt.show()


def run_experiment(config_override=None):
    """Run experiment comparing Standard Transformer and Spectron."""
    # Base config for the Transformer model
    transformer_config = TransformerConfig(
        dim=128,
        num_heads=2,
        num_layers=2,
        max_seq_len=256,
        vocab_size=8192 + 3,  # Align with MQAR data generation needs
        learning_rate=3e-4,
        weight_decay=1e-1,
        warmup_steps=2000,
        dtype=jnp.float32,
        seq_len=256,
        num_pairs=32,
        alpha=0.1,
        train_size=100000,
        val_size=3000,
        bsz=64,
        num_epochs=64,
        seed=1746,
    )

    # Apply overrides if any to the base config
    if config_override:
        for key, value in config_override.items():
            setattr(transformer_config, key, value)

    # Create Spectron config with aligned parameters
    spectron_config = SpectronConfig(
        dim=transformer_config.dim,
        num_heads=1,  # Spectron typically uses 1 head
        num_layers=transformer_config.num_layers,
        seq_len=transformer_config.seq_len,
        vocab_size=transformer_config.vocab_size,
        bsz=transformer_config.bsz,
        dtype=transformer_config.dtype,
        num_eigh=32,
        use_tensordot=True,
        inter_dim=4 * transformer_config.dim,
        weight_tying=False,
        bias=False,
        eps=1e-5,
        # Training parameters
        learning_rate=transformer_config.learning_rate,
        weight_decay=transformer_config.weight_decay,
        warmup_steps=transformer_config.warmup_steps,
        # Task parameters
        num_pairs=transformer_config.num_pairs,
        alpha=transformer_config.alpha,
        train_size=transformer_config.train_size,
        val_size=transformer_config.val_size,
        num_epochs=transformer_config.num_epochs,
        seed=transformer_config.seed,
    )

    print(f"Transformer config: {transformer_config.__dict__}")
    print(f"Spectron config: {spectron_config.__dict__}")

    # --- Data Generation ---
    rng = jax.random.PRNGKey(transformer_config.seed)
    print("Generating training data...")
    rng, data_rng = jax.random.split(rng)
    train_inputs, train_targets, rng = generate_jax_mqar_data(
        data_rng,
        transformer_config.train_size,
        transformer_config.seq_len,
        transformer_config.vocab_size,
        transformer_config.num_pairs,
        transformer_config.alpha,
    )
    print("Generating validation data...")
    rng, data_rng = jax.random.split(rng)
    val_inputs, val_targets, rng = generate_jax_mqar_data(
        data_rng,
        transformer_config.val_size,
        transformer_config.seq_len,
        transformer_config.vocab_size,
        transformer_config.num_pairs,
        transformer_config.alpha,
    )

    # --- Train Spectron ---
    print("\nTraining Spectron...")
    spectron_model = Spectron(spectron_config)
    print(f"- Parameters: ~{spectron_model.count_params() / 1e6:.2f}M")
    sp_state, sp_train_metrics, sp_val_metrics = train_model(
        "spectron", spectron_config, spectron_model, train_inputs, train_targets, val_inputs, val_targets
    )

    # --- Train Transformer ---
    print("\nTraining Transformer...")
    transformer_model = Transformer(transformer_config)
    print(f"- Parameters: ~{transformer_model.count_params() / 1e6:.2f}M")
    tf_state, tf_train_metrics, tf_val_metrics = train_model(
        "transformer", transformer_config, transformer_model, train_inputs, train_targets, val_inputs, val_targets
    )

    # --- Compare Models ---
    print("\nComparing Models...")
    compare_models(
        transformer_config,
        spectron_config,
        tf_train_metrics,
        tf_val_metrics,
        sp_train_metrics,
        sp_val_metrics,
    )

    # Return results
    return {
        "transformer": {
            "config": transformer_config,
            "state": tf_state,
            "train_metrics": tf_train_metrics,
            "val_metrics": tf_val_metrics,
        },
        "spectron": {
            "config": spectron_config,
            "state": sp_state,
            "train_metrics": sp_train_metrics,
            "val_metrics": sp_val_metrics,
        },
    }


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"JAX version: {jax.__version__}")
    print(f"Flax version: {flax.__version__}")
    print(f"Optax version: {optax.__version__}")
    results = run_experiment()

    # # === Train Step Micro-benchmark (jax_scan_attn_mqar_separate.py) ===
    # import time
    # import numpy as np  # Need numpy for mean
    # from tqdm.auto import tqdm  # Need tqdm

    # print("\nRunning train_step Micro-benchmark (Spectron)...")

    # # Config (matching run_experiment)
    # key = jax.random.PRNGKey(123)  # Use a different seed for clarity
    # bench_config = SpectronConfig(
    #     dim=128,
    #     num_heads=1,
    #     num_layers=2,
    #     seq_len=256,
    #     vocab_size=8195,
    #     bsz=64,
    #     dtype=jnp.float32,
    #     num_eigh=32,
    #     use_tensordot=True,
    #     inter_dim=512,
    #     eps=1e-5,
    #     # Need other params for create_train_state
    #     learning_rate=3e-4,
    #     weight_decay=1e-1,
    #     warmup_steps=2000,
    #     num_epochs=1,  # Not relevant for benchmark
    #     train_size=128064,  # Increase to ensure decay_steps >= warmup_steps
    # )

    # # Dummy data
    # dummy_inputs = jnp.ones((bench_config.bsz, bench_config.seq_len), dtype=jnp.int32)
    # dummy_targets = jnp.ones((bench_config.bsz, bench_config.seq_len), dtype=jnp.int32)

    # # Instantiate model
    # model = Spectron(bench_config)

    # # Create Train State
    # key, init_key = jax.random.split(key)
    # state = create_train_state(bench_config, model, init_key)

    # # train_step function is already JITted

    # # Warm-up
    # print("Running JIT warm-up for train_step...")
    # state, metrics = train_step(state, dummy_inputs, dummy_targets)
    # jax.tree_util.tree_map(lambda x: x.block_until_ready(), (state, metrics))
    # print("Warm-up complete.")

    # # Benchmark
    # num_runs = 50
    # timings = []
    # print(f"Running train_step benchmark ({num_runs} runs)...")
    # # Use a new state for each run to avoid potential state changes affecting timing?
    # # Or reuse state? Reusing state is closer to actual training.
    # current_state = state
    # for _ in tqdm(range(num_runs)):
    #     start_time = time.time()
    #     current_state, metrics = train_step(current_state, dummy_inputs, dummy_targets)
    #     # Block on outputs of train_step (new state and metrics)
    #     jax.tree_util.tree_map(lambda x: x.block_until_ready(), (current_state, metrics))
    #     end_time = time.time()
    #     timings.append(end_time - start_time)

    # avg_time = np.mean(timings)
    # print(f"--- train_step Benchmark Results (Spectron) ---")
    # print(f"Average train_step time over {num_runs} runs: {avg_time:.6f} seconds")
    # print(f"-------------------------------------------------")
