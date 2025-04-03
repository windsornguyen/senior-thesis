import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Any, Tuple
import jax.scipy as jsp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import inspect
from functools import partial, wraps

# Set platform-specific defaults
jax.config.update("jax_default_matmul_precision", "bfloat16")

# ------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------


class TransformerConfig:
    """Configuration class for the Transformer model, similar to PyTorch's approach."""

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.0,
        max_seq_len: int = 2048,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        attention_type: str = "vanilla",  # "vanilla" or "scan"
        # Dataset configuration
        seq_len: int = 256,
        num_pairs: int = 3,  # Number of key-value pairs to memorize
        num_queries: int = 2,  # Number of queries to test recall
        train_size: int = 100000,
        val_size: int = 3000,
        batch_size: int = 64,
        num_epochs: int = 32,
        seed: int = 1746,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.attention_type = attention_type

        # Dataset configuration
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed


# ------------------------------------------------------------------------
# Better Data Generation from PyTorch implementation
# ------------------------------------------------------------------------


def generate_assoc_recall_pytorch(
    num_examples: int = 100000,
    sequence_len: int = 128,
    vocab_size: int = 8192,
    num_pairs: int = 4,
    random_non_queries: bool = True,
    num_queries: int = 3,
    seed: int = 1746,
):
    """
    Generates synthetic data for the associative recall task using PyTorch.
    This is directly taken from the original implementation.
    """
    # Basic sanity checks.
    assert sequence_len % 2 == 0, "sequence_len must be even"
    assert vocab_size > sequence_len, "vocab_size must be greater than sequence_len"
    assert num_pairs * 2 + num_queries <= sequence_len, "sequence_len must be >= 2*num_pairs + num_queries"

    torch.manual_seed(seed)

    # The first part of the sequence is reserved for key–value pairs.
    context_size = num_pairs * 2

    # Create unique keys for each example.
    key_vocab_size = vocab_size // 2  # keys come from the first half (skipping 0 for clarity)
    possible_keys = torch.arange(1, key_vocab_size, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_keys = torch.rand(num_examples, key_vocab_size - 1)
    _, key_perm = rand_keys.sort(dim=1)
    keys = possible_keys.gather(1, key_perm[:, :num_pairs])

    # Create corresponding values from the second half of the vocabulary.
    possible_values = torch.arange(key_vocab_size, vocab_size, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_values = torch.rand(num_examples, vocab_size - key_vocab_size)
    _, value_perm = rand_values.sort(dim=1)
    values = possible_values.gather(1, value_perm[:, :num_pairs])

    # Build the key–value sequence (keys in even positions, values in odd positions).
    kvs = torch.empty(num_examples, context_size, dtype=torch.long)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Initialize inputs and targets with length sequence_len + 1 to account for shifting.
    inputs = torch.zeros(num_examples, sequence_len + 1, dtype=torch.long)
    targets = torch.full((num_examples, sequence_len + 1), -100, dtype=torch.long)

    # Insert the key–value pairs at the beginning.
    inputs[:, :context_size] = kvs

    # Prepare advanced indexing: rows for each example.
    rows = torch.arange(num_examples, dtype=torch.long).unsqueeze(1).expand(-1, num_queries)

    # Sample key–value pair indices (without replacement) for the queries.
    possible_idx = torch.arange(num_pairs, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_idx = torch.rand(num_examples, num_pairs)
    _, idx_perm = rand_idx.sort(dim=1)
    chosen_idxs = possible_idx.gather(1, idx_perm[:, :num_queries])

    # Select query keys and corresponding target values.
    queries = keys.gather(1, chosen_idxs)
    query_labels = values.gather(1, chosen_idxs)

    # Randomly choose positions up to sequence_len - 1
    pos_choices = torch.arange(context_size, sequence_len, dtype=torch.long).unsqueeze(0).expand(num_examples, -1)
    rand_pos = torch.rand(num_examples, sequence_len - context_size)
    _, pos_perm = rand_pos.sort(dim=1)
    query_pos = pos_choices.gather(1, pos_perm[:, :num_queries])

    # Insert queries into inputs and their labels into targets at query_pos + 1
    inputs[rows, query_pos] = queries
    targets[rows, query_pos + 1] = query_labels  # Shift target to next position

    # Shift inputs/targets by one to get final shapes
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]

    # Optionally replace filler zeros (which are not queries) with random tokens.
    if random_non_queries:
        mask = inputs == 0
        if mask.any():
            inputs[mask] = torch.randint(0, vocab_size, (mask.sum().item(),), dtype=torch.long)

    return inputs.numpy(), targets.numpy()


def generate_jax_assoc_recall(
    key: jax.random.PRNGKey,
    num_examples: int = 10000,
    sequence_len: int = 128,
    vocab_size: int = 8192,
    num_pairs: int = 4,
    random_non_queries: bool = True,
    num_queries: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Wrapper that uses the PyTorch implementation and converts to JAX arrays.
    """
    # Extract seed from key
    seed = jax.random.randint(key, (), 0, 2**31 - 1).item()

    # Generate data using PyTorch implementation
    inputs_np, targets_np = generate_assoc_recall_pytorch(
        num_examples=num_examples,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        num_pairs=num_pairs,
        random_non_queries=random_non_queries,
        num_queries=num_queries,
        seed=seed,
    )

    # Convert to JAX arrays
    inputs_jax = jnp.array(inputs_np, dtype=jnp.int32)
    targets_jax = jnp.array(targets_np, dtype=jnp.int32)

    # Update key
    new_key = jax.random.split(key)[0]

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
    """Standard scaled dot-product attention with causal mask, similar to PyTorch's Attention class."""

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


def enforce_associativity(op=None, *,
                        check_flag="check_associativity",
                        sample_key="z_sample",
                        tol=1e-4,
                        eq_fn=None):
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
#     """
#     Optimized scan attention that integrates content-awareness, selective sharpening,
#     and learnable parameters for improved associative recall performance.
#     """

#     dim: int
#     num_heads: int
#     beta_init: float = 1.0  # Initial softplus sharpness
#     temp_init: float = 0.5  # Initial temperature scaling
#     sharp_init: float = 4.0  # Initial output sharpening
#     content_scaling: bool = True  # Whether to use content-aware scaling
#     position_scaling: bool = True  # Whether to use position-dependent scaling
#     eps: float = 1e-6

#     def setup(self):
#         assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
#         self.head_dim = self.dim // self.num_heads

#         # Standard attention projections
#         self.wq = nn.Dense(self.dim)
#         self.wk = nn.Dense(self.dim)
#         self.wv = nn.Dense(self.dim)
#         self.wo = nn.Dense(self.dim)

#         # Learnable parameters (per head)
#         self.log_beta = self.param(
#             "log_beta", lambda key, shape: jnp.ones(shape) * jnp.log(self.beta_init), (self.num_heads,)
#         )

#         self.log_temp = self.param(
#             "log_temp", lambda key, shape: jnp.ones(shape) * jnp.log(self.temp_init), (self.num_heads,)
#         )

#         self.log_sharp = self.param(
#             "log_sharp", lambda key, shape: jnp.ones(shape) * jnp.log(self.sharp_init), (self.num_heads,)
#         )

#         if self.content_scaling:
#             # Learnable importance scales for content-awareness
#             self.log_importance_scales = self.param(
#                 "log_importance_scales", lambda key, shape: jnp.zeros(shape), (self.num_heads, 3)
#             )  # 3 factors: variance, magnitude, sparsity

#     def softplus(self, x, beta):
#         """Softplus activation with configurable beta parameter."""
#         return (1.0 / beta) * jnp.log(1.0 + jnp.exp(beta * x))

#     def compute_importance(self, x, scales):
#         """Compute content-aware importance scores."""
#         # Unpack importance scales
#         var_scale, mag_scale, sparsity_scale = scales

#         # 1. Distinctiveness: high variance = more distinctive pattern
#         variance = jnp.var(x, axis=-1, keepdims=True)
#         variance_score = jnp.tanh(var_scale * variance)

#         # 2. Signal strength: higher magnitude = stronger signal
#         magnitude = jnp.mean(jnp.abs(x), axis=-1, keepdims=True)
#         magnitude_score = jnp.tanh(mag_scale * magnitude)

#         # 3. Focus: measure how concentrated the pattern is
#         l1 = jnp.sum(jnp.abs(x), axis=-1, keepdims=True)
#         l2 = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + self.eps)
#         sparsity = l1 / (l2 + self.eps)
#         sparsity_score = jnp.tanh(sparsity_scale * sparsity)

#         # Combine scores (1.0 + ... ensures the base case has importance 1.0)
#         importance = 1.0 + variance_score * magnitude_score * sparsity_score
#         return importance

#     @enforce_associativity
#     def selective_combine_fn(self, x, y, importance_scales=None):
#         """Content-aware selective combine function."""
#         m_x, log_N_x, log_D_x = x
#         m_y, log_N_y, log_D_y = y

#         # Apply content-aware scaling if enabled
#         if importance_scales is not None:
#             # Compute importance scores for both segments
#             # Importance scaling in an associative way, without inferring head from shape
#             # We directly compute importance based on the content without trying to extract head info
#             importance_x = self.compute_content_importance(m_x)
#             importance_y = self.compute_content_importance(m_y)

#             # Apply importance in log domain
#             log_importance_x = jnp.log(importance_x)
#             log_importance_y = jnp.log(importance_y)

#             # Scale log values
#             log_N_x = log_N_x + log_importance_x[..., None]
#             log_D_x = log_D_x + log_importance_x
#             log_N_y = log_N_y + log_importance_y[..., None]
#             log_D_y = log_D_y + log_importance_y

#         # Compute new reference point
#         m_new = jnp.maximum(m_x, m_y)

#         # Shift values to new log-domain reference
#         log_N_x_shifted = log_N_x + (m_x - m_new)[..., None]
#         log_N_y_shifted = log_N_y + (m_y - m_new)[..., None]
#         log_D_x_shifted = log_D_x + (m_x - m_new)
#         log_D_y_shifted = log_D_y + (m_y - m_new)

#         # Use logsumexp for stable addition
#         log_N_new = jsp.special.logsumexp(jnp.stack([log_N_x_shifted, log_N_y_shifted]), axis=0)
#         log_D_new = jsp.special.logsumexp(jnp.stack([log_D_x_shifted, log_D_y_shifted]), axis=0)

#         return m_new, log_N_new, log_D_new

#     def compute_content_importance(self, x):
#         """Compute importance based purely on content features, without requiring head-specific scales.
#         This function is used inside the scan and must be associative."""
#         # Compute simple content-based features that don't require learned parameters

#         # 1. Distinctiveness: high variance = more distinctive pattern
#         variance = jnp.var(x, axis=-1, keepdims=True)
#         variance_score = jnp.tanh(variance)

#         # 2. Signal strength: higher magnitude = stronger signal
#         magnitude = jnp.mean(jnp.abs(x), axis=-1, keepdims=True)
#         magnitude_score = jnp.tanh(magnitude)

#         # 3. Focus: measure how concentrated the pattern is
#         l1 = jnp.sum(jnp.abs(x), axis=-1, keepdims=True)
#         l2 = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + self.eps)
#         sparsity = l1 / (l2 + self.eps)
#         sparsity_score = jnp.tanh(sparsity)

#         # Combine scores with fixed weights of 1.0
#         # This ensures the function is fully associative with no shape-dependent behavior
#         importance = 1.0 + variance_score * magnitude_score * sparsity_score
#         return importance

#     def __call__(self, x, training=False):
#         batch_size, seq_len, _ = x.shape

#         # Linear projections
#         q = self.wq(x)
#         k = self.wk(x)
#         v = self.wv(x)

#         # Reshape for multi-head attention
#         q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

#         # Transpose for batched operations
#         q = q.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
#         k = k.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
#         v = v.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

#         # Get learnable parameters
#         beta = jnp.exp(self.log_beta)  # shape: (num_heads,)
#         temp = jnp.exp(self.log_temp)  # shape: (num_heads,)
#         sharp = jnp.exp(self.log_sharp)  # shape: (num_heads,)

#         # Reshape for broadcasting
#         beta = beta[:, None, None]  # shape: (num_heads, 1, 1)
#         temp = temp[:, None, None]  # shape: (num_heads, 1, 1)
#         sharp = sharp[:, None, None]  # shape: (num_heads, 1, 1)

#         # Note: we'll apply learned importance scaling after the scan to maintain associativity
#         # Flag whether to use content scaling
#         use_content_scaling = self.content_scaling

#         # Apply temperature scaling
#         q = q / temp
#         k = k / temp

#         # Apply feature map with learnable beta
#         Q_feat = self.softplus(q, beta)
#         K_feat = self.softplus(k, beta)

#         # Initialize scan components
#         m_initial = jnp.log(K_feat + self.eps)
#         num_initial = jnp.einsum("bhtp,bhtn->bhtpn", v, K_feat)
#         denom_initial = K_feat

#         # Create combine function with fixed importance scales
#         if use_content_scaling:
#             combine_fn = partial(self.selective_combine_fn, importance_scales=True)
#         else:
#             combine_fn = partial(self.selective_combine_fn, importance_scales=None)

#         # Use associative scan for the main computation
#         (m_cum, num_cum, denom_cum) = jax.lax.associative_scan(
#             combine_fn, (m_initial, num_initial, denom_initial), axis=2
#         )

#         # Compute attention output
#         Y_num = jnp.einsum("bhtp,bhtpq->bhtq", Q_feat, num_cum)
#         Y_den = jnp.einsum("bhtp,bhtp->bht", Q_feat, denom_cum)

#         # Apply normalization with epsilon for stability
#         Y = Y_num / (Y_den[..., None] + self.eps)

#         # Apply learned importance scaling here if enabled (outside the scan)
#         if self.content_scaling and hasattr(self, "log_importance_scales"):
#             # Get per-head importance scales
#             importance_scales = jnp.exp(self.log_importance_scales)  # shape: (num_heads, 3)

#             # For each head, compute a scaling factor based on content features
#             content_features = jnp.concatenate(
#                 [
#                     jnp.var(Y, axis=-1, keepdims=True),  # variance
#                     jnp.mean(jnp.abs(Y), axis=-1, keepdims=True),  # magnitude
#                     jnp.sum(jnp.abs(Y), axis=-1, keepdims=True)
#                     / (jnp.sqrt(jnp.sum(Y**2, axis=-1, keepdims=True)) + self.eps),  # sparsity
#                 ],
#                 axis=-1,
#             )  # shape: (batch, heads, seq_len, 3)

#             # Apply learned scales to features and combine
#             # We reshape importance_scales to (1, num_heads, 1, 3) for broadcasting
#             scaled_features = content_features * importance_scales.reshape(1, -1, 1, 3)
#             importance = 1.0 + jnp.prod(jnp.tanh(scaled_features), axis=-1, keepdims=True)

#             # Apply importance scaling
#             Y = Y * importance

#         # Apply sharpening
#         sign = jnp.sign(Y)
#         Y = sign * jnp.abs(Y) ** sharp

#         # Apply position-dependent scaling if enabled
#         if self.position_scaling:
#             # Scale outputs based on position (later positions get higher weight)
#             position_scaling = jnp.arange(1, seq_len + 1) / seq_len
#             position_scaling = position_scaling.reshape(1, 1, seq_len, 1)
#             Y = Y * position_scaling

#         # Reshape output
#         Y = Y.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
#         Y = Y.reshape(batch_size, seq_len, self.dim)  # (batch_size, seq_len, dim)

#         # Output projection
#         out = self.wo(Y)  # (batch_size, seq_len, dim)
#         return out

class ScanAttention(nn.Module):
    """Scan attention with learnable parameters for better associative recall performance."""

    dim: int
    num_heads: int
    beta_init: float = 1.0        # Initial sharpness value
    temp_init: float = 0.5        # Initial temperature value
    sharp_init: float = 4.0       # Initial sharpening value
    position_scaling: bool = True # Whether to apply position-dependent scaling
    eps: float = 1e-5

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads

        # Standard attention projections
        self.wq = nn.Dense(self.dim)
        self.wk = nn.Dense(self.dim)
        self.wv = nn.Dense(self.dim)
        self.wo = nn.Dense(self.dim)
        
        # Learnable parameters (per head)
        # Using log parameterization to ensure positivity
        self.log_beta = self.param('log_beta', 
                                  lambda key, shape: jnp.ones(shape) * jnp.log(self.beta_init),
                                  (self.num_heads,))
        
        self.log_temp = self.param('log_temp',
                                  lambda key, shape: jnp.ones(shape) * jnp.log(self.temp_init),
                                  (self.num_heads,))
        
        self.log_sharp = self.param('log_sharp',
                                   lambda key, shape: jnp.ones(shape) * jnp.log(self.sharp_init),
                                   (self.num_heads,))

    def softplus(self, x, beta):
        """Softplus activation with configurable beta parameter."""
        return (1.0 / beta) * jnp.log(1.0 + jnp.exp(beta * x))
    
    @enforce_associativity
    def combine_fn(self, x, y):
        """Bounded, numerically stable combine function for log-domain scan."""
        m_x, log_N_x, log_D_x = x
        m_y, log_N_y, log_D_y = y

        # Compute new max for stability
        m_new = jnp.maximum(m_x, m_y)

        # Shift values to new log-domain reference
        log_N_x_shifted = log_N_x + (m_x - m_new)[..., None]
        log_N_y_shifted = log_N_y + (m_y - m_new)[..., None]

        log_D_x_shifted = log_D_x + (m_x - m_new)
        log_D_y_shifted = log_D_y + (m_y - m_new)

        # Use logsumexp for stable addition
        log_N_new = jsp.special.logsumexp(jnp.stack([log_N_x_shifted, log_N_y_shifted]), axis=0)
        log_D_new = jsp.special.logsumexp(jnp.stack([log_D_x_shifted, log_D_y_shifted]), axis=0)

        return m_new, log_N_new, log_D_new

    def __call__(self, x, training=False):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for batched operations
        q = q.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        # Get learnable parameters
        beta = jnp.exp(self.log_beta)      # shape: (num_heads,)
        temp = jnp.exp(self.log_temp)      # shape: (num_heads,)
        sharp = jnp.exp(self.log_sharp)    # shape: (num_heads,)
        
        # Reshape for broadcasting
        beta = beta[:, None, None]         # shape: (num_heads, 1, 1)
        temp = temp[:, None, None]         # shape: (num_heads, 1, 1)
        sharp = sharp[:, None, None]       # shape: (num_heads, 1, 1)
        
        # Apply temperature scaling
        q = q / temp
        k = k / temp
        
        # Apply feature map with learnable beta
        Q_feat = self.softplus(q, beta)
        K_feat = self.softplus(k, beta)

        # Initialize scan components
        m_initial = jnp.log(K_feat + self.eps)
        num_initial = jnp.einsum("bhtp,bhtn->bhtpn", v, K_feat)
        denom_initial = K_feat
        
        # Use associative scan for the main computation
        (m_cum, num_cum, denom_cum) = jax.lax.associative_scan(
            self.combine_fn, (m_initial, num_initial, denom_initial), axis=2
        )

        # Compute attention output
        Y_num = jnp.einsum("bhtp,bhtpq->bhtq", Q_feat, num_cum)
        Y_den = jnp.einsum("bhtp,bhtp->bht", Q_feat, denom_cum)
        
        # Apply normalization with epsilon for stability
        Y = Y_num / (Y_den[..., None] + self.eps)
        
        # Apply sharpening
        sign = jnp.sign(Y)
        Y = sign * jnp.abs(Y) ** sharp
        
        # Apply position-dependent scaling if enabled
        if self.position_scaling:
            # Scale outputs based on position (later positions get higher weight)
            position_scaling = jnp.arange(1, seq_len + 1) / seq_len
            position_scaling = position_scaling.reshape(1, 1, seq_len, 1)
            Y = Y * position_scaling

        # Reshape output
        Y = Y.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        Y = Y.reshape(batch_size, seq_len, self.dim)  # (batch_size, seq_len, dim)

        # Output projection
        out = self.wo(Y)  # (batch_size, seq_len, dim)
        return out


# ------------------------------------------------------------------------
# MLP Feed-Forward Layer
# ------------------------------------------------------------------------


class FeedForward(nn.Module):
    """MLP feed-forward layer with SiLU activation, similar to PyTorch's MLP."""

    dim: int
    expansion_factor: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        hidden_dim = self.dim * self.expansion_factor
        x = nn.Dense(hidden_dim, name="fc1")(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.dim, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


# ------------------------------------------------------------------------
# Transformer Layers and Full Model
# ------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """A single transformer layer with attention and feed-forward, including dropout."""

    dim: int
    num_heads: int
    attention_class: Any  # VanillaAttention or ScanAttention
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=False):
        # Attention block
        residual = x
        x = nn.LayerNorm()(x)
        x = self.attention_class(dim=self.dim, num_heads=self.num_heads)(x, training=training)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = x + residual

        # Feed-forward block
        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForward(dim=self.dim, dropout_rate=self.dropout_rate)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    """Complete transformer model based on the PyTorch implementation's architecture."""

    config: TransformerConfig

    def setup(self):
        # Select attention class based on config
        if self.config.attention_type == "vanilla":
            attention_class = VanillaAttention
        elif self.config.attention_type == "scan":
            attention_class = ScanAttention
        else:
            raise ValueError(f"Unknown attention type: {self.config.attention_type}")

        self.token_embedding = nn.Embed(self.config.vocab_size, self.config.dim)
        self.pos_encoding = PositionalEncoding(self.config.dim, self.config.max_seq_len)
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

        # Create transformer layers
        self.layers = [
            TransformerLayer(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                attention_class=attention_class,
                dropout_rate=self.config.dropout_rate,
            )
            for _ in range(self.config.num_layers)
        ]

        self.norm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size)

    def __call__(self, x, training=False):
        # Get token embeddings and add positional encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # Apply dropout to embeddings
        x = self.dropout(x, deterministic=not training)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, training=training)

        # Final norm and projection to vocab
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def count_params(self):
        """Count the total number of parameters in the model."""
        params = jnp.array([1])  # Dummy array for init
        init_key = jax.random.PRNGKey(0)
        variables = self.init(init_key, jnp.ones((1, 1), dtype=jnp.int32))
        return sum(x.size for x in jax.tree_util.tree_leaves(variables))


# ------------------------------------------------------------------------
# Training and Evaluation Functions - Based on PyTorch implementation style
# ------------------------------------------------------------------------


@jax.jit
def cross_entropy_loss(logits, targets):
    """Compute cross entropy loss with masked tokens."""
    vocab_size = logits.shape[-1]

    # Create one-hot encoded targets
    onehot_targets = jax.nn.one_hot(targets, vocab_size)

    # Compute cross entropy loss, ignoring padded tokens (-100)
    mask = (targets != -100).astype(logits.dtype)

    # Compute standard cross entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(onehot_targets * log_probs, axis=-1)

    # Apply mask and compute mean
    loss = jnp.sum(loss * mask) / (jnp.sum(mask) + 1e-8)

    return loss


@jax.jit
def compute_metrics(logits, targets):
    """Compute loss and accuracy metrics."""
    loss = cross_entropy_loss(logits, targets)

    # Compute accuracy on non-padding tokens
    preds = jnp.argmax(logits, axis=-1)
    mask = targets != -100
    correct = (preds == targets) * mask
    accuracy = jnp.sum(correct) / (jnp.sum(mask) + 1e-8)

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
    """Create initial training state with AdamW optimizer."""
    # Create a dummy input for model initialization
    dummy_input = jnp.ones((config.batch_size, config.seq_len), dtype=jnp.int32)

    # Initialize model
    variables = model.init(key, dummy_input)
    params = variables["params"]

    # Create optimizer with AdamW configuration
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay, b1=0.9, b2=0.999, eps=1e-8),
    )

    # Create training state
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_model(config, model, train_inputs, train_targets, val_inputs, val_targets):
    """Train model with a step-based approach similar to PyTorch implementation."""

    # Initialize random keys
    rng = jax.random.PRNGKey(config.seed)
    rng, init_key, train_key, dropout_key = jax.random.split(rng, 4)

    # Create train state (model + optimizer)
    state = create_train_state(config, model, init_key)

    # Create batch iterators
    train_iter = get_batch_iterator(train_inputs, train_targets, config.batch_size, shuffle=True)
    val_iter = get_batch_iterator(val_inputs, val_targets, config.batch_size, shuffle=False)

    # Training metrics
    train_metrics = []
    val_metrics = []  # Ensure this is initialized as a list
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
                if step % 50 == 0:
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

    return state, train_metrics, val_metrics


# ------------------------------------------------------------------------
# Main Script to run the experiment
# ------------------------------------------------------------------------


def run_experiment():
    # Create configurations for both attention types
    vanilla_config = TransformerConfig(attention_type="vanilla")
    scan_config = TransformerConfig(attention_type="scan")

    # Initialize RNG key
    key = jax.random.PRNGKey(vanilla_config.seed)

    # Generate datasets using the PyTorch-based implementation
    print("\nGenerating datasets...")
    train_inputs, train_targets, key = generate_jax_assoc_recall(
        key=key,
        num_examples=vanilla_config.train_size,
        sequence_len=vanilla_config.seq_len,
        vocab_size=vanilla_config.vocab_size,
        num_pairs=vanilla_config.num_pairs,
        random_non_queries=True,
        num_queries=vanilla_config.num_queries,
    )

    val_inputs, val_targets, key = generate_jax_assoc_recall(
        key=key,
        num_examples=vanilla_config.val_size,
        sequence_len=vanilla_config.seq_len,
        vocab_size=vanilla_config.vocab_size,
        num_pairs=vanilla_config.num_pairs,
        random_non_queries=True,
        num_queries=vanilla_config.num_queries,
    )

    # Create models
    vanilla_model = Transformer(vanilla_config)
    scan_model = Transformer(scan_config)

    # Print model info
    print("\nModel architecture:")
    print(f"- Vocabulary size: {vanilla_config.vocab_size}")
    print(f"- Model dimension: {vanilla_config.dim}")
    print(f"- Number of heads: {vanilla_config.num_heads}")
    print(f"- Number of layers: {vanilla_config.num_layers}")
    print(f"- Parameters: ~{vanilla_model.count_params() / 1e6:.2f}M")

    # Train scan attention model
    print("\nTraining Scan Attention model...")
    scan_state, scan_train, scan_val = train_model(
        scan_config, scan_model, train_inputs, train_targets, val_inputs, val_targets
    )

    # Train vanilla attention model
    print("\nTraining Vanilla Attention model...")
    vanilla_state, vanilla_train, vanilla_val = train_model(
        vanilla_config, vanilla_model, train_inputs, train_targets, val_inputs, val_targets
    )

    # Plot results
    plot_results(vanilla_val, scan_val)

    return vanilla_state, scan_state, vanilla_val, scan_val


def plot_results(vanilla_metrics, scan_metrics):
    """Plot training curves comparing the two attention mechanisms."""
    # Extract metrics
    vanilla_steps = [m["step"] for m in vanilla_metrics]
    vanilla_loss = [m["loss"] for m in vanilla_metrics]
    vanilla_acc = [m["accuracy"] * 100 for m in vanilla_metrics]

    scan_steps = [m["step"] for m in scan_metrics]
    scan_loss = [m["loss"] for m in scan_metrics]
    scan_acc = [m["accuracy"] * 100 for m in scan_metrics]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot loss
    ax1.plot(vanilla_steps, vanilla_loss, "b-", label="Vanilla")
    ax1.plot(scan_steps, scan_loss, "r-", label="Scan")
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(vanilla_steps, vanilla_acc, "b-", label="Vanilla")
    ax2.plot(scan_steps, scan_acc, "r-", label="Scan")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Set title
    plt.suptitle("Vanilla vs Scan-Based Attention on Associative Recall", fontsize=14)

    plt.tight_layout()
    plt.savefig("associative_recall_comparison.png")
    plt.show()


# Create batched dataset iterator with JAX RNG handling
def get_batch_iterator(inputs, targets, batch_size, shuffle=True):
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


if __name__ == "__main__":
    # Enable x64 precision for numerical stability if needed
    # jax.config.update("jax_enable_x64", True)

    # Print available devices
    print("JAX devices available:", jax.devices())
    print("Default JAX device:", jax.devices()[0])

    # Run the experiment
    vanilla_state, scan_state, vanilla_metrics, scan_metrics = run_experiment()

    # Get final accuracies
    final_vanilla_acc = vanilla_metrics[-1]["accuracy"] * 100
    final_scan_acc = scan_metrics[-1]["accuracy"] * 100

    print("\nExperiment complete!")
    print(f"Vanilla Attention Final Accuracy: {final_vanilla_acc:.2f}%")
    print(f"Scan-Based Attention Final Accuracy: {final_scan_acc:.2f}%")
    print(f"Improvement: {final_scan_acc - final_vanilla_acc:.2f} percentage points")
    
    # Print learned hyperparameters from the scan attention model
    print("\nLearned ScanAttention hyperparameters:")
    params = scan_state.params
    
    # Get number of layers by counting the layer params
    num_layers = sum(1 for key in params.keys() if key.startswith('layers_'))
    
    # Extract hyperparameters from all layers
    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx + 1}:")
        layer_params = params[f'layers_{layer_idx}']['attention']
        
        # Get beta values (softplus sharpness)
        log_beta = layer_params['log_beta']
        beta = jnp.exp(log_beta)
        print(f"- Beta (softplus sharpness): {beta}")
        
        # Get temperature values
        log_temp = layer_params['log_temp']
        temp = jnp.exp(log_temp)
        print(f"- Temperature: {temp}")
        
        # Get sharpness values
        log_sharp = layer_params['log_sharp']
        sharp = jnp.exp(log_sharp)
        print(f"- Sharpness: {sharp}")
        
        # Get importance scales if content scaling is enabled
        if 'log_importance_scales' in layer_params:
            log_importance_scales = layer_params['log_importance_scales']
            importance_scales = jnp.exp(log_importance_scales)
            
            # Print per-head importance scales
            for head_idx, head_scales in enumerate(importance_scales):
                var_scale, mag_scale, sparsity_scale = head_scales
                print(f"  Head {head_idx + 1} importance scales:")
                print(f"  - Variance (distinctiveness): {var_scale:.4f}")
                print(f"  - Magnitude (signal strength): {mag_scale:.4f}")
                print(f"  - Sparsity (focus): {sparsity_scale:.4f}")
