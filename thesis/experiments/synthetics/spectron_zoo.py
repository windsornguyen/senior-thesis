# ===----------------------------------------------------------------------=== #
# File: spectron.py
# Author(s): Windsor Nguyen
# Description: A linear attention model implemented in JAX/Flax.
# ===----------------------------------------------------------------------=== #

import math

import jax
import jax.numpy as jnp
import jax.nn as nn
import flax.linen as lnn
import jax.lax as lax

from flax import struct
from typing import Optional, Tuple, Union
from functools import partial


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


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
    eig_vals = eig_vals[-K:]
    eig_vecs = eig_vecs[:, -K:]
    eig_vecs = eig_vecs * eig_vals**0.25
    return eig_vecs


@jax.jit
def tensordot_conv(f: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Perform causal 1D convolution via FFT for multi-headed inputs.

    Args:
        f: Spectral filters of shape [L, D].
        u: Input sequences of shape [B, H, L, h].

    Returns:
        Convolved sequences of shape [B, H, L, h].
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
    """Compute FFT-based convolution for multi-headed inputs.

    Args:
        v: Spectral filters of shape [L, K].
        u: Inputs of shape [B, H, L, h].

    Returns:
        Convolution output of shape [B, H, L, K, h].
    """
    L = u.shape[2]  # Sequence length from input u

    # Basic convolution for two 1D arrays of length L, truncated to L
    tr_conv = lambda filter_k, channel_h: jax.scipy.signal.convolve(channel_h, filter_k, method="fft")[:L]

    # Convolve one filter [L] with all h channels [L, h] -> output [L, h]
    # Maps tr_conv over the second axis (h) of the second argument (u_slice)
    # Broadcasts the first argument (filter_k)
    conv_filter_with_channels = jax.vmap(tr_conv, in_axes=(None, 1), out_axes=1)

    # Convolve all K filters [L, K] with all h channels [L, h] -> output [L, K, h]
    # Maps conv_filter_with_channels over the first axis (K) of the first argument (v)
    # Broadcasts the second argument (u_slice)
    conv_all_filters_channels = jax.vmap(conv_filter_with_channels, in_axes=(1, None), out_axes=1)

    # Apply to one head: input u1 [L, h], filters v [L, K] -> output [L, K, h]
    # Pass v first to match the axes mapping in conv_all_filters_channels
    conv_one_head = lambda u1, v_filters: conv_all_filters_channels(v_filters, u1)

    # Map over heads H: input u_h [H, L, h], filters v [L, K] -> output [H, L, K, h]
    # Maps conv_one_head over the first axis (H) of the first argument (u_slice)
    # Broadcasts the second argument (v)
    conv_heads = jax.vmap(conv_one_head, in_axes=(0, None), out_axes=0)

    # Map over batch B: input u [B, H, L, h], filters v [L, K] -> output [B, H, L, K, h]
    # Maps conv_heads over the first axis (B) of the first argument (u)
    # Broadcasts the second argument (v)
    conv_batch = jax.vmap(conv_heads, in_axes=(0, None), out_axes=0)

    return conv_batch(u, v)


@partial(jax.jit, static_argnames=("p", "axis"))
def normalize(
    input: jnp.ndarray, p: float = 2.0, axis: Union[int, Tuple[int, ...]] = 1, eps: float = 1e-12
) -> jnp.ndarray:
    """Normalize input along specified dimension.

    Args:
        input: Input array.
        p: Norm order (default: 2.0 for L2 norm).
        axis: Dimension(s) to normalize over.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized array.

    """
    norm = jnp.linalg.norm(input, ord=p, axis=axis, keepdims=True)
    return input / jnp.maximum(norm, eps)


@jax.jit
def combine_fn(x: Tuple, y: Tuple) -> Tuple:
    """Combine function for scan: (m, s, n, Z, g)."""
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y

    m_new = jnp.maximum(m_x, m_y)
    exp_x = jnp.exp(m_x - m_new)
    exp_y = jnp.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y

    # n_x is always [L, h], exp_x is [L]. Broadcast exp_x to [L, 1].
    n_new = n_x * exp_x[..., None] + n_y * exp_y[..., None]

    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new


@jax.jit
def scan_fn(qk_slice: jnp.ndarray, v_norm_slice: jnp.ndarray, Z_slice: jnp.ndarray, g_slice: jnp.ndarray):
    """Process a single slice for scanning using v_norm for n state."""
    # Initial leaves: (scores, initial_s, original_v_norm, Z_slice, g_slice)
    leaves = (qk_slice, jnp.ones_like(qk_slice), v_norm_slice, Z_slice, g_slice)
    return jax.lax.associative_scan(combine_fn, leaves, axis=0)


@jax.jit
def batched_scan_fn(sim, v_norm, gated_Z, gates):
    """Applies the full scan across batch and head dimensions."""
    # v_norm has shape [B, H, L, h]
    return jax.vmap(jax.vmap(scan_fn, in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))(sim, v_norm, gated_Z, gates)


@jax.jit
def softplus(x, beta):
    """Softplus with configurable beta."""
    return (1.0 / beta) * jnp.log(1.0 + jnp.exp(beta * x))


# ===----------------------------------------------------------------------=== #
#                                   Model Definitions
# ===----------------------------------------------------------------------=== #


class AssociativeAttention(lnn.Module):
    """Associative attention mechanism with spectral filtering and gating."""

    dim: int
    num_heads: int
    seq_len: int
    spectral_basis: jnp.ndarray
    use_tensordot: bool = True
    eps: float = 1e-5

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.wq = lnn.Dense(self.dim)
        self.wk = lnn.Dense(self.dim)
        self.wv = lnn.Dense(self.dim)
        self.wo = lnn.Dense(self.dim)

        if self.use_tensordot:
            self.tensordot_proj = lnn.Dense(self.dim)

        self.wg_z = lnn.Dense(1, use_bias=True)
        self.gate_proj = lnn.Dense(self.dim, use_bias=True)

        self.kv_norm_scale = self.param(
            "kv_norm_scale",
            lambda rng: jnp.ones((1, 1, 1, self.head_dim, self.head_dim)),
        )
        self.qk_norm_scale = self.param(
            "qk_norm_scale", lambda rng: jnp.full((1, self.num_heads, 1), 1 / jnp.sqrt(self.head_dim))
        )

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim

        q = self.wq(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)

        sim = jnp.einsum("bhld,bhld->bhl", q, k) * self.qk_norm_scale

        k_norm = normalize(k, p=2.0, axis=-1, eps=self.eps)
        v_norm = normalize(v, p=2.0, axis=-1, eps=self.eps)

        if self.use_tensordot:
            filters = self.tensordot_proj(self.spectral_basis)
            k_filtered = tensordot_conv(filters, k_norm)
            v_filtered = tensordot_conv(filters, v_norm)
        else:
            k_filtered = stu_conv(self.spectral_basis, k_norm)
            v_filtered = stu_conv(self.spectral_basis, v_norm)

        Z = (
            jnp.einsum("bhlkd,bhlke->bhlde", v_filtered, k_filtered)
            if not self.use_tensordot
            else jnp.einsum("bhld,bhle->bhlde", v_filtered, k_filtered)
        ) * self.kv_norm_scale

        gate_input_z = Z.reshape(*Z.shape[:3], -1)
        gates_logits_z = self.wg_z(gate_input_z)
        gates_z = nn.relu(gates_logits_z) ** 2 + self.eps
        gates_z = gates_z[..., None]

        gated_Z = gates_z * Z

        m_scan, s_scan, n_scan, Z_scan, g_scan = batched_scan_fn(sim, v_norm, gated_Z, gates_z)

        # val_path = n_scan / jnp.maximum(s_scan, self.eps)[..., None]

        softmax_weights = jnp.exp(sim - m_scan)[..., None, None] / (s_scan[..., None, None] + self.eps)
        gated_weights = Z_scan / (g_scan + self.eps)
        attn_weights = gated_weights * (1.0 + nn.silu(softmax_weights))

        # Query from the attention weights
        ctxt = jnp.einsum("bhld,bhlde->bhle", q, attn_weights)  # [B, H, L, h]

        # Reshape and project
        ctxt_norm = normalize(ctxt, axis=-1)
        output = ctxt_norm.transpose(0, 2, 1, 3).reshape(B, L, D)
        output = self.wo(output)

        return output


# @jax.jit
# def torch_conv(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
#     """
#     Computes a 1D convolution with same padding using fftconvolve.

#     Args:
#         v: Kernel of shape [L, K] (length matches input length)
#         u: Input of shape [B, L, D] (batch, length, channels)

#     Returns:
#         Output of shape [B, L, K, D]
#     """
#     conv_1d = lambda x: jax.scipy.signal.convolve(x[:, None], v[:, None], mode="same", method="fft")[:, 0]
#     conv_batched = lambda x: jax.vmap(conv_1d, in_axes=0, out_axes=0)(x)
#     conv_full = lambda x: jax.vmap(conv_batched, in_axes=2, out_axes=2)(x)
#     return conv_full(u)


# class AssociativeAttention(lnn.Module):
#     """Associative attention mechanism with spectral filtering and gating."""

#     dim: int
#     num_heads: int
#     seq_len: int
#     spectral_filters: jnp.ndarray
#     use_tensordot: bool = True
#     eps: float = 1e-5

#     def setup(self):
#         self.head_dim = self.dim // self.num_heads
#         self.wq = lnn.Dense(self.dim)
#         self.wk = lnn.Dense(self.dim)
#         self.wv = lnn.Dense(self.dim)
#         self.wo = lnn.Dense(self.dim)
#         self.r = self.dim  # Keep it simple for now
#         self.n = nearest_power_of_two(2 * self.seq_len - 1, round_up=True)
#         self.k = 24

#         if self.use_tensordot:
#             self.tensordot_proj = lnn.Dense(self.dim)

#         # self.wg_z = lnn.Dense(1, use_bias=True)
#         self.wg = lnn.Dense(self.dim, use_bias=True)

#         # self.kv_norm_scale = self.param(
#         #     "kv_norm_scale",
#         #     lambda rng: jnp.ones((1, 1, 1, self.head_dim, self.head_dim)),
#         # )
#         # self.qk_norm_scale = self.param(
#         #     "qk_norm_scale", lambda rng: jnp.full((1, self.num_heads, 1), 1 / jnp.sqrt(self.head_dim))
#         # )

#         if self.use_tensordot:
#             self.M_inputs = self.param("M_inputs", nn.initializers.zeros, (self.dim, self.r))
#             self.M_filters = self.param("M_filters", nn.initializers.zeros, (self.k, self.r))
#             self.out_proj_stu = lnn.Dense(features=self.dim, use_bias=True)
#         else:
#             self.M_phi_plus = self.param("M_phi_plus", nn.initializers.zeros, (self.k, self.dim, self.dim))
#             if not self.use_hankel_L:
#                 self.M_phi_minus = self.param("M_phi_minus", nn.initializers.zeros, (self.k, self.dim, self.dim))

#     def stu_conv(
#         self, u: jnp.ndarray, v: jnp.ndarray, n: int, use_tensordot: bool = True
#     ) -> tuple[jnp.ndarray, jnp.ndarray]:
#         """
#         Performs FFT-based convolution with causal alignment using negative featurization.

#         Args:
#             u: Input array of shape (B, seq_len, d_in).
#             v: Kernel array; shape (seq_len, d_out) if use_tensordot is True.
#             n: FFT length (usually 2*seq_len - 1 for linear convolution).
#             use_tensordot: Flag to control kernel reshaping.

#         Returns:
#             Tuple (U_plus, U_minus):
#             - U_plus: Primary convolution output.
#             - U_minus: Secondary output, corrected by the sign array.
#         """
#         bsz, seq_len, d_in = u.shape
#         sgn = jnp.full((1, seq_len, 1), 1, dtype=u.dtype)
#         sgn = sgn.at[:, 1::2].multiply(-1)  # Apply negative featurization.

#         if use_tensordot:
#             _, d_out = v.shape
#             v = v.reshape(1, -1, d_out, 1)
#         else:
#             _, K = v.shape
#             sgn = jnp.expand_dims(sgn, axis=-1)
#             v = v.reshape(1, -1, K, 1, 1)
#             u = jnp.expand_dims(u, axis=2)
#             u = jnp.tile(u, (1, 1, K, 1))

#         v = jnp.fft.rfft(v, n=n, axis=1)
#         U = jnp.stack([u, u * sgn], axis=-1)
#         U = jnp.fft.rfft(U, n=n, axis=1)
#         U_conv = jnp.fft.irfft(v * U, n=n, axis=1)[:, :seq_len]
#         U_plus, U_minus = jnp.split(U_conv, 2, axis=-1)

#         # Have to manually squeeze since JAX doesn't have unbind
#         U_plus = jnp.squeeze(U_plus, axis=-1)
#         U_minus = jnp.squeeze(U_minus, axis=-1)

#         U_minus = U_minus * sgn

#         return U_plus, U_minus

#     def compute_stu_features(self, u: jnp.ndarray) -> jnp.ndarray:
#         if self.use_tensordot:
#             # Project first
#             u_proj = u @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
#             # p_coeffs_conv = p_coeffs_conv @ self.M_inputs  # (B, L, d_in) x (d_in, r) -> (B, L, r)
#             phi_proj = self.spectral_filters @ self.M_filters  # (L, K) x (K, r) -> (L, r)

#             # Then, convolve: (B, L, r) ⊗ (L, r) -> (B, L, r)
#             spectral_plus, spectral_minus = self.stu_conv(u_proj, phi_proj, self.n, self.use_tensordot)

#             # Final output
#             # out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
#             out = spectral_plus + spectral_minus  # No hankel_L for now
#             # out = self.out_proj_stu(out + p_coeffs_conv)
#         else:
#             # Convolve first to get featurized inputs: (B, L, d_in) x (L, K) -> (B, L, K, d_in)
#             U_plus, U_minus = self.stu_conv(u, self.spectral_filters, self.n, self.use_tensordot)

#             # Compute sum-product of featurized inputs and M matrices over the K filters
#             B, L, K, d_in = U_plus.shape

#             # Spectral output: (B, L, K * d_in) x (K * d_in, d_out) -> (B, L, d_out)
#             spectral_plus = U_plus.view(B, L, K * d_in) @ self.M_phi_plus.view(K * d_in, self.d_out)

#             if not self.use_hankel_L:
#                 spectral_minus = U_minus.view(B, L, K * d_in) @ self.M_phi_minus.view(K * d_in, self.d_model)

#             # out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
#             out = spectral_plus + spectral_minus  # No hankel_L for now

#         return out

#     def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
#         B, L, D = x.shape
#         H, h = self.num_heads, self.head_dim

#         x_tilde = self.compute_stu_features(x)
#         q = self.wq(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)
#         k = self.wk(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)
#         v = self.wv(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)

#         Z = jnp.einsum("bhtp,bhtn->bhtpn", v, k)
#         H = jnp.cumsum(Z, axis=2)
#         Y = jnp.einsum("bhtp,bhtpn->bhtn", q, H)  # (B, H, T, d_head)

#         # Merge heads
#         Y_attn = Y.transpose(0, 2, 1, 3).reshape(B, L, D)  # (B, T, d)

#         # Compute gate values
#         gate_values = nn.sigmoid(self.wg(x))  # (B, T, d)

#         # Combine branches using element-wise gating
#         Y_combined = gate_values * Y_attn + (1 - gate_values) * x_tilde

#         # Final projection and normalization
#         return self.wo(Y_combined)


class FeedForward(lnn.Module):
    """Feed-forward network with GELU activation."""

    dim: int
    inter_dim: int

    def setup(self):
        self.dense1 = lnn.Dense(self.inter_dim)
        self.dense2 = lnn.Dense(self.dim)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)
        return x


class SpectronBlock(lnn.Module):
    """Decoder layer with associative attention and feed-forward network.

    Args:
        dim: Model dimension.
        inter_dim: Intermediate dimension for feed-forward network.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        spectral_filters: Spectral filters for attention.
        use_tensordot: Whether to use tensordot convolution.
        eps: Small value for numerical stability.
    """

    dim: int
    inter_dim: int
    num_heads: int
    seq_len: int
    spectral_filters: jnp.ndarray
    use_tensordot: bool = True
    eps: float = 1e-5

    def setup(self):
        self.norm1 = lnn.LayerNorm(epsilon=self.eps)
        self.attn = AssociativeAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            spectral_filters=self.spectral_filters,
            use_tensordot=self.use_tensordot,
            eps=self.eps,
        )
        self.norm2 = lnn.LayerNorm(epsilon=self.eps)
        self.ffn = FeedForward(dim=self.dim, inter_dim=self.inter_dim)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, training=training)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x, training=training)
        x = x + residual
        return x


# ===----------------------------------------------------------------------=== #
#                                Configurations
# ===----------------------------------------------------------------------=== #


@struct.dataclass
class SpectronConfig:
    """Configuration for the Spectron model.

    Args:
        spectral_filters: Spectral filters for attention.
        spectral_filters_fft: Precomputed FFT of spectral filters (optional).
        bsz: Batch size.
        dim: Model dimension.
        num_heads: Number of attention heads.
        num_local_heads: Number of local attention heads (defaults to num_heads).
        num_layers: Number of transformer layers.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        inter_dim: Intermediate dimension for feed-forward (optional).
        mlp_scale: Scaling factor for intermediate dimension.
        use_tensordot: Whether to use tensordot convolution.
        weight_tying: Whether to tie input/output embeddings.
        bias: Whether to use bias in layers.
        eps: Small value for numerical stability.
        dtype: Data type for computations.
    """

    # Non-default fields first
    spectral_filters: jnp.ndarray

    # Fields with defaults
    model_type: str = "Spectron"
    spectral_filters_fft: Optional[jnp.ndarray] = None
    bsz: int = 1
    dim: int = 128
    num_heads: int = 2
    num_local_heads: Optional[int] = -1
    num_layers: int = 2
    seq_len: int = 512
    vocab_size: int = 1024
    inter_dim: Optional[int] = None
    mlp_scale: Optional[float] = 4.0
    use_tensordot: Optional[bool] = True
    weight_tying: Optional[bool] = True
    bias: Optional[bool] = False
    eps: float = 1e-5

    dtype: jnp.dtype = jnp.float32

    def __post_init__(self):
        if self.num_local_heads == -1:
            object.__setattr__(self, "num_local_heads", self.num_heads)
        if self.inter_dim is None:
            hidden_dim = self.mlp_scale * self.dim
            num_hidden = int(2 * hidden_dim / 3)
            object.__setattr__(self, "inter_dim", self._find_multiple(num_hidden, 256))

    @staticmethod
    def _find_multiple(n: int, k: int) -> int:
        """Find the nearest multiple of k greater than or equal to n."""
        return ((n + k - 1) // k) * k

    @classmethod
    def from_name(cls, name: str, K: int = 64):
        r"""Create a configuration from a preset name.

        Args:
            name: Preset name in ["synthetic"].
            K: Number of spectral filters.

        Returns:
            SpectronConfig instance.

        Raises:
            ValueError: If the preset name is unknown.

        """
        presets = {
            "synthetic": {
                "dim": 128,
                "num_heads": 4,
                "num_layers": 2,
                "seq_len": 256,
                "vocab_size": 1000,
                "inter_dim": 512,
                "use_tensordot": False,
                "weight_tying": False,
                "bias": False,
            },
        }
        if name not in presets:
            raise ValueError(f"Unknown preset name: {name}")

        # Generate spectral filters
        configs = presets[name]
        configs["spectral_filters"] = get_spectral_filters(
            seq_len=configs["seq_len"], K=K, use_hankel_L=False, dtype=jnp.float32
        )
        return cls(**configs)


# ===----------------------------------------------------------------------=== #
#                                 Main Model
# ===----------------------------------------------------------------------=== #


class Spectron(lnn.Module):
    """Spectron transformer model for causal language modeling.

    Args:
        config: Spectron configuration.
    """

    config: SpectronConfig

    def setup(self):
        self.tok_emb = lnn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)
        self.layers = [
            SpectronBlock(
                dim=self.config.dim,
                inter_dim=self.config.inter_dim,
                num_heads=self.config.num_heads,
                seq_len=self.config.seq_len,
                spectral_filters=self.config.spectral_filters,
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


# --- Chebyshev Polynomial Helpers (Needed for STU path) ---
def poly_mul_x(poly):
    return [0] + poly


def poly_scale(poly, factor):
    return [c * factor for c in poly]


def poly_sub(poly1, poly2):
    length = max(len(poly1), len(poly2))
    result = []
    for i in range(length):
        coef1 = poly1[i] if i < len(poly1) else 0
        coef2 = poly2[i] if i < len(poly2) else 0
        result.append(coef1 - coef2)
    return result


def chebyshev_coeff(n):
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    T_nm2, T_nm1 = [1], [0, 1]
    for _ in range(2, n + 1):
        term = poly_scale(poly_mul_x(T_nm1), 2)
        T_n = poly_sub(term, T_nm2)
        T_nm2, T_nm1 = T_nm1, T_n
    return T_n


def get_monic_chebyshev_coeffs(n: int) -> jnp.ndarray:
    coeffs = jnp.array(chebyshev_coeff(n), dtype=jnp.complex128)
    if n > 0:
        coeffs = coeffs / (2.0 ** (n - 1))
    return coeffs.real  # Assuming real coeffs are sufficient


def get_opt_degree(seq_len: int) -> int:
    return int(math.ceil((7 / 6) * math.log2(seq_len)))


# --- FFT Convolution Helpers (Needed for STU path) ---
def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    if x <= 0:
        return 1
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


def fft_conv(u: jnp.ndarray, v: jnp.ndarray, mode: str = "full", causal: bool = False) -> jnp.ndarray:
    """JAX implementation of FFT convolution, mirroring torch version structure."""
    B, L, d_u = u.shape
    F, d_v = v.shape  # Assumes filter v is [F, d]
    assert d_u == d_v, "Input and filter dimensions must match"

    conv_len = L + F - 1
    fft_len = nearest_power_of_two(conv_len, round_up=True)

    # Pad u (input) along sequence dim
    u_padded = jnp.pad(u, ((0, 0), (0, fft_len - L), (0, 0)))
    # Pad v (filter) along sequence dim
    v_padded = jnp.pad(v, ((0, fft_len - F), (0, 0)))

    # FFT
    U_fft = jnp.fft.rfft(u_padded, n=fft_len, axis=1)  # [B, fft_len//2+1, d]
    V_fft = jnp.fft.rfft(v_padded, n=fft_len, axis=0)  # [fft_len//2+1, d]

    # Convolution theorem (element-wise multiplication in frequency domain)
    # Broadcast V_fft to match U_fft batch dimension
    conv_fft = U_fft * V_fft[None, ...]  # [B, fft_len//2+1, d]

    # Inverse FFT
    conv_result = jnp.fft.irfft(conv_fft, n=fft_len, axis=1)  # [B, fft_len, d]

    # --- Slicing based on mode and causality ---
    if causal:
        start_idx = F - 1  # Shift for causality
    else:
        start_idx = 0

    if mode == "full":
        end_idx = start_idx + conv_len
    elif mode == "same":
        end_idx = start_idx + L
    elif mode == "valid":
        end_idx = start_idx + L - F + 1
    else:
        raise ValueError(f"Invalid mode '{mode}'")

    # Slice the result
    result = jax.lax.dynamic_slice_in_dim(conv_result, start_idx, end_idx - start_idx, axis=1)

    return result.astype(u.dtype)


def stu_conv(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Compute FFT-based convolution for multi-headed inputs.

    Args:
        v: Spectral filters of shape [L, K].
        u: Inputs of shape [B, H, L, h].

    Returns:
        Convolution output of shape [B, H, L, K, h].
    """
    L = u.shape[2]  # Sequence length from input u

    # Basic convolution for two 1D arrays of length L, truncated to L
    tr_conv = lambda filter_k, channel_h: jax.scipy.signal.convolve(channel_h, filter_k, method="fft")[:L]

    # Convolve one filter [L] with all h channels [L, h] -> output [L, h]
    # Maps tr_conv over the second axis (h) of the second argument (u_slice)
    # Broadcasts the first argument (filter_k)
    conv_filter_with_channels = jax.vmap(tr_conv, in_axes=(None, 1), out_axes=1)

    # Convolve all K filters [L, K] with all h channels [L, h] -> output [L, K, h]
    # Maps conv_filter_with_channels over the first axis (K) of the first argument (v)
    # Broadcasts the second argument (u_slice)
    conv_all_filters_channels = jax.vmap(conv_filter_with_channels, in_axes=(1, None), out_axes=1)

    # Apply to one head: input u1 [L, h], filters v [L, K] -> output [L, K, h]
    # Pass v first to match the axes mapping in conv_all_filters_channels
    conv_one_head = lambda u1, v_filters: conv_all_filters_channels(v_filters, u1)

    # Map over heads H: input u_h [H, L, h], filters v [L, K] -> output [H, L, K, h]
    # Maps conv_one_head over the first axis (H) of the first argument (u_slice)
    # Broadcasts the second argument (v)
    conv_heads = jax.vmap(conv_one_head, in_axes=(0, None), out_axes=0)

    # Map over batch B: input u [B, H, L, h], filters v [L, K] -> output [B, H, L, K, h]
    # Maps conv_heads over the first axis (B) of the first argument (u)
    # Broadcasts the second argument (v)
    conv_batch = jax.vmap(conv_heads, in_axes=(0, None), out_axes=0)

    return conv_batch(u, v)


# ===----------------------------------------------------------------------=== #
# Simple Spectron Components (Mimicking PyTorch train_copy.py version)
# ===----------------------------------------------------------------------=== #


class SimpleSpectralAttention(lnn.Module):
    """Simpler Spectral Attention combining Linear Attention and STU features."""

    dim: int
    num_heads: int
    seq_len: int
    spectral_filters: jnp.ndarray  # Shape [L, K]
    use_hankel_L: bool = False
    eps: float = 1e-5

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        K = self.spectral_filters.shape[1]  # Number of spectral filters

        # --- STU Path Components ---
        # Chebyshev polynomial convolution kernel
        poly_degree = get_opt_degree(self.seq_len)
        coeffs = get_monic_chebyshev_coeffs(poly_degree)  # Real coeffs
        # Shape [poly_degree+1, 1] -> expand to [poly_degree+1, dim]
        self.p_coeffs_kernel = coeffs[:, None].repeat(self.dim, axis=1)
        # Linear layers to project convolved features (matching non-tensordot path)
        self.M_phi_plus = lnn.Dense(self.dim, use_bias=False)  # Input K*dim
        self.M_phi_minus = lnn.Dense(self.dim, use_bias=False)  # Input K*dim

        # --- Linear Attention Path Components ---
        self.q_proj = lnn.Dense(self.dim)
        self.k_proj = lnn.Dense(self.dim)
        self.v_proj = lnn.Dense(self.dim)
        self.o_proj = lnn.Dense(self.dim)  # Output projection for combined result

        # --- Gating Component ---
        self.gate_proj = lnn.Dense(self.dim, use_bias=True)  # Gate based on input x

    def compute_stu_features(self, u: jnp.ndarray) -> jnp.ndarray:
        """Compute STU features mimicking PyTorch version's non-tensordot path."""
        B, T, d = u.shape
        K = self.spectral_filters.shape[1]

        # 1. Polynomial convolution
        # Ensure kernel has shape [F, d]
        # p_kernel = jnp.array(self.p_coeffs_kernel, dtype=u.dtype)  # Shape [Degree+1, d]
        # p_conv = -fft_conv(u, p_kernel, mode="same", causal=True)  # Shape [B, T, d]

        # 2. Spectral convolution (using simplified stu_conv)
        # Input u [B, T, d], filter v [T, K] -> Outputs U_plus, U_minus [B, T, K, d]
        U_plus, U_minus = stu_conv(self.spectral_filters, u)

        # 3. Project spectral features
        # Reshape U_plus/minus to [B*T, K*d] for Dense layer
        U_plus_flat = U_plus.reshape(B * T, K * d)
        spectral_plus = self.M_phi_plus(U_plus_flat).reshape(B, T, d)

        if not self.use_hankel_L:  # Match PyTorch logic
            U_minus_flat = U_minus.reshape(B * T, K * d)
            spectral_minus = self.M_phi_minus(U_minus_flat).reshape(B, T, d)
            spectral_out = spectral_plus + spectral_minus
        else:
            spectral_out = spectral_plus

        # 4. Combine with polynomial features and normalize
        out = spectral_out
        return out

    def compute_linear_attention(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Compute Linear Attention: Q @ associative_scan(K @ V.T)"""
        B, H, T, h = q.shape

        # Apply non-linearities (optional, e.g., GeLU as in PyTorch version)
        q_act = q  # JAX uses precise GeLU by default
        k_act = k

        Z = jnp.einsum("bhsd, bhse -> bhsde", k_act, v)  # Outer product k_act, v -> [B, H, T, h, h]

        # Causal cumulative sum of Z using associative scan
        Z_accum = lax.cumsum(Z, axis=2)

        # Compute final output Y = Q @ Z_accum
        Y = jnp.einsum("bhqd,bhqde->bhqe", q_act, Z_accum)  # [B, H, T, h]

        # Merge heads
        Y_attn = Y.transpose(0, 2, 1, 3).reshape(B, T, self.dim)  # [B, T, D]
        return Y_attn

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, T, D = x.shape

        # --- Branch 1: STU Features ---
        # Note: Uses original x as input u
        x_tilde = self.compute_stu_features(x)  # [B, T, D]

        # --- Branch 2: Linear Attention ---
        # Project Q, K, V from original x
        q_lin = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_lin = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_lin = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Compute linear attention output
        Y_attn = self.compute_linear_attention(q_lin, k_lin, v_lin)  # [B, T, D]

        # --- Combine Branches ---
        # Compute gate values based on original input x
        gate_logits = self.gate_proj(x)  # [B, T, D]
        gate_values = nn.sigmoid(gate_logits)  # [B, T, D]

        # Combine using element-wise gating
        Y_combined = gate_values * Y_attn + (1 - gate_values) * x_tilde  # [B, T, D]

        # Final projection
        output = self.o_proj(Y_combined)
        return output


class SimpleSpectralAttentionLayer(lnn.Module):
    """Combines SimpleSpectralAttention and MLP with residuals/norms."""

    dim: int
    inter_dim: int
    num_heads: int
    seq_len: int
    spectral_filters: jnp.ndarray
    use_hankel_L: bool = False
    eps: float = 1e-5

    def setup(self):
        self.norm1 = lnn.LayerNorm(epsilon=self.eps)
        self.attn = SimpleSpectralAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            spectral_filters=self.spectral_filters,
            use_hankel_L=self.use_hankel_L,
            eps=self.eps,
        )
        self.norm2 = lnn.LayerNorm(epsilon=self.eps)
        self.ffn = FeedForward(dim=self.dim, inter_dim=self.inter_dim)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Pre-normalization convention
        attn_out = self.attn(self.norm1(x), training=training)
        x = x + attn_out  # Residual connection 1

        ffn_out = self.ffn(self.norm2(x), training=training)
        x = x + ffn_out  # Residual connection 2
        return x


class SimpleSpectron(lnn.Module):
    """JAX implementation mirroring the simpler PyTorch Spectron (train_copy.py)."""

    config: SpectronConfig  # Reuse config for compatibility

    def setup(self):
        self.tok_emb = lnn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)
        # Optional: Add Positional Embeddings like in PyTorch version
        position = jnp.arange(self.config.seq_len)[None, :, None]  # [1, L, 1]
        div_term = jnp.exp(jnp.arange(0, self.config.dim, 2) * (-math.log(10000.0) / self.config.dim))  # [D/2]
        pe = jnp.zeros((1, self.config.seq_len, self.config.dim))  # [1, L, D]
        pe = pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(position * div_term))
        self.pos_emb = pe  # Store as buffer [1, L, D]

        # Dropout can be added here if desired using config.dropout

        self.layers = [
            SimpleSpectralAttentionLayer(
                dim=self.config.dim,
                inter_dim=self.config.inter_dim,
                num_heads=self.config.num_heads,
                seq_len=self.config.seq_len,
                spectral_filters=self.config.spectral_filters,
                use_hankel_L=False,  # Mimic train_copy.py setting? Check config usage there.
                eps=self.config.eps,
            )
            for _ in range(self.config.num_layers)
        ]
        self.norm = lnn.LayerNorm(epsilon=self.config.eps)  # Final norm

        # Output projection (respecting weight tying config)
        if self.config.weight_tying:
            self.lm_head = lambda x: jnp.dot(x, self.tok_emb.embedding.T)
        else:
            self.lm_head = lnn.Dense(self.config.vocab_size, use_bias=self.config.bias)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # x shape [B, L]
        B, L = x.shape
        assert L == self.config.seq_len, f"Input seq len {L} != Config seq len {self.config.seq_len}"

        x_emb = self.tok_emb(x)
        # Add positional embeddings
        x_emb = x_emb + self.pos_emb[:, :L, :]  # Slice PE in case L < config.seq_len

        # Apply dropout if added

        h = x_emb
        for layer in self.layers:
            h = layer(h, training=training)

        h = self.norm(h)  # Apply final norm
        logits = self.lm_head(h)
        return logits.astype(self.config.dtype)  # Cast output


# --- FeedForward --- (Needed by Transformer)
class FeedForward(lnn.Module):
    """Standard FeedForward layer."""

    dim: int
    inter_dim: int
    dropout_rate: float = 0.0  # Add dropout

    @lnn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        y = lnn.Dense(features=self.inter_dim)(x)
        y = nn.gelu(y)
        y = lnn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)  # Add dropout
        y = lnn.Dense(features=self.dim)(y)
        y = lnn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)  # Add dropout
        return y


# ===----------------------------------------------------------------------=== #
# Vanilla Transformer Components
# ===----------------------------------------------------------------------=== #


@struct.dataclass
class TransformerConfig:
    """Configuration for the vanilla Transformer model."""

    dim: int
    num_heads: int
    num_layers: int
    inter_dim: int  # Dimension of the MLP intermediate layer (often multiple of dim)
    vocab_size: int
    seq_len: int  # Max sequence length for positional embeddings
    dropout_rate: float = 0.1
    eps: float = 1e-5  # Epsilon for RMSNorm
    weight_tying: bool = False
    dtype: jnp.dtype = jnp.float32


# ===----------------------------------------------------------------------=== #
# Pure JAX Style Transformer Components within Flax Modules
# ===----------------------------------------------------------------------=== #


class RMSNorm(lnn.Module):
    """Root Mean Square Layer Normalization."""

    dim: Optional[int] = None
    eps: float = 1e-6  # Default epsilon, will be overridden by config ideally
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Typically, the RMSNorm scale parameter (`w` in the example) is initialized to 1.
        # It's a learnable parameter.
        # Initializer function must accept rng key
        self.weight = self.param(
            "weight",
            lambda rng: jnp.ones(self.dim or -1, dtype=self.param_dtype),  # Use self.dim or infer from input
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies RMSNorm normalization."""
        # Calculate RMS: sqrt(mean(x^2))
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        # Normalize: x / RMS
        # return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        # Simpler: return x / rms
        return x / rms  # Use the calculated rms

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies RMSNorm layer."""
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))  # Promote for stability
        normed_x = self._norm(x)
        output = self.weight * normed_x
        return output.astype(self.dtype)


class AttentionPureJax(lnn.Module):
    """Attention mechanism based on user-provided pure JAX example."""

    config: TransformerConfig

    def setup(self):
        self.head_dim = self.config.dim // self.config.num_heads
        if self.config.dim % self.config.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        # Define projections using Flax Dense layers (handles initialization)
        # No bias usually used in these layers for pure transformers
        dense = partial(
            lnn.Dense,
            features=self.config.dim,
            use_bias=False,  # Typically False
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,  # Match param dtype
            kernel_init=nn.initializers.xavier_uniform(),
        )

        self.wq = dense()
        self.wk = dense()
        self.wv = dense()
        self.wo = dense()  # Output projection

        self.dropout = lnn.Dropout(rate=self.config.dropout_rate)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, L, D = x.shape
        H = self.config.num_heads
        K = self.head_dim  # Size of each head key/value/query

        # Apply projections
        query = self.wq(x)  # [B, L, D]
        key = self.wk(x)  # [B, L, D]
        value = self.wv(x)  # [B, L, D]

        # Reshape for multi-head attention [B, L, H, K]
        query = query.reshape(B, L, H, K)
        key = key.reshape(B, L, H, K)
        value = value.reshape(B, L, H, K)

        # Transpose for einsum: [B, H, L, K]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Scaled dot-product: (B, H, L, K) @ (B, H, K, L) -> (B, H, L, L)
        # Using einsum as in the example:
        scores = jnp.einsum("bhik,bhjk->bhij", query, key) / math.sqrt(K)

        # Apply causal mask
        # Ensure mask is broadcastable [1, 1, L, L] or [B, 1, L, L]
        if mask.ndim == 2:  # [L, L]
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:  # [B, L, L] - unlikely from make_causal_mask
            mask = mask[:, None, :, :]

        scores = jnp.where(mask, scores, jnp.finfo(self.config.dtype).min)

        # Softmax and dropout
        weights = nn.softmax(scores, axis=-1)  # [B, H, L, L]
        weights = self.dropout(weights, deterministic=not training)

        # Apply weights to values: (B, H, L, L) @ (B, H, L, K) -> (B, H, L, K)
        # Using einsum as in the example (transposed value):
        # wtd_values_blhk = jnp.einsum('blhk,bhlm->blhk', value_bhLk, weights_bhLL) # Doesn't match dims
        # Corrected einsum for (B,H,L,L) @ (B,H,L,K) -> (B,H,L,K)
        wtd_values = jnp.einsum("bhij,bhjk->bhik", weights, value)

        # Reshape and project output
        # Transpose back: [B, L, H, K]
        wtd_values = wtd_values.transpose(0, 2, 1, 3)
        # Reshape: [B, L, D]
        wtd_values = wtd_values.reshape(B, L, D)

        # Final output projection
        out = self.wo(wtd_values)
        out = self.dropout(out, deterministic=not training)  # Dropout on final output

        return out


class FeedForwardSwiGLU(lnn.Module):
    """SwiGLU Feed-Forward Network."""

    config: TransformerConfig

    def setup(self):
        # Hidden dim usually different from embed dim, often multiple like 4*dim
        # The example uses inter_dim for w2 (output), implies w1/w3 go to inter_dim
        hidden_dim = self.config.inter_dim  # Use inter_dim for clarity
        embed_dim = self.config.dim

        dense = partial(
            lnn.Dense,
            use_bias=False,  # Typically False
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )

        self.w1 = dense(features=hidden_dim)  # Gate projection
        self.w3 = dense(features=hidden_dim)  # Linear projection
        self.w2 = dense(features=embed_dim)  # Output projection back to embed_dim

        self.dropout = lnn.Dropout(rate=self.config.dropout_rate)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        gate = self.w1(x)  # (B, L, H)
        hidden = self.w3(x)  # (B, L, H)
        activated_hidden = nn.silu(gate) * hidden  # SwiGLU # (B, L, H)
        outputs = self.w2(activated_hidden)  # (B, L, D)
        outputs = self.dropout(outputs, deterministic=not training)
        return outputs


class TransformerLayerPureJax(lnn.Module):
    """Transformer Layer using pure JAX style components."""

    config: TransformerConfig

    def setup(self):
        # Use LayerNorm instead of RMSNorm
        self.attn_norm = lnn.LayerNorm(epsilon=self.config.eps, dtype=self.config.dtype)
        self.attention = AttentionPureJax(config=self.config)
        # Use LayerNorm instead of RMSNorm
        self.ffn_norm = lnn.LayerNorm(epsilon=self.config.eps, dtype=self.config.dtype)
        self.ffn = FeedForwardSwiGLU(config=self.config)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Pre-Normalization and Residual Connection for Attention
        attn_out = self.attention(self.attn_norm(x), mask=mask, training=training)
        x = x + attn_out

        # Pre-Normalization and Residual Connection for FFN
        ffn_out = self.ffn(self.ffn_norm(x), training=training)
        x = x + ffn_out
        return x


class Transformer(lnn.Module):
    """Transformer Model using pure JAX style components."""

    config: TransformerConfig

    def setup(self):
        self.tok_emb = lnn.Embed(
            num_embeddings=self.config.vocab_size, features=self.config.dim, dtype=self.config.dtype
        )
        # Simple Sinusoidal Positional Embeddings
        position = jnp.arange(self.config.seq_len)[None, :, None]  # [1, L, 1]
        div_term = jnp.exp(jnp.arange(0, self.config.dim, 2) * (-math.log(10000.0) / self.config.dim))  # [D/2]
        pe = jnp.zeros((1, self.config.seq_len, self.config.dim), dtype=self.config.dtype)  # [1, L, D]
        pe = pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(position * div_term))
        self.pos_emb = pe  # Store as buffer [1, L, D]

        self.dropout = lnn.Dropout(rate=self.config.dropout_rate)

        # Instantiate layers
        self.layers = [TransformerLayerPureJax(config=self.config) for _ in range(self.config.num_layers)]
        # Final normalization layer - use LayerNorm instead of RMSNorm
        self.norm = lnn.LayerNorm(epsilon=self.config.eps, dtype=self.config.dtype)

        # Output projection (LM Head)
        if not self.config.weight_tying:
            self.lm_head = lnn.Dense(self.config.vocab_size, use_bias=False, dtype=self.config.dtype)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # x shape [B, L] (token indices)
        B, L = x.shape
        if L > self.config.seq_len:
            raise ValueError(f"Input seq len {L} > configured max seq len {self.config.seq_len}")

        # 1. Embeddings (Token + Positional)
        h = self.tok_emb(x)  # [B, L, D]
        pos_emb_sliced = jax.lax.dynamic_slice_in_dim(self.pos_emb, 0, L, axis=1)  # Slice PE [1, L, D]
        h = h + pos_emb_sliced  # Add positional encoding
        h = self.dropout(h, deterministic=not training)

        # 2. Create Causal Mask
        causal_mask = lnn.make_causal_mask(x, dtype=jnp.bool_)  # Ensure boolean mask

        # 3. Apply Transformer Layers sequentially
        # Note: Using a Python loop is simpler than lax.scan with Flax modules
        for layer in self.layers:
            h = layer(h, mask=causal_mask, training=training)

        # 4. Final Normalization
        h = self.norm(h)

        # 5. LM Head
        if self.config.weight_tying:
            # Use the transpose of the token embeddings
            # Ensure compatible dtypes for matmul
            if h.dtype != self.tok_emb.embedding.dtype:
                h = h.astype(self.tok_emb.embedding.dtype)
            logits = h @ self.tok_emb.embedding.T
        else:
            logits = self.lm_head(h)

        # Ensure final output has the expected dtype (e.g., float32 for loss)
        return logits.astype(jnp.float32)


if __name__ == "__main__":
    import jax.random as random
    from jax import profiler

    # Check devices
    print("Available devices:", jax.devices())
    print("Default backend:", jax.default_backend())

    # Select GPU device if available
    gpu_device = jax.devices("gpu")[0] if "gpu" in jax.extend.backend.get_backend().platform else jax.devices("cpu")[0]

    # Test presets
    presets = ["synthetic"]
    rng = random.PRNGKey(0)

    for preset in presets:
        print(f"\nTesting preset: {preset}")
        config = SpectronConfig.from_name(preset, K=24)

        # Place spectral filters on GPU
        config = config.replace(
            spectral_filters=jax.device_put(config.spectral_filters, gpu_device).astype(config.dtype)
        )

        # Initialize model
        model = Spectron(config)

        # Count parameters using model.init
        rng, init_rng = random.split(rng)
        dummy_input_for_init = jnp.ones((1, config.seq_len), dtype=jnp.int32)
        variables = model.init(init_rng, dummy_input_for_init, training=False)
        params = variables["params"]  # We usually only need params for apply
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

        # Create actual dummy input for the forward pass
        rng, input_rng = random.split(rng)
        dummy_input = random.randint(input_rng, (config.bsz, config.seq_len), 0, config.vocab_size, dtype=jnp.int32)
        dummy_input = jax.device_put(dummy_input, gpu_device)
        params = jax.device_put(params, gpu_device)  # Ensure params are on the correct device

        # Define the core forward function operating on params and inputs
        # `model` is captured via closure
        @partial(jax.jit, static_argnames=("training",))
        def fwd(p, x, training: bool):
            return model.apply({"params": p}, x, training=training)

        # Run forward pass with profiling
        with profiler.TraceAnnotation("model_apply"):
            # Pass only params, input, and static args to the jitted function
            logits = fwd(params, dummy_input, training=False)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Output mean: {jnp.mean(logits):.4f}")
        print(f"Output std: {jnp.std(logits):.4f}")
