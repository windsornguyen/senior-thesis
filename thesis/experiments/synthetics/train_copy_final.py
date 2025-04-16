# Standard library imports
import math
import inspect
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, Union, Tuple

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp  # Added for autocast

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optax  # Add optax import
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchtune.modules import RotaryPositionalEmbeddings

# Local registry import
from thesis.experiments.synthetics.registry import registry

# Flash Attention imports
from flash_attn import flash_attn_func

# Model imports
from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from fla.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models import DeltaNetConfig, DeltaNetForCausalLM
from fla.models import RetNetConfig, RetNetForCausalLM
from fla.models import GLAConfig, GLAForCausalLM
from fla.models import LinearAttentionConfig, LinearAttentionForCausalLM
from fla.models import TransformerConfig, TransformerForCausalLM

# Flash FFT imports
try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError:
    FlashFFTConv = None
    flash_fft_available = False

# Flash Linear Attention imports for benchmarking
from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from fla.models.mamba2.modeling_mamba2 import Mamba2Model, Mamba2ForCausalLM, Mamba2Output, Mamba2CausalLMOutput

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_leaves
from flax import linen as linen_nn
from dataclasses import dataclass
from functools import partial
from flax.training import train_state  # For managing JAX training state

from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy loss that ignores -100 labels."""
    return F.cross_entropy(logits, labels, ignore_index=-100)


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


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


@dataclass
class SpectronConfig:
    dim: int = 1024
    num_eigh: int = 32
    num_heads: int = 4
    num_local_heads: Optional[int] = -1
    num_layers: int = 12
    seq_len: int = 4096
    vocab_size: int = 200064
    inter_dim: Optional[int] = 4096
    mlp_scale: float = 12.0
    weight_tying: bool = True
    use_tensordot: bool = True
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
class SpectronMLP(linen_nn.Module):
    config: SpectronConfig

    def setup(self):
        self.w1 = linen_nn.Dense(self.config.inter_dim)
        self.w2 = linen_nn.Dense(self.config.dim)

    def __call__(self, x):
        return self.w2(linen_nn.gelu(self.w1(x)))


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
class AssociativeAttention(linen_nn.Module):
    config: SpectronConfig
    spectral_filters: jnp.ndarray
    use_tensordot: bool

    def setup(self):
        self.head_dim = self.config.dim // self.config.num_heads
        self.wq = linen_nn.Dense(self.config.dim)
        self.wk = linen_nn.Dense(self.config.dim)
        self.wv = linen_nn.Dense(self.config.dim)
        self.wo = linen_nn.Dense(self.config.dim)
        self.wg = linen_nn.Dense(1)
        self.eps = 1e-5

        if self.use_tensordot:
            self.tensordot_proj = linen_nn.Dense(self.config.dim)

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
        gates = (linen_nn.relu(gates_logits) ** 2 + self.eps)[..., None]
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
class SpectronBlock(linen_nn.Module):
    config: SpectronConfig
    spectral_filters: jnp.ndarray

    def setup(self):
        self.aa_norm = linen_nn.LayerNorm(epsilon=self.config.eps)
        self.mlp_norm = linen_nn.LayerNorm(epsilon=self.config.eps)
        self.aa = AssociativeAttention(
            config=self.config, spectral_filters=self.spectral_filters, use_tensordot=self.config.use_tensordot
        )
        self.mlp = SpectronMLP(config=self.config)

    def __call__(self, x, training: bool = False):
        x = x + self.aa(self.aa_norm(x), training=training)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# Spectron
class Spectron(linen_nn.Module):
    config: SpectronConfig
    spectral_filters: jnp.ndarray

    def setup(self):
        self.tok_emb = linen_nn.Embed(self.config.vocab_size, self.config.dim)
        self.layers = [SpectronBlock(self.config, self.spectral_filters) for _ in range(self.config.num_layers)]
        self.norm_f = linen_nn.LayerNorm(epsilon=self.config.eps)
        # If we're tying weights, we want lm_head to project to the embedding dimension,
        # and then later use a dot product with the tied embeddings.
        if self.config.weight_tying:
            self.lm_head = linen_nn.Dense(self.config.dim, use_bias=False)
        else:
            self.lm_head = linen_nn.Dense(self.config.vocab_size, use_bias=False)

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

    def estimate_mfu(self, tokens_per_second: float, flops_per_gpu_second: float, params) -> float:
        """
        Calculate the Mean Frequency Utilization (MFU) of the model based on per-token FLOP cost.

        The per-token FLOP cost is computed from three main components:

        1. Parameter Cost:
            - Each parameter contributes 6 FLOPs per token.
            - Total = 6 * N, where N is the total number of model parameters.

        2. ScanAttention Cost:
            - For each layer (L) and head (H) with head dimension h:
                • q·k dot product costs: 2 * h FLOPs
                • Outer product(s) and gating cost: 6 * h^2 FLOPs
            - For all layers and heads, total = L * H * (2 * h + 6 * h^2)
            - Note: This cost scales with the number of layers and heads, not with the sequence length T.

        3. STU Cost (if enabled):
            - When using STU (i.e. when self.config.use_tensordot is False),
            an extra cost is incurred per token:
                L * [8 * d * (c_rfft * log2(2 * T)) + 12 * d]
            - Here, d is the model dimension, T is the sequence length, and c_rfft is a constant (default 1).

        Args:
            tokens_per_second (float): Observed throughput in tokens per second.
            flops_per_gpu_second (float): Peak achievable FLOPs per second of the GPU (or aggregate, if appropriate).
            params: The model's parameters pytree

        Returns:
            float: Estimated MFU (fraction between 0 and 1).
        """
        import math  # Add math import for log2

        # Extract model configuration.
        L = self.config.num_layers
        H = self.config.num_heads
        D = self.config.dim
        h = D // H  # head dimension
        T = self.config.seq_len

        # (1) Parameter cost: each parameter costs 6 FLOPs per token.
        N = get_num_params(params)
        param_cost = 6.0 * N

        # (2) ScanAttention cost per token:
        #    For each (layer, head): 2*h (for q·k) + 6*h^2 (for outer product & gating).
        scan_attention_cost = L * H * (2.0 * h + 6.0 * (h**2))

        # (3) STU cost (if enabled, assumed active when use_tensordot is False).
        c_rfft = 1.0  # constant factor for FFT-based convolution
        stu_cost = L * (8.0 * D * (c_rfft * math.log2(2.0 * T)) + 12.0 * D)

        # Total FLOPs per token.
        flops_per_token = param_cost + scan_attention_cost + stu_cost

        # Theoretical peak tokens per second.
        theoretical_peak_tokens_per_sec = flops_per_gpu_second / flops_per_token

        # MFU: observed tokens/sec divided by the theoretical peak tokens/sec.
        mfu = tokens_per_second / theoretical_peak_tokens_per_sec

        return float(mfu)


def get_hankel_matrix_jax(n: int, use_hankel_L: bool = False) -> jnp.ndarray:
    """Get the Hankel matrix for STU spectral filters using JAX.

    Args:
        n: sequence length
        use_hankel_L: whether to use the L-operator version of the Hankel matrix

    Returns:
        Hankel matrix of shape (n, n)
    """
    entries = jnp.arange(1, n + 1, dtype=jnp.float32)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)

    return Z


def get_spectral_filters_jax(n: int, k: int) -> jnp.ndarray:
    """Get spectral filters for STU using JAX.

    Args:
        n: sequence length
        k: number of eigenvalues/vectors to use

    Returns:
        spectral filters of shape (n, k)
    """
    try:
        print("Computing spectral filters using JAX on CPU...")
        # Move computation to CPU
        with jax.default_device(jax.devices("cpu")[0]):
            H = get_hankel_matrix_jax(n)
            eig_vals, eig_vecs = jnp.linalg.eigh(H)
            eig_vecs_k = eig_vecs[:, -k:]  # Take last k eigenvectors

        # Convert back to default device (GPU if available)
        return jax.device_put(eig_vecs_k)
    except Exception as e:
        print(f"Error in get_spectral_filters_jax: {str(e)}")
        print("Falling back to random initialization...")
        # Initialize random filters as fallback
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, (n, k)) / jnp.sqrt(k)


def get_hankel_matrix_torch(n: int, use_hankel_L: bool = False, device: str = "cuda") -> torch.Tensor:
    """Get the Hankel matrix for STU spectral filters using PyTorch.

    Args:
        n: sequence length
        use_hankel_L: whether to use the L-operator version of the Hankel matrix
        device: device to put the matrix on

    Returns:
        Hankel matrix of shape (n, n)
    """
    device = torch.device(device)
    entries = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    i_plus_j = entries.unsqueeze(1) + entries.unsqueeze(0)

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)

    return Z.to(device)


def get_spectral_filters_torch(n: int, k: int, device: str = "cuda") -> torch.Tensor:
    """Get spectral filters for STU using PyTorch.

    Args:
        n: sequence length
        k: number of eigenvalues/vectors to use
        device: device to put the filters on

    Returns:
        spectral filters of shape (n, k)
    """
    try:
        print("Computing spectral filters using PyTorch...")
        device = torch.device(device)
        H = get_hankel_matrix_torch(n, device=device)
        eig_vals, eig_vecs = torch.linalg.eigh(H)
        eig_vecs_k = eig_vecs[:, -k:]  # Take last k eigenvectors
        return eig_vecs_k.to(device)
    except Exception as e:
        print(f"Error in get_spectral_filters_torch: {str(e)}")
        print("Falling back to random initialization...")
        # Initialize random filters as fallback
        return torch.randn(n, k, device=device) / math.sqrt(k)


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


def init_model(config: SpectronConfig, rng):
    spectral_filters = get_spectral_filters_jax(config.seq_len, config.num_eigh)[1]
    model = Spectron(config, spectral_filters)
    params = model.init(rng, jnp.ones((1, config.seq_len), dtype=jnp.int32))["params"]
    return model, params


# Constants
IGNORE_IDX = -1
SEED = 1746
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")


# Configuration classes
@dataclass
class BaseConfigForCausalLM:
    """Base configuration class for causal language models"""

    model_type: str = "base_model"


@dataclass
class FlashSTUConfig(BaseConfigForCausalLM):
    model_type: str = "FlashSTU"
    bsz: int = 1
    dim: int = 1024
    r: int = 1024
    num_heads: int = 12
    num_local_heads: int = -1
    num_layers: int = 12
    seq_len: int = 4096
    n: int = 8192
    window_size: int = 2048
    vocab_size: int = 200064
    inter_dim: int = 3072
    mlp_scale: float = 12.0
    weight_tying: bool = True
    bias: bool = False
    rope_theta: float = 10000.0
    softcap: float = 50.0
    num_eigh: int = 24
    use_hankel_L: bool = False
    use_flash_fft: bool = flash_fft_available
    use_tensordot: bool = True
    use_attn: bool = True
    use_alibi: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    device: torch.device = None

    def __post_init__(self):
        if self.num_local_heads == -1:
            self.num_local_heads = self.num_heads
        if self.inter_dim is None:
            hidden_dim = self.mlp_scale * self.dim
            num_hidden = int(2 * hidden_dim / 3)
            self.inter_dim = find_multiple(num_hidden, 256)
        self.head_dim = self.dim // self.num_heads


# Base components
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.inter_dim)
        self.w2 = nn.Linear(config.inter_dim, config.dim)
        self.w2.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


# Attention components
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.wo.SCALE_INIT = 1

        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_local_heads = config.num_local_heads

        self.rotary_emb = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=config.seq_len,
            base=config.rope_theta,
        )

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        q, k = self.rotary_emb(q), self.rotary_emb(k)

        y = flash_attn_func(
            q=q,
            k=k,
            v=v,
            causal=True,
        )

        out = y.reshape(bsz, seq_len, -1)
        out = self.wo(out)

        return out


class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.wo.SCALE_INIT = 1

        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_local_heads = config.num_local_heads
        self.window_size = config.window_size
        self.softcap = config.softcap

        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if config.use_alibi else None
        self.rotary_emb = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=config.seq_len,
            base=config.rope_theta,
        )

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_local_heads, self.head_dim)

        if self.alibi_slopes is None:
            q, k = self.rotary_emb(q), self.rotary_emb(k)

        y = flash_attn_func(
            q=q,
            k=k,
            v=v,
            causal=True,
            window_size=(self.window_size, 0),
            alibi_slopes=self.alibi_slopes,
        )

        out = y.reshape(bsz, seq_len, -1)
        out = self.wo(out)

        return out

    def _generate_slopes(self, n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, num_heads: int, interpolation_factor: float = 0.25):
        if math.log2(num_heads).is_integer():
            slopes = self._generate_slopes(num_heads)
        else:
            n = nearest_power_of_two(num_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: num_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=torch.device("cuda"))
        slopes = slopes * interpolation_factor
        return slopes


# Layer components
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SlidingWindowAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swa_norm = nn.LayerNorm(config.dim)
        self.swa = SlidingWindowAttention(config)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        # Autocast handles precision
        x = x + self.swa(self.swa_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# Utility functions
def create_lr_lambda(warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * step / warmup_steps
        else:
            lr = max_lr - (max_lr - min_lr) * (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return lr / max_lr

    return lr_lambda


# May be different depending on task
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


"""

optim; adamw
max lr 3e-4
min lr 3e-5 (linear up to max lr at 10% of total steps, linear down to min lr rest of way)
gradient clipping 1.0
weight decay 1e-2

"""


# STU components
class STU(nn.Module):
    def __init__(self, config, spectral_filters=None):
        super().__init__()
        self.n = config.n
        self.num_eigh = config.num_eigh
        self.d_in = config.dim
        self.d_out = config.dim
        self.r = config.r
        self.use_hankel_L = config.use_hankel_L
        self.use_tensordot = config.use_tensordot
        self.flash_fft = (
            FlashFFTConv(self.n, dtype=torch.bfloat16) if config.use_flash_fft and flash_fft_available else None
        )

        # Ensure spectral filters are on the correct device
        if spectral_filters is not None:
            if not isinstance(spectral_filters, torch.Tensor):
                spectral_filters = torch.from_numpy(np.array(spectral_filters))
            self.spectral_filters = spectral_filters
        else:
            self.spectral_filters = None

        # Parameters will be moved to the correct device by the parent model's .to(device) call
        if self.use_tensordot:
            self.M_inputs = nn.Parameter(torch.zeros(self.d_in, self.d_out))
            self.M_filters = nn.Parameter(torch.zeros(self.num_eigh, self.d_in))
        else:
            self.M_phi_plus = nn.Parameter(torch.zeros(self.num_eigh, self.d_in, self.d_out))
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.zeros(self.num_eigh, self.d_in, self.d_out))

    def forward(self, x):
        B, L, D = x.shape
        x = x.to(self.spectral_filters.device)

        if self.use_tensordot:
            # Autocast should handle dtypes, parameters moved by .to(device)
            x_proj = x @ self.M_inputs
            phi_proj = self.spectral_filters @ self.M_filters

            if self.flash_fft:
                spectral_plus, spectral_minus = self.flash_conv(
                    x_proj, phi_proj.to(torch.float32), self.flash_fft, self.use_tensordot
                )
            else:
                spectral_plus, spectral_minus = self.conv(x_proj, phi_proj, self.n, self.use_tensordot)
        else:
            # Convolve inputs and filters first, then contract over (K, D) dims
            if self.flash_fft:
                U_plus, U_minus = self.flash_conv(x, self.spectral_filters, self.flash_fft, self.use_tensordot)
            else:
                U_plus, U_minus = self.conv(x, self.spectral_filters, self.n, self.use_tensordot)

            B, L, K, D = U_plus.shape
            spectral_plus = U_plus.reshape(B, L, K * D) @ self.M_phi_plus.reshape(K * D, self.d_out)
            if not self.use_hankel_L:
                spectral_minus = U_minus.reshape(B, L, K * D) @ self.M_phi_minus.reshape(K * D, self.d_out)

        out = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
        return out

    def conv(
        self, u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs convolution via FFT with causal alignment using a negative featurization."""
        bsz, seq_len, d_in = u.shape

        # Ensure sgn is created on the same device as input u
        sgn = torch.full((1, seq_len, 1), 1, device=u.device)
        sgn[:, 1::2] *= -1  # Apply negative featurization: multiply every other element by -1.

        if use_tensordot:
            _, d_out = v.shape
            v = v.view(1, -1, d_out, 1).to(torch.float32).contiguous()
        else:
            _, K = v.shape
            sgn = sgn.unsqueeze(-1)
            v = v.view(1, -1, K, 1, 1).to(torch.float32).contiguous()  # (bsz, seq_len, K, d_in, stack)
            u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

        v = torch.fft.rfft(v, n=n, dim=1)

        U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
        U = torch.fft.rfft(U, n=n, dim=1)

        # Slicing the first seq_len outputs yields the proper causal convolution given the negative modulation.
        U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
        U_plus, U_minus = torch.unbind(U_conv, dim=-1)
        U_minus = U_minus * sgn

        return U_plus, U_minus

    def flash_conv(
        self, u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_tensordot: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flash FFT convolution."""
        bsz, seq_len, d_in = u.shape
        _, K = v.shape

        padded_len = nearest_power_of_two(seq_len, round_up=True)
        pad_len = padded_len - seq_len

        # Ensure sgn is created on the same device as input u
        sgn = torch.full((1, 1, padded_len), 1, device=u.device)
        sgn[:, :, 1::2] = -1

        if use_tensordot:
            u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16)
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32)
            u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
        else:
            u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1)
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1)
            u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

        U_conv = flash_fft(u_conv, v_padded)

        # Trim the output back to the original sequence length
        U_conv = U_conv[..., :seq_len]
        u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

        if use_tensordot:
            u_minus = u_minus * sgn[:, :, :seq_len]
            U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
        else:
            sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
            U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
            U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

        return U_plus, U_minus


class STULayer(nn.Module):
    def __init__(self, config, spectral_filters=None):
        super().__init__()
        self.stu_norm = nn.LayerNorm(config.dim)
        self.stu = STU(config, spectral_filters)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        # Autocast handles precision
        x = x + self.stu(self.stu_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# Main model classes
class FlashSTU(nn.Module):
    def __init__(self, config, spectral_filters=None):
        super().__init__()
        print(f"Initializing FlashSTU model with config:")
        print(f"  dim: {config.dim}")
        print(f"  num_heads: {config.num_heads}")
        print(f"  num_layers: {config.num_layers}")
        print(f"  seq_len: {config.seq_len}")
        print(f"  num_eigh: {config.num_eigh}")

        self.config = config
        self.spectral_filters = spectral_filters

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList()

        for layer_idx in range(config.num_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, spectral_filters))
            else:
                if config.use_attn:
                    self.layers.append(SlidingWindowAttentionLayer(config))
                else:
                    self.layers.append(STULayer(config, spectral_filters))

        self.norm_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if self.config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = self.config.dim**-0.5

    def init_weights(self, module):
        std = self.std
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs) -> CausalLMOutput:
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = cross_entropy_loss(logits.flatten(0, 1), labels.flatten(0, 1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )


# Model Registry
class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, name: str, model_cls):
        self.models[name] = model_cls

    def get(self, name: str):
        if name not in self.models:
            raise ValueError(f"Model {name} not found. Available models: {list(self.models.keys())}")
        return self.models[name]

    def list_models(self):
        return list(self.models.keys())


# Initialize global registry
MODEL_REGISTRY = ModelRegistry()

# Register models
# MODEL_REGISTRY.register("flash_stu", FlashSTU)
# MODEL_REGISTRY.register("transformer", TransformerForCausalLM)
# MODEL_REGISTRY.register("mamba", Mamba2ForCausalLM)
MODEL_REGISTRY.register("rwkv7", RWKV7ForCausalLM)
# MODEL_REGISTRY.register("deltanet", DeltaNetForCausalLM)
# MODEL_REGISTRY.register("retnet", RetNetForCausalLM)
# MODEL_REGISTRY.register("gla", GLAForCausalLM)
# MODEL_REGISTRY.register("linear_attention", LinearAttentionForCausalLM)

# Register Spectron (JAX model)
MODEL_REGISTRY.register("spectron", Spectron)


def create_model_config(model_name: str, base_config: dict) -> Any:
    """Create appropriate config object for each model type."""
    if model_name == "flash_stu":
        return FlashSTUConfig(
            dim=base_config["hidden_size"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            seq_len=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            inter_dim=base_config["inter_dim"],
            use_flash_fft=flash_fft_available,
            use_tensordot=True,
            use_attn=True,
        )
    elif model_name == "transformer":
        return TransformerConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            num_attention_heads=base_config["num_heads"],
            max_position_embeddings=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            intermediate_size=base_config["inter_dim"],
        )
    elif model_name == "retnet":
        return RetNetConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            num_attention_heads=base_config["num_heads"],
            max_position_embeddings=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            intermediate_size=base_config["inter_dim"],
            hidden_act="swish",
        )
    elif model_name == "gla":
        return GLAConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            num_attention_heads=base_config["num_heads"],
            max_position_embeddings=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            intermediate_size=base_config["inter_dim"],
            hidden_act="swish",
        )
    elif model_name == "linear_attention":
        config = LinearAttentionConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            num_attention_heads=base_config["num_heads"],
            max_position_embeddings=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            intermediate_size=base_config["inter_dim"],
        )
        # Ensure flags that control layer output are False for training test
        config.output_attentions = False
        config.use_cache = False
        # Add missing hidden_act attribute required by the model implementation
        config.hidden_act = "swish"
        config.use_lower_bound = False
        return config
    elif model_name == "mamba":
        return Mamba2Config(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            vocab_size=base_config["vocab_size"],
            max_position_embeddings=base_config["max_seq_len"],
            intermediate_size=base_config["inter_dim"],
            hidden_act="swish",
        )
    elif model_name == "rwkv7":
        return RWKV7Config(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            max_position_embeddings=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            attn_mode="chunk",
            hidden_act="swish",
        )
    elif model_name == "deltanet":
        return DeltaNetConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_layers"],
            num_attention_heads=base_config["num_heads"],
            max_position_embeddings=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            intermediate_size=base_config["inter_dim"],
            hidden_act="swish",
        )
    elif model_name == "spectron":
        # Use the SpectronConfig defined earlier in the file
        return SpectronConfig(
            dim=base_config["hidden_size"],
            num_heads=base_config["num_heads"],
            num_layers=base_config["num_layers"],
            seq_len=base_config["max_seq_len"],
            vocab_size=base_config["vocab_size"],
            inter_dim=base_config["inter_dim"],
            # num_eigh, mlp_scale, weight_tying, use_tensordot, bias, eps use defaults from SpectronConfig
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def train_pytorch_model(
    model_name: str,
    config_obj: Any,  # Renamed from model_config, receives the specific config object
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    device: str = "cuda",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
):
    print(f"\n{'=' * 60}")
    print(f"Setting up {model_name.upper()} model")
    print(f"{'=' * 60}")

    device = torch.device("cuda")  # Explicitly use CUDA

    try:
        # Get model class and create appropriate config
        model_cls = MODEL_REGISTRY.get(model_name)
        # config_obj is already the specific config object created in main
        print(f"Received config object of type: {type(config_obj)}")

        if model_name == "flash_stu":
            # Initialize STU filters first
            # seq_len = config_obj.seq_len # Assuming seq_len is needed here from config
            # print(f"\nInitializing spectral filters for sequence length {seq_len}...")
            # spectral_filters = get_spectral_filters_torch(n=seq_len, k=config_obj.num_eigh).to(device)
            # Spectral filters should ideally be created in main and passed if needed,
            # or handled within FlashSTU itself if possible.
            # Let's assume FlashSTU handles its filters internally for now, or pass seq_len if needed.

            # Re-calculate filters here if absolutely necessary:
            print(f"\nInitializing spectral filters for sequence length {config_obj.seq_len}...")
            spectral_filters = get_spectral_filters_torch(n=config_obj.seq_len, k=config_obj.num_eigh).to(device)

            # Create model with filters
            print("Creating FlashSTU model with spectral filters...")
            model = model_cls(config_obj, spectral_filters).to(device)
            model.apply(model.init_weights)
        else:
            print(f"Creating {model_name} model...")
            model = model_cls(config_obj).to(device)

        print("\nModel created successfully!")

        # NOTE: DeltaNet currently skipped due to internal issues.
        # No explicit casting needed here; autocast will handle precision.

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {num_params:,}")
        print(f"Model dtype: {next(model.parameters()).dtype}")  # Print model dtype

        # Setup optimizer
        print("\nSetting up optimizer...")
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        print(f"\nStarting training for {num_epochs} epochs...")
        model.train()
        # Initialize GradScaler
        scaler = amp.GradScaler()
        total_steps = len(train_loader) * num_epochs
        global_step = 0

        # LR Scheduler (Example: linear warmup and decay)
        warmup_steps = int(0.1 * total_steps)
        lr_lambda = create_lr_lambda(warmup_steps, total_steps, learning_rate, learning_rate / 10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        latest_val_acc = 0.0  # Variable to hold latest validation accuracy
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            epoch_loss = 0.0
            # Wrap train_loader with tqdm for a progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch in pbar:
                try:
                    # Move batch to the correct device
                    input_ids = batch[0].to(device)
                    labels = batch[1].to(device)

                    optimizer.zero_grad()
                    # Autocast context
                    with amp.autocast(dtype=torch.bfloat16):
                        outputs = model(input_ids=input_ids, labels=labels)

                        # Handle different output formats
                        if isinstance(outputs, dict):
                            loss = outputs.get("loss")
                            if loss is None:
                                # Try to find loss under a different key if needed, or raise error
                                raise KeyError(
                                    f"Model {model_name} output dict missing 'loss' key. Keys: {outputs.keys()}"
                                )
                        elif isinstance(outputs, CausalLMOutputWithPast):
                            loss = outputs.loss  # Handle Transformer output type
                        elif isinstance(outputs, CausalLMOutput):
                            loss = outputs.loss
                        else:
                            print(f"Unexpected output type: {type(outputs)}")
                            print(f"Output contents: {outputs}")
                            raise ValueError(f"Model {model_name} returned unexpected output type")

                        if loss is None:
                            raise ValueError(f"Loss is None for model {model_name}")

                    # Scale loss and call backward
                    scaler.scale(loss).backward()

                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    # Scaler step optimizer
                    scaler.step(optimizer)

                    # Update scaler
                    scaler.update()

                    scheduler.step()  # Update learning rate

                    # Check for NaNs in loss
                    if torch.isnan(loss):
                        print(
                            f"\n!!! NaN loss detected at step {global_step}. Stopping training for {model_name}. !!!"
                        )
                        return False  # Stop training this model

                    epoch_loss += loss.item()
                    global_step += 1

                    # Update tqdm progress bar description with current loss, LR and latest val acc
                    pbar.set_postfix(
                        {
                            "Loss": loss.item(),
                            "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                            "Val Acc": f"{latest_val_acc:.2f}%",
                        }
                    )

                except Exception as e:
                    print(f"\nError during training step {global_step}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if hasattr(e, "__traceback__"):
                        import traceback

                        print("\nTraceback:")
                        traceback.print_tb(e.__traceback__)
                    # Optionally decide whether to continue to next batch or stop
                    # return False # Stop training this model on error
                    continue  # Continue to next batch

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

            # --- Validation Step --- #
            print(f"Running validation...")
            # compute_token_level_accuracy sets model.eval() and model.train() internally
            val_loss = 0.0
            val_steps = 0

            # Calculate accuracy using the helper function
            latest_val_acc = compute_token_level_accuracy(model, val_loader, device)
            print(f"Validation Accuracy: {latest_val_acc:.2f}%")

            # Calculate validation loss separately (optional, can be combined)
            with torch.no_grad():
                model.eval()  # Ensure model is in eval mode for loss calc
                for val_batch in tqdm(val_loader, desc="Validation Loss Calc"):  # Use a separate tqdm or remove
                    val_input_ids = val_batch[0].to(device)
                    val_labels = val_batch[1].to(device)
                    # Use autocast for validation inference as well
                    with amp.autocast(dtype=torch.bfloat16):
                        val_outputs = model(input_ids=val_input_ids, labels=val_labels)
                        # Handle different output formats for loss
                        current_val_loss = None
                        if isinstance(val_outputs, dict):
                            current_val_loss = val_outputs.get("loss")
                        elif isinstance(val_outputs, (CausalLMOutput, CausalLMOutputWithPast)):
                            current_val_loss = val_outputs.loss

                        if current_val_loss is not None:
                            val_loss += current_val_loss.item()
                            val_steps += 1
                        else:
                            print(
                                f"Warning: Validation loss not found for batch. Output keys: {val_outputs.keys() if isinstance(val_outputs, dict) else type(val_outputs)}"
                            )

            if val_steps > 0:
                avg_val_loss = val_loss / val_steps
                print(f"Validation Loss: {avg_val_loss:.4f}")
            model.train()  # Ensure model is back in train mode
            # --- End Validation Step --- #

        print(f"\nTraining finished successfully for {model_name}!")
        return True  # Indicate success

    except Exception as e:
        print(f"\nError during model setup or training loop initiation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback

            print("\nTraceback:")
            traceback.print_tb(e.__traceback__)
        return False


class TrainState(train_state.TrainState):
    # Simple TrainState for Optax
    pass


def train_spectron_model(
    model_name: str,
    model_config: Any,  # Should be SpectronConfig
    train_inputs_jax: jnp.ndarray,
    train_labels_jax: jnp.ndarray,
    val_inputs_jax: jnp.ndarray,
    val_labels_jax: jnp.ndarray,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,  # Note: Optax uses weight decay differently
    grad_clip: float = 1.0,
    seed: int = SEED,
) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Setting up JAX {model_name.upper()} model")
    print(f"{'=' * 60}")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    try:
        print("Creating Spectron config and model (JAX)...")
        # Config is already created and passed in as model_config
        seq_len = model_config.seq_len

        print(f"\nInitializing JAX spectral filters for sequence length {seq_len}...")
        spectral_filters_jax = get_spectral_filters_jax(seq_len, model_config.num_eigh)

        print("Initializing Spectron model parameters (JAX)...")
        dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
        model = Spectron(config=model_config, spectral_filters=spectral_filters_jax)
        params = model.init(init_rng, dummy_input)["params"]

        num_params_jax = get_num_params(params)
        print(f"Total parameters: {num_params_jax:,}")

        print("\nSetting up Optax optimizer...")
        # Example: AdamW with linear schedule (needs adjustment)
        num_train_steps = (train_inputs_jax.shape[0] // batch_size) * num_epochs
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=int(0.1 * num_train_steps),
            decay_steps=num_train_steps,
            end_value=learning_rate / 10.0,
        )
        # AdamW in Optax needs mask for weight decay (apply only to non-bias/norm params)
        # Simple Adam for now, weight decay needs more care
        optimizer = optax.adam(learning_rate=lr_schedule)
        tx = optimizer
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        print("\nDefining JAX training step...")

        def compute_loss(params, batch_inputs, batch_labels):
            logits = state.apply_fn({"params": params}, batch_inputs)["logits"]
            # Use Optax's cross entropy which handles integer labels
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, model_config.vocab_size), batch_labels.reshape(-1)
            )
            # Mask out ignore index (-100)
            mask = batch_labels.reshape(-1) != -100
            loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1)
            return loss

        @jax.jit
        def train_step(state, batch_inputs, batch_labels):
            loss_fn = lambda params: compute_loss(params, batch_inputs, batch_labels)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            # Gradient clipping (example)
            grads = optax.clip_by_global_norm(grad_clip).update(grads, state, None)[0]
            state = state.apply_gradients(grads=grads)
            return state, loss

        @jax.jit
        def compute_accuracy_jax(params, batch_inputs, batch_labels):
            logits = state.apply_fn({"params": params}, batch_inputs)["logits"]
            predictions = jnp.argmax(logits, axis=-1)
            mask = batch_labels != -100
            num_correct = jnp.sum((predictions == batch_labels) * mask)
            num_targets = jnp.sum(mask)
            return num_correct, num_targets

        print(f"\nStarting JAX training for {num_epochs} epochs...")
        num_train_examples = train_inputs_jax.shape[0]
        steps_per_epoch = num_train_examples // batch_size
        global_step = 0

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            rng, epoch_rng = jax.random.split(rng)
            # Shuffle data indices for the epoch
            perm = jax.random.permutation(epoch_rng, num_train_examples)
            shuffled_inputs = train_inputs_jax[perm]
            shuffled_labels = train_labels_jax[perm]

            epoch_loss = 0.0
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")

            for step in pbar:
                try:
                    start_idx = step * batch_size
                    end_idx = start_idx + batch_size
                    batch_inputs = shuffled_inputs[start_idx:end_idx]
                    batch_labels = shuffled_labels[start_idx:end_idx]

                    state, loss = train_step(state, batch_inputs, batch_labels)

                    epoch_loss += loss
                    global_step += 1
                    current_lr = lr_schedule(global_step)  # Get LR from schedule
                    pbar.set_postfix({"Loss": f"{loss:.4f}", "LR": f"{current_lr:.2e}"})

                except Exception as e:
                    print(f"\nError during JAX training step {global_step}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if hasattr(e, "__traceback__"):
                        import traceback

                        print("\nTraceback:")
                        traceback.print_tb(e.__traceback__)
                    return False  # Stop training on error

            avg_epoch_loss = epoch_loss / steps_per_epoch
            print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

            # --- JAX Validation Step --- #
            print(f"Running JAX validation...")
            val_loss = 0.0
            total_correct = 0
            total_targets = 0
            num_val_examples = val_inputs_jax.shape[0]
            val_steps_per_epoch = num_val_examples // batch_size
            for val_step in tqdm(range(val_steps_per_epoch), desc="Validation"):
                val_start_idx = val_step * batch_size
                val_end_idx = val_start_idx + batch_size
                val_batch_inputs = val_inputs_jax[val_start_idx:val_end_idx]
                val_batch_labels = val_labels_jax[val_start_idx:val_end_idx]
                # Use the compute_loss function (no gradients needed)
                # Note: Assumes compute_loss uses state.params, which might not be ideal if BN exists
                # A separate eval_step function using state.apply_fn with train=False is better practice
                val_loss += compute_loss(state.params, val_batch_inputs, val_batch_labels)
                # Compute accuracy
                num_correct, num_targets = compute_accuracy_jax(state.params, val_batch_inputs, val_batch_labels)
                total_correct += num_correct
                total_targets += num_targets

            avg_val_loss = val_loss / val_steps_per_epoch
            val_accuracy = (total_correct / jnp.maximum(total_targets, 1)) * 100.0
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            # --- End JAX Validation Step --- #

        print(f"\nJAX Training finished successfully for {model_name}!")
        return True

    except Exception as e:
        print(f"\nError during JAX model setup or training loop initiation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback

            print("\nTraceback:")
            traceback.print_tb(e.__traceback__)
        return False


def compute_token_level_accuracy(model, loader, device):
    """
    Compute token-level accuracy while ignoring special tokens and target_ignore_idx.
    This is a general metric that considers all valid positions.
    """
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            # Use autocast for evaluation inference
            with amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=inputs, labels=None)  # Don't need labels for inference
                # Handle different output formats to get logits
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                elif isinstance(outputs, (CausalLMOutput, CausalLMOutputWithPast)):
                    logits = outputs.logits
                else:
                    logits = outputs  # Assume raw logits if not a standard output object

                if logits is None:
                    print(
                        f"Warning: Could not extract logits during validation for model {model.__class__.__name__}. Skipping batch."
                    )
                    continue

            predictions = logits.argmax(dim=-1)
            valid_mask = targets != -100
            match = (predictions == targets) & valid_mask
            correct_tokens += match.sum().item()
            total_tokens += valid_mask.sum().item()

    token_acc = 100.0 * correct_tokens / (total_tokens if total_tokens > 0 else 1)
    # print(f"Overall Token-Level Accuracy: {token_acc:.2f}%") # Optional: print inside function
    model.train()  # Set model back to train mode
    return token_acc


def main():
    # Set device
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Common hyperparameters - smaller architecture
    config = {
        "hidden_size": 128,  # Reduced from 256
        "num_heads": 1,  # Reduced from 4
        "num_layers": 2,  # Reduced from 4
        "max_seq_len": 256,  # Set to desired sequence length
        "vocab_size": 128,
        "inter_dim": 512,
    }

    print("Model configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    # Create Associative Recall task datasets
    print("\nGenerating Associative Recall datasets via registry...")
    num_train = 12800
    num_test = num_train // 10  # Using a smaller validation set
    batch_size = 128

    # Use registry to create Torch DataLoaders
    train_loader, val_loader = registry.create_data_loaders(
        task_name="noisy_in_context_recall",
        batch_size=batch_size,
        num_train=num_train,
        num_test=num_test,
        backend="torch",
        device=device,  # device ignored by torch backend, but good practice
        in_memory=True,
        # Task-specific args:
        sequence_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
        random_non_queries=True,
    )
    print("Torch DataLoaders created.")

    # --- JAX Data Conversion (if Spectron JAX is still being tested) ---
    # NOTE: This assumes the registry created datasets accessible via loader.dataset
    if "spectron" in MODEL_REGISTRY.list_models():
        print("\nConverting data to JAX arrays...")
        try:
            # Access dataset objects from loaders
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset

            # Assuming MemoryDataset structure with .inputs and .targets NumPy arrays
            train_inputs_jax = jnp.asarray(train_dataset.inputs)
            train_labels_jax = jnp.asarray(train_dataset.targets)
            val_inputs_jax = jnp.asarray(val_dataset.inputs)
            val_labels_jax = jnp.asarray(val_dataset.targets)

            print("JAX data shapes:")
            print(f"  Train inputs: {train_inputs_jax.shape}, Train labels: {train_labels_jax.shape}")
            print(f"  Val inputs: {val_inputs_jax.shape}, Val labels: {val_labels_jax.shape}")
        except AttributeError as e:
            print(f"Error accessing .inputs/.targets from loader.dataset: {e}")
            print("Cannot convert data for JAX model. Ensure registry creates MemoryDataset.")
            # Optionally remove spectron from models_to_test if conversion fails
        except Exception as e:
            print(f"Error converting data to JAX: {e}")
            # Optionally remove spectron from models_to_test if conversion fails
    # ---------------------------------------------------------------------

    # Print example batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch shapes and devices:")
    print(f"Input shape: {sample_batch[0].shape}, device: {sample_batch[0].device}")
    print(f"Target shape: {sample_batch[1].shape}, device: {sample_batch[1].device}")

    # Test each model in the registry
    results = {}
    # models_to_test = ["spectron"]  # Clear model filter
    for model_name in MODEL_REGISTRY.list_models():
        # Skip problematic models
        if model_name == "deltanet":
            print(f"\n{'=' * 50}")
            print(f"Skipping {model_name} due to known issues...")
            print(f"{'=' * 50}")
            results[model_name] = "Skipped"
            continue

        print(f"\n{'=' * 50}")
        print(f"Testing {model_name}")
        print(f"{'=' * 50}")

        # Get the correct config for the model
        model_specific_config = create_model_config(model_name, config)

        if model_name == "spectron":
            success = train_spectron_model(
                model_name=model_name,
                model_config=model_specific_config,
                train_inputs_jax=train_inputs_jax,
                train_labels_jax=train_labels_jax,
                val_inputs_jax=val_inputs_jax,
                val_labels_jax=val_labels_jax,
                num_epochs=64,
                batch_size=32,
            )
        else:
            # Pass the specific config object to the PyTorch trainer
            success = train_pytorch_model(
                model_name=model_name,
                config_obj=model_specific_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=64,
            )
        results[model_name] = "✓ Passed" if success else "✗ Failed"

    # Print final results
    print("\nTest Results:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"{model_name:20s}: {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()
