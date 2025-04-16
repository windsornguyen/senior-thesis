import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torchaudio
import jax
import jax.numpy as jnp
import flax.linen as flax_nn
import numpy as np
from jax.scipy.signal import convolve
from thesis.experiments.utils.assoc_scan import associative_scan


def check(x):
    print(f"x | shape {x.shape} |")


# PyTorch Implementation
def conv_torch(filters: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    def conv1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.fftconvolve(x, y)[: x.shape[0]]

    conv_over_features = torch.vmap(conv1d, in_dims=(None, 1), out_dims=1)
    conv_over_heads = torch.vmap(lambda f, k: conv_over_features(f, k), in_dims=(0, 0), out_dims=0)
    filters_T = filters.T
    conv_over_batch = torch.vmap(lambda keys_batch: conv_over_heads(filters_T, keys_batch), in_dims=0, out_dims=0)
    return conv_over_batch(keys)


class ScanAttentionTorch(torch_nn.Module):
    def __init__(self, dim, num_heads, seq_len, spectral_basis, eps=1e-5):
        super(ScanAttentionTorch, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.spectral_basis = spectral_basis  # Already moved to cuda below!
        self.eps = eps
        self.head_dim = dim // num_heads

        self.wq = torch_nn.Linear(dim, dim)
        self.wk = torch_nn.Linear(dim, dim)
        self.wv = torch_nn.Linear(dim, dim)
        self.wo = torch_nn.Linear(dim, dim)
        self.gate_proj = torch_nn.Linear(dim * dim, 1)

        self.beta = torch_nn.Parameter(torch.tensor(1.0))
        self.alpha = torch_nn.Parameter(torch.tensor(1.0))
        self.scaleup = torch_nn.Parameter(torch.ones(1, num_heads, 1, dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        h = self.head_dim
        q = self.wq(x).view(B, L, H, h).transpose(1, 2)
        k = self.wk(x).view(B, L, H, h).transpose(1, 2)
        v = self.wv(x).view(B, L, H, h).transpose(1, 2)

        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        # Convolve with spectral basis (which is on cuda)
        k = conv_torch(self.spectral_basis, k)
        v = conv_torch(self.spectral_basis, v)

        Z = torch.einsum("bhsd,bhse->bhsde", v, k) * self.scaleup
        gate_input = Z.view(*Z.shape[:3], -1)
        gates_logits = self.gate_proj(gate_input)
        gates = F.relu(gates_logits) ** 2 + self.eps
        gates = gates.unsqueeze(-1)

        # Multiply the original Z by the gates.
        gated_Z = gates * Z  # [B, H, L, h, h]

        # Prepare for associative scan
        gated_Z = gated_Z.permute(0, 1, 3, 4, 2)  # [B, H, h, h, L]
        gated_Z_flat = gated_Z.reshape(B, H * h * h, L)  # [B, H * h * h, L]

        gates_flat = gates.squeeze(-1).squeeze(-1)  # now [B, H, L]
        gates_flat = gates_flat.repeat_interleave(h * h, dim=1)  #  [B, H * h * h, L]

        cumul_gated_Z_flat, cumul_gates_flat = associative_scan(gated_Z_flat, gates_flat)

        cumul_gated_Z = cumul_gated_Z_flat.reshape(B, H, h, h, L).permute(0, 1, 4, 2, 3)
        cumul_gates = cumul_gates_flat.reshape(B, H, h, h, L).permute(0, 1, 4, 2, 3)

        attn_weights = cumul_gated_Z / (cumul_gates + self.eps)

        ctxt = torch.einsum("bhli,bhldo->bhlo", q, attn_weights)
        unit_ctxt = F.normalize(ctxt, p=2, dim=3, eps=self.eps)
        output = unit_ctxt.transpose(1, 2).reshape(B, L, D)
        return self.wo(output)


# JAX Implementation
def normalize_jax(x, axis=-1):
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True)


@jax.jit
def causal_conv(filters: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    conv_1d = lambda x, f: convolve(x, f, mode="full")[: x.shape[0]]
    conv_filters = lambda x: jax.vmap(conv_1d, in_axes=(None, 1), out_axes=1)(x, filters)
    conv_channels = lambda x: jax.vmap(conv_filters, in_axes=1, out_axes=2)(x)
    conv_heads = lambda x: jax.vmap(conv_channels, in_axes=0, out_axes=0)(x)
    conv_batch = lambda x: jax.vmap(conv_heads, in_axes=0, out_axes=0)(x)
    output = conv_batch(inputs)
    B, H, S, K, D_per_H = output.shape
    return output.reshape(B, H, S, K * D_per_H)


class ScanAttentionJax(flax_nn.Module):
    dim: int
    num_heads: int
    seq_len: int
    spectral_basis: jnp.ndarray
    eps: float = 1e-5

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.wq = flax_nn.Dense(self.dim)
        self.wk = flax_nn.Dense(self.dim)
        self.wv = flax_nn.Dense(self.dim)
        self.wo = flax_nn.Dense(self.dim)
        self.gate_proj = flax_nn.Dense(1)
        self.scaleup = self.param("scaleup", lambda rng: jnp.ones((1, self.num_heads, 1, self.dim, self.dim)))

    def __call__(self, x, training=False):
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        k = normalize_jax(k, axis=-1)
        v = normalize_jax(v, axis=-1)

        # The spectral_basis is assumed to already be on the correct device (GPU)
        k = causal_conv(self.spectral_basis, k)
        v = causal_conv(self.spectral_basis, v)

        Z = jnp.einsum("bhsd,bhse->bhsde", v, k) * self.scaleup
        gate_input = Z.reshape(*Z.shape[:3], -1)
        gates_logits = self.gate_proj(gate_input)
        gates = jax.nn.relu(gates_logits) ** 2 + self.eps
        gates = gates[..., None]

        gated_Z = gates * Z

        def combine_fn(carry, next_val):
            sum_gated_Z, sum_gates = carry
            next_gated_Z, next_gates = next_val
            return (sum_gated_Z + next_gated_Z, sum_gates + next_gates)

        cumulative_gated_Z, cumulative_gates = jax.lax.associative_scan(combine_fn, (gated_Z, gates), axis=2)
        attn_weights = cumulative_gated_Z / (cumulative_gates + self.eps)

        output_raw = jnp.einsum("bhsd,bhsde->bhse", q, attn_weights)
        output_norm = normalize_jax(output_raw, axis=-1)
        output_normalized = output_norm.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        return self.wo(output_normalized)


# Determine CUDA device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example inputs and weight sharing
batch_size = 2
seq_len = 256
dim = 64
num_heads = 1
K = num_heads

# Generate random input tensor and spectral basis
x_np = np.random.rand(batch_size, seq_len, dim).astype(np.float32)
spectral_basis_np = np.random.rand(seq_len, K).astype(np.float32)

# PyTorch: move tensors to CUDA
x_torch = torch.from_numpy(x_np).to(device)
spectral_basis_torch = torch.from_numpy(spectral_basis_np).to(device)

# JAX: put inputs and spectral basis on GPU if available
gpu_devices = jax.devices("gpu")
gpu_device = gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]
x_jax = jax.device_put(jnp.array(x_np), device=gpu_device)
spectral_basis_jax = jax.device_put(jnp.array(spectral_basis_np), device=gpu_device)

# Initialize PyTorch model and move it to CUDA
model_torch = ScanAttentionTorch(dim=dim, num_heads=num_heads, seq_len=seq_len, spectral_basis=spectral_basis_torch)
model_torch.to(device)

# Extract PyTorch weights (they reside on CUDA, but we convert them to numpy for sharing)
torch_weights = {
    "wq": {
        "kernel": model_torch.wq.weight.data.cpu().numpy(),  # move to CPU before converting
        "bias": model_torch.wq.bias.data.cpu().numpy(),
    },
    "wk": {"kernel": model_torch.wk.weight.data.cpu().numpy(), "bias": model_torch.wk.bias.data.cpu().numpy()},
    "wv": {"kernel": model_torch.wv.weight.data.cpu().numpy(), "bias": model_torch.wv.bias.data.cpu().numpy()},
    "wo": {"kernel": model_torch.wo.weight.data.cpu().numpy(), "bias": model_torch.wo.bias.data.cpu().numpy()},
    "gate_proj": {
        "kernel": model_torch.gate_proj.weight.data.cpu().numpy(),  # Shape: [1, dim * dim]
        "bias": model_torch.gate_proj.bias.data.cpu().numpy(),
    },
    "beta": model_torch.beta.data.cpu().numpy(),
    "alpha": model_torch.alpha.data.cpu().numpy(),
    "scaleup": model_torch.scaleup.data.cpu().numpy(),
}

# Convert to JAX-compatible params structure (Flax Dense layers expect kernels to be transposed)
jax_params = {
    "params": {
        "wq": {"kernel": jnp.array(torch_weights["wq"]["kernel"].T), "bias": jnp.array(torch_weights["wq"]["bias"])},
        "wk": {"kernel": jnp.array(torch_weights["wk"]["kernel"].T), "bias": jnp.array(torch_weights["wk"]["bias"])},
        "wv": {"kernel": jnp.array(torch_weights["wv"]["kernel"].T), "bias": jnp.array(torch_weights["wv"]["bias"])},
        "wo": {"kernel": jnp.array(torch_weights["wo"]["kernel"].T), "bias": jnp.array(torch_weights["wo"]["bias"])},
        "gate_proj": {
            "kernel": jnp.array(torch_weights["gate_proj"]["kernel"].T),
            "bias": jnp.array(torch_weights["gate_proj"]["bias"]),
        },
        "beta": jnp.array(torch_weights["beta"]),
        "alpha": jnp.array(torch_weights["alpha"]),
        "scaleup": jnp.array(torch_weights["scaleup"]),
    }
}
# Optionally, move the params to the GPU device
jax_params = jax.device_put(jax_params, device=gpu_device)

# Initialize JAX model
model_jax = ScanAttentionJax(dim=dim, num_heads=num_heads, seq_len=seq_len, spectral_basis=spectral_basis_jax)

# Forward pass with shared weights
output_torch = model_torch(x_torch)
output_jax = model_jax.apply(jax_params, x_jax)

# Convert JAX output to PyTorch for comparison (no need to move output_jax to host—the conversion does that)
output_jax_torch = torch.from_numpy(np.array(output_jax))

# Check if outputs are close
result = torch.allclose(output_jax_torch.to(device), output_torch, rtol=1e-3, atol=1e-5)
print(f"Outputs are close: {result}")

# Check shapes and a sample of outputs
check(output_torch)
check(output_jax)
print(f"Sample output_torch: {output_torch[0, 0, :5]}")
print(f"Sample output_jax: {output_jax[0, 0, :5]}")
