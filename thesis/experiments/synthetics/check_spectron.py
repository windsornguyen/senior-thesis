import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taf

import jax
import jax.numpy as jnp
import jax.scipy.signal
import flax.linen as nn_flax
import numpy as np

# Import your associative scan (placeholder) for Torch.
from thesis.experiments.utils.assoc_scan.kernel import associative_scan as associative_scan_torch


def get_hankel(seq_len: int, use_hankel_L: bool = False, device=None) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z.to(device=device)


def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False, device: torch.device = None, dtype=torch.float32
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L, device)
    # Perform eigh on CPU if necessary as it might be more stable
    if device is not None and device.type == "cuda":
        Z_cpu = Z.cpu()
        sigma, phi = torch.linalg.eigh(Z_cpu)
        sigma = sigma.to(device)
        phi = phi.to(device)
    else:
        sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k = phi_k * sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


def get_hankel_matrix_jax(n: int) -> jnp.ndarray:
    i = jnp.arange(1, n + 1)
    j = jnp.arange(1, n + 1)
    I, J = jnp.meshgrid(i, j, indexing="ij")
    return 2 / ((I + J) ** 3 - (I + J))


def get_spectral_filters_jax(n: int, k: int) -> jnp.ndarray:
    eig_vals, eig_vecs = jnp.linalg.eigh(get_hankel_matrix_jax(n))
    # Return only the eigenvectors (the spectral basis) matching the Torch version.
    return eig_vecs[:, -k:]


# --- Torch version ---
class SpectralAttentionTorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.seq_len = config.seq_len
        self.head_dim = config.dim // config.num_heads

        # Spectral basis: shape [seq_len, num_eigh]
        filters = get_spectral_filters(
            config.seq_len, config.num_eigh, config.use_hankel_L, config.device, config.dtype
        )
        self.register_buffer("spectral_basis", filters)

        # Standard Q, K, V projections.
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        # Gate projection: maps [head_dim*head_dim] to [1]
        self.wg = nn.Linear(self.head_dim**2, 1)
        self.eps = 1e-5

    def conv(self, filters: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # keys: [B, num_heads, seq_len, head_dim]
        # Make sure filters and keys are on the same device
        filters = filters.to(keys.device)

        conv1d = lambda f, k: taf.fftconvolve(k, f)[: k.shape[0]]
        fconv = torch.vmap(conv1d, in_dims=(None, 1), out_dims=1)

        def conv_head(filter_vec: torch.Tensor, key_seq: torch.Tensor) -> torch.Tensor:
            return fconv(filter_vec, key_seq)

        hconv = torch.vmap(conv_head, in_dims=(0, 0), out_dims=0)
        filters_T = filters.transpose(0, 1)  # [num_heads, seq_len]
        bconv = torch.vmap(lambda keys_batch: hconv(filters_T, keys_batch), in_dims=0, out_dims=0)
        return bconv(keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        h = self.head_dim

        # Ensure spectral_basis is on the same device as input x
        spectral_basis = self.spectral_basis.to(x.device)

        # Compute Q, K, V projections. (Output shape: [B, H, L, head_dim])
        q = self.wq(x).reshape(B, L, H, h).transpose(1, 2)
        k = self.wk(x).reshape(B, L, H, h).transpose(1, 2) * (h**-0.5)
        v = self.wv(x).reshape(B, L, H, h).transpose(1, 2)

        # Apply spectral filter (conv) to keys and values.
        k_conv = self.conv(spectral_basis, k)
        v_conv = self.conv(spectral_basis, v)

        # Outer product: [B, H, L, h, h]
        Z = torch.einsum("bhlm,bhln->bhlmn", v_conv, k_conv)

        # Gate projection.
        gate_input = Z.flatten(-2, -1)  # [B, H, L, h*h]
        gates_logits = self.wg(gate_input)  # [B, H, L, 1]
        gates = (F.relu(gates_logits) ** 2) + self.eps  # [B, H, L, 1]
        gates = gates.unsqueeze(-1)  # [B, H, L, 1, 1]

        # Gate Z.
        gated_Z = gates * Z  # [B, H, L, h, h]

        # Prepare for associative scan.
        gated_Z = gated_Z.permute(0, 1, 3, 4, 2)  # [B, H, h, h, L]
        gated_Z_flat = gated_Z.reshape(B, H * h * h, L)  # [B, H*h*h, L]
        gates_flat = gates.squeeze(-1).squeeze(-1)  # [B, H, L]
        gates_flat = gates_flat.repeat_interleave(h * h, dim=1)  # [B, H*h*h, L]

        cumul_gated_Z_flat, cumul_gates_flat = associative_scan_torch(gated_Z_flat, gates_flat)
        cumul_gated_Z = cumul_gated_Z_flat.reshape(B, H, h, h, L).permute(0, 1, 4, 2, 3)
        cumul_gates = cumul_gates_flat.reshape(B, H, h, h, L).permute(0, 1, 4, 2, 3)

        attn_weights = cumul_gated_Z / (cumul_gates + self.eps)
        ctxt = torch.einsum("bhsd,bhlde->bhse", q, attn_weights)
        unit_ctxt = F.normalize(ctxt, p=2, dim=3, eps=self.eps)
        output = unit_ctxt.transpose(1, 2).reshape(B, L, D)
        return self.wo(output)


# --- JAX version using Flax ---
class ScanAttentionJAX(nn_flax.Module):
    dim: int
    num_heads: int
    seq_len: int
    spectral_basis: jnp.ndarray  # Expected shape: [seq_len, num_eigh]
    eps: float = 1e-5

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.wq = nn_flax.Dense(self.dim)
        self.wk = nn_flax.Dense(self.dim)
        self.wv = nn_flax.Dense(self.dim)
        self.wo = nn_flax.Dense(self.dim)
        self.gate_proj = nn_flax.Dense(1)
        self.beta = self.param("beta", lambda rng: jnp.array(1.0))

    def conv(self, filters: jnp.ndarray, keys: jnp.ndarray) -> jnp.ndarray:
        # keys: [B, num_heads, seq_len, head_dim]
        def conv1d(f, k):
            return jax.scipy.signal.convolve(k, f, method="fft")[: k.shape[0]]

        conv_over_features = jax.vmap(conv1d, in_axes=(None, 1), out_axes=1)

        def conv_head(filter_seq, key_seq):
            return conv_over_features(filter_seq, key_seq)

        conv_over_heads = jax.vmap(conv_head, in_axes=(0, 0), out_axes=0)
        filters_T = filters.T  # [num_heads, seq_len]
        conv_over_batch = jax.vmap(lambda keys_batch: conv_over_heads(filters_T, keys_batch), in_axes=0, out_axes=0)
        return conv_over_batch(keys)

    def __call__(self, x, training=False):
        B, L, _ = x.shape
        H = self.num_heads
        h = self.head_dim

        q = self.wq(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, H, h).transpose(0, 2, 1, 3) / jnp.sqrt(h)
        v = self.wv(x).reshape(B, L, H, h).transpose(0, 2, 1, 3)

        k_conv = self.conv(self.spectral_basis, k)
        v_conv = self.conv(self.spectral_basis, v)
        Z = jnp.einsum("bhsd,bhse->bhsde", v_conv, k_conv)

        gate_input = Z.reshape(B, H, L, -1)  # [B, H, L, h*h]
        gates_logits = self.gate_proj(gate_input)  # [B, H, L, 1]
        gates = (jax.nn.relu(gates_logits) ** 2) + self.eps  # [B, H, L, 1]
        gates = gates[..., None]  # [B, H, L, 1, 1]

        gated_Z = gates * Z  # [B, H, L, h, h]

        def combine_fn(carry, next_val):
            sum_gated_Z, sum_gates = carry
            next_gated_Z, next_gates = next_val
            return (sum_gated_Z + next_gated_Z, sum_gates + next_gates)

        cumulative_gated_Z, cumulative_gates = jax.lax.associative_scan(combine_fn, (gated_Z, gates), axis=2)
        attn_weights = cumulative_gated_Z / (cumulative_gates + self.eps)
        output_raw = jnp.einsum("bhsd,bhsde->bhse", q, attn_weights)
        output_norm = jnp.linalg.norm(output_raw, axis=3, keepdims=True)
        output_normalized = output_raw / jnp.maximum(output_norm, self.eps)
        output_normalized = output_normalized.transpose(0, 2, 1, 3).reshape(B, L, self.dim)
        return self.wo(output_normalized)


# --- Helper to set all parameters to a constant value ---
def set_torch_weights_to_constant(model, constant=1.0):
    for name, param in model.named_parameters():
        with torch.no_grad():
            param.fill_(constant)


def set_flax_weights_to_constant(params, constant=1.0):
    return jax.tree.map(lambda p: jnp.full_like(p, constant), params)


# --- Main test ---
def main():
    # Fixed config.
    class Config:
        pass

    print("=== STANDARD TEST CASE ===")
    config = Config()
    config.dim = 64
    config.num_heads = 4
    config.seq_len = 16
    config.num_eigh = 4  # For simplicity, match num_heads.
    config.use_hankel_L = False
    config.use_tensordot = False
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.dtype = torch.float32

    # Create fixed dummy input (same for Torch and JAX)
    np.random.seed(0)
    x_np = np.random.randn(2, config.seq_len, config.dim).astype(np.float32)
    x_torch = torch.tensor(x_np, device=config.device)
    x_jax = jnp.array(x_np)

    # --- Torch test ---
    torch_model = SpectralAttentionTorch(config).to(config.device)
    set_torch_weights_to_constant(torch_model, constant=1.0)
    out_torch = torch_model(x_torch)
    print("Torch output shape:", out_torch.shape)
    print("Torch output sample:", out_torch[0, 0, :5].detach().cpu().numpy())

    # --- JAX test ---
    spectral_basis_jax = get_spectral_filters_jax(config.seq_len, config.num_eigh)
    jax_model = ScanAttentionJAX(
        dim=config.dim, num_heads=config.num_heads, seq_len=config.seq_len, spectral_basis=spectral_basis_jax, eps=1e-5
    )
    key = jax.random.PRNGKey(0)
    variables = jax_model.init(key, x_jax)
    # Set all parameters to constant 1.0.
    params = variables["params"]
    params = set_flax_weights_to_constant(params, constant=1.0)
    variables = {"params": params}
    variables = nn_flax.FrozenDict(variables)
    out_jax = jax_model.apply(variables, x_jax)
    print("JAX output shape:", out_jax.shape)
    print("JAX output sample:", np.array(out_jax[0, 0, :5]))

    # Challenging test cases
    test_cases = [
        {"name": "LARGER BATCH", "batch": 8, "seq_len": 16, "dim": 64, "heads": 4, "eigh": 4},
        {"name": "LONGER SEQUENCE", "batch": 2, "seq_len": 128, "dim": 64, "heads": 4, "eigh": 4},
        {"name": "MORE HEADS", "batch": 2, "seq_len": 16, "dim": 128, "heads": 16, "eigh": 16},
        {"name": "HIGH DIM", "batch": 2, "seq_len": 16, "dim": 256, "heads": 8, "eigh": 8},
        {"name": "EXTREME CASE", "batch": 4, "seq_len": 64, "dim": 128, "heads": 16, "eigh": 16},
        {"name": "SUPER EXTREME", "batch": 8, "seq_len": 256, "dim": 256, "heads": 32, "eigh": 32},
    ]

    import time

    results = []

    for case in test_cases:
        try:
            # Skip super extreme case if not enough GPU memory
            if case["name"] == "SUPER EXTREME" and torch.cuda.is_available():
                # Get available GPU memory in MB
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_memory_mb = free_memory / (1024 * 1024)

                # Estimate memory requirement (rough estimate)
                # Each test needs about batch * seq_len * dim * num_heads * 4 (float32) * ~10 (for temp tensors)
                required_mb = case["batch"] * case["seq_len"] * case["dim"] * case["heads"] * 4 * 10 / (1024 * 1024)

                if free_memory_mb < required_mb * 1.5:  # Add 50% safety margin
                    print(
                        f"\n=== SKIPPING {case['name']} (REQUIRES ~{required_mb:.0f}MB, ONLY {free_memory_mb:.0f}MB AVAILABLE) ==="
                    )
                    continue

            print(f"\n=== {case['name']} ===")
            config = Config()
            config.dim = case["dim"]
            config.num_heads = case["heads"]
            config.seq_len = case["seq_len"]
            config.num_eigh = case["eigh"]
            config.use_hankel_L = False
            config.use_tensordot = False
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config.dtype = torch.float32

            # Create new input
            x_np = np.random.randn(case["batch"], config.seq_len, config.dim).astype(np.float32)
            x_torch = torch.tensor(x_np, device=config.device)
            x_jax = jnp.array(x_np)

            # Torch test
            torch_model = SpectralAttentionTorch(config).to(config.device)
            set_torch_weights_to_constant(torch_model, constant=1.0)

            # Warmup
            _ = torch_model(x_torch)
            torch.cuda.synchronize()

            # Timing
            start_time = time.time()
            out_torch = torch_model(x_torch)
            torch.cuda.synchronize()
            torch_time = time.time() - start_time

            print(f"Torch output shape: {out_torch.shape}, Time: {torch_time:.6f} sec")
            print(f"Torch output mean: {out_torch.mean().item()}")

            # JAX test
            spectral_basis_jax = get_spectral_filters_jax(config.seq_len, config.num_eigh)
            jax_model = ScanAttentionJAX(
                dim=config.dim,
                num_heads=config.num_heads,
                seq_len=config.seq_len,
                spectral_basis=spectral_basis_jax,
                eps=1e-5,
            )
            key = jax.random.PRNGKey(0)
            variables = jax_model.init(key, x_jax)
            # Set all parameters to constant 1.0.
            params = variables["params"]
            params = set_flax_weights_to_constant(params, constant=1.0)
            variables = {"params": params}
            variables = nn_flax.FrozenDict(variables)

            # Warmup
            _ = jax_model.apply(variables, x_jax)

            # Timing
            start_time = time.time()
            out_jax = jax_model.apply(variables, x_jax)
            jax_time = time.time() - start_time

            print(f"JAX output shape: {out_jax.shape}, Time: {jax_time:.6f} sec")
            print(f"JAX output mean: {float(jnp.mean(out_jax))}")

            # Store results
            results.append(
                {
                    "case": case["name"],
                    "torch_time": torch_time,
                    "jax_time": jax_time,
                    "torch_shape": list(out_torch.shape),
                    "jax_shape": list(out_jax.shape),
                    "torch_mean": float(out_torch.mean().item()),
                    "jax_mean": float(jnp.mean(out_jax)),
                }
            )

            # Clean up to free memory
            del torch_model, out_torch, jax_model, out_jax, x_torch, x_jax
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in {case['name']}: {str(e)}")

    # Print summary table
    print("\n=== SUMMARY ===")
    if results:
        print(
            f"{'Case':<15} {'Torch Time':<12} {'JAX Time':<12} {'Speed Ratio':<12} {'Torch Mean':<12} {'JAX Mean':<12}"
        )
        for result in results:
            speed_ratio = result["torch_time"] / result["jax_time"] if result["jax_time"] > 0 else float("inf")
            print(
                f"{result['case']:<15} {result['torch_time']:<12.6f} {result['jax_time']:<12.6f} {speed_ratio:<12.2f} {result['torch_mean']:<12.6f} {result['jax_mean']:<12.6f}"
            )
    else:
        print("No successful test results to display.")


if __name__ == "__main__":
    main()
