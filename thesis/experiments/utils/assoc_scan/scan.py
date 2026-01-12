import jax
import jax.numpy as jnp
import torch
import numpy as np


def combine_fn(carry, next_state):
    m_x, s_x, n_x, Z_x, g_x = carry
    m_y, s_y, n_y, Z_y, g_y = next_state
    m_new = jnp.maximum(m_x, m_y)
    exp_x = jnp.exp(m_x - m_new)
    exp_y = jnp.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return (m_new, s_new, n_new, Z_new, g_new), None


def scan_fn(qk_slice, v_slice, Z_slice, g_slice):
    leaves = (
        qk_slice,  # [L]
        jnp.ones_like(qk_slice),  # [L]
        v_slice,  # [L, h]
        Z_slice,  # [L, h, h]
        g_slice,  # [L]
    )

    # Transpose based on tensor dimension
    def transpose_leaves(x):
        if x.ndim == 1:
            return x  # No transpose for 1D
        elif x.ndim == 2:
            return jnp.transpose(x, (1, 0))  # [L, h] -> [h, L]
        else:
            return jnp.transpose(x, (2, 0, 1))  # [L, h, h] -> [h, h, L]

    leaves = tuple(transpose_leaves(x) for x in leaves)
    result, _ = jax.lax.associative_scan(combine_fn, leaves, axis=-1)

    # Reverse transpose
    def reverse_transpose(x):
        if x.ndim == 1:
            return x
        elif x.ndim == 2:
            return jnp.transpose(x, (1, 0))
        else:
            return jnp.transpose(x, (1, 2, 0))

    return tuple(reverse_transpose(x) for x in result)


def batched_scan_fn(sim, v, gated_Z, gates_z):
    B, H, L = sim.shape
    sim_flat = sim.reshape(B * H, L)
    v_flat = v.reshape(B * H, L, v.shape[-1])
    gated_Z_flat = gated_Z.reshape(B * H, L, gated_Z.shape[-2], gated_Z.shape[-1])
    gates_z_flat = gates_z.reshape(B * H, L)

    scan_all = jax.vmap(scan_fn, in_axes=(0, 0, 0, 0), out_axes=0)
    max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = scan_all(sim_flat, v_flat, gated_Z_flat, gates_z_flat)

    return (
        max_cumul.reshape(B, H, L),
        norm_cumul.reshape(B, H, L),
        v_cumul.reshape(B, H, L, v.shape[-1]),
        Z_cumul.reshape(B, H, L, gated_Z.shape[-2], gated_Z.shape[-1]),
        gate_cumul.reshape(B, H, L),
    )


# Test data
B, H, L, h = 2, 4, 8, 16
torch.manual_seed(0)
q = torch.randn(B, H, L, h)
k = torch.randn(B, H, L, h)
q, k = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1)
sim = torch.einsum("bhld,bhld->bhl", q, k) * (h**-0.5)  # [B, H, L]
v = torch.randn(B, H, L, h)
Z = torch.einsum("bhsn,bhsp->bhspn", k, v)
gates_z = torch.randn(B, H, L)

# Convert to JAX
sim_jax = jnp.array(sim.numpy())
v_jax = jnp.array(v.numpy())
Z_jax = jnp.array(Z.numpy())
gates_z_jax = jnp.array(gates_z.numpy())

# Run scan
max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = batched_scan_fn(sim_jax, v_jax, Z_jax, gates_z_jax)

# Precompute softmax
softmax_sim = torch.softmax(sim, dim=-1)  # [B, H, L]
cumsum_softmax = torch.cumsum(softmax_sim, dim=-1)  # [B, H, L]

# Compare
norm_cumul_torch = torch.tensor(np.array(norm_cumul))
print("norm_cumul (scan):", norm_cumul_torch)
print("cumsum(softmax(sim)):", cumsum_softmax)
print("Max absolute difference:", torch.max(torch.abs(norm_cumul_torch - cumsum_softmax)).item())
