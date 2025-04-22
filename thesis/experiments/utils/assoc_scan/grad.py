import jax
import jax.numpy as jnp
import torch
import numpy as np

"""
Gradient verification script for the attention combine function.

Verified Derivatives (PyTorch vs JAX):
- dm_new/dm_x
- dm_new/dm_y
- ds_new/dm_x
- ds_new/dm_y
- ds_new/ds_x
- ds_new/ds_y
- dn_new/dm_x
- dn_new/dn_x
- dn_new/dm_y
- dn_new/dn_y

All required derivatives have been successfully verified.
"""


# JAX version of the attention combine function (Only m_new needed)
def attention_combine_fn_jax(h_prev, x_curr):
    """JAX version of the attention combine function.
    Computes m_new, s_new, and n_new for grad check.
    """
    m_x, s_x, n_x, *_ = h_prev  # Need m_x, s_x, n_x
    m_y, s_y, n_y, *_ = x_curr  # Need m_y, s_y, n_y

    # Compute new maximum
    m_new = jnp.maximum(m_x, m_y)

    # Scale factors
    exp_x = jnp.exp(m_x - m_new)
    exp_y = jnp.exp(m_y - m_new)

    # Update softmax components
    s_new = s_x * exp_x + s_y * exp_y

    # Uncommented: Calculate n_new
    exp_x_broadcast = jnp.expand_dims(exp_x, axis=1)
    exp_y_broadcast = jnp.expand_dims(exp_y, axis=1)
    n_new = n_x * exp_x_broadcast + n_y * exp_y_broadcast

    # Return m_new, s_new, and n_new
    return m_new, s_new, n_new


# PyTorch Jacobian computation (copied and adapted)
# Simplified to only compute derivatives for m_new w.r.t m_x
def compute_dm_new_dm_x_torch(h_prev, x_curr):
    """Computes d(m_new) / d(m_x).

    Formula: 1.0 if m_x >= m_y else 0.0
         d(max(m_x, m_y)) / d(m_x)
    """
    m_x, *_ = h_prev
    m_y, *_ = x_curr

    # Recompute intermediates needed for m_new derivatives
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )

    # Derivative dm_new / dm_x
    dm_new_dm_x = torch.where(m_x_safe >= m_y_safe, 1.0, 0.0).to(m_x.dtype)

    return dm_new_dm_x


# PyTorch Jacobian computation
# Simplified to only compute derivatives for m_new w.r.t m_y
def compute_dm_new_dm_y_torch(h_prev, x_curr):
    """Computes d(m_new) / d(m_y).

    Formula: 1.0 if m_x < m_y else 0.0
         = d(max(m_x, m_y)) / d(m_y)
    """
    m_x, *_ = h_prev
    m_y, *_ = x_curr

    # Recompute intermediates needed for m_new derivatives
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )

    # Derivative dm_new / dm_y
    dm_new_dm_y = torch.where(m_x_safe < m_y_safe, 1.0, 0.0).to(m_y.dtype)

    return dm_new_dm_y


# PyTorch Jacobian computation
# Simplified to only compute derivatives for m_new w.r.t m_x
def compute_ds_new_dm_x_torch(h_prev, x_curr):
    """Computes d(s_new) / d(m_x).

    s_new = s_x * exp(m_x - m_new) + s_y * exp(m_y - m_new)
    Formula: d(s_new)/d(m_x) = s_x*d(exp_x)/d(m_x) + s_y*d(exp_y)/d(m_x)
        d(exp_x)/d(m_x) = exp_x * (1 - d(m_new)/d(m_x))
        d(exp_y)/d(m_x) = exp_y * (0 - d(m_new)/d(m_x))
    """
    m_x, s_x, *_ = h_prev
    m_y, s_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    exp_x = torch.exp(m_x_safe - m_new_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe)
    dm_new_dm_x = torch.where(m_x_safe >= m_y_safe, 1.0, 0.0).to(m_x.dtype)
    dexp_x_dm_x = exp_x * (1.0 - dm_new_dm_x)
    dexp_y_dm_x = exp_y * (-dm_new_dm_x)
    ds_new_dm_x = s_x * dexp_x_dm_x + s_y * dexp_y_dm_x
    return ds_new_dm_x


# PyTorch Jacobian computation
# Simplified to only compute derivatives for m_new w.r.t m_y
def compute_ds_new_dm_y_torch(h_prev, x_curr):
    """Computes d(s_new) / d(m_y).

    s_new = s_x * exp(m_x - m_new) + s_y * exp(m_y - m_new)
    Formula: d(s_new)/d(m_y) = s_x*d(exp_x)/d(m_y) + s_y*d(exp_y)/d(m_y)
        d(exp_x)/d(m_y) = exp_x * (0 - d(m_new)/d(m_y))
        d(exp_y)/d(m_y) = exp_y * (1 - d(m_new)/d(m_y))
    """
    m_x, s_x, *_ = h_prev
    m_y, s_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    exp_x = torch.exp(m_x_safe - m_new_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe)
    dm_new_dm_y = torch.where(m_x_safe < m_y_safe, 1.0, 0.0).to(m_y.dtype)
    dexp_x_dm_y = exp_x * (-dm_new_dm_y)
    dexp_y_dm_y = exp_y * (1.0 - dm_new_dm_y)
    ds_new_dm_y = s_x * dexp_x_dm_y + s_y * dexp_y_dm_y
    return ds_new_dm_y


def compute_ds_new_ds_x_torch(h_prev, x_curr):
    """Computes d(s_new) / d(s_x).

    s_new = s_x * exp(m_x - m_new) + s_y * exp(m_y - m_new)
    Formula: d(s_new)/d(s_x) = exp(m_x - m_new)
    """
    m_x, s_x, *_ = h_prev
    m_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    exp_x = torch.exp(m_x_safe - m_new_safe).to(s_x.dtype)  # Ensure dtype matches s_x
    return exp_x


def compute_ds_new_ds_y_torch(h_prev, x_curr):
    """Computes d(s_new) / d(s_y).

    s_new = s_x * exp(m_x - m_new) + s_y * exp(m_y - m_new)
    Formula: d(s_new)/d(s_y) = exp(m_y - m_new)
    """
    m_x, *_ = h_prev
    m_y, s_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe).to(s_y.dtype)  # Ensure dtype matches s_y
    return exp_y


def compute_dn_new_dm_x_torch(h_prev, x_curr):
    """Computes d(n_new) / d(m_x).

    n_new = n_x * broadcast(exp(m_x - m_new)) + n_y * broadcast(exp(m_y - m_new))
    Formula: d(n_new)/d(m_x) = n_x*d(broadcast(exp_x))/d(m_x) + n_y*d(broadcast(exp_y))/d(m_x)
        Uses same d(exp_x)/d(m_x) and d(exp_y)/d(m_x) as ds_new/dm_x, requires broadcasting.
    """
    m_x, _, n_x, *_ = h_prev
    m_y, _, n_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)

    exp_x = torch.exp(m_x_safe - m_new_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe)
    dm_new_dm_x = torch.where(m_x_safe >= m_y_safe, 1.0, 0.0).to(m_x.dtype)

    dexp_x_dm_x = exp_x * (1.0 - dm_new_dm_x)
    dexp_y_dm_x = exp_y * (-dm_new_dm_x)

    # Broadcast derivatives of exp to match n shape (B, D_n, D_m)
    dexp_x_dm_x_b = dexp_x_dm_x.unsqueeze(1)
    dexp_y_dm_x_b = dexp_y_dm_x.unsqueeze(1)

    dn_new_dm_x = n_x * dexp_x_dm_x_b + n_y * dexp_y_dm_x_b
    return dn_new_dm_x.to(n_x.dtype)


def compute_dn_new_dn_x_torch(h_prev, x_curr):
    """Computes d(n_new) / d(n_x).

    n_new = n_x * broadcast(exp(m_x - m_new)) + n_y * broadcast(exp(m_y - m_new))
    Formula: d(n_new)/d(n_x) = broadcast(exp(m_x - m_new))
    """
    m_x, _, n_x, *_ = h_prev
    m_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)

    exp_x = torch.exp(m_x_safe - m_new_safe)

    # Broadcast exp_x and expand to match n shape (B, D_n, D_m)
    dn_new_dn_x = exp_x.unsqueeze(1).expand(-1, n_x.shape[1], -1)
    return dn_new_dn_x.to(n_x.dtype)


def compute_dn_new_dm_y_torch(h_prev, x_curr):
    """Computes d(n_new) / d(m_y).

    n_new = n_x * broadcast(exp(m_x - m_new)) + n_y * broadcast(exp(m_y - m_new))
    Formula: d(n_new)/d(m_y) = n_x*d(broadcast(exp_x))/d(m_y) + n_y*d(broadcast(exp_y))/d(m_y)
        Uses same d(exp_x)/d(m_y) and d(exp_y)/d(m_y) as ds_new/dm_y, requires broadcasting.
    """
    m_x, _, n_x, *_ = h_prev
    m_y, _, n_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)

    exp_x = torch.exp(m_x_safe - m_new_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe)
    dm_new_dm_y = torch.where(m_x_safe < m_y_safe, 1.0, 0.0).to(m_y.dtype)

    dexp_x_dm_y = exp_x * (-dm_new_dm_y)
    dexp_y_dm_y = exp_y * (1.0 - dm_new_dm_y)

    # Broadcast derivatives of exp to match n shape (B, D_n, D_m)
    dexp_x_dm_y_b = dexp_x_dm_y.unsqueeze(1)
    dexp_y_dm_y_b = dexp_y_dm_y.unsqueeze(1)

    dn_new_dm_y = n_x * dexp_x_dm_y_b + n_y * dexp_y_dm_y_b
    return dn_new_dm_y.to(n_x.dtype)


def compute_dn_new_dn_y_torch(h_prev, x_curr):
    """Computes d(n_new) / d(n_y).

    n_new = n_x * broadcast(exp(m_x - m_new)) + n_y * broadcast(exp(m_y - m_new))
    Formula: d(n_new)/d(n_y) = broadcast(exp(m_y - m_new))
    """
    m_x, *_ = h_prev
    m_y, _, n_y, *_ = x_curr
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)

    exp_y = torch.exp(m_y_safe - m_new_safe)

    # Broadcast exp_y and expand to match n shape (B, D_n, D_m)
    dn_new_dn_y = exp_y.unsqueeze(1).expand(-1, n_y.shape[1], -1)
    return dn_new_dn_y.to(n_y.dtype)


# --- Helper functions for constructing full Jacobians ---


def make_diagonal_jacobian(diag_tensor, B, D_out, D_in):
    """Creates a diagonal Jacobian (B, D_out, B, D_in) from a diagonal (B, D)."""
    assert D_out == D_in, "Diagonal Jacobian requires D_out == D_in"
    assert diag_tensor.shape == (B, D_out), f"Input shape mismatch: {diag_tensor.shape}"
    jac = torch.zeros(
        B, D_out, B, D_in, device=diag_tensor.device, dtype=diag_tensor.dtype
    )
    b_idx, d_idx = torch.meshgrid(torch.arange(B), torch.arange(D_out), indexing="ij")
    jac[b_idx, d_idx, b_idx, d_idx] = diag_tensor
    return jac


def make_broadcast_jacobian_n_m(deriv_tensor, B, D_n, D_m):
    """Creates Jacobian (B, D_n, D_m, B, D_m) from (B, D_n, D_m)."""
    assert deriv_tensor.shape == (B, D_n, D_m), (
        f"Input shape mismatch: {deriv_tensor.shape}"
    )
    jac = torch.zeros(
        B, D_n, D_m, B, D_m, device=deriv_tensor.device, dtype=deriv_tensor.dtype
    )
    b_idx, i_idx, j_idx = torch.meshgrid(
        torch.arange(B), torch.arange(D_n), torch.arange(D_m), indexing="ij"
    )
    jac[b_idx, i_idx, j_idx, b_idx, j_idx] = deriv_tensor
    return jac


def make_broadcast_jacobian_n_n(deriv_tensor, B, D_n, D_m):
    """Creates Jacobian (B, D_n, D_m, B, D_n, D_m) from (B, D_n, D_m)."""
    assert deriv_tensor.shape == (B, D_n, D_m), (
        f"Input shape mismatch: {deriv_tensor.shape}"
    )
    jac = torch.zeros(
        B, D_n, D_m, B, D_n, D_m, device=deriv_tensor.device, dtype=deriv_tensor.dtype
    )
    b_idx, i_idx, j_idx = torch.meshgrid(
        torch.arange(B), torch.arange(D_n), torch.arange(D_m), indexing="ij"
    )
    jac[b_idx, i_idx, j_idx, b_idx, i_idx, j_idx] = deriv_tensor
    return jac


def make_broadcast_jacobian_s_m(deriv_tensor, B, D_s, D_m):
    """Creates Jacobian (B, D_s, B, D_m) from (B, D_s). Assumes D_s == D_m."""
    assert D_s == D_m, "This helper assumes D_s == D_m for ds/dm derivatives"
    assert deriv_tensor.shape == (B, D_s), f"Input shape mismatch: {deriv_tensor.shape}"
    jac = torch.zeros(
        B, D_s, B, D_m, device=deriv_tensor.device, dtype=deriv_tensor.dtype
    )
    b_idx, d_idx = torch.meshgrid(torch.arange(B), torch.arange(D_s), indexing="ij")
    jac[b_idx, d_idx, b_idx, d_idx] = deriv_tensor  # Place on diagonal as D_s == D_m
    return jac


# --- Full PyTorch function with Jacobian calculation ---


def attention_combine_fn_torch_full(h_prev, x_curr):
    """PyTorch version computing outputs and full Jacobians."""
    m_x, s_x, n_x, *_ = h_prev
    m_y, s_y, n_y, *_ = x_curr

    B, D_m = m_x.shape
    _, D_s = s_x.shape
    _, D_n, D_n_extra_kv = n_x.shape
    assert D_s == D_m, "Code assumes D_s == D_m"
    assert D_n_extra_kv == D_m, "Code assumes D_n_extra_kv == D_m"

    # --- Forward Pass ---
    m_x_safe = torch.nan_to_num(
        m_x, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_y_safe = torch.nan_to_num(
        m_y, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
    )
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    # Use safe m_new for intermediates, but return original-like m_new
    m_new = torch.maximum(m_x, m_y)

    exp_x = torch.exp(m_x_safe - m_new_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe)

    s_new = s_x * exp_x + s_y * exp_y

    exp_x_broadcast = exp_x.unsqueeze(1)
    exp_y_broadcast = exp_y.unsqueeze(1)
    n_new = n_x * exp_x_broadcast + n_y * exp_y_broadcast

    outputs = (m_new, s_new, n_new)

    # --- Backward Pass (Compute individual derivatives) ---
    dm_new_dm_x = compute_dm_new_dm_x_torch(h_prev, x_curr)
    dm_new_dm_y = compute_dm_new_dm_y_torch(h_prev, x_curr)
    ds_new_dm_x = compute_ds_new_dm_x_torch(h_prev, x_curr)
    ds_new_ds_x = compute_ds_new_ds_x_torch(h_prev, x_curr)
    ds_new_dm_y = compute_ds_new_dm_y_torch(h_prev, x_curr)
    ds_new_ds_y = compute_ds_new_ds_y_torch(h_prev, x_curr)
    dn_new_dm_x = compute_dn_new_dm_x_torch(h_prev, x_curr)
    dn_new_dn_x = compute_dn_new_dn_x_torch(h_prev, x_curr)
    dn_new_dm_y = compute_dn_new_dm_y_torch(h_prev, x_curr)
    dn_new_dn_y = compute_dn_new_dn_y_torch(h_prev, x_curr)

    # --- Assemble Jacobians ---

    # Jacobian w.r.t h_prev = (m_x, s_x, n_x, Z_x, g_x)
    # We only compute non-zero for (m_x, s_x, n_x)

    # Derivatives of m_new w.r.t (m_x, s_x, n_x)
    jac_m_h_m = make_diagonal_jacobian(dm_new_dm_x, B, D_m, D_m)
    jac_m_h_s = torch.zeros(
        B, D_m, B, D_s, device=m_x.device, dtype=m_x.dtype
    )  # dm/dsx = 0
    jac_m_h_n = torch.zeros(
        B, D_m, B, D_n, D_n_extra_kv, device=m_x.device, dtype=m_x.dtype
    )  # dm/dnx = 0
    jac_m_h_Z = torch.zeros(
        B, D_m, B, dummy_Z.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_m_h_g = torch.zeros(
        B, D_m, B, dummy_g.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_m_h = (jac_m_h_m, jac_m_h_s, jac_m_h_n, jac_m_h_Z, jac_m_h_g)

    # Derivatives of s_new w.r.t (m_x, s_x, n_x)
    jac_s_h_m = make_broadcast_jacobian_s_m(ds_new_dm_x, B, D_s, D_m)
    jac_s_h_s = make_diagonal_jacobian(ds_new_ds_x, B, D_s, D_s)
    jac_s_h_n = torch.zeros(
        B, D_s, B, D_n, D_n_extra_kv, device=m_x.device, dtype=m_x.dtype
    )  # ds/dnx = 0
    jac_s_h_Z = torch.zeros(
        B, D_s, B, dummy_Z.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_s_h_g = torch.zeros(
        B, D_s, B, dummy_g.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_s_h = (jac_s_h_m, jac_s_h_s, jac_s_h_n, jac_s_h_Z, jac_s_h_g)

    # Derivatives of n_new w.r.t (m_x, s_x, n_x)
    jac_n_h_m = make_broadcast_jacobian_n_m(dn_new_dm_x, B, D_n, D_m)
    jac_n_h_s = torch.zeros(
        B, D_n, D_m, B, D_s, device=m_x.device, dtype=m_x.dtype
    )  # dn/dsx = 0
    jac_n_h_n = make_broadcast_jacobian_n_n(dn_new_dn_x, B, D_n, D_m)
    jac_n_h_Z = torch.zeros(
        B, D_n, D_m, B, dummy_Z.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_n_h_g = torch.zeros(
        B, D_n, D_m, B, dummy_g.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_n_h = (jac_n_h_m, jac_n_h_s, jac_n_h_n, jac_n_h_Z, jac_n_h_g)

    # Combine Jacobians for h_prev
    jac_h_torch = (jac_m_h, jac_s_h, jac_n_h)

    # Jacobian w.r.t x_curr = (m_y, s_y, n_y, Z_y, g_y)
    # Derivatives of m_new w.r.t (m_y, s_y, n_y)
    jac_m_x_m = make_diagonal_jacobian(dm_new_dm_y, B, D_m, D_m)
    jac_m_x_s = torch.zeros(
        B, D_m, B, D_s, device=m_x.device, dtype=m_x.dtype
    )  # dm/dsy = 0
    jac_m_x_n = torch.zeros(
        B, D_m, B, D_n, D_n_extra_kv, device=m_x.device, dtype=m_x.dtype
    )  # dm/dny = 0
    jac_m_x_Z = torch.zeros(
        B, D_m, B, dummy_Z.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_m_x_g = torch.zeros(
        B, D_m, B, dummy_g.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_m_x = (jac_m_x_m, jac_m_x_s, jac_m_x_n, jac_m_x_Z, jac_m_x_g)

    # Derivatives of s_new w.r.t (m_y, s_y, n_y)
    jac_s_x_m = make_broadcast_jacobian_s_m(ds_new_dm_y, B, D_s, D_m)
    jac_s_x_s = make_diagonal_jacobian(ds_new_ds_y, B, D_s, D_s)
    jac_s_x_n = torch.zeros(
        B, D_s, B, D_n, D_n_extra_kv, device=m_x.device, dtype=m_x.dtype
    )  # ds/dny = 0
    jac_s_x_Z = torch.zeros(
        B, D_s, B, dummy_Z.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_s_x_g = torch.zeros(
        B, D_s, B, dummy_g.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_s_x = (jac_s_x_m, jac_s_x_s, jac_s_x_n, jac_s_x_Z, jac_s_x_g)

    # Derivatives of n_new w.r.t (m_y, s_y, n_y)
    jac_n_x_m = make_broadcast_jacobian_n_m(dn_new_dm_y, B, D_n, D_m)
    jac_n_x_s = torch.zeros(
        B, D_n, D_m, B, D_s, device=m_x.device, dtype=m_x.dtype
    )  # dn/dsy = 0
    jac_n_x_n = make_broadcast_jacobian_n_n(dn_new_dn_y, B, D_n, D_m)
    jac_n_x_Z = torch.zeros(
        B, D_n, D_m, B, dummy_Z.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_n_x_g = torch.zeros(
        B, D_n, D_m, B, dummy_g.shape[1], device=m_x.device, dtype=m_x.dtype
    )  # Dummy
    jac_n_x = (jac_n_x_m, jac_n_x_s, jac_n_x_n, jac_n_x_Z, jac_n_x_g)

    # Combine Jacobians for x_curr
    jac_x_torch = (jac_m_x, jac_s_x, jac_n_x)

    return outputs, jac_h_torch, jac_x_torch


# Helper to convert pytrees between JAX and PyTorch
# ... (conversion helpers unchanged) ...
def tree_map_device(tree, device=None, dtype=None):
    if isinstance(tree, torch.Tensor):
        return tree.to(device=device, dtype=dtype)
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map_device(leaf, device, dtype) for leaf in tree)
    elif isinstance(tree, dict):
        return {k: tree_map_device(v, device, dtype) for k, v in tree.items()}
    # Add other container types if needed
    return tree


def torch_to_jax(tree):
    if isinstance(tree, torch.Tensor):
        return jnp.array(tree.cpu().detach().numpy())
    elif isinstance(tree, (list, tuple)):
        return type(tree)(torch_to_jax(leaf) for leaf in tree)
    elif isinstance(tree, dict):
        return {k: torch_to_jax(v) for k, v in tree.items()}
    return tree


def jax_to_torch(tree, device="cpu", dtype=torch.float32):
    if isinstance(tree, jnp.ndarray):
        return torch.from_numpy(np.array(tree)).to(device=device, dtype=dtype)
    elif isinstance(tree, (list, tuple)):
        return type(tree)(jax_to_torch(leaf, device, dtype) for leaf in tree)
    elif isinstance(tree, dict):
        return {k: jax_to_torch(v, device, dtype) for k, v in tree.items()}
    return tree


if __name__ == "__main__":
    # Configuration
    # B = 1
    # D_m = 1 # Simpler dimension
    B = 2  # Larger Batch
    D_m = 2  # Larger Dim
    D_s = D_m
    # D_n = 1 # Simpler dimension
    D_n = 3  # Larger Dim
    D_n_extra_kv = D_m
    D_Z = 1
    D_g = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE_TORCH = torch.float64
    DTYPE_JAX = jnp.float64

    print(f"Device: {DEVICE}, DType Torch: {DTYPE_TORCH}, DType JAX: {DTYPE_JAX}")

    # --- Generate Sample Data ---
    # print("\n--- Using Case 1: m_x > m_y ---")
    # m_x_val, m_y_val = 2.0, 1.0
    # s_x_val, s_y_val = 3.0, 4.0 # Not needed for this test
    # n_x_val, n_y_val = 5.0, 6.0 # Not needed for this test

    # Use randn for dimensions specified in config
    m_x_torch = torch.randn(B, D_m, device=DEVICE, dtype=DTYPE_TORCH)
    s_x_torch = torch.randn(B, D_s, device=DEVICE, dtype=DTYPE_TORCH)
    m_y_torch = torch.randn(B, D_m, device=DEVICE, dtype=DTYPE_TORCH)
    s_y_torch = torch.randn(B, D_s, device=DEVICE, dtype=DTYPE_TORCH)
    # Use actual n tensors now
    n_x_torch = torch.randn(B, D_n, D_n_extra_kv, device=DEVICE, dtype=DTYPE_TORCH)
    n_y_torch = torch.randn(B, D_n, D_n_extra_kv, device=DEVICE, dtype=DTYPE_TORCH)

    # Create dummy tensors for unused inputs
    dummy_Z = torch.zeros(B, D_Z, device=DEVICE, dtype=DTYPE_TORCH)
    dummy_g = torch.zeros(B, D_g, device=DEVICE, dtype=DTYPE_TORCH)

    # Need tuples for function signatures, only m_x/m_y/s_x/s_y are used
    h_prev_torch = (m_x_torch, s_x_torch, n_x_torch, dummy_Z, dummy_g)
    x_curr_torch = (m_y_torch, s_y_torch, n_y_torch, dummy_Z, dummy_g)

    h_prev_jax = torch_to_jax(h_prev_torch)
    x_curr_jax = torch_to_jax(x_curr_torch)

    # --- Calculate PyTorch Outputs and Jacobians ---
    print("\nCalculating PyTorch Outputs and Jacobians...")
    outputs_torch, jac_h_torch, jac_x_torch = attention_combine_fn_torch_full(
        h_prev_torch, x_curr_torch
    )
    print(" -> Done.")

    # --- Calculate JAX Jacobians using jacrev ---
    print("\nCalculating JAX Jacobians (m_new, s_new, n_new)...")

    # Differentiate output 0 (m_new) & 1 (s_new) & 2 (n_new) w.r.t input 0 (m_x) & 1 (s_x) of h_prev
    def combine_fn_wrt_h_prev(hp):
        return attention_combine_fn_jax(hp, x_curr_jax)

    # Differentiate w.r.t arg 0 = h_prev = (m_x, s_x, ...)
    jac_h_jax_tree = jax.jacrev(combine_fn_wrt_h_prev, argnums=0)(h_prev_jax)

    # Extract d(m_new)/d(m_x) [0][0]
    dm_new_dm_x_jax = jac_h_jax_tree[0][0]
    # Extract d(s_new)/d(m_x) [1][0]
    ds_new_dm_x_jax = jac_h_jax_tree[1][0]
    # Extract d(s_new)/d(s_x) [1][1]
    ds_new_ds_x_jax = jac_h_jax_tree[1][1]
    # Extract d(n_new)/d(m_x) [2][0]
    dn_new_dm_x_jax = jac_h_jax_tree[2][0]
    # Extract d(n_new)/d(n_x) [2][2]
    dn_new_dn_x_jax = jac_h_jax_tree[2][2]

    # Differentiate output 0 (m_new) & 1 (s_new) & 2 (n_new) w.r.t input 0 (m_y) & 1 (s_y) of x_curr
    def combine_fn_wrt_x_curr(xc):
        return attention_combine_fn_jax(h_prev_jax, xc)

    # Differentiate w.r.t arg 0 = x_curr = (m_y, s_y, ...)
    jac_x_jax_tree = jax.jacrev(combine_fn_wrt_x_curr, argnums=0)(x_curr_jax)

    # Extract d(m_new)/d(m_y) [0][0]
    dm_new_dm_y_jax = jac_x_jax_tree[0][0]
    # Extract d(s_new)/d(m_y) [1][0]
    ds_new_dm_y_jax = jac_x_jax_tree[1][0]
    # Extract d(s_new)/d(s_y) [1][1]
    ds_new_ds_y_jax = jac_x_jax_tree[1][1]
    # Extract d(n_new)/d(m_y) [2][0]
    dn_new_dm_y_jax = jac_x_jax_tree[2][0]
    # Extract d(n_new)/d(n_y) [2][2]
    dn_new_dn_y_jax = jac_x_jax_tree[2][2]

    print(f"   Raw JAX Jacobian shape (dm_new/dm_x): {dm_new_dm_x_jax.shape}")
    print(f"   Raw JAX Jacobian shape (ds_new/dm_x): {ds_new_dm_x_jax.shape}")
    print(f"   Raw JAX Jacobian shape (ds_new/ds_x): {ds_new_ds_x_jax.shape}")
    print(f"   Raw JAX Jacobian shape (dm_new/dm_y): {dm_new_dm_y_jax.shape}")
    print(f"   Raw JAX Jacobian shape (ds_new/dm_y): {ds_new_dm_y_jax.shape}")
    print(f"   Raw JAX Jacobian shape (ds_new/ds_y): {ds_new_ds_y_jax.shape}")
    print(f"   Raw JAX Jacobian shape (dn_new/dm_x): {dn_new_dm_x_jax.shape}")
    print(f"   Raw JAX Jacobian shape (dn_new/dn_x): {dn_new_dn_x_jax.shape}")
    print(f"   Raw JAX Jacobian shape (dn_new/dm_y): {dn_new_dm_y_jax.shape}")
    print(f"   Raw JAX Jacobian shape (dn_new/dn_y): {dn_new_dn_y_jax.shape}")

    # --- Calculate JAX Forward Pass for comparison ---
    print("\nCalculating JAX Forward Pass...")
    outputs_jax = attention_combine_fn_jax(h_prev_jax, x_curr_jax)
    outputs_jax_torch = jax_to_torch(outputs_jax, device=DEVICE, dtype=DTYPE_TORCH)
    print(" -> Done.")

    # --- Compare Jacobians ---
    print("\n--- Comparing Forward Outputs ---")
    output_names = ["m_new", "s_new", "n_new"]
    all_outputs_match = True
    for i, (name, out_torch, out_jax_torch) in enumerate(
        zip(output_names, outputs_torch, outputs_jax_torch)
    ):
        print(f"Comparing {name}...")
        try:
            torch.testing.assert_close(out_torch, out_jax_torch, rtol=1e-5, atol=1e-5)
            print(f"  -> {name} MATCHES")
        except Exception as e:
            print(f"  -> {name} MISMATCH DETECTED: {e}")
            print(f"     Torch shape: {out_torch.shape}, dtype: {out_torch.dtype}")
            print(
                f"     JAX shape: {out_jax_torch.shape}, dtype: {out_jax_torch.dtype}"
            )
            all_outputs_match = False
    print(f"\nForward Outputs {'MATCH' if all_outputs_match else 'DO NOT MATCH'}")

    # --- Recursive comparison for Jacobian structures ---
    def compare_jacobian_trees(jac_torch, jac_jax, prefix=""):
        if isinstance(jac_torch, torch.Tensor):
            print(f"Comparing {prefix}...")
            try:
                # Convert JAX tensor on the fly
                jac_jax_torch = jax_to_torch(jac_jax, device=DEVICE, dtype=DTYPE_TORCH)
                torch.testing.assert_close(
                    jac_torch, jac_jax_torch, rtol=1e-5, atol=1e-5
                )
                print(f"  -> {prefix} MATCHES")
                return True
            except Exception as e:
                print(f"  -> {prefix} MISMATCH DETECTED: {e}")
                print(f"     Torch shape: {jac_torch.shape}, dtype: {jac_torch.dtype}")
                if hasattr(jac_jax, "shape"):
                    print(f"     JAX shape: {jac_jax.shape}, dtype: {jac_jax.dtype}")
                else:
                    print(f"     JAX is not an array: {type(jac_jax)}")
                return False
        elif isinstance(jac_torch, (list, tuple)):
            results = []
            for i, (sub_torch, sub_jax) in enumerate(zip(jac_torch, jac_jax)):
                results.append(
                    compare_jacobian_trees(sub_torch, sub_jax, prefix=f"{prefix}[{i}]")
                )
            return all(results)
        else:
            print(f"Skipping comparison for type {type(jac_torch)} at {prefix}")
            return True  # Or False, depending on whether non-tensor/sequence types are expected

    print("\n--- Comparing Jacobian w.r.t h_prev ---")
    jac_h_match = compare_jacobian_trees(jac_h_torch, jac_h_jax_tree, "jac_h")
    print(f"\nJacobian w.r.t h_prev {'MATCHES' if jac_h_match else 'DOES NOT MATCH'}")

    print("\n--- Comparing Jacobian w.r.t x_curr ---")
    jac_x_match = compare_jacobian_trees(jac_x_torch, jac_x_jax_tree, "jac_x")
    print(f"\nJacobian w.r.t x_curr {'MATCHES' if jac_x_match else 'DOES NOT MATCH'}")

    # --- Comparison logic commented out ---
    # passed_h = compare_jacobians(jac_h_torch_tuple, jac_h_jax_tuple, names_h)
    # print(f"\n -> Jacobians w.r.t h_prev {'MATCH' if passed_h else 'DO NOT MATCH'}")
    #
    # passed_x = compare_jacobians(jac_x_torch_tuple, jac_x_jax_tuple, names_x)
    # print(f"\n -> Jacobians w.r.t x_curr {'MATCH' if passed_x else 'DO NOT MATCH'}")
