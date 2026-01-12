#!/usr/bin/env python3
"""
repro_vjp_failure.py
--------------------

Drop-in reproduction of the analytical-vs-numerical VJP **shape/logic bug**
that appears as soon as the numerator `n` is not a scalar.

How to run:
    $ python repro_vjp_failure.py           # CPU
    $ CUDA_VISIBLE_DEVICES=0 python repro_vjp_failure.py   # GPU

Expected tail of output:

    >>> Adversarial non-scalar test (h=4) ...
    dL/dinput_0: MISMATCH (Num: -5.960512, Ana: -1.506484)
    Traceback (most recent call last):
      ...
    RuntimeError: The size of tensor a (1) must match the size of tensor b (4) ...

The crash pinpoints the place where the vector contribution coming from *v_n* was
not reduced to a scalar before being added to the gradient of *m_x/m_y*.
"""

import torch
from torch.autograd import functional as F

# ------------------------ Core combine ------------------------ #
def combine_fn_logic_torch(m_x, s_x, n_x, Z_x, g_x,
                           m_y, s_y, n_y, Z_y, g_y):
    m_new      = torch.maximum(m_x, m_y)

    # log-sum-exp stabilisation
    m_x_safe   = torch.where(torch.isinf(m_x),
                             torch.full_like(m_x, -1e10), m_x)
    m_y_safe   = torch.where(torch.isinf(m_y),
                             torch.full_like(m_y, -1e10), m_y)
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    exp_x      = torch.exp(m_x_safe - m_new_safe)
    exp_y      = torch.exp(m_y_safe - m_new_safe)

    s_new = s_x * exp_x        + s_y * exp_y
    n_new = n_x * exp_x        + n_y * exp_y        # <-- broadcasting OK
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new


def combine_fn_ref_torch(x, y):
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y
    return combine_fn_logic_torch(m_x, s_x, n_x, Z_x, g_x,
                                  m_y, s_y, n_y, Z_y, g_y)

# ------------------------ Helpers ------------------------ #
def get_exps(m_x, m_y):
    m_new = torch.maximum(m_x, m_y)
    exp_x = torch.exp(m_x - m_new)
    exp_y = torch.exp(m_y - m_new)
    return exp_x, exp_y, m_new

def dm_new_dm_x(m_x, m_y): return (m_x >= m_y).float()
def dm_new_dm_y(m_x, m_y): return (m_x <  m_y).float()

def ds_new_ds_x(m_x, m_y): return get_exps(m_x, m_y)[0]
def ds_new_ds_y(m_x, m_y): return get_exps(m_x, m_y)[1]
def dn_new_dn_x(m_x, m_y): return get_exps(m_x, m_y)[0]
def dn_new_dn_y(m_x, m_y): return get_exps(m_x, m_y)[1]

def ds_new_dm_x(m_x, s_x, m_y, s_y):
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    mask = dm_new_dm_x(m_x, m_y)
    return s_x * exp_x * (1 - mask) - s_y * exp_y * mask

def ds_new_dm_y(m_x, s_x, m_y, s_y):
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    mask = dm_new_dm_y(m_x, m_y)
    return -s_x * exp_x * mask + s_y * exp_y * (1 - mask)

def dn_new_dm_x(m_x, n_x, m_y, n_y):
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    mask = dm_new_dm_x(m_x, m_y)
    return n_x * exp_x * (1 - mask) - n_y * exp_y * mask

def dn_new_dm_y(m_x, n_x, m_y, n_y):
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    mask = dm_new_dm_y(m_x, m_y)
    return -n_x * exp_x * mask + n_y * exp_y * (1 - mask)

# ------------------------ YOUR original analytical VJP (buggy) ------------------------ #
def analytical_vjp_combine_fn(inputs, v_outputs):
    (m_x, s_x, n_x, Z_x, g_x,
     m_y, s_y, n_y, Z_y, g_y) = inputs
    v_m, v_s, v_n, v_Z, v_g = v_outputs

    # ---------- w.r.t. x ----------
    grad_m_x = (
        v_m * dm_new_dm_x(m_x, m_y)
        + v_s * ds_new_dm_x(m_x, s_x, m_y, s_y)
        + (v_n * dn_new_dm_x(m_x, n_x, m_y, n_y)).sum(dim=-1)  # ← parentheses added
    )
    grad_s_x = v_s * ds_new_ds_x(m_x, m_y)
    grad_n_x = v_n * dn_new_dn_x(m_x, m_y)
    grad_Z_x = v_Z
    grad_g_x = v_g

    # ---------- w.r.t. y ----------
    grad_m_y = (
        v_m * dm_new_dm_y(m_x, m_y)
        + v_s * ds_new_dm_y(m_x, s_x, m_y, s_y)
        + (v_n * dn_new_dm_y(m_x, n_x, m_y, n_y)).sum(dim=-1)  # ← parentheses added
    )
    grad_s_y = v_s * ds_new_ds_y(m_x, m_y)
    grad_n_y = v_n * dn_new_dn_y(m_x, m_y)
    grad_Z_y = v_Z
    grad_g_y = v_g

    return (
        grad_m_x, grad_s_x, grad_n_x, grad_Z_x, grad_g_x,
        grad_m_y, grad_s_y, grad_n_y, grad_Z_y, grad_g_y
    )


# ------------------------ Test harness ------------------------ #
def run_vjp_test(name, combine_fn, analytical_vjp_fn,
                 inputs, v_outputs):
    print(f"\n>>> {name} ...")
    # numerical VJP
    inputs_req_grad = tuple(t.detach().clone().requires_grad_(True)
                            for t in inputs)
    _, vjp_num = F.vjp(combine_fn, inputs_req_grad, v_outputs)

    # analytical VJP
    vjp_ana = analytical_vjp_fn(tuple(t.detach() for t in inputs),
                                v_outputs)

    for i, (num, ana) in enumerate(zip(vjp_num, vjp_ana)):
        try:
            match = torch.allclose(num, ana, atol=1e-5, rtol=1e-5)
        except RuntimeError as e:
            print(f"dL/dinput_{i}: SHAPE-ERROR • {e}")
            raise
        print(f"dL/dinput_{i}: {'MATCH' if match else 'MISMATCH'} "
              f"(Num: {num.flatten()[0]:.6f}, "
              f"Ana: {ana.flatten()[0]:.6f})")

# ------------------------ Adversarial non-scalar test ------------------------ #
if __name__ == "__main__":
    torch.manual_seed(2025)
    dtype  = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h = 4                    # ----> non-scalar numerator dimension
    big = 50.0               # make logits roughly comparable

    # Left operand (x)
    m_x = torch.randn(1, dtype=dtype, device=device) * big
    s_x = torch.randn(1, dtype=dtype, device=device)
    n_x = torch.randn(h, dtype=dtype, device=device)          # VECTOR
    Z_x = torch.randn(1, dtype=dtype, device=device)
    g_x = torch.randn(1, dtype=dtype, device=device)

    # Right operand (y)
    m_y = torch.randn(1, dtype=dtype, device=device) * big
    s_y = torch.randn(1, dtype=dtype, device=device)
    n_y = torch.randn(h, dtype=dtype, device=device)          # VECTOR
    Z_y = torch.randn(1, dtype=dtype, device=device)
    g_y = torch.randn(1, dtype=dtype, device=device)

    inputs = (m_x, s_x, n_x, Z_x, g_x,
              m_y, s_y, n_y, Z_y, g_y)

    # Up-stream VJP seed (same shapes as outputs)
    v_m = torch.randn(1, dtype=dtype, device=device)
    v_s = torch.randn(1, dtype=dtype, device=device)
    v_n = torch.randn(h, dtype=dtype, device=device)          # VECTOR
    v_Z = torch.randn(1, dtype=dtype, device=device)
    v_g = torch.randn(1, dtype=dtype, device=device)
    v_outputs = (v_m, v_s, v_n, v_Z, v_g)

    # run
    run_vjp_test(f"Adversarial non-scalar test (h={h})",
                 combine_fn_logic_torch,
                 analytical_vjp_combine_fn,
                 inputs, v_outputs)
