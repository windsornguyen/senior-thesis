import torch
import torch.autograd.functional as F

from thesis.experiments.utils.assoc_scan.kernel import associative_scan
from typing import Tuple


def combine_fn_logic_torch(m_x, s_x, n_x, Z_x, g_x, m_y, s_y, n_y, Z_y, g_y):
    # m_new = max(m_x, m_y)
    m_new = torch.maximum(m_x, m_y)
    # Stabilize exp: exp(m_x - m_new), exp(m_y - m_new)
    m_x_safe = torch.where(torch.isinf(m_x), torch.full_like(m_x, -1e10), m_x)
    m_y_safe = torch.where(torch.isinf(m_y), torch.full_like(m_y, -1e10), m_y)
    m_new_safe = torch.maximum(m_x_safe, m_y_safe)
    exp_x = torch.exp(m_x_safe - m_new_safe)
    exp_y = torch.exp(m_y_safe - m_new_safe)
    # s_new = s_x * exp(m_x - m_new) + s_y * exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    # n_new = n_x * exp(m_x - m_new) + n_y * exp(m_y - m_new)
    n_new = n_x * exp_x + n_y * exp_y
    # Z_new = Z_x + Z_y
    Z_new = Z_x + Z_y
    # g_new = g_x + g_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new


def combine_fn_ref_torch(x, y):
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y
    return combine_fn_logic_torch(m_x, s_x, n_x, Z_x, g_x, m_y, s_y, n_y, Z_y, g_y)


# Helper functions for VJP
def get_exps(m_x, m_y):
    # Stabilize exp: m_new = max(m_x, m_y), exp_x = exp(m_x - m_new), exp_y = exp(m_y - m_new)
    m_x_safe = torch.where(torch.isinf(m_x), torch.full_like(m_x, -1e10), m_x)
    m_y_safe = torch.where(torch.isinf(m_y), torch.full_like(m_y, -1e10), m_y)
    m_new = torch.maximum(m_x_safe, m_y_safe)
    exp_x = torch.exp(m_x_safe - m_new)
    exp_y = torch.exp(m_y_safe - m_new)
    return exp_x, exp_y, m_new


def dm_new_dm_x(m_x, m_y):
    # ∂m_new/∂m_x = 1 if m_x >= m_y else 0
    return torch.where(m_x >= m_y, torch.ones_like(m_x), torch.zeros_like(m_x))


def dm_new_dm_y(m_x, m_y):
    # ∂m_new/∂m_y = 1 if m_x < m_y else 0
    return torch.where(m_x < m_y, torch.ones_like(m_x), torch.zeros_like(m_x))


def ds_new_ds_x(m_x, m_y):
    # ∂s_new/∂s_x = exp(m_x - m_new)
    exp_x, _, _ = get_exps(m_x, m_y)
    return exp_x


def ds_new_ds_y(m_x, m_y):
    # ∂s_new/∂s_y = exp(m_y - m_new)
    _, exp_y, _ = get_exps(m_x, m_y)
    return exp_y


def ds_new_dm_x(m_x, s_x, m_y, s_y):
    # ∂s_new/∂m_x = s_x * exp(m_x - m_new) * (1 - ∂m_new/∂m_x) + s_y * exp(m_y - m_new) * (-∂m_new/∂m_x)
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    dm_new_dm_x_ = dm_new_dm_x(m_x, m_y)
    return s_x * exp_x * (1.0 - dm_new_dm_x_) - s_y * exp_y * dm_new_dm_x_


def ds_new_dm_y(m_x, s_x, m_y, s_y):
    # ∂s_new/∂m_y = s_x * exp(m_x - m_new) * (-∂m_new/∂m_y) + s_y * exp(m_y - m_new) * (1 - ∂m_new/∂m_y)
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    dm_new_dm_y_ = dm_new_dm_y(m_x, m_y)
    return -s_x * exp_x * dm_new_dm_y_ + s_y * exp_y * (1.0 - dm_new_dm_y_)


def dn_new_dn_x(m_x, m_y):
    # ∂n_new/∂n_x = exp(m_x - m_new)
    exp_x, _, _ = get_exps(m_x, m_y)
    return exp_x


def dn_new_dn_y(m_x, m_y):
    # ∂n_new/∂n_y = exp(m_y - m_new)
    _, exp_y, _ = get_exps(m_x, m_y)
    return exp_y


def dn_new_dm_x(m_x, n_x, m_y, n_y):
    # ∂n_new/∂m_x = n_x * exp(m_x - m_new) * (1 - ∂m_new/∂m_x) + n_y * exp(m_y - m_new) * (-∂m_new/∂m_x)
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    dm_new_dm_x_ = dm_new_dm_x(m_x, m_y)
    return n_x * exp_x * (1.0 - dm_new_dm_x_) - n_y * exp_y * dm_new_dm_x_


def dn_new_dm_y(m_x, n_x, m_y, n_y):
    # ∂n_new/∂m_y = n_x * exp(m_x - m_new) * (-∂m_new/∂m_y) + n_y * exp(m_y - m_new) * (1 - ∂m_new/∂m_y)
    exp_x, exp_y, _ = get_exps(m_x, m_y)
    dm_new_dm_y_ = dm_new_dm_y(m_x, m_y)
    return -n_x * exp_x * dm_new_dm_y_ + n_y * exp_y * (1.0 - dm_new_dm_y_)


def analytical_vjp_combine_fn(inputs, v_outputs):
    m_x, s_x, n_x, Z_x, g_x, m_y, s_y, n_y, Z_y, g_y = inputs
    v_m, v_s, v_n, v_Z, v_g = v_outputs

    # Jacobians w.r.t. x
    dm_new_dm_x_ = dm_new_dm_x(m_x, m_y)
    ds_new_dm_x_ = ds_new_dm_x(m_x, s_x, m_y, s_y)
    dn_new_dm_x_ = dn_new_dm_x(m_x, n_x, m_y, n_y)
    ds_new_ds_x_ = ds_new_ds_x(m_x, m_y)
    dn_new_dn_x_ = dn_new_dn_x(m_x, m_y)
    grad_m_x = v_m * dm_new_dm_x_ + v_s * ds_new_dm_x_ + v_n * dn_new_dm_x_
    grad_s_x = v_s * ds_new_ds_x_
    grad_n_x = v_n * dn_new_dn_x_
    grad_Z_x = v_Z
    grad_g_x = v_g

    # Jacobians w.r.t. y
    dm_new_dm_y_ = dm_new_dm_y(m_x, m_y)
    ds_new_dm_y_ = ds_new_dm_y(m_x, s_x, m_y, s_y)
    dn_new_dm_y_ = dn_new_dm_y(m_x, n_x, m_y, n_y)
    ds_new_ds_y_ = ds_new_ds_y(m_x, m_y)
    dn_new_dn_y_ = dn_new_dn_y(m_x, m_y)
    grad_m_y = v_m * dm_new_dm_y_ + v_s * ds_new_dm_y_ + v_n * dn_new_dm_y_
    grad_s_y = v_s * ds_new_ds_y_
    grad_n_y = v_n * dn_new_dn_y_
    grad_Z_y = v_Z
    grad_g_y = v_g

    return grad_m_x, grad_s_x, grad_n_x, grad_Z_x, grad_g_x, grad_m_y, grad_s_y, grad_n_y, grad_Z_y, grad_g_y


def run_vjp_test(test_name, combine_fn, analytical_vjp_fn, inputs, v_outputs):
    inputs_req_grad = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    try:
        _, vjp_num = F.vjp(combine_fn, inputs_req_grad, v_outputs)
    except Exception as e:
        print(f"Numerical VJP error: {e}")
        return False
    try:
        vjp_ana = analytical_vjp_fn(tuple(t.detach() for t in inputs), v_outputs)
    except Exception as e:
        print(f"Analytical VJP error: {e}")
        return False
    all_match = True
    for i, (num, ana) in enumerate(zip(vjp_num, vjp_ana)):
        match = torch.allclose(num, ana, atol=1e-5, rtol=1e-5)
        print(f"dL/dinput_{i}: {'MATCH' if match else 'MISMATCH'} (Num: {num.item():.6f}, Ana: {ana.item():.6f})")
        all_match &= match
    print(f">>> {test_name} {'PASSED' if all_match else 'FAILED'} <<<")
    return all_match


from thesis.experiments.utils.assoc_scan.kernel import associative_scan as assoc_scan_ref


def combine_fn_ref(x, y):
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y
    m_new = torch.maximum(m_x, m_y)
    exp_x = torch.exp(m_x - m_new)
    exp_y = torch.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x.unsqueeze(-1) + n_y * exp_y.unsqueeze(-1)
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new


def scan_fn(
    qk_slice: torch.Tensor, v_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs associative scan over one (B,H) stream.

    Args:
        qk_slice: [L] similarity logits
        v_slice: [L, h] L2-normalized V (first-order numerator)
        Z_slice: [L, h, h] gated outer-product accumulator
        g_slice: [L] scalar gate sequence
    """
    leaves = (
        qk_slice,
        torch.ones_like(qk_slice),  # s starts at 1.0
        v_slice,
        Z_slice,
        g_slice,
    )
    # Assuming assoc_scan_ref is a correct reference (e.g., sequential scan)
    return assoc_scan_ref(combine_fn=combine_fn_ref, xs=leaves, dim=0, combine_mode="generic")


def batched_scan_fn(
    sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs scan_fn independently for every (B,H) stream.

    Args:
        sim: [B, H, L] similarity logits
        v: [B, H, L, h] L2-normalized V (first-order numerator)
        gated_Z: [B, H, L, h, h] gated outer-product accumulator
        gates_z: [B, H, L] scalar gate sequence

    Returns:
        Tuple of (max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul) with shapes:
        - max_cumul: [B, H, L]
        - norm_cumul: [B, H, L]
        - v_cumul: [B, H, L, h]
        - Z_cumul: [B, H, L, h, h]
        - gate_cumul: [B, H, L]
    """
    B, H, L, h = v.shape
    # Prepare inputs for vmap
    sim_bh = sim.reshape(-1, L)
    v_bh = v.reshape(-1, L, h)
    gated_Z_bh = gated_Z.reshape(-1, L, h, h)
    gates_z_bh = gates_z.reshape(-1, L)

    # vmap over the flattened B*H dimension
    scan_vmapped = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    m_res, s_res, n_res, Z_res, g_res = scan_vmapped(sim_bh, v_bh, gated_Z_bh, gates_z_bh)

    # Reshape results back to [B, H, ...]
    return (
        m_res.reshape(B, H, L),
        s_res.reshape(B, H, L),
        n_res.reshape(B, H, L, h),
        Z_res.reshape(B, H, L, h, h),
        g_res.reshape(B, H, L),
    )

def wrapped_fn_unpacked(*all_inputs):
    x = all_inputs[:5]
    y = all_inputs[5:]
    return combine_fn_ref_torch(x, y)

if __name__ == "__main__":
    from torch.autograd import gradcheck

    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1746)  # For reproducibility

    inputs = tuple(torch.randn(1, dtype=dtype, device=device, requires_grad=True) * 1e3 for _ in range(10))
    v_outputs = tuple(torch.randn(1, dtype=dtype, device=device) * 1e2 for _ in range(5))

    print(">>> Running analytical vs. numerical VJP test...")
    run_vjp_test("Hard Test: Random Inputs", combine_fn_logic_torch, analytical_vjp_combine_fn, inputs, v_outputs)

    print(">>> Checking for NaNs/Infs before gradcheck...")
    for i, inp in enumerate(inputs):
        if not torch.all(torch.isfinite(inp)):
            raise ValueError(f"Input {i} contains NaNs or Infs!")

    print(">>> Running gradcheck...")
    inputs = tuple(t.clone().double().requires_grad_(True) for t in inputs)
    ok = gradcheck(wrapped_fn_unpacked, inputs, eps=1e-7, atol=1e-5, rtol=1e-4)
    print("Gradcheck:", "PASSED" if ok else "FAILED")
    assert ok
