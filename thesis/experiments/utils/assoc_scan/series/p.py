# -*- Author: Windsor Nguyen -*-
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

@triton.jit
def combine_fn_ref(m_x, s_x, n_x, m_y, s_y, n_y):
    """Combines states for softmax-related scan."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x[:, None] + n_y * exp_y[:, None]
    return m_new, s_new, n_new

@triton.jit
def additive_combine_fn_ref(x, y):
    """Combines states for additive scan (Z or g)."""
    return x + y

@triton.jit
def backward_combine_fn_ref(m_x, s_x, n_x, m_y, s_y, n_y):
    """Combines states for backward softmax-related scan."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x[:, None] + n_y * exp_y[:, None]
    return m_new, s_new, n_new

def get_softmax_configs(seq_len=None):
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [32, 64, 128, 256, 512]
        for s in [2, 3]
        for w in [4]
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [32, 64, 128]]
        )
    return [conf for conf in configs if keep_config(conf, seq_len)]

def keep_config(conf, seq_len=None):
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if BLOCK_SIZE >= 512 and num_warps < 4:
        return False
    if BLOCK_SIZE < 64 and num_warps > 4:
        return False
    if seq_len is not None and BLOCK_SIZE > seq_len:
        return False
    return True

@triton.autotune(
    configs=[conf for conf in get_softmax_configs(seq_len=32)],
    key=["batch_size", "num_heads", "seq_len"],
)
@triton.jit
def scan_fn_triton_kernel(
    sim_ptr,
    v_ptr,
    gated_Z_ptr,
    gates_z_ptr,
    weights_ptr,
    max_cumul_ptr,
    norm_cumul_ptr,
    v_cumul_ptr,
    Z_cumul_ptr,
    gate_cumul_ptr,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    HEAD_DIM: tl.constexpr,
    stride_b: int,
    stride_h: int,
    stride_l: int,
    v_stride_b: int,
    v_stride_h: int,
    v_stride_l: int,
    v_stride_h1: int,
    Z_stride_b: int,
    Z_stride_h: int,
    Z_stride_l: int,
    Z_stride_h1: int,
    Z_stride_h2: int,
    gates_z_stride_b: int,
    gates_z_stride_h: int,
    gates_z_stride_l: int,
    weights_stride_b: int,
    weights_stride_h: int,
    weights_stride_l: int,
    max_cumul_stride_b: int,
    max_cumul_stride_h: int,
    max_cumul_stride_l: int,
    norm_cumul_stride_b: int,
    norm_cumul_stride_h: int,
    norm_cumul_stride_l: int,
    v_cumul_stride_b: int,
    v_cumul_stride_h: int,
    v_cumul_stride_l: int,
    v_cumul_stride_h1: int,
    Z_cumul_stride_b: int,
    Z_cumul_stride_h: int,
    Z_cumul_stride_l: int,
    Z_cumul_stride_h1: int,
    Z_cumul_stride_h2: int,
    gate_cumul_stride_b: int,
    gate_cumul_stride_h: int,
    gate_cumul_stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_head * stride_h
    v_offset = pid_batch * v_stride_b + pid_head * v_stride_h
    Z_offset = pid_batch * Z_stride_b + pid_head * Z_stride_h
    gates_z_offset = pid_batch * gates_z_stride_b + pid_head * gates_z_stride_h
    weights_offset = pid_batch * weights_stride_b + pid_head * weights_stride_h
    max_cumul_offset = pid_batch * max_cumul_stride_b + pid_head * max_cumul_stride_h
    norm_cumul_offset = pid_batch * norm_cumul_stride_b + pid_head * norm_cumul_stride_h
    v_cumul_offset = pid_batch * v_cumul_stride_b + pid_head * v_cumul_stride_h
    Z_cumul_offset = pid_batch * Z_cumul_stride_b + pid_head * Z_cumul_stride_h
    gate_cumul_offset = pid_batch * gate_cumul_stride_b + pid_head * gate_cumul_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    s_i = tl.full((), 0.0, dtype=DTYPE)
    n_i = tl.zeros([HEAD_DIM], dtype=DTYPE)
    Z_i = tl.zeros([HEAD_DIM, HEAD_DIM], dtype=DTYPE)
    g_i = tl.full((), 0.0, dtype=DTYPE)

    # First pass: Compute m_i, s_i, n_i, Z_i, g_i
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        v_offsets = v_offset + (start_col + offs)[:, None] * v_stride_l + \
                    tl.arange(0, HEAD_DIM)[None, :] * v_stride_h1
        gates_z_offsets = gates_z_offset + (start_col + offs) * gates_z_stride_l

        qk = tl.load(sim_ptr + offsets, mask=mask, other=float("-inf"))
        v = tl.load(v_ptr + v_offsets, mask=mask[:, None], other=0.0)
        gates_z = tl.load(gates_z_ptr + gates_z_offsets, mask=mask, other=0.0)
        m = qk
        s = tl.exp(qk - m)
        n = v * tl.exp(qk - m)[:, None]

        # Scan for m, s, n
        m, s, n = tl.associative_scan(
            (m, s, n), axis=0, combine_fn=combine_fn_ref
        )

        # Load gated_Z
        Z_offs = Z_offset + (start_col + offs)[:, None, None] * Z_stride_l + \
                 tl.arange(0, HEAD_DIM)[None, :, None] * Z_stride_h1 + \
                 tl.arange(0, HEAD_DIM)[None, None, :] * Z_stride_h2
        gated_Z = tl.load(gated_Z_ptr + Z_offs, mask=mask[:, None, None], other=0.0)

        # Separate scans for gated_Z and gates_z
        Z = tl.associative_scan(gated_Z, axis=0, combine_fn=additive_combine_fn_ref)
        g = tl.associative_scan(gates_z, axis=0, combine_fn=additive_combine_fn_ref)

        # Store per-position outputs
        tl.store(max_cumul_ptr + max_cumul_offset + (start_col + offs) * max_cumul_stride_l, m, mask=mask)
        tl.store(norm_cumul_ptr + norm_cumul_offset + (start_col + offs) * norm_cumul_stride_l, s, mask=mask)
        tl.store(v_cumul_ptr + v_cumul_offset + (start_col + offs)[:, None] * v_cumul_stride_l + \
                 tl.arange(0, HEAD_DIM)[None, :] * v_cumul_stride_h1, n, mask=mask[:, None])
        tl.store(Z_cumul_ptr + Z_cumul_offset + (start_col + offs)[:, None, None] * Z_cumul_stride_l + \
                 tl.arange(0, HEAD_DIM)[None, :, None] * Z_cumul_stride_h1 + \
                 tl.arange(0, HEAD_DIM)[None, None, :] * Z_cumul_stride_h2, Z, mask=mask[:, None, None])
        tl.store(gate_cumul_ptr + gate_cumul_offset + (start_col + offs) * gate_cumul_stride_l, g, mask=mask)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_col - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        s_local = tl.where(offs == last_idx, s, 0.0)
        n_local = tl.where(offs == last_idx, n, 0.0)
        g_local = tl.where(offs == last_idx, g, 0.0)
        Z_local = tl.where(offs[:, None, None] == last_idx, Z, 0.0)
        m_local = tl.max(m_local, axis=0)
        s_local = tl.sum(s_local, axis=0)
        n_local = tl.sum(n_local, axis=0)
        g_local = tl.sum(g_local, axis=0)
        Z_local = tl.sum(Z_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        s_i = s_i * tl.exp(old_m_i - m_i) + s_local * tl.exp(m_local - m_i)
        n_i = n_i * tl.exp(old_m_i - m_i) + n_local * tl.exp(m_local - m_i)
        g_i = g_i + g_local
        Z_i = Z_i + Z_local

    # Second pass: Compute weights
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        qk = tl.load(sim_ptr + offsets, mask=mask, other=float("-inf"))
        weights = tl.exp(qk - m_i) / (s_i + 1e-6)
        tl.store(weights_ptr + weights_offset + (start_col + offs) * weights_stride_l, weights, mask=mask)

@triton.autotune(
    configs=[conf for conf in get_softmax_configs(seq_len=32)],
    key=["batch_size", "num_heads", "seq_len"],
)
@triton.jit
def bwd_scan_fn_triton_kernel(
    sim_ptr, v_ptr, gated_Z_ptr, gates_z_ptr,
    grad_weights_ptr, grad_max_cumul_ptr, grad_norm_cumul_ptr, grad_v_cumul_ptr, grad_Z_cumul_ptr, grad_gate_cumul_ptr,
    grad_sim_ptr, grad_v_ptr, grad_gated_Z_ptr, grad_gates_z_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_l: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_l: tl.constexpr,
    v_stride_h1: tl.constexpr,
    Z_stride_b: tl.constexpr,
    Z_stride_h: tl.constexpr,
    Z_stride_l: tl.constexpr,
    Z_stride_h1: tl.constexpr,
    Z_stride_h2: tl.constexpr,
    gates_z_stride_b: tl.constexpr,
    gates_z_stride_h: tl.constexpr,
    gates_z_stride_l: tl.constexpr,
    grad_weights_stride_b: tl.constexpr,
    grad_weights_stride_h: tl.constexpr,
    grad_weights_stride_l: tl.constexpr,
    grad_max_cumul_stride_b: tl.constexpr,
    grad_max_cumul_stride_h: tl.constexpr,
    grad_max_cumul_stride_l: tl.constexpr,
    grad_norm_cumul_stride_b: tl.constexpr,
    grad_norm_cumul_stride_h: tl.constexpr,
    grad_norm_cumul_stride_l: tl.constexpr,
    grad_v_cumul_stride_b: tl.constexpr,
    grad_v_cumul_stride_h: tl.constexpr,
    grad_v_cumul_stride_l: tl.constexpr,
    grad_v_cumul_stride_h1: tl.constexpr,
    grad_Z_cumul_stride_b: tl.constexpr,
    grad_Z_cumul_stride_h: tl.constexpr,
    grad_Z_cumul_stride_l: tl.constexpr,
    grad_Z_cumul_stride_h1: tl.constexpr,
    grad_Z_cumul_stride_h2: tl.constexpr,
    grad_gate_cumul_stride_b: tl.constexpr,
    grad_gate_cumul_stride_h: tl.constexpr,
    grad_gate_cumul_stride_l: tl.constexpr,
    grad_sim_stride_b: tl.constexpr,
    grad_sim_stride_h: tl.constexpr,
    grad_sim_stride_l: tl.constexpr,
    grad_v_stride_b: tl.constexpr,
    grad_v_stride_h: tl.constexpr,
    grad_v_stride_l: tl.constexpr,
    grad_v_stride_h1: tl.constexpr,
    grad_Z_stride_b: tl.constexpr,
    grad_Z_stride_h: tl.constexpr,
    grad_Z_stride_l: tl.constexpr,
    grad_Z_stride_h1: tl.constexpr,
    grad_Z_stride_h2: tl.constexpr,
    grad_gates_z_stride_b: tl.constexpr,
    grad_gates_z_stride_h: tl.constexpr,
    grad_gates_z_stride_l: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    row_offset = pid_b * stride_b + pid_h * stride_h
    v_offset = pid_b * v_stride_b + pid_h * v_stride_h
    Z_offset = pid_b * Z_stride_b + pid_h * Z_stride_h
    gates_z_offset = pid_b * gates_z_stride_b + pid_h * gates_z_stride_h
    grad_weights_offset = pid_b * grad_weights_stride_b + pid_h * grad_weights_stride_h
    grad_max_cumul_offset = pid_b * grad_max_cumul_stride_b + pid_h * grad_max_cumul_stride_h
    grad_norm_cumul_offset = pid_b * grad_norm_cumul_stride_b + pid_h * grad_norm_cumul_stride_h
    grad_v_cumul_offset = pid_b * grad_v_cumul_stride_b + pid_h * grad_v_cumul_stride_h
    grad_Z_cumul_offset = pid_b * grad_Z_cumul_stride_b + pid_h * grad_Z_cumul_stride_h
    grad_gate_cumul_offset = pid_b * grad_gate_cumul_stride_b + pid_h * grad_gate_cumul_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    s_i = tl.full((), 0.0, dtype=DTYPE)
    n_i = tl.zeros([HEAD_DIM], dtype=DTYPE)
    Z_i = tl.zeros([HEAD_DIM, HEAD_DIM], dtype=DTYPE)
    g_i = tl.full((), 0.0, dtype=DTYPE)

    # First pass: Recompute m_i, s_i, n_i, Z_i, g_i, and compute gradient accumulations
    for k in range(tl.cdiv(seq_len, BLOCK_SIZE) - 1, -1, -1):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        v_idx = v_offset + (start + offs)[:, None] * v_stride_l + \
                tl.arange(0, HEAD_DIM)[None, :] * v_stride_h1
        gates_z_idx = gates_z_offset + (start + offs) * gates_z_stride_l
        grad_weights_idx = grad_weights_offset + (start + offs) * grad_weights_stride_l
        grad_max_cumul_idx = grad_max_cumul_offset + (start + offs) * grad_max_cumul_stride_l
        grad_norm_cumul_idx = grad_norm_cumul_offset + (start + offs) * grad_norm_cumul_stride_l
        grad_v_cumul_idx = grad_v_cumul_offset + (start + offs)[:, None] * grad_v_cumul_stride_l + \
                           tl.arange(0, HEAD_DIM)[None, :] * grad_v_cumul_stride_h1

        qk = tl.load(sim_ptr + idx, mask=mask, other=float("-inf"))
        v = tl.load(v_ptr + v_idx, mask=mask[:, None], other=0.0)
        gates_z = tl.load(gates_z_ptr + gates_z_idx, mask=mask, other=0.0)
        g_weights = tl.load(grad_weights_ptr + grad_weights_idx, mask=mask, other=0.0)
        g_max_cumul = tl.load(grad_max_cumul_ptr + grad_max_cumul_idx, mask=mask, other=0.0)
        g_norm_cumul = tl.load(grad_norm_cumul_ptr + grad_norm_cumul_idx, mask=mask, other=0.0)
        g_v_cumul = tl.load(grad_v_cumul_ptr + grad_v_cumul_idx, mask=mask[:, None], other=0.0)
        m = qk
        s = tl.exp(qk - m)
        n = v * tl.exp(qk - m)[:, None]

        # Scan for m, s, n (forward)
        m, s, n = tl.associative_scan(
            (m, s, n), axis=0, combine_fn=backward_combine_fn_ref
        )

        # Load gated_Z
        Z_offs = Z_offset + (start + offs)[:, None, None] * Z_stride_l + \
                 tl.arange(0, HEAD_DIM)[None, :, None] * Z_stride_h1 + \
                 tl.arange(0, HEAD_DIM)[None, None, :] * Z_stride_h2
        gated_Z = tl.load(gated_Z_ptr + Z_offs, mask=mask[:, None, None], other=0.0)

        # Reverse scans for gated_Z and gates_z gradients
        rev_idx = seq_len - 1 - (start + offs)
        safe_idx = tl.where(mask, rev_idx, 0)
        Z_grad_offs = Z_offset + safe_idx[:, None, None] * grad_Z_stride_l + \
                      tl.arange(0, HEAD_DIM)[None, :, None] * grad_Z_stride_h1 + \
                      tl.arange(0, HEAD_DIM)[None, None, :] * grad_Z_stride_h2
        gates_z_grad_offs = gates_z_offset + safe_idx * grad_gates_z_stride_l

        g_Z = tl.load(grad_Z_cumul_ptr + Z_grad_offs, mask=mask[:, None, None], other=0.0)
        g_gates_z = tl.load(grad_gate_cumul_ptr + gates_z_grad_offs, mask=mask, other=0.0)

        g_Z = tl.associative_scan(g_Z, axis=0, combine_fn=additive_combine_fn_ref, reverse=False)
        g_gates_z = tl.associative_scan(g_gates_z, axis=0, combine_fn=additive_combine_fn_ref, reverse=False)

        tl.store(grad_Z_cumul_ptr + Z_grad_offs, g_Z, mask=mask[:, None, None])
        tl.store(grad_gate_cumul_ptr + gates_z_grad_offs, g_gates_z, mask=mask)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        s_local = tl.where(offs == last_idx, s, 0.0)
        n_local = tl.where(offs == last_idx, n, 0.0)
        g_local = tl.where(offs == last_idx, gates_z, 0.0)
        Z_local = tl.where(offs[:, None, None] == last_idx, gated_Z, 0.0)
        m_local = tl.max(m_local, axis=0)
        s_local = tl.sum(s_local, axis=0)
        n_local = tl.sum(n_local, axis=0)
        g_local = tl.sum(g_local, axis=0)
        Z_local = tl.sum(Z_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        s_i = s_i * tl.exp(old_m_i - m_i) + s_local * tl.exp(m_local - m_i)
        n_i = n_i * tl.exp(old_m_i - m_i) + n_local * tl.exp(m_local - m_i)
        g_i = g_i + g_local
        Z_i = Z_i + Z_local

    # Load upstream gradients
    g_norm_cumul = tl.load(grad_norm_cumul_ptr + grad_norm_cumul_offset + (seq_len - 1) * grad_norm_cumul_stride_l)
    g_gate_cumul = tl.load(grad_gate_cumul_ptr + grad_gate_cumul_offset + (seq_len - 1) * grad_gate_cumul_stride_l)
    g_Z_cumul_offs = grad_Z_cumul_offset + (seq_len - 1)[:, None, None] * grad_Z_cumul_stride_l + \
                     tl.arange(0, HEAD_DIM)[None, :, None] * grad_Z_cumul_stride_h1 + \
                     tl.arange(0, HEAD_DIM)[None, None, :] * grad_Z_cumul_stride_h2
    g_Z_cumul = tl.load(grad_Z_cumul_ptr + g_Z_cumul_offs)

    # Compute dot product for weights gradient
    dot = n_i / (s_i + 1e-6)

    # Second pass: Compute gradients
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        v_idx = v_offset + (start + offs)[:, None] * v_stride_l + \
                tl.arange(0, HEAD_DIM)[None, :] * v_stride_h1
        gates_z_idx = gates_z_offset + (start + offs) * gates_z_stride_l
        grad_weights_idx = grad_weights_offset + (start + offs) * grad_weights_stride_l
        grad_max_cumul_idx = grad_max_cumul_offset + (start + offs) * grad_max_cumul_stride_l
        grad_norm_cumul_idx = grad_norm_cumul_offset + (start + offs) * grad_norm_cumul_stride_l
        grad_v_cumul_idx = grad_v_cumul_offset + (start + offs)[:, None] * grad_v_cumul_stride_l + \
                           tl.arange(0, HEAD_DIM)[None, :] * grad_v_cumul_stride_h1
        grad_Z_cumul_idx = grad_Z_cumul_offset + (start + offs)[:, None, None] * grad_Z_cumul_stride_l + \
                           tl.arange(0, HEAD_DIM)[None, :, None] * grad_Z_cumul_stride_h1 + \
                           tl.arange(0, HEAD_DIM)[None, None, :] * grad_Z_cumul_stride_h2
        grad_gates_z_idx = grad_gate_cumul_offset + (start + offs) * grad_gate_cumul_stride_l

        qk = tl.load(sim_ptr + idx, mask=mask, other=float("-inf"))
        v = tl.load(v_ptr + v_idx, mask=mask[:, None], other=0.0)
        g_weights = tl.load(grad_weights_ptr + grad_weights_idx, mask=mask, other=0.0)
        g_max_cumul = tl.load(grad_max_cumul_ptr + grad_max_cumul_idx, mask=mask, other=0.0)
        g_norm_cumul = tl.load(grad_norm_cumul_ptr + grad_norm_cumul_idx, mask=mask, other=0.0)
        g_v_cumul = tl.load(grad_v_cumul_ptr + grad_v_cumul_idx, mask=mask[:, None], other=0.0)
        p = tl.exp(qk - m_i) / (s_i + 1e-6)

        # Update max_indicator
        is_max = (qk >= m_i).to(DTYPE)
        max_indicator = tl.where(mask, is_max, 0.0)

        # Gradients
        g_sim = p * (g_weights - dot) - g_norm_cumul * max_indicator
        g_v = g_v_cumul * tl.exp(qk - m_i)[:, None]
        g_gates_z = g_gate_cumul
        g_gated_Z = tl.where(mask[:, None, None], g_Z_cumul, 0.0)

        tl.store(grad_sim_ptr + pid_b * grad_sim_stride_b + pid_h * grad_sim_stride_h + (start + offs) * grad_sim_stride_l, g_sim, mask=mask)
        tl.store(grad_v_ptr + pid_b * grad_v_stride_b + pid_h * grad_v_stride_h + (start + offs)[:, None] * grad_v_stride_l + \
                 tl.arange(0, HEAD_DIM)[None, :] * grad_v_stride_h1, g_v, mask=mask[:, None])
        tl.store(grad_gates_z_ptr + pid_b * grad_gates_z_stride_b + pid_h * grad_gates_z_stride_h + (start + offs) * grad_gates_z_stride_l, g_gates_z, mask=mask)
        tl.store(grad_gated_Z_ptr + pid_b * grad_Z_stride_b + pid_h * grad_Z_stride_h + (start + offs)[:, None, None] * grad_Z_stride_l + \
                 tl.arange(0, HEAD_DIM)[None, :, None] * grad_Z_stride_h1 + \
                 tl.arange(0, HEAD_DIM)[None, None, :] * grad_Z_stride_h2, g_gated_Z, mask=mask[:, None, None])

class BatchedScanFnTriton(torch.autograd.Function):
    @staticmethod
    def forward(sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
        batch_size, num_heads, seq_len = sim.shape
        assert v.shape == (batch_size, num_heads, seq_len, gated_Z.shape[-1]), "v must have shape [B, H, L, h]"
        assert gated_Z.shape == (batch_size, num_heads, seq_len, gated_Z.shape[-1], gated_Z.shape[-1]), "gated_Z must have shape [B, H, L, h, h]"
        assert gates_z.shape == sim.shape, "gates_z must have the same shape as sim"
        head_dim = gated_Z.shape[-1]
        sim = sim.contiguous().cuda()
        v = v.contiguous().cuda()
        gated_Z = gated_Z.contiguous().cuda()
        gates_z = gates_z.contiguous().cuda()
        weights = torch.empty_like(sim)
        max_cumul = torch.empty_like(sim)
        norm_cumul = torch.empty_like(sim)
        v_cumul = torch.empty_like(v)
        Z_cumul = torch.empty_like(gated_Z)
        gate_cumul = torch.empty_like(gates_z)
        grid = (batch_size, num_heads)
        triton_dtype = dtype_map.get(sim.dtype, tl.float32)

        scan_fn_triton_kernel[grid](
            sim,
            v,
            gated_Z,
            gates_z,
            weights,
            max_cumul,
            norm_cumul,
            v_cumul,
            Z_cumul,
            gate_cumul,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            sim.stride(0),
            sim.stride(1),
            sim.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            gated_Z.stride(0),
            gated_Z.stride(1),
            gated_Z.stride(2),
            gated_Z.stride(3),
            gated_Z.stride(4),
            gates_z.stride(0),
            gates_z.stride(1),
            gates_z.stride(2),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            max_cumul.stride(0),
            max_cumul.stride(1),
            max_cumul.stride(2),
            norm_cumul.stride(0),
            norm_cumul.stride(1),
            norm_cumul.stride(2),
            v_cumul.stride(0),
            v_cumul.stride(1),
            v_cumul.stride(2),
            v_cumul.stride(3),
            Z_cumul.stride(0),
            Z_cumul.stride(1),
            Z_cumul.stride(2),
            Z_cumul.stride(3),
            Z_cumul.stride(4),
            gate_cumul.stride(0),
            gate_cumul.stride(1),
            gate_cumul.stride(2),
            DTYPE=triton_dtype,
        )
        return weights, max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul

    @staticmethod
    def setup_context(ctx, inputs, output):
        sim, v, gated_Z, gates_z = inputs
        weights, max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = output
        ctx.save_for_backward(sim, v, gated_Z, gates_z, weights, max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul)
        ctx.triton_dtype = dtype_map.get(sim.dtype, tl.float32)
        ctx.shape = sim.shape
        ctx.head_dim = gated_Z.shape[-1]
        ctx.dtype = sim.dtype

    @staticmethod
    def backward(ctx, grad_weights, grad_max_cumul, grad_norm_cumul, grad_v_cumul, grad_Z_cumul, grad_gate_cumul):
        sim, v, gated_Z, gates_z, weights, max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = ctx.saved_tensors
        grad_sim = torch.empty_like(sim)
        grad_v = torch.empty_like(v)
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates_z = torch.empty_like(gates_z)
        b, h, l = ctx.shape
        head_dim = ctx.head_dim
        grid = (b, h)

        bwd_scan_fn_triton_kernel[grid](
            sim, v, gated_Z, gates_z,
            grad_weights, grad_max_cumul, grad_norm_cumul, grad_v_cumul, grad_Z_cumul, grad_gate_cumul,
            grad_sim, grad_v, grad_gated_Z, grad_gates_z,
            b, h, l, head_dim,
            sim.stride(0), sim.stride(1), sim.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            gated_Z.stride(0), gated_Z.stride(1), gated_Z.stride(2), gated_Z.stride(3), gated_Z.stride(4),
            gates_z.stride(0), gates_z.stride(1), gates_z.stride(2),
            grad_weights.stride(0), grad_weights.stride(1), grad_weights.stride(2),
            grad_max_cumul.stride(0), grad_max_cumul.stride(1), grad_max_cumul.stride(2),
            grad_norm_cumul.stride(0), grad_norm_cumul.stride(1), grad_norm_cumul.stride(2),
            grad_v_cumul.stride(0), grad_v_cumul.stride(1), grad_v_cumul.stride(2), grad_v_cumul.stride(3),
            grad_Z_cumul.stride(0), grad_Z_cumul.stride(1), grad_Z_cumul.stride(2),
            grad_Z_cumul.stride(3), grad_Z_cumul.stride(4),
            grad_gate_cumul.stride(0), grad_gate_cumul.stride(1), grad_gate_cumul.stride(2),
            grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
            grad_gated_Z.stride(0), grad_gated_Z.stride(1), grad_gated_Z.stride(2),
            grad_gated_Z.stride(3), grad_gated_Z.stride(4),
            grad_gates_z.stride(0), grad_gates_z.stride(1), grad_gates_z.stride(2),
            DTYPE=ctx.triton_dtype,
        )
        return grad_sim, grad_v, grad_gated_Z, grad_gates_z

def batched_scan_fn_triton(sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
    return BatchedScanFnTriton.apply(sim, v, gated_Z, gates_z)

@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)

if __name__ == "__main__":
    # Gradient check
    torch.manual_seed(1746)
    batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 8
    sim = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float64, device="cuda", requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float64, device="cuda", requires_grad=True)
    gated_Z = torch.randn(batch_size, num_heads, seq_len, head_dim, head_dim, dtype=torch.float64, device="cuda", requires_grad=True)
    gates_z = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float64, device="cuda", requires_grad=True)
    torch.autograd.gradcheck(
        BatchedScanFnTriton.apply,
        inputs=(sim, v, gated_Z, gates_z),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-5,
        fast_mode=True
    )
    print("Gradient check passed")

    # Performance test
    batch_size, num_heads, seq_len, head_dim = 16, 32, 16000, 16
    sim = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
    gated_Z = torch.randn(batch_size, num_heads, seq_len, head_dim, head_dim, dtype=torch.float32, device="cuda")
    gates_z = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device="cuda")
    weights, max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = batched_scan_fn_triton(sim, v, gated_Z, gates_z)
    ref = torch_softmax(sim, dim=-1)
    print(f"Max absolute error: {(weights - ref).abs().max().item()}")
    t1 = do_bench(lambda: batched_scan_fn_triton(sim, v, gated_Z, gates_z))
    t2 = do_bench(lambda: torch_softmax(sim, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = (sim.numel() + v.numel() + gated_Z.numel() + gates_z.numel() + weights.numel() + \
                 max_cumul.numel() + norm_cumul.numel() + v_cumul.numel() + Z_cumul.numel() + gate_cumul.numel()) * sim.element_size()
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")