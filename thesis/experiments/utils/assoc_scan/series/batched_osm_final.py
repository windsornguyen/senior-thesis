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
def softmax_combine_fn(m_x, d_x, sim_x, m_y, d_y, sim_y):
    """Combines states for softmax-related scan."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    sim_new = sim_x * exp_x + sim_y * exp_y
    return m_new, d_new, sim_new


@triton.jit
def additive_combine_fn(x, y):
    """Combines states for additive scan (Z or g)."""
    return x + y


@triton.jit
def backward_combine_fn(m_x, d_x, n_x, sim_x, m_y, d_y, n_y, sim_y):
    """Combines states for backward softmax-related scan."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    sim_new = sim_x * exp_x + sim_y * exp_y
    return m_new, d_new, n_new, sim_new


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
def fwd_online_softmax_2pass_kernel(
    x_ptr,
    sim_ptr,
    Z_ptr,
    g_ptr,
    out_ptr,
    sim_score_ptr,
    Z_cumul_ptr,
    gate_cumul_ptr,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    HEAD_DIM: tl.constexpr,
    stride_b: int,
    stride_h: int,
    stride_l: int,
    sim_stride_b: int,
    sim_stride_h: int,
    sim_stride_l: int,
    Z_stride_b: int,
    Z_stride_h: int,
    Z_stride_l: int,
    Z_stride_h1: int,
    Z_stride_h2: int,
    g_stride_b: int,
    g_stride_h: int,
    g_stride_l: int,
    out_stride_b: int,
    out_stride_h: int,
    out_stride_l: int,
    sim_score_stride_b: int,
    sim_score_stride_h: int,
    sim_score_stride_l: int,
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
    sim_offset = pid_batch * sim_stride_b + pid_head * sim_stride_h
    Z_offset = pid_batch * Z_stride_b + pid_head * Z_stride_h
    g_offset = pid_batch * g_stride_b + pid_head * g_stride_h
    out_offset = pid_batch * out_stride_b + pid_head * out_stride_h
    sim_score_offset = pid_batch * sim_score_stride_b + pid_head * sim_score_stride_h
    Z_cumul_offset = pid_batch * Z_cumul_stride_b + pid_head * Z_cumul_stride_h
    gate_cumul_offset = pid_batch * gate_cumul_stride_b + pid_head * gate_cumul_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    sim_i = tl.full((), 0.0, dtype=DTYPE)
    Z_i = tl.zeros([HEAD_DIM, HEAD_DIM], dtype=DTYPE)
    g_i = tl.full((), 0.0, dtype=DTYPE)

    # First pass: Compute m_i, d_i', sim_i, Z_i, g_i
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        sim_offsets = sim_offset + (start_col + offs) * sim_stride_l
        g_offsets = g_offset + (start_col + offs) * g_stride_l

        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)
        sim_state = tl.exp(sim - m)

        # Scan for m, d, sim
        m, d, sim_state = tl.associative_scan((m, d, sim_state), axis=0, combine_fn=softmax_combine_fn)

        # Load Z
        Z_offs = (
            Z_offset
            + (start_col + offs)[:, None, None] * Z_stride_l
            + tl.arange(0, HEAD_DIM)[None, :, None] * Z_stride_h1
            + tl.arange(0, HEAD_DIM)[None, None, :] * Z_stride_h2
        )
        Z = tl.load(Z_ptr + Z_offs, mask=mask[:, None, None], other=0.0)

        # Separate scans for Z and g
        Z = tl.associative_scan(Z, axis=0, combine_fn=additive_combine_fn)
        g_state = tl.associative_scan(g, axis=0, combine_fn=additive_combine_fn)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_col - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        sim_local = tl.where(offs == last_idx, sim_state, 0.0)
        g_local = tl.where(offs == last_idx, g_state, 0.0)
        Z_local = tl.where(offs[:, None, None] == last_idx, Z, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)
        sim_local = tl.sum(sim_local, axis=0)
        g_local = tl.sum(g_local, axis=0)
        Z_local = tl.sum(Z_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        sim_i = sim_i * tl.exp(old_m_i - m_i) + sim_local * tl.exp(m_local - m_i)
        g_i = g_i + g_local
        Z_i = Z_i + Z_local

    # Store cumulative outputs
    tl.store(sim_score_ptr + sim_score_offset, sim_i)
    tl.store(gate_cumul_ptr + gate_cumul_offset, g_i)
    Z_cumul_offs = (
        Z_cumul_offset
        + tl.arange(0, HEAD_DIM)[:, None] * Z_cumul_stride_h1
        + tl.arange(0, HEAD_DIM)[None, :] * Z_cumul_stride_h2
    )
    tl.store(Z_cumul_ptr + Z_cumul_offs, Z_i)

    # Second pass: Compute softmax outputs
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        tl.store(out_ptr + out_offset + (start_col + offs) * out_stride_l, softmax_out, mask=mask)


@triton.autotune(
    configs=[conf for conf in get_softmax_configs(seq_len=32)],
    key=["batch_size", "num_heads", "seq_len"],
)
@triton.jit
def bwd_online_softmax_2pass_kernel(
    x_ptr,
    sim_ptr,
    Z_ptr,
    g_ptr,
    grad_softmax_ptr,
    grad_sim_score_ptr,
    grad_Z_cumul_ptr,
    grad_gate_cumul_ptr,
    grad_x_ptr,
    grad_sim_ptr,
    grad_Z_ptr,
    grad_g_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_l: tl.constexpr,
    sim_stride_b: tl.constexpr,
    sim_stride_h: tl.constexpr,
    sim_stride_l: tl.constexpr,
    Z_stride_b: tl.constexpr,
    Z_stride_h: tl.constexpr,
    Z_stride_l: tl.constexpr,
    Z_stride_h1: tl.constexpr,
    Z_stride_h2: tl.constexpr,
    g_stride_b: tl.constexpr,
    g_stride_h: tl.constexpr,
    g_stride_l: tl.constexpr,
    grad_softmax_stride_b: tl.constexpr,
    grad_softmax_stride_h: tl.constexpr,
    grad_softmax_stride_l: tl.constexpr,
    grad_sim_score_stride_b: tl.constexpr,
    grad_sim_score_stride_h: tl.constexpr,
    grad_sim_score_stride_l: tl.constexpr,
    grad_Z_cumul_stride_b: tl.constexpr,
    grad_Z_cumul_stride_h: tl.constexpr,
    grad_Z_cumul_stride_l: tl.constexpr,
    grad_Z_cumul_stride_h1: tl.constexpr,
    grad_Z_cumul_stride_h2: tl.constexpr,
    grad_gate_cumul_stride_b: tl.constexpr,
    grad_gate_cumul_stride_h: tl.constexpr,
    grad_gate_cumul_stride_l: tl.constexpr,
    grad_x_stride_b: tl.constexpr,
    grad_x_stride_h: tl.constexpr,
    grad_x_stride_l: tl.constexpr,
    grad_sim_stride_b: tl.constexpr,
    grad_sim_stride_h: tl.constexpr,
    grad_sim_stride_l: tl.constexpr,
    grad_Z_stride_b: tl.constexpr,
    grad_Z_stride_h: tl.constexpr,
    grad_Z_stride_l: tl.constexpr,
    grad_Z_stride_h1: tl.constexpr,
    grad_Z_stride_h2: tl.constexpr,
    grad_g_stride_b: tl.constexpr,
    grad_g_stride_h: tl.constexpr,
    grad_g_stride_l: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    row_offset = pid_b * stride_b + pid_h * stride_h
    sim_offset = pid_b * sim_stride_b + pid_h * sim_stride_h
    Z_offset = pid_b * Z_stride_b + pid_h * Z_stride_h
    g_offset = pid_b * g_stride_b + pid_h * g_stride_h
    grad_softmax_offset = pid_b * grad_softmax_stride_b + pid_h * grad_softmax_stride_h
    grad_sim_score_offset = pid_b * grad_sim_score_stride_b + pid_h * grad_sim_score_stride_h
    grad_Z_cumul_offset = pid_b * grad_Z_cumul_stride_b + pid_h * grad_Z_cumul_stride_h
    grad_gate_cumul_offset = pid_b * grad_gate_cumul_stride_b + pid_h * grad_gate_cumul_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    n_i = tl.full((), 0.0, dtype=DTYPE)
    sim_i = tl.full((), 0.0, dtype=DTYPE)
    Z_i = tl.zeros([HEAD_DIM, HEAD_DIM], dtype=DTYPE)
    g_i = tl.full((), 0.0, dtype=DTYPE)

    # First pass: Recompute m_i, d_i', sim_i, Z_i, g_i, and compute n_i
    for k in range(tl.cdiv(seq_len, BLOCK_SIZE) - 1, -1, -1):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        sim_idx = sim_offset + (start + offs) * sim_stride_l
        g_idx = g_offset + (start + offs) * g_stride_l
        grad_softmax_idx = grad_softmax_offset + (start + offs) * grad_softmax_stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_idx, mask=mask, other=0.0)
        g = tl.load(g_ptr + g_idx, mask=mask, other=0.0)
        g_softmax = tl.load(grad_softmax_ptr + grad_softmax_idx, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)
        n = g_softmax * d
        sim_state = tl.exp(sim - m)

        # Scan for m, d, n, sim (forward)
        m, d, n, sim_state = tl.associative_scan((m, d, n, sim_state), axis=0, combine_fn=backward_combine_fn)

        # Load Z
        Z_offs = (
            Z_offset
            + (start + offs)[:, None, None] * Z_stride_l
            + tl.arange(0, HEAD_DIM)[None, :, None] * Z_stride_h1
            + tl.arange(0, HEAD_DIM)[None, None, :] * Z_stride_h2
        )
        Z = tl.load(Z_ptr + Z_offs, mask=mask[:, None, None], other=0.0)

        # Reverse scans for Z and g gradients
        rev_idx = seq_len - 1 - (start + offs)
        safe_idx = tl.where(mask, rev_idx, 0)
        Z_grad_offs = (
            Z_offset
            + safe_idx[:, None, None] * grad_Z_stride_l
            + tl.arange(0, HEAD_DIM)[None, :, None] * grad_Z_stride_h1
            + tl.arange(0, HEAD_DIM)[None, None, :] * grad_Z_stride_h2
        )
        g_grad_offs = g_offset + safe_idx * grad_g_stride_l

        g_Z = tl.load(grad_Z_ptr + Z_grad_offs, mask=mask[:, None, None], other=0.0)
        g_g = tl.load(grad_g_ptr + g_grad_offs, mask=mask, other=0.0)

        g_Z = tl.associative_scan(g_Z, axis=0, combine_fn=additive_combine_fn, reverse=False)
        g_g = tl.associative_scan(g_g, axis=0, combine_fn=additive_combine_fn, reverse=False)

        tl.store(grad_Z_ptr + Z_grad_offs, g_Z, mask=mask[:, None, None])
        tl.store(grad_g_ptr + g_grad_offs, g_g, mask=mask)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        n_local = tl.where(offs == last_idx, n, 0.0)
        sim_local = tl.where(offs == last_idx, sim_state, 0.0)
        g_local = tl.where(offs == last_idx, g, 0.0)
        Z_local = tl.where(offs[:, None, None] == last_idx, Z, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)
        n_local = tl.sum(n_local, axis=0)
        sim_local = tl.sum(sim_local, axis=0)
        g_local = tl.sum(g_local, axis=0)
        Z_local = tl.sum(Z_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        n_i = n_i * tl.exp(old_m_i - m_i) + n_local * tl.exp(m_local - m_i)
        sim_i = sim_i * tl.exp(old_m_i - m_i) + sim_local * tl.exp(m_local - m_i)
        g_i = g_i + g_local
        Z_i = Z_i + Z_local

    # Load upstream gradients
    g_sim_score = tl.load(grad_sim_score_ptr + grad_sim_score_offset)
    g_gate_cumul = tl.load(grad_gate_cumul_ptr + grad_gate_cumul_offset)
    g_Z_cumul_offs = (
        grad_Z_cumul_offset
        + tl.arange(0, HEAD_DIM)[:, None] * grad_Z_cumul_stride_h1
        + tl.arange(0, HEAD_DIM)[None, :] * grad_Z_cumul_stride_h2
    )
    g_Z_cumul = tl.load(grad_Z_cumul_ptr + g_Z_cumul_offs)

    # Compute dot product for softmax gradient
    dot = n_i / (d_i + 1e-6)

    # Indicator for whether x_i is the maximum
    max_indicator = tl.zeros([BLOCK_SIZE], dtype=DTYPE)

    # Second pass: Compute gradients
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        sim_idx = sim_offset + (start + offs) * sim_stride_l
        g_idx = g_offset + (start + offs) * g_stride_l
        grad_softmax_idx = grad_softmax_offset + (start + offs) * grad_softmax_stride_l
        grad_Z_offs = (
            Z_offset
            + (start + offs)[:, None, None] * grad_Z_stride_l
            + tl.arange(0, HEAD_DIM)[None, :, None] * grad_Z_stride_h1
            + tl.arange(0, HEAD_DIM)[None, None, :] * grad_Z_stride_h2
        )

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_idx, mask=mask, other=0.0)
        g = tl.load(g_ptr + g_idx, mask=mask, other=0.0)
        g_softmax = tl.load(grad_softmax_ptr + grad_softmax_idx, mask=mask, other=0.0)
        p = tl.exp(x - m_i) / (d_i + 1e-6)

        # Update max_indicator
        is_max = (x >= m_i).to(DTYPE)
        max_indicator = tl.where(mask, is_max, max_indicator)

        # Gradients
        gx = p * (g_softmax - dot) - g_sim_score * sim_i * max_indicator
        g_sim = g_sim_score * tl.exp(sim - m_i)
        g_g = g_gate_cumul
        g_Z = tl.where(mask[:, None, None], g_Z_cumul, 0.0)

        tl.store(
            grad_x_ptr + pid_b * grad_x_stride_b + pid_h * grad_x_stride_h + (start + offs) * grad_x_stride_l,
            gx,
            mask=mask,
        )
        tl.store(
            grad_sim_ptr + pid_b * grad_sim_stride_b + pid_h * grad_sim_stride_h + (start + offs) * grad_sim_stride_l,
            g_sim,
            mask=mask,
        )
        tl.store(
            grad_g_ptr + pid_b * grad_g_stride_b + pid_h * grad_g_stride_h + (start + offs) * grad_g_stride_l,
            g_g,
            mask=mask,
        )
        tl.store(grad_Z_ptr + grad_Z_offs, g_Z, mask=mask[:, None, None])


class OnlineSoftmax2Pass(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, sim: torch.Tensor, Z: torch.Tensor, g: torch.Tensor):
        batch_size, num_heads, seq_len = x.shape
        assert sim.shape == x.shape, "Similarity tensor must have the same shape as input"
        assert Z.shape[:3] == x.shape, "Z must have shape [B, H, L, h, h]"
        assert g.shape == x.shape, "Gate tensor must have the same shape as input"
        head_dim = Z.shape[-1]
        x = x.contiguous().cuda()
        sim = sim.contiguous().cuda()
        Z = Z.contiguous().cuda()
        g = g.contiguous().cuda()
        softmax = torch.empty_like(x)
        sim_score = torch.empty(batch_size, num_heads, 1, dtype=x.dtype, device="cuda")
        Z_cumul = torch.empty(batch_size, num_heads, 1, head_dim, head_dim, dtype=x.dtype, device="cuda")
        gate_cumul = torch.empty(batch_size, num_heads, 1, dtype=x.dtype, device="cuda")
        grid = (batch_size, num_heads)
        triton_dtype = dtype_map.get(x.dtype, tl.float32)

        fwd_online_softmax_2pass_kernel[grid](
            x,
            sim,
            Z,
            g,
            softmax,
            sim_score,
            Z_cumul,
            gate_cumul,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            sim.stride(0),
            sim.stride(1),
            sim.stride(2),
            Z.stride(0),
            Z.stride(1),
            Z.stride(2),
            Z.stride(3),
            Z.stride(4),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            softmax.stride(0),
            softmax.stride(1),
            softmax.stride(2),
            sim_score.stride(0),
            sim_score.stride(1),
            sim_score.stride(2),
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
        return softmax, sim_score, Z_cumul, gate_cumul

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, sim, Z, g = inputs
        softmax, sim_score, Z_cumul, gate_cumul = output
        ctx.save_for_backward(x, sim, Z, g, softmax, sim_score, Z_cumul, gate_cumul)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)
        ctx.shape = x.shape
        ctx.head_dim = Z.shape[-1]
        ctx.dtype = x.dtype

    @staticmethod
    def backward(ctx, grad_softmax, grad_sim_score, grad_Z_cumul, grad_gate_cumul):
        x, sim, Z, g, softmax, sim_score, Z_cumul, gate_cumul = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        grad_sim = torch.empty_like(sim)
        grad_Z = torch.empty_like(Z)
        grad_g = torch.empty_like(g)
        b, h, l = ctx.shape
        head_dim = ctx.head_dim
        grid = (b, h)

        bwd_online_softmax_2pass_kernel[grid](
            x,
            sim,
            Z,
            g,
            grad_softmax,
            grad_sim_score,
            grad_Z_cumul,
            grad_gate_cumul,
            grad_x,
            grad_sim,
            grad_Z,
            grad_g,
            b,
            h,
            l,
            head_dim,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            sim.stride(0),
            sim.stride(1),
            sim.stride(2),
            Z.stride(0),
            Z.stride(1),
            Z.stride(2),
            Z.stride(3),
            Z.stride(4),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            grad_softmax.stride(0),
            grad_softmax.stride(1),
            grad_softmax.stride(2),
            grad_sim_score.stride(0),
            grad_sim_score.stride(1),
            grad_sim_score.stride(2),
            grad_Z_cumul.stride(0),
            grad_Z_cumul.stride(1),
            grad_Z_cumul.stride(2),
            grad_Z_cumul.stride(3),
            grad_Z_cumul.stride(4),
            grad_gate_cumul.stride(0),
            grad_gate_cumul.stride(1),
            grad_gate_cumul.stride(2),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_x.stride(2),
            grad_sim.stride(0),
            grad_sim.stride(1),
            grad_sim.stride(2),
            grad_Z.stride(0),
            grad_Z.stride(1),
            grad_Z.stride(2),
            grad_Z.stride(3),
            grad_Z.stride(4),
            grad_g.stride(0),
            grad_g.stride(1),
            grad_g.stride(2),
            DTYPE=ctx.triton_dtype,
        )
        return grad_x, grad_sim, grad_Z, grad_g


def online_softmax_2pass(x: torch.Tensor, sim: torch.Tensor, Z: torch.Tensor, g: torch.Tensor):
    return OnlineSoftmax2Pass.apply(x, sim, Z, g)


@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)


# --- Reference Implementation and Utilities ---
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    leaves = (qk_slice, torch.ones_like(qk_slice), v_slice, Z_slice, g_slice)
    return assoc_scan_ref(combine_fn=combine_fn_ref, xs=leaves, dim=0, combine_mode="generic")


def batched_scan_fn(
    sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run scan_fn independently for every (B,H) stream.

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
    B, H = sim.shape[0], sim.shape[1]
    # Flatten B and H dimensions: (B, H, ...) -> (B*H, ...)
    sim_flat = sim.flatten(0, 1)  # (B*H, L)
    v_flat = v.flatten(0, 1)  # (B*H, L, h)
    gated_Z_flat = gated_Z.flatten(0, 1)  # (B*H, L, h, h)
    gates_z_flat = gates_z.flatten(0, 1)  # (B*H, L)

    # Apply scan_fn to each (B*H) stream
    scan_all = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    result = scan_all(sim_flat, v_flat, gated_Z_flat, gates_z_flat)  # Tuple of 5 tensors

    # Reshape each output tensor back to (B, H, ...)
    return tuple(t.reshape(B, H, *t.shape[1:]) for t in result)


if __name__ == "__main__":
    # Gradient check
    torch.manual_seed(1746)
    batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 8
    x = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float64, device="cuda", requires_grad=True)
    sim = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float64, device="cuda", requires_grad=True)
    Z = torch.randn(
        batch_size, num_heads, seq_len, head_dim, head_dim, dtype=torch.float64, device="cuda", requires_grad=True
    )
    g = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float64, device="cuda", requires_grad=True)
    torch.autograd.gradcheck(
        OnlineSoftmax2Pass.apply,
        inputs=(x, sim, Z, g),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-5,
        fast_mode=True,
    )
    print("Gradient check passed")

    """
    gated Z <=> Z_cumul / gate_cumul
    exact sm attn: weights <=> our softmax output
    norm_cumul <=> d_i (internally, not returned for now, should return d_i for all i)
    v_cumul: we don't have? since we don't add v?
    """

    # Performance test
    B, H, L, D, h = 16, 32, 16000, 16, 16
    q = torch.randn(B, H, L, h, dtype=torch.float32, device="cuda")
    m = torch.randn(B, H, L, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, L, h, dtype=torch.float32, device="cuda")
    Z = torch.randn(B, H, L, h, h, dtype=torch.float32, device="cuda")
    g_ref = torch.randn(B, H, L, dtype=torch.float32, device="cuda")

    n = torch.randn(B, H, L, h, dtype=torch.float32, device="cuda")

    x = torch.randn(B, H, L, dtype=torch.float32, device="cuda")
    sim = torch.randn(B, H, L, dtype=torch.float32, device="cuda")
    g = torch.randn(B, H, L, dtype=torch.float32, device="cuda")
    softmax, sim_score, Z_cumul, gate_cumul = online_softmax_2pass(x, sim, Z, g)
    ref = torch_softmax(x, dim=-1)
    print(f"Max absolute error: {(softmax - ref).abs().max().item()}")

    max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = batched_scan_fn(sim=sim, v=v, gated_Z=Z, gates_z=g_ref)

    ref_linear_attn = v_cumul / (norm_cumul[..., None] + 1e-6)
    ref_weights = torch.exp(sim - max_cumul) / (norm_cumul + 1e-6)
    ref_H = Z_cumul / (gate_cumul.unsqueeze(-1).unsqueeze(-1) + 1e-6)

    ref_Y_base = torch.einsum("bhtp,bhtpn->bhtn", q, ref_H)
    ref_Y_lin = torch.einsum("bhtp,bhtpn->bhtn", q, ref_linear_attn.unsqueeze(-2))
    ref_Y = ref_Y_base + (ref_Y_lin - ref_Y_base) * ref_weights[..., None]

    print(f"Max absolute error: {(sim_score - ref_linear_attn).abs().max().item()}")

    t1 = do_bench(lambda: online_softmax_2pass(x, sim, Z, g))
    t2 = do_bench(lambda: torch_softmax(x, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = (
        x.numel()
        + sim.numel()
        + Z.numel()
        + g.numel()
        + softmax.numel()
        + sim_score.numel()
        + Z_cumul.numel()
        + gate_cumul.numel()
    ) * x.element_size()
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")
