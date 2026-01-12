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
def softmax_combine_fn(m_x, d_x, Z_x, g_x, m_y, d_y, Z_y, g_y):
    """Combines two states for forward online softmax scan with Z and g."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    Z_new = Z_x + Z_y  # Cumulative sum for Z
    g_new = g_x + g_y  # Cumulative sum for g
    return m_new, d_new, Z_new, g_new


@triton.jit
def backward_combine_fn(m_x, d_x, n_x, Z_x, g_x, m_y, d_y, n_y, Z_y, g_y):
    """Combines two states for backward online softmax scan with Z and g."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    Z_new = Z_x + Z_y  # Cumulative sum for Z gradients
    g_new = g_x + g_y  # Cumulative sum for g gradients
    return m_new, d_new, n_new, Z_new, g_new


def get_softmax_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024, 2048, 4096]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
        )
    return configs


def keep_config(conf, seq_len=None):
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    if seq_len is not None and BLOCK_SIZE > seq_len:
        return False
    return True


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["batch_size", "num_heads", "seq_len", "head_dim"],
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
    g_cumul_ptr,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: tl.constexpr,
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
    g_cumul_stride_b: int,
    g_cumul_stride_h: int,
    g_cumul_stride_l: int,
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
    g_cumul_offset = pid_batch * g_cumul_stride_b + pid_head * g_cumul_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    Z_cumul_i = tl.zeros([head_dim, head_dim], dtype=DTYPE)
    g_cumul_i = tl.zeros([1], dtype=DTYPE)

    # First pass: Compute m_i, d_i, Z_cumul_i, and g_cumul_i using associative scan
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        x_offsets = start_offset + (start_col + offs) * stride_l
        Z_offsets = Z_offset + (start_col + offs) * Z_stride_l
        g_offsets = g_offset + (start_col + offs) * g_stride_l
        Z_cumul_offsets = Z_cumul_offset + (start_col + offs) * Z_cumul_stride_l
        g_cumul_offsets = g_cumul_offset + (start_col + offs) * g_cumul_stride_l

        x = tl.load(x_ptr + x_offsets, mask=mask, other=float("-inf"))
        Z = tl.load(
            Z_ptr
            + Z_offsets[:, None, None]
            + tl.arange(0, head_dim)[None, :, None] * Z_stride_h1
            + tl.arange(0, head_dim)[None, None, :] * Z_stride_h2,
            mask=mask[:, None, None],
            other=0.0,
        )
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)

        # Perform scan over m, d, Z, g
        m, d, Z, g = tl.associative_scan((m, d, Z, g), axis=0, combine_fn=softmax_combine_fn)

        # Store cumulative Z and g
        tl.store(
            Z_cumul_ptr
            + Z_cumul_offsets[:, None, None]
            + tl.arange(0, head_dim)[None, :, None] * Z_cumul_stride_h1
            + tl.arange(0, head_dim)[None, None, :] * Z_cumul_stride_h2,
            Z,
            mask=mask[:, None, None],
        )
        tl.store(g_cumul_ptr + g_cumul_offsets, g, mask=mask)

        # Update global accumulators
        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_col - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        Z_local = tl.where(offs[:, None, None] == last_idx, Z, 0.0)
        g_local = tl.where(offs == last_idx, g, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)
        Z_local = tl.sum(Z_local, axis=0)
        g_local = tl.sum(g_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        Z_cumul_i = Z_cumul_i + Z_local
        g_cumul_i = g_cumul_i + g_local

    # Second pass: Compute softmax outputs and sim_score
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        x_offsets = start_offset + (start_col + offs) * stride_l
        sim_offsets = sim_offset + (start_col + offs) * sim_stride_l
        out_offsets = out_offset + (start_col + offs) * out_stride_l
        sim_score_offsets = sim_score_offset + (start_col + offs) * sim_score_stride_l

        x = tl.load(x_ptr + x_offsets, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_offsets, mask=mask, other=0.0)
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        sim_score_out = tl.exp(sim - m_i)
        tl.store(out_ptr + out_offsets, softmax_out, mask=mask)
        tl.store(sim_score_ptr + sim_score_offsets, sim_score_out, mask=mask)


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["batch_size", "num_heads", "seq_len", "head_dim"],
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
    grad_g_cumul_ptr,
    grad_x_ptr,
    grad_sim_ptr,
    grad_Z_ptr,
    grad_g_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
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
    grad_g_cumul_stride_b: tl.constexpr,
    grad_g_cumul_stride_h: tl.constexpr,
    grad_g_cumul_stride_l: tl.constexpr,
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
    grad_g_cumul_offset = pid_b * grad_g_cumul_stride_b + pid_h * grad_g_cumul_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    n_i = tl.full((), 0.0, dtype=DTYPE)
    Z_cumul_i = tl.zeros([head_dim, head_dim], dtype=DTYPE)
    g_cumul_i = tl.zeros([1], dtype=DTYPE)

    # First pass: Recompute m_i, d_i, compute n_i, Z_cumul_i, g_cumul_i
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        grad_softmax_idx = grad_softmax_offset + (start + offs) * grad_softmax_stride_l
        Z_idx = Z_offset + (start + offs) * Z_stride_l
        g_idx = g_offset + (start + offs) * g_stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        g_softmax = tl.load(grad_softmax_ptr + grad_softmax_idx, mask=mask, other=0.0)
        Z = tl.load(
            Z_ptr
            + Z_idx[:, None, None]
            + tl.arange(0, head_dim)[None, :, None] * Z_stride_h1
            + tl.arange(0, head_dim)[None, None, :] * Z_stride_h2,
            mask=mask[:, None, None],
            other=0.0,
        )
        g = tl.load(g_ptr + g_idx, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)
        n = g_softmax * d

        m, d, n, Z, g = tl.associative_scan((m, d, n, Z, g), axis=0, combine_fn=backward_combine_fn)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        n_local = tl.where(offs == last_idx, n, 0.0)
        Z_local = tl.where(offs[:, None, None] == last_idx, Z, 0.0)
        g_local = tl.where(offs == last_idx, g, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)
        n_local = tl.sum(n_local, axis=0)
        Z_local = tl.sum(Z_local, axis=0)
        g_local = tl.sum(g_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        n_i = n_i * tl.exp(old_m_i - m_i) + n_local * tl.exp(m_local - m_i)
        Z_cumul_i = Z_cumul_i + Z_local
        g_cumul_i = g_cumul_i + g_local

    # Compute dot product for softmax gradient
    dot = n_i / (d_i + 1e-6)

    # Second pass: Compute gradients for x, sim, Z, and g
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        sim_idx = sim_offset + (start + offs) * sim_stride_l
        Z_idx = Z_offset + (start + offs) * Z_stride_l
        g_idx = g_offset + (start + offs) * g_stride_l
        grad_softmax_idx = grad_softmax_offset + (start + offs) * grad_softmax_stride_l
        grad_sim_score_idx = grad_sim_score_offset + (start + offs) * grad_sim_score_stride_l
        grad_Z_cumul_idx = grad_Z_cumul_offset + (start + offs) * grad_Z_cumul_stride_l
        grad_g_cumul_idx = grad_g_cumul_offset + (start + offs) * grad_g_cumul_stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_idx, mask=mask, other=0.0)
        g_softmax = tl.load(grad_softmax_ptr + grad_softmax_idx, mask=mask, other=0.0)
        g_sim_score = tl.load(grad_sim_score_ptr + grad_sim_score_idx, mask=mask, other=0.0)
        grad_Z_cumul = tl.load(
            grad_Z_cumul_ptr
            + grad_Z_cumul_idx[:, None, None]
            + tl.arange(0, head_dim)[None, :, None] * grad_Z_cumul_stride_h1
            + tl.arange(0, head_dim)[None, None, :] * grad_Z_cumul_stride_h2,
            mask=mask[:, None, None],
            other=0.0,
        )
        grad_g_cumul = tl.load(grad_g_cumul_ptr + grad_g_cumul_idx, mask=mask, other=0.0)
        p = tl.exp(x - m_i) / (d_i + 1e-6)

        # Compute max_indicator
        is_max = (x == m_i).to(DTYPE)
        max_count = tl.sum(is_max, axis=0)
        max_indicator = tl.where(max_count > 0, is_max / max_count, 0.0)

        # Gradient for sim
        g_sim = g_sim_score * tl.exp(sim - m_i)

        # Gradient for x
        sim_score_grad = tl.sum(g_sim_score * tl.exp(sim - m_i), axis=0)
        gx = p * (g_softmax - dot) - max_indicator * sim_score_grad

        # Gradient for Z and g (cumulative sum, so gradient is direct pass-through)
        g_Z = grad_Z_cumul
        g_g = grad_g_cumul

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
            grad_Z_ptr
            + Z_idx[:, None, None]
            + tl.arange(0, head_dim)[None, :, None] * grad_Z_stride_h1
            + tl.arange(0, head_dim)[None, None, :] * grad_Z_stride_h2,
            g_Z,
            mask=mask[:, None, None],
        )
        tl.store(
            grad_g_ptr + pid_b * grad_g_stride_b + pid_h * grad_g_stride_h + (start + offs) * grad_g_stride_l,
            g_g,
            mask=mask,
        )


class OnlineSoftmax2Pass(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, sim: torch.Tensor, Z: torch.Tensor, g: torch.Tensor):
        batch_size, num_heads, seq_len = x.shape
        head_dim = Z.shape[-1]  # h from [B, H, L, h, h]
        assert sim.shape == x.shape, "Similarity tensor must have the same shape as input"
        assert Z.shape == (batch_size, num_heads, seq_len, head_dim, head_dim), "Z must have shape [B, H, L, h, h]"
        assert g.shape == (batch_size, num_heads, seq_len), "gates must have shape [B, H, L]"
        x = x.contiguous().cuda()
        sim = sim.contiguous().cuda()
        Z = Z.contiguous().cuda()
        g = g.contiguous().cuda()
        softmax = torch.empty_like(x)
        sim_score = torch.empty_like(x)
        Z_cumul = torch.empty_like(Z)
        g_cumul = torch.empty_like(g)
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
            g_cumul,
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
            g_cumul.stride(0),
            g_cumul.stride(1),
            g_cumul.stride(2),
            DTYPE=triton_dtype,
            head_dim=head_dim,
        )
        return softmax, sim_score, Z_cumul, g_cumul

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, sim, Z, g = inputs
        softmax, sim_score, Z_cumul, g_cumul = output
        ctx.save_for_backward(x, sim, Z, g, softmax, sim_score, Z_cumul, g_cumul)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)
        ctx.shape = x.shape
        ctx.head_dim = Z.shape[-1]
        ctx.dtype = x.dtype

    @staticmethod
    def backward(ctx, grad_softmax, grad_sim_score, grad_Z_cumul, grad_g_cumul):
        x, sim, Z, g, softmax, sim_score, Z_cumul, g_cumul = ctx.saved_tensors
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
            grad_g_cumul,
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
            grad_g_cumul.stride(0),
            grad_g_cumul.stride(1),
            grad_g_cumul.stride(2),
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
            head_dim=head_dim,
        )
        return grad_x, grad_sim, grad_Z, grad_g


def online_softmax_2pass(x: torch.Tensor, sim: torch.Tensor, Z: torch.Tensor, g: torch.Tensor):
    return OnlineSoftmax2Pass.apply(x, sim, Z, g)


@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)


if __name__ == "__main__":
    # Gradient check
    torch.manual_seed(1746)
    batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 16
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
        fast_mode=False,
    )
    print("Gradient check passed")

    # Performance test
    batch_size, num_heads, seq_len, head_dim = 16, 32, 16000, 64
    x = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device="cuda")
    sim = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device="cuda")
    Z = torch.randn(batch_size, num_heads, seq_len, head_dim, head_dim, dtype=torch.float32, device="cuda")
    g = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device="cuda")
    softmax, sim_score, Z_cumul, g_cumul = online_softmax_2pass(x, sim, Z, g)
    ref = torch_softmax(x, dim=-1)
    print(f"Max absolute error: {(softmax - ref).abs().max().item()}")
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
        + g_cumul.numel()
    ) * x.element_size()
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")
