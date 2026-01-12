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
    """Combines two states for forward online softmax scan with similarity score."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    sim_new = sim_x * exp_x + sim_y * exp_y
    return m_new, d_new, sim_new

@triton.jit
def backward_combine_fn(m_x, d_x, n_x, sim_x, m_y, d_y, n_y, sim_y):
    """Combines two states for backward online softmax scan with similarity score."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    sim_new = sim_x * exp_x + sim_y * exp_y
    return m_new, d_new, n_new, sim_new

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
    key=["batch_size", "num_heads", "seq_len"],
)
@triton.jit
def fwd_online_softmax_2pass_kernel(
    x_ptr,
    sim_ptr,
    out_ptr,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    stride_b: int,
    stride_h: int,
    stride_l: int,
    sim_stride_b: int,
    sim_stride_h: int,
    sim_stride_l: int,
    out_stride_b: int,
    out_stride_h: int,
    out_stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_head * stride_h
    sim_offset = pid_batch * sim_stride_b + pid_head * sim_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    sim_i = tl.full((), 0.0, dtype=DTYPE)

    # First pass: Compute m_i, d_i', and sim_i using associative scan
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        sim_offsets = sim_offset + (start_col + offs) * sim_stride_l

        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_offsets, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)
        sim_state = tl.exp(sim - m)
        m, d, sim_state = tl.associative_scan((m, d, sim_state), axis=0, combine_fn=softmax_combine_fn)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_col - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        sim_local = tl.where(offs == last_idx, sim_state, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)
        sim_local = tl.sum(sim_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        sim_i = sim_i * tl.exp(old_m_i - m_i) + sim_local * tl.exp(m_local - m_i)

    # Second pass: Compute softmax outputs
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        tl.store(out_ptr + pid_batch * out_stride_b + pid_head * out_stride_h + (start_col + offs) * out_stride_l, softmax_out, mask=mask)

@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["batch_size", "num_heads", "seq_len"],
)
@triton.jit
def bwd_online_softmax_2pass_kernel(
    x_ptr, sim_ptr, grad_out_ptr, grad_x_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_l: tl.constexpr,
    sim_stride_b: tl.constexpr,
    sim_stride_h: tl.constexpr,
    sim_stride_l: tl.constexpr,
    grad_stride_b: tl.constexpr,
    grad_stride_h: tl.constexpr,
    grad_stride_l: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    row_offset = pid_b * stride_b + pid_h * stride_h
    sim_offset = pid_b * sim_stride_b + pid_h * sim_stride_h

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    n_i = tl.full((), 0.0, dtype=DTYPE)
    sim_i = tl.full((), 0.0, dtype=DTYPE)

    # First pass: Recompute m_i, d_i', sim_i, and compute n_i
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l
        sim_idx = sim_offset + (start + offs) * sim_stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        sim = tl.load(sim_ptr + sim_idx, mask=mask, other=0.0)
        g = tl.load(grad_out_ptr + idx, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)
        n = g * d
        sim_state = tl.exp(sim - m)
        m, d, n, sim_state = tl.associative_scan((m, d, n, sim_state), axis=0, combine_fn=backward_combine_fn)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        n_local = tl.where(offs == last_idx, n, 0.0)
        sim_local = tl.where(offs == last_idx, sim_state, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)
        n_local = tl.sum(n_local, axis=0)
        sim_local = tl.sum(sim_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        n_i = n_i * tl.exp(old_m_i - m_i) + n_local * tl.exp(m_local - m_i)
        sim_i = sim_i * tl.exp(old_m_i - m_i) + sim_local * tl.exp(m_local - m_i)

    dot = n_i / (d_i + 1e-6)

    # Second pass: Compute gradients
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        g = tl.load(grad_out_ptr + idx, mask=mask, other=0.0)
        p = tl.exp(x - m_i) / (d_i + 1e-6)
        gx = p * (g - dot)
        tl.store(grad_x_ptr + pid_b * grad_stride_b + pid_h * grad_stride_h + (start + offs) * grad_stride_l, gx, mask=mask)

class OnlineSoftmax2Pass(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, sim: torch.Tensor):
        batch_size, num_heads, seq_len = x.shape
        assert sim.shape == x.shape, "Similarity tensor must have the same shape as input"
        x = x.contiguous().cuda()
        sim = sim.contiguous().cuda()
        out = torch.empty_like(x)
        grid = (batch_size, num_heads)
        triton_dtype = dtype_map.get(x.dtype, tl.float32)

        fwd_online_softmax_2pass_kernel[grid](
            x,
            sim,
            out,
            batch_size,
            num_heads,
            seq_len,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            sim.stride(0),
            sim.stride(1),
            sim.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            DTYPE=triton_dtype,
        )
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, sim = inputs
        ctx.save_for_backward(x, sim, output)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)
        ctx.shape = x.shape
        ctx.dtype = x.dtype

    @staticmethod
    def backward(ctx, grad_out):
        x, sim, out = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        b, h, l = ctx.shape
        grid = (b, h)

        bwd_online_softmax_2pass_kernel[grid](
            x, sim, grad_out, grad_x,
            b, h, l,
            x.stride(0), x.stride(1), x.stride(2),
            sim.stride(0), sim.stride(1), sim.stride(2),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
            DTYPE=ctx.triton_dtype,
        )
        return grad_x, None  # No gradient for sim

def online_softmax_2pass(x: torch.Tensor, sim: torch.Tensor):
    return OnlineSoftmax2Pass.apply(x, sim)

@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)

if __name__ == "__main__":
    # Gradient check
    torch.manual_seed(1746)
    x = torch.randn(2, 8, 64, dtype=torch.float64, device="cuda", requires_grad=True)
    sim = torch.randn(2, 8, 64, dtype=torch.float64, device="cuda")
    torch.autograd.gradcheck(
        OnlineSoftmax2Pass.apply,
        inputs=(x, sim),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-5,
        fast_mode=True
    )
    print("Gradient check passed")

    # Performance test
    x = torch.randn(16, 32, 16000, dtype=torch.float32, device="cuda")
    sim = torch.randn(16, 32, 16000, dtype=torch.float32, device="cuda")
    out = online_softmax_2pass(x, sim)
    ref = torch_softmax(x, dim=-1)
    print(f"Max absolute error: {(out - ref).abs().max().item()}")
    t1 = do_bench(lambda: online_softmax_2pass(x, sim))
    t2 = do_bench(lambda: torch_softmax(x, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = x.numel() * x.element_size() * 5  # x, sim, out, read x again, read sim again
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")
