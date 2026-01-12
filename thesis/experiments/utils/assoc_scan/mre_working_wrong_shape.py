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
def softmax_combine_fn(m_x, d_x, n_x, Z_x, g_x, m_y, d_y, n_y, Z_y, g_y):
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, d_new, n_new, Z_new, g_new


@triton.jit
def backward_combine_fn(m_x, d_x, n_x, m_y, d_y, n_y):
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    return m_new, d_new, n_new


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
    key=["feature_size", "seq_len"],
)
@triton.jit
def fwd_online_softmax_kernel(
    x_ptr,
    z_ptr,
    g_ptr,
    out_ptr,
    feature_size: int,
    seq_len: int,
    stride_f: int,
    stride_l: int,
    out_stride_f: int,
    out_stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_feature = tl.program_id(0)
    start_offset = pid_feature * stride_f

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l

        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        z = tl.load(z_ptr + offsets, mask=mask, other=float("-inf"))
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)

        # Initialize state for the scan
        m = x
        d = tl.exp(x - m)
        n = g * d
        Z = z
        g_state = g

        # Single scan pass
        m, d, n, Z, g_state = tl.associative_scan((m, d, n, Z, g_state), axis=0, combine_fn=softmax_combine_fn)

        # Compute final output
        p = tl.exp(x - m) / (d + 1e-6)
        out = p * (g - n / (d + 1e-6))
        tl.store(out_ptr + pid_feature * out_stride_f + (start_col + offs) * out_stride_l, out, mask=mask)


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["feature_size", "seq_len"],
)
@triton.jit
def bwd_online_softmax_kernel(
    x_ptr,
    z_ptr,
    g_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_z_ptr,
    grad_g_ptr,
    feature_size: tl.constexpr,
    seq_len: tl.constexpr,
    stride_f: tl.constexpr,
    stride_l: tl.constexpr,
    grad_stride_f: tl.constexpr,
    grad_stride_l: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_f = tl.program_id(0)
    row_offset = pid_f * stride_f

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)
    n_i = tl.full((), 0.0, dtype=DTYPE)
    Z_i = tl.full((), 0.0, dtype=DTYPE)
    g_i = tl.full((), 0.0, dtype=DTYPE)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        z = tl.load(z_ptr + idx, mask=mask, other=float("-inf"))
        g = tl.load(g_ptr + idx, mask=mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + idx, mask=mask, other=0.0)
        m = x
        d = tl.exp(x - m)
        n = g * d
        m, d, n = tl.associative_scan((m, d, n), axis=0, combine_fn=backward_combine_fn)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        n_local = tl.where(offs == last_idx, n, 0.0)
        Z_local = tl.where(offs == last_idx, z, 0.0)
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
        Z_i = Z_i + Z_local
        g_i = g_i + g_local

    dot = n_i / (d_i + 1e-6)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        z = tl.load(z_ptr + idx, mask=mask, other=float("-inf"))
        g = tl.load(g_ptr + idx, mask=mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + idx, mask=mask, other=0.0)
        p = tl.exp(x - m_i) / (d_i + 1e-6)

        # Compute gradients and multiply by grad_out
        gx = p * (g - dot) * grad_out
        gz = p * grad_out
        gg = p * grad_out

        tl.store(grad_x_ptr + pid_f * grad_stride_f + (start + offs) * grad_stride_l, gx, mask=mask)
        tl.store(grad_z_ptr + pid_f * grad_stride_f + (start + offs) * grad_stride_l, gz, mask=mask)
        tl.store(grad_g_ptr + pid_f * grad_stride_f + (start + offs) * grad_stride_l, gg, mask=mask)


class OnlineSoftmaxMDScan(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, z: torch.Tensor, g: torch.Tensor):
        # Get input shape and validate
        input_shape = x.shape
        assert x.ndim >= 2, "Input must have at least 2 dimensions (..., L)"
        L = x.shape[-1]
        batch_dims = input_shape[:-1]
        batch_size = torch.prod(torch.tensor(batch_dims)).item()

        # Flatten batch/feature dims
        x_flat = x.reshape(batch_size, L)
        z_flat = z.reshape(batch_size, L)
        g_flat = g.reshape(batch_size, L)
        x_flat = x_flat.contiguous().cuda()
        z_flat = z_flat.contiguous().cuda()
        g_flat = g_flat.contiguous().cuda()

        # Prepare output
        out_flat = torch.empty_like(x_flat)
        grid = (batch_size,)
        triton_dtype = dtype_map.get(x.dtype, tl.float32)

        fwd_online_softmax_kernel[grid](
            x_flat,
            z_flat,
            g_flat,
            out_flat,
            batch_size,
            L,
            x_flat.stride(0),
            x_flat.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
            DTYPE=triton_dtype,
        )

        # Reshape output to match input shape
        out = out_flat.reshape(*input_shape)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, z, g = inputs
        ctx.save_for_backward(x, z, g, output)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)
        ctx.input_shape = x.shape
        ctx.dtype = x.dtype

    @staticmethod
    def backward(ctx, grad_out):
        x, z, g, out = ctx.saved_tensors
        input_shape = ctx.input_shape
        L = input_shape[-1]
        batch_dims = input_shape[:-1]
        batch_size = torch.prod(torch.tensor(batch_dims)).item()

        # Flatten batch/feature dims
        x_flat = x.reshape(batch_size, L)
        z_flat = z.reshape(batch_size, L)
        g_flat = g.reshape(batch_size, L)
        grad_out_flat = grad_out.reshape(batch_size, L)
        grad_x_flat = torch.empty_like(x_flat)
        grad_z_flat = torch.empty_like(z_flat)
        grad_g_flat = torch.empty_like(g_flat)

        grid = (batch_size,)

        bwd_online_softmax_kernel[grid](
            x_flat,
            z_flat,
            g_flat,
            grad_out_flat,
            grad_x_flat,
            grad_z_flat,
            grad_g_flat,
            batch_size,
            L,
            x_flat.stride(0),
            x_flat.stride(1),
            grad_x_flat.stride(0),
            grad_x_flat.stride(1),
            DTYPE=ctx.triton_dtype,
        )

        # Reshape gradients to match input shapes
        grad_x = grad_x_flat.reshape(*input_shape)
        grad_z = grad_z_flat.reshape(*input_shape)
        grad_g = grad_g_flat.reshape(*input_shape)
        return grad_x, grad_z, grad_g


def online_softmax_scan(x: torch.Tensor, z: torch.Tensor, g: torch.Tensor):
    return OnlineSoftmaxMDScan.apply(x, z, g)


@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)


if __name__ == "__main__":
    # Gradient check
    torch.manual_seed(1746)
    x = torch.randn(32, 64, dtype=torch.float64, device="cuda", requires_grad=True)
    z = torch.randn(32, 64, dtype=torch.float64, device="cuda", requires_grad=True)
    g = torch.randn(32, 64, dtype=torch.float64, device="cuda", requires_grad=True)
    torch.autograd.gradcheck(
        OnlineSoftmaxMDScan.apply, inputs=(x, z, g), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5, fast_mode=True
    )
    print("Gradient check passed")

    # Performance test with fp32
    print("\nRunning performance test with fp32:")
    x = torch.randn(2048, 16000, dtype=torch.float32, device="cuda")
    z = torch.randn(2048, 16000, dtype=torch.float32, device="cuda")
    g = torch.randn(2048, 16000, dtype=torch.float32, device="cuda")
    out = online_softmax_scan(x, z, g)
    ref = torch_softmax(x, dim=-1)
    print(f"Max absolute error: {(out - ref).abs().max().item()}")
    t1 = do_bench(lambda: online_softmax_scan(x, z, g))
    t2 = do_bench(lambda: torch_softmax(x, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = x.numel() * x.element_size() * 4
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")

    # Performance test with fp64
    print("\nRunning performance test with fp64:")
    x = torch.randn(2048, 16000, dtype=torch.float64, device="cuda")
    z = torch.randn(2048, 16000, dtype=torch.float64, device="cuda")
    g = torch.randn(2048, 16000, dtype=torch.float64, device="cuda")
    out = online_softmax_scan(x, z, g)
    ref = torch_softmax(x, dim=-1)
    print(f"Max absolute error: {(out - ref).abs().max().item()}")
    t1 = do_bench(lambda: online_softmax_scan(x, z, g))
    t2 = do_bench(lambda: torch_softmax(x, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = x.numel() * x.element_size() * 4
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")
