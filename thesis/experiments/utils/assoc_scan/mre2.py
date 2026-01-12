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


# Analytical derivative kernels
@triton.jit
def kernel_dm_new_dm_x(m_x, m_y):
    return tl.where(m_x >= m_y, 1.0, 0.0)


@triton.jit
def kernel_dm_new_dm_y(m_x, m_y):
    return tl.where(m_x < m_y, 1.0, 0.0)


@triton.jit
def kernel_get_exps(m_x, m_y):
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    return exp_x, exp_y, m_new


@triton.jit
def kernel_ds_new_dm_x(m_x, s_x, m_y, s_y):
    exp_x, exp_y, m_new = kernel_get_exps(m_x, m_y)
    dm_new_dm_x_ = kernel_dm_new_dm_x(m_x, m_y)
    dexp_x_dm_x = exp_x * (1.0 - dm_new_dm_x_)
    dexp_y_dm_x = exp_y * (-dm_new_dm_x_)
    return s_x * dexp_x_dm_x + s_y * dexp_y_dm_x


@triton.jit
def kernel_ds_new_dm_y(m_x, s_x, m_y, s_y):
    exp_x, exp_y, m_new = kernel_get_exps(m_x, m_y)
    dm_new_dm_y_ = kernel_dm_new_dm_y(m_x, m_y)
    dexp_x_dm_y = exp_x * (-dm_new_dm_y_)
    dexp_y_dm_y = exp_y * (1.0 - dm_new_dm_y_)
    return s_x * dexp_x_dm_y + s_y * dexp_y_dm_y


@triton.jit
def kernel_ds_new_ds_x(m_x, m_y):
    exp_x, _, _ = kernel_get_exps(m_x, m_y)
    return exp_x


@triton.jit
def kernel_ds_new_ds_y(m_x, m_y):
    _, exp_y, _ = kernel_get_exps(m_x, m_y)
    return exp_y


@triton.jit
def kernel_dn_new_dm_x(m_x, n_x, m_y, n_y):
    exp_x, exp_y, m_new = kernel_get_exps(m_x, m_y)
    dm_new_dm_x_ = kernel_dm_new_dm_x(m_x, m_y)
    dexp_x_dm_x = exp_x * (1.0 - dm_new_dm_x_)
    dexp_y_dm_x = exp_y * (-dm_new_dm_x_)
    return n_x * dexp_x_dm_x + n_y * dexp_y_dm_x


@triton.jit
def kernel_dn_new_dn_x(m_x, m_y):
    exp_x, _, _ = kernel_get_exps(m_x, m_y)
    return exp_x


@triton.jit
def kernel_dn_new_dm_y(m_x, n_x, m_y, n_y):
    exp_x, exp_y, m_new = kernel_get_exps(m_x, m_y)
    dm_new_dm_y_ = kernel_dm_new_dm_y(m_x, m_y)
    dexp_x_dm_y = exp_x * (-dm_new_dm_y_)
    dexp_y_dm_y = exp_y * (1.0 - dm_new_dm_y_)
    return n_x * dexp_x_dm_y + n_y * dexp_y_dm_y


@triton.jit
def kernel_dn_new_dn_y(m_x, m_y):
    _, exp_y, _ = kernel_get_exps(m_x, m_y)
    return exp_y


@triton.jit
def combine_add(a0, b0, a1, b1):
    """Combine function for additive scan of Z and g gradients."""
    return a0 + a1, b0 + b1


@triton.jit
def softmax_combine_fn(m_x, d_x, m_y, d_y):
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    return m_new, d_new


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

    m_i = tl.full((), float("-inf"), dtype=DTYPE)
    d_i = tl.full((), 0.0, dtype=DTYPE)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l

        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        m = x
        d = tl.exp(x - m)
        m, d = tl.associative_scan((m, d), axis=0, combine_fn=softmax_combine_fn)

        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_col - 1)
        m_local = tl.where(offs == last_idx, m, float("-inf"))
        d_local = tl.where(offs == last_idx, d, 0.0)
        m_local = tl.max(m_local, axis=0)
        d_local = tl.sum(d_local, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        tl.store(out_ptr + pid_feature * out_stride_f + (start_col + offs) * out_stride_l, softmax_out, mask=mask)


@triton.jit
def vjp_scan_combine_fn(
    gm_x,
    gd_x,
    gn_x,
    m_x,
    d_x,
    n_x,  # Left operand (cumulative state)
    gm_y,
    gd_y,
    gn_y,
    m_y,
    d_y,
    n_y,  # Right operand (current element's initial state)
):
    """
    Associative combine function for the forward scan implementing the VJP.
    State: (gm, gd, gn, m, d, n)
    Output is the new cumulative state.
    """
    # 1. Compute combined forward state (m_new, d_new, n_new)
    m_new, d_new, n_new = backward_combine_fn(m_x, d_x, n_x, m_y, d_y, n_y)

    # 2. Compute Jacobians P_y = d(new)/d(y) needed for pullback B_y(g_x) = g_x @ P_y^T
    dm_new_dm_y = kernel_dm_new_dm_y(m_x, m_y)
    dd_new_dm_y = kernel_ds_new_dm_y(m_x, d_x, m_y, d_y)  # s->d
    dn_new_dm_y = kernel_dn_new_dm_y(m_x, n_x, m_y, n_y)
    dm_new_dd_y = 0.0
    dd_new_dd_y = kernel_ds_new_ds_y(m_x, m_y)  # s->d
    dn_new_dd_y = 0.0
    dm_new_dn_y = 0.0
    dd_new_dn_y = 0.0
    dn_new_dn_y = kernel_dn_new_dn_y(m_x, m_y)

    # 3. Compute B_y(g_x) = (gm_x, gd_x, gn_x) @ P_y^T
    gm_proj = gm_x * dm_new_dm_y + gd_x * dd_new_dm_y + gn_x * dn_new_dm_y
    gd_proj = gm_x * dm_new_dd_y + gd_x * dd_new_dd_y + gn_x * dn_new_dd_y
    gn_proj = gm_x * dm_new_dn_y + gd_x * dd_new_dn_y + gn_x * dn_new_dn_y

    # 4. Compute G_new = B_y(G_x) + G_y
    gm_new = gm_proj + gm_y
    gd_new = gd_proj + gd_y
    gn_new = gn_proj + gn_y

    # 5. Return new state (gm_new, gd_new, gn_new, m_new, d_new, n_new)
    return gm_new, gd_new, gn_new, m_new, d_new, n_new


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["feature_size", "seq_len"],
)
@triton.jit
def bwd_softmax_single_pass_kernel(
    x_ptr,
    grad_out_ptr,
    grad_x_ptr,
    feature_size: tl.constexpr,
    seq_len: tl.constexpr,
    x_stride_f: tl.constexpr,
    x_stride_l: tl.constexpr,
    gout_stride_f: tl.constexpr,
    gout_stride_l: tl.constexpr,
    gx_stride_f: tl.constexpr,
    gx_stride_l: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_f = tl.program_id(0)
    # Calculate base offsets for the current feature row
    row_base_offset_x = pid_f * x_stride_f
    row_base_offset_gout = pid_f * gout_stride_f
    row_base_offset_gx = pid_f * gx_stride_f

    # Carry state S = (gm, gd, gn, m, d, n)
    carry_gm = tl.zeros((1,), dtype=DTYPE)
    carry_gd = tl.zeros((1,), dtype=DTYPE)
    carry_gn = tl.zeros((1,), dtype=DTYPE)
    carry_m = tl.full((1,), float("-inf"), dtype=DTYPE)  # Ensure correct type for max
    carry_d = tl.zeros((1,), dtype=DTYPE)
    carry_n = tl.zeros((1,), dtype=DTYPE)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col

        # Calculate integer offsets for the current block within the row
        # Assuming x_stride_l == gout_stride_l == gx_stride_l for simplicity
        # If strides can differ, calculate separately for x, gout, gx
        block_row_offsets = (start_col + offs) * x_stride_l

        # --- Create pointer tensors ---
        x_pointers = x_ptr + row_base_offset_x + block_row_offsets
        gout_pointers = grad_out_ptr + row_base_offset_gout + block_row_offsets
        gx_pointers = grad_x_ptr + row_base_offset_gx + block_row_offsets

        # Load inputs x and grad_out (g) using pointer tensors
        x = tl.load(x_pointers, mask=mask, other=float("-inf"))
        g = tl.load(gout_pointers, mask=mask, other=0.0)

        # Initialize element state S_y = (gm=0, gd=0, gn=0, m=x, d=1.0, n=g)
        m_y = x
        d_y = tl.full(m_y.shape, 1.0, dtype=DTYPE)  # d=exp(x-m)=exp(0)=1
        n_y = g * d_y  # n = g * d
        gm_y = tl.zeros(m_y.shape, dtype=DTYPE)
        gd_y = tl.zeros(m_y.shape, dtype=DTYPE)
        gn_y = tl.zeros(m_y.shape, dtype=DTYPE)

        # --- Intra-block Scan ---
        gm_scan, gd_scan, gn_scan, m_scan, d_scan, n_scan = tl.associative_scan(
            (gm_y, gd_y, gn_y, m_y, d_y, n_y), axis=0, combine_fn=vjp_scan_combine_fn
        )

        # --- Combine with Carry ---
        final_gm, final_gd, final_gn, final_m, final_d, final_n = vjp_scan_combine_fn(
            carry_gm,
            carry_gd,
            carry_gn,
            carry_m,
            carry_d,
            carry_n,  # Carry state
            gm_scan,
            gd_scan,
            gn_scan,
            m_scan,
            d_scan,
            n_scan,  # Block scan results
        )

        # --- Store Result ---
        # Store the resulting gradient component gm as grad_x using pointer tensor
        tl.store(gx_pointers, final_gm, mask=mask)

        # --- Update Carry ---
        last_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_col - 1)
        if last_idx >= 0:
            last_mask = (offs == last_idx) & mask
            carry_gm = tl.sum(tl.where(last_mask, final_gm, 0.0), axis=0, keep_dims=True)
            carry_gd = tl.sum(tl.where(last_mask, final_gd, 0.0), axis=0, keep_dims=True)
            carry_gn = tl.sum(tl.where(last_mask, final_gn, 0.0), axis=0, keep_dims=True)
            # Use tl.where with -inf for max reduction, ensures correct handling if last element masked out
            carry_m = tl.max(tl.where(last_mask, final_m, float("-inf")), axis=0, keep_dims=True)
            carry_d = tl.sum(tl.where(last_mask, final_d, 0.0), axis=0, keep_dims=True)
            carry_n = tl.sum(tl.where(last_mask, final_n, 0.0), axis=0, keep_dims=True)


class ScanMax(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor):
        feature_size, seq_len = x.shape
        x = x.contiguous()
        out = torch.empty_like(x)
        grid = (feature_size,)
        triton_dtype = dtype_map.get(x.dtype, tl.float32)

        fwd_online_softmax_kernel[grid](
            x,
            out,
            feature_size,
            seq_len,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            DTYPE=triton_dtype,
        )
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        x = inputs[0]
        ctx.save_for_backward(x)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)
        ctx.shape = x.shape
        ctx.dtype = x.dtype

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        f, L = ctx.shape
        grid = (f,)

        bwd_softmax_single_pass_kernel[grid](
            x,
            grad_out,
            grad_x,
            f,
            L,
            x.stride(0),
            x.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_x.stride(0),
            grad_x.stride(1),
            DTYPE=ctx.triton_dtype,
        )
        return grad_x


def scan_max(x: torch.Tensor):
    return ScanMax.apply(x)


@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)


if __name__ == "__main__":
    # Gradient check
    torch.manual_seed(1746)
    x = torch.randn(32, 64, dtype=torch.float64, device="cuda", requires_grad=True)
    torch.autograd.gradcheck(ScanMax.apply, inputs=x, eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5, fast_mode=True)
    print("Gradient check passed")

    # Performance test
    x = torch.randn(4096, 16384, dtype=torch.float32, device="cuda")
    out = scan_max(x)
    ref = torch_softmax(x, dim=-1)
    print(f"Max absolute error: {(out - ref).abs().max().item()}")
    t1 = do_bench(lambda: scan_max(x))
    t2 = do_bench(lambda: torch_softmax(x, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = x.numel() * x.element_size() * 4
    ideal_time_ms = num_bytes / 3.35e12 * 1e3
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")
