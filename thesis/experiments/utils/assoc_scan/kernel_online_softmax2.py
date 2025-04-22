"""leaf integration"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


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


def get_softmax_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024, 2048, 4096]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3)
             for bs in [64, 128, 256]]
        )
    return configs


def keep_config(conf, n_cols=None):
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    if n_cols is not None and BLOCK_SIZE > n_cols:
        return False
    return True


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["n_rows", "n_cols"],
)
@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr, m_leaf_ptr, s_leaf_ptr, n_leaf_ptr, Z_leaf_ptr, g_leaf_ptr,
    n_rows, n_cols, x_stride_0, out_stride_0,
    m_leaf_stride_0, s_leaf_stride_0, n_leaf_stride_0, Z_leaf_stride_0, g_leaf_stride_0,
    BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    pid = tl.program_id(0)  # Row index
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_offset = pid * x_stride_0

    # Initialize global max and denominator
    m_i = float("-inf")
    d_i = 0.0
    # Initialize dummy variables
    n_i = 0.0
    Z_i = 0.0
    g_i = 0.0

    # First pass: scan to compute m_N, s_N, n_N, Z_N, g_N
    for k in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        boundary_mask = offs < n_cols - start_col
        x = tl.load(x_ptr + row_offset + start_col + offs, mask=boundary_mask & mask, other=float("-inf"))
        m0 = x
        d0 = tl.where(mask, 1.0, 0.0)
        n0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        Z0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        g0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Scan within chunk
        m_out, d_out, n_out, Z_out, g_out = tl.associative_scan(
            (m0, d0, n0, Z0, g0), axis=0, combine_fn=softmax_combine_fn, reverse=False
        )

        # Select last valid element in chunk
        chunk_size = tl.minimum(BLOCK_SIZE, n_cols - start_col)
        last_idx = chunk_size - 1
        last_mask = offs == last_idx
        m_local = tl.max(tl.where(last_mask, m_out, float("-inf")), axis=0)
        d_local = tl.sum(tl.where(last_mask, d_out, 0.0), axis=0)
        n_local = tl.sum(tl.where(last_mask, n_out, 0.0), axis=0)
        Z_local = tl.sum(tl.where(last_mask, Z_out, 0.0), axis=0)
        g_local = tl.sum(tl.where(last_mask, g_out, 0.0), axis=0)

        # Update global leaves
        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)
        n_i = n_i * tl.exp(old_m_i - m_i) + n_local * tl.exp(m_local - m_i)
        Z_i = Z_i + Z_local
        g_i = g_i + g_local

        # Store leaf outputs for this chunk
        tl.store(m_leaf_ptr + pid * m_leaf_stride_0 + start_col + offs, m_out, mask=boundary_mask & mask)
        tl.store(s_leaf_ptr + pid * s_leaf_stride_0 + start_col + offs, d_out, mask=boundary_mask & mask)
        tl.store(n_leaf_ptr + pid * n_leaf_stride_0 + start_col + offs, n_out, mask=boundary_mask & mask)
        tl.store(Z_leaf_ptr + pid * Z_leaf_stride_0 + start_col + offs, Z_out, mask=boundary_mask & mask)
        tl.store(g_leaf_ptr + pid * g_leaf_stride_0 + start_col + offs, g_out, mask=boundary_mask & mask)

    # Second pass: compute softmax output exp(x - m_N) / s_N
    for k in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        boundary_mask = offs < n_cols - start_col
        x = tl.load(x_ptr + row_offset + start_col + offs, mask=boundary_mask & mask, other=float("-inf"))
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        tl.store(out_ptr + pid * out_stride_0 + start_col + offs, softmax_out, mask=boundary_mask & mask)


def online_softmax(x):
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    m_leaf = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)
    s_leaf = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)
    n_leaf = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)
    Z_leaf = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)
    g_leaf = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_rows,)
    online_softmax_kernel[grid](
        x, out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf,
        n_rows, n_cols, x.stride(0), out.stride(0),
        m_leaf.stride(0), s_leaf.stride(0), n_leaf.stride(0),
        Z_leaf.stride(0), g_leaf.stride(0)
    )
    return out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf


if __name__ == "__main__":
    x = torch.randn(2048, 4096, dtype=torch.float32, device="cuda")
    out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf = online_softmax(x)
    ref = torch.softmax(x, dim=1)
    print(f"Max absolute error: {(out - ref).abs().max().item()}")
    t1 = do_bench(lambda: online_softmax(x))
    t2 = do_bench(lambda: torch.softmax(x, dim=1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = x.numel() * x.element_size() + out.numel() * x.element_size()
    ideal_time_ms = num_bytes / 3.35e12 * 1e3  # On H100
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")