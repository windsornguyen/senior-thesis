import torch
import triton
import triton.language as tl
from typing import Tuple
from triton.testing import do_bench
from thesis.experiments.utils.assoc_scan.kernel_1d import associative_scan as cumulative_sum_scan
from thesis.experiments.utils.assoc_scan.kernel import associative_scan as generic_associative_scan

dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def softmax_combine_fn(m_x, d_x, m_y, d_y):
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    return m_new, d_new


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
def fwd_softmax_md_kernel(
    x_ptr,
    out_ptr,
    m_leaf_ptr,
    s_leaf_ptr,
    feature_size: int,
    seq_len: int,
    stride_f: int,
    stride_l: int,
    out_stride_f: int,
    out_stride_l: int,
    m_leaf_stride_f: int,
    m_leaf_stride_l: int,
    s_leaf_stride_f: int,
    s_leaf_stride_l: int,
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
        m_local = tl.max(x, axis=0)
        exp_x = tl.exp(x - m_local)
        d_local = tl.sum(exp_x, axis=0)

        old_m_i = m_i
        m_i = tl.maximum(m_i, m_local)
        d_i = d_i * tl.exp(old_m_i - m_i) + d_local * tl.exp(m_local - m_i)

        m_leaf = tl.where(mask, m_i, float("-inf"))
        s_leaf = tl.where(mask, d_i, 0.0)
        tl.store(m_leaf_ptr + pid_feature * m_leaf_stride_f + (start_col + offs) * m_leaf_stride_l, m_leaf, mask=mask)
        tl.store(s_leaf_ptr + pid_feature * s_leaf_stride_f + (start_col + offs) * s_leaf_stride_l, s_leaf, mask=mask)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start_col
        offsets = start_offset + (start_col + offs) * stride_l
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        tl.store(out_ptr + pid_feature * out_stride_f + (start_col + offs) * out_stride_l, softmax_out, mask=mask)


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["feature_size", "seq_len"],
)
@triton.jit
def bwd_online_softmax_kernel(
    x_ptr,
    grad_out_ptr,
    grad_x_ptr,
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

    m = tl.full((), float("-inf"), dtype=DTYPE)
    d = tl.full((), 0.0, dtype=DTYPE)
    n = tl.full((), 0.0, dtype=DTYPE)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        g = tl.load(grad_out_ptr + idx, mask=mask, other=0.0)

        m_loc = tl.max(x, axis=0)
        exp_x = tl.exp(x - m_loc)
        d_loc = tl.sum(exp_x, axis=0)
        n_loc = tl.sum(g * exp_x, axis=0)

        m_new = tl.maximum(m, m_loc)
        coef_x, coef_y = tl.exp(m - m_new), tl.exp(m_loc - m_new)
        d, n = d * coef_x + d_loc * coef_y, n * coef_x + n_loc * coef_y
        m = m_new

    dot = n / (d + 1e-6)

    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len - start
        idx = row_offset + (start + offs) * stride_l

        x = tl.load(x_ptr + idx, mask=mask, other=float("-inf"))
        g = tl.load(grad_out_ptr + idx, mask=mask, other=0.0)
        p = tl.exp(x - m) / (d + 1e-6)
        gx = p * (g - dot)

        tl.store(grad_x_ptr + pid_f * grad_stride_f + (start + offs) * grad_stride_l, gx, mask=mask)


class OnlineSoftmaxMDScan(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor):
        feature_size, seq_len = x.shape
        x = x.contiguous().cuda()
        out = torch.empty_like(x)
        m_leaf = torch.empty_like(x)
        s_leaf = torch.empty_like(x)

        grid = (feature_size,)
        triton_dtype = dtype_map.get(x.dtype, tl.float32)

        fwd_softmax_md_kernel[grid](
            x,
            out,
            m_leaf,
            s_leaf,
            feature_size,
            seq_len,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            m_leaf.stride(0),
            m_leaf.stride(1),
            s_leaf.stride(0),
            s_leaf.stride(1),
            DTYPE=triton_dtype,
        )
        return out, m_leaf, s_leaf

    @staticmethod
    def setup_context(ctx, inputs, output):
        x = inputs[0]
        out, m_leaf, s_leaf = output
        ctx.save_for_backward(x, out)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)
        ctx.shape = x.shape
        ctx.dtype = x.dtype

    @staticmethod
    def backward(ctx, grad_out, grad_m_leaf, grad_s_leaf):
        x, out = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        f, L = ctx.shape
        grid = (f,)

        bwd_online_softmax_kernel[grid](
            x,
            grad_out,
            grad_x,
            f,
            L,
            x.stride(0),
            x.stride(1),
            grad_x.stride(0),
            grad_x.stride(1),
            DTYPE=dtype_map[x.dtype],
        )
        return grad_x


def online_softmax_scan(x: torch.Tensor):
    return OnlineSoftmaxMDScan.apply(x)


@torch.compile(mode="reduce-overhead")
def torch_softmax(x: torch.Tensor, dim: int = -1):
    return torch.softmax(x, dim=dim)


def online_softmax_multigate_scan(x, v, gated_Z, gates_z):
    initial_shape = x.shape
    L_dim_index = -1
    if len(initial_shape) > 2:
        bf_size = initial_shape[0] * initial_shape[1]
        x_flat = x.reshape(bf_size, initial_shape[L_dim_index])
        v_flat = v.reshape(bf_size, initial_shape[L_dim_index], v.shape[-1])
        Z_flat = gated_Z.reshape(bf_size, initial_shape[L_dim_index], gated_Z.shape[-2], gated_Z.shape[-1])
        g_flat = gates_z.reshape(bf_size, initial_shape[L_dim_index])
        L = initial_shape[L_dim_index]
    else:
        x_flat, v_flat, Z_flat, g_flat = x, v, gated_Z, gates_z
        L = initial_shape[L_dim_index]

    out_flat, m_leaf_flat, s_leaf_flat = OnlineSoftmaxMDScan.apply(x_flat)

    Z_leaf = torch.cumsum(gated_Z, dim=-3)
    g_leaf = torch.cumsum(gates_z, dim=-1)

    m_leaf = m_leaf_flat.reshape(*initial_shape[:-1], L)
    v_scaled = v * torch.exp(m_leaf).unsqueeze(-1)
    v_scaled_cumsum = torch.cumsum(v_scaled, dim=L_dim_index)
    n_leaf = torch.exp(-m_leaf).unsqueeze(-1) * v_scaled_cumsum

    if len(initial_shape) > 2:
        out = out_flat.reshape(initial_shape)
        m_leaf = m_leaf
        s_leaf = s_leaf_flat.reshape(initial_shape)
    else:
        out = out_flat
        s_leaf = s_leaf_flat

    return out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf


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
    return (m_new, s_new, n_new, Z_new, g_new)


def scan_fn(
    qk_slice: torch.Tensor, v_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Runs the associative scan over **one** (B,H) stream.

    Args
    ----
    qk_slice : [L]          similarity logits for this stream
    v_slice  : [L, h]       L2‑normalised V (first‑order numerator)
    Z_slice  : [L, h, h]    gated outer‑product accumulator
    g_slice  : [L]          scalar gate sequence
    """
    leaves = (
        qk_slice,  # m (initialised to logits)
        torch.ones_like(qk_slice),  # s (denominator starts at 1)
        v_slice,  # n  (V numerator)
        Z_slice,  # Z
        g_slice,  # g
    )
    return generic_associative_scan(combine_fn=combine_fn_ref, xs=leaves, dim=0, combine_mode="generic")


def batched_scan_fn(
    sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    batch_size, feature_size, seq_len, h = 4, 2, 128, 16
    B, F, L, H = batch_size, feature_size, seq_len, h

    torch.manual_seed(1746)
    x_grad = torch.randn(F, L, dtype=torch.float64, device="cuda", requires_grad=True)
    # Skip grad check for now, leave this here but commented out
    # torch.autograd.gradcheck(
    #     OnlineSoftmaxMDScan.apply, inputs=x_grad, eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5, fast_mode=True
    # )
    # print("Gradient check passed")

    print("Generating data...")
    x = torch.randn(B, F, L, dtype=torch.float32, device="cuda")
    v = torch.randn(B, F, L, H, dtype=torch.float32, device="cuda")
    gated_Z = torch.randn(B, F, L, H, H, dtype=torch.float32, device="cuda")
    gates_z = torch.randn(B, F, L, dtype=torch.float32, device="cuda")
    print("Data generated.")

    print("Running reference scan...")
    leaves = (x, torch.ones_like(x), v, gated_Z, gates_z)
    ref_leaves = batched_scan_fn(x, v, gated_Z, gates_z)
    ref_m, ref_s, ref_n, ref_Z, ref_g = ref_leaves
    print("Reference scan finished.")

    print("Running online_softmax_multigate_scan...")
    out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf = online_softmax_multigate_scan(x, v, gated_Z, gates_z)
    print("Scan finished.")

    print("Checking output shapes:")
    print(f"  out: {out.shape}")
    print(f"  m_leaf: {m_leaf.shape}")
    print(f"  s_leaf: {s_leaf.shape}")
    print(f"  n_leaf: {n_leaf.shape}")
    print(f"  Z_leaf: {Z_leaf.shape}")
    print(f"  g_leaf: {g_leaf.shape}")

    print("Comparing Triton results with reference scan...")
    assert torch.allclose(m_leaf, ref_m, atol=1e-5), "m_leaf mismatch with reference"
    print("m_leaf check passed.")
    assert torch.allclose(s_leaf, ref_s, atol=1e-5), "s_leaf mismatch with reference"
    print("s_leaf check passed.")
    assert torch.allclose(n_leaf, ref_n, atol=1e-5), "n_leaf mismatch with reference"
    print("n_leaf check passed.")
    assert torch.allclose(Z_leaf, ref_Z, atol=1e-5), "Z_leaf mismatch with reference"
    print("Z_leaf check passed.")
    assert torch.allclose(g_leaf, ref_g, atol=1e-5), "g_leaf mismatch with reference"
    print("g_leaf check passed.")
    print("Leaf comparisons passed.")

    print("Performing basic checks...")
    assert torch.allclose(out.sum(dim=-1), torch.ones_like(out[..., 0]), atol=1e-5), "Softmax output doesn't sum to 1"
    print("Softmax check passed.")
    ref_Z_leaf_cumsum = torch.cumsum(gated_Z, dim=-3)
    assert torch.allclose(Z_leaf, ref_Z_leaf_cumsum, atol=1e-5), "Z_leaf mismatch with torch.cumsum"
    print("Z_leaf cumsum check passed.")
    ref_g_leaf_cumsum = torch.cumsum(gates_z, dim=-1)
    assert torch.allclose(g_leaf, ref_g_leaf_cumsum, atol=1e-5), "g_leaf mismatch with torch.cumsum"
    print("g_leaf cumsum check passed.")

    x_flat = x.reshape(-1, L)
    t_softmax = do_bench(lambda: OnlineSoftmaxMDScan.apply(x_flat))
    print(f"OnlineSoftmaxMDScan Time: {t_softmax:.3f} ms")

    t_cumsum_Z = do_bench(lambda: torch.cumsum(gated_Z, dim=2))
    t_cumsum_g = do_bench(lambda: torch.cumsum(gates_z, dim=2))
    t_n_leaf = do_bench(
        lambda: torch.exp(-m_leaf).unsqueeze(-1) * torch.cumsum(v * torch.exp(m_leaf).unsqueeze(-1), dim=2)
    )
    print(f"Torch Cumsum Z Time: {t_cumsum_Z:.3f} ms")
    print(f"Torch Cumsum g Time: {t_cumsum_g:.3f} ms")
    print(f"Torch n_leaf Calc Time: {t_n_leaf:.3f} ms")

    sim = torch.randn(batch_size, feature_size, seq_len, dtype=torch.float32, device="cuda")
    v = torch.randn(batch_size, feature_size, seq_len, h, dtype=torch.float32, device="cuda")
    gated_Z = torch.randn(batch_size, feature_size, seq_len, h, h, dtype=torch.float32, device="cuda")
    gates_z = torch.randn(batch_size, feature_size, seq_len, dtype=torch.float32, device="cuda")

    # Reference scan
    leaves = (sim, torch.ones_like(sim), v, gated_Z, gates_z)
    ref_leaves = batched_scan_fn(sim, v, gated_Z, gates_z)
    ref_m, ref_s, ref_n, ref_Z, ref_g = ref_leaves

    # Triton scan
    out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf = online_softmax_scan(sim, v, gated_Z, gates_z)
