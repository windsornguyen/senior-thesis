# -*- Author: Windsor Nguyen -*-
"""
Slower than single tl.associative_scan call and gives incorrect outputs...
"""

import math
import torch
import triton
import triton.language as tl

from typing import Tuple


# Mapping of PyTorch dtypes to Triton dtypes
dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Constants
TILE: tl.constexpr = 16
SMALL_SEQ_THRESHOLD: tl.constexpr = 2 * TILE

# Utility
def grid(streams: int, n_tiles: int = 1):
    return (streams, n_tiles) if n_tiles > 1 else (streams,)

def get_scan_configs():
    """Generate Triton configurations for autotuning."""
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3)
             for bs in [64, 128, 256]]
        )
    return configs

def keep_config(conf, seq_len=None):
    """Filter out invalid Triton configurations."""
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    if seq_len is not None and BLOCK_SIZE > seq_len:
        return False
    return True

@triton.jit
def combine_fn_logic(m_x, s_x, n_x, Z_x, g_x, m_y, s_y, n_y, Z_y, g_y):
    """Combines two states for associative scan."""
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new

# Single scan kernel (for small sequences)
@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["batch_size", "feature_size_h2", "seq_len"],
)
@triton.jit
def single_scan(
    m_ptr, out_m_ptr,
    s_ptr, out_s_ptr,
    n_ptr, out_n_ptr,
    Z_ptr, out_Z_ptr,
    g_ptr, out_g_ptr,
    batch_size: int,
    feature_size_h2: int,
    seq_len: int,
    stride_b: int,
    stride_f: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Single CTA-wide scan for small sequences (L <= 4096)."""
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    offset = pid_batch * stride_b + pid_feature * stride_f

    offs = offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < seq_len

    # Load entire tile
    m = tl.load(m_ptr + offs * stride_l, mask=mask, other=0.0)
    s = tl.load(s_ptr + offs * stride_l, mask=mask, other=0.0)
    n = tl.load(n_ptr + offs * stride_l, mask=mask, other=0.0)
    Z = tl.load(Z_ptr + offs * stride_l, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs * stride_l, mask=mask, other=0.0)

    # Single inclusive scan
    res_m, res_s, res_n, res_Z, res_g = tl.associative_scan((m, s, n, Z, g), axis=0, combine_fn=combine_fn_logic, reverse=False
)

    # Store results
    tl.store(out_m_ptr + offs * stride_l, res_m, mask=mask)
    tl.store(out_s_ptr + offs * stride_l, res_s, mask=mask)
    tl.store(out_n_ptr + offs * stride_l, res_n, mask=mask)
    tl.store(out_Z_ptr + offs * stride_l, res_Z, mask=mask)
    tl.store(out_g_ptr + offs * stride_l, res_g, mask=mask)

# Forward tile scan (hierarchical, pass 1)
@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["feature_size_h2", "tile_len"],
)
@triton.jit
def fwd_tile_scan(
    m_ptr, s_ptr, n_ptr, Z_ptr, g_ptr,
    out_m_ptr, out_s_ptr, out_n_ptr, out_Z_ptr, out_g_ptr,
    carry_m_ptr, carry_s_ptr, carry_n_ptr, carry_Z_ptr, carry_g_ptr,
    tile_len: tl.constexpr,
    seq_stride: tl.int32,
    tile_stride: tl.int32,
    feature_size_h2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scans one tile (≤ 2048 tokens) and writes carry to aux buffer."""
    pid = tl.program_id(0)
    itile = tl.program_id(1)
    offset_tile = itile * tile_len

    offs = pid * seq_stride + offset_tile + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < tile_len

    # Load tile once
    m = tl.load(m_ptr + offs * seq_stride, mask=mask, other=0.0)
    s = tl.load(s_ptr + offs * seq_stride, mask=mask, other=0.0)
    n = tl.load(n_ptr + offs * seq_stride, mask=mask, other=0.0)
    Z = tl.load(Z_ptr + offs * seq_stride, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs * seq_stride, mask=mask, other=0.0)

    # Single inclusive scan
    m, s, n, Z, g = tl.associative_scan(
        (m, s, n, Z, g), axis=0, combine_fn=combine_fn_logic
    )

    # Write results
    tl.store(out_m_ptr + offs * seq_stride, m, mask=mask)
    tl.store(out_s_ptr + offs * seq_stride, s, mask=mask)
    tl.store(out_n_ptr + offs * seq_stride, n, mask=mask)
    tl.store(out_Z_ptr + offs * seq_stride, Z, mask=mask)
    tl.store(out_g_ptr + offs * seq_stride, g, mask=mask)

    # Write last element to carry buffer
    last_off  = pid * seq_stride + offset_tile + tile_len - 1  # scalar index
    carry_idx = pid * tile_stride + itile                      # scalar index in carry buf

    # fetch the already‑scanned values we just wrote to global memory
    last_m = tl.load(out_m_ptr + last_off * seq_stride)
    last_s = tl.load(out_s_ptr + last_off * seq_stride)
    last_n = tl.load(out_n_ptr + last_off * seq_stride)
    last_Z = tl.load(out_Z_ptr + last_off * seq_stride)
    last_g = tl.load(out_g_ptr + last_off * seq_stride)

    # store into the carry buffers (no mask needed – single‑element write)
    tl.store(carry_m_ptr + carry_idx, last_m)
    tl.store(carry_s_ptr + carry_idx, last_s)
    tl.store(carry_n_ptr + carry_idx, last_n)
    tl.store(carry_Z_ptr + carry_idx, last_Z)
    tl.store(carry_g_ptr + carry_idx, last_g)

# Carry scan (hierarchical, pass 2)
@triton.jit
def combine_add(xm, xs, xn, xZ, xg, ym, ys, yn, yZ, yg):
    return xm + ym, xs + ys, xn + yn, xZ + yZ, xg + yg

@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["feature_size_h2", "n_tiles"],
)
@triton.jit
def carry_scan(
    carry_m_ptr, carry_s_ptr, carry_n_ptr, carry_Z_ptr, carry_g_ptr,
    prefix_m_ptr, prefix_s_ptr, prefix_n_ptr, prefix_Z_ptr, prefix_g_ptr,
    n_tiles: tl.constexpr,
    tile_stride: tl.int32,
    feature_size_h2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scans the carry buffer to compute prefixes for each tile."""
    pid = tl.program_id(0)
    offs = pid * n_tiles + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_tiles

    cm = tl.load(carry_m_ptr + offs, mask=mask, other=0.0)
    cs = tl.load(carry_s_ptr + offs, mask=mask, other=0.0)
    cn = tl.load(carry_n_ptr + offs, mask=mask, other=0.0)
    cZ = tl.load(carry_Z_ptr + offs, mask=mask, other=0.0)
    cg = tl.load(carry_g_ptr + offs, mask=mask, other=0.0)

    pm, ps, pn, pZ, pg = tl.associative_scan(
        (cm, cs, cn, cZ, cg), axis=0, combine_fn=combine_add
    )

    tl.store(prefix_m_ptr + offs, pm, mask=mask)
    tl.store(prefix_s_ptr + offs, ps, mask=mask)
    tl.store(prefix_n_ptr + offs, pn, mask=mask)
    tl.store(prefix_Z_ptr + offs, pZ, mask=mask)
    tl.store(prefix_g_ptr + offs, pg, mask=mask)

# Add carry (hierarchical, pass 3)
@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["feature_size_h2", "tile_len"],
)
@triton.jit
def fwd_add_carry(
    out_m_ptr, out_s_ptr, out_n_ptr, out_Z_ptr, out_g_ptr,
    prefix_m_ptr, prefix_s_ptr, prefix_n_ptr, prefix_Z_ptr, prefix_g_ptr,
    tile_len: tl.constexpr,
    seq_stride: tl.int32,
    tile_stride: tl.int32,
    feature_size_h2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adds scanned prefixes to each tile's elements."""
    pid = tl.program_id(0)
    itile = tl.program_id(1)
    offset_tile = itile * tile_len

    offs = pid * seq_stride + offset_tile + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < tile_len

    pm = tl.load(prefix_m_ptr + pid * tile_stride + itile)
    ps = tl.load(prefix_s_ptr + pid * tile_stride + itile)
    pn = tl.load(prefix_n_ptr + pid * tile_stride + itile)
    pZ = tl.load(prefix_Z_ptr + pid * tile_stride + itile)
    pg = tl.load(prefix_g_ptr + pid * tile_stride + itile)

    m = tl.load(out_m_ptr + offs * seq_stride, mask=mask, other=0.0)
    s = tl.load(out_s_ptr + offs * seq_stride, mask=mask, other=0.0)
    n = tl.load(out_n_ptr + offs * seq_stride, mask=mask, other=0.0)
    Z = tl.load(out_Z_ptr + offs * seq_stride, mask=mask, other=0.0)
    g = tl.load(out_g_ptr + offs * seq_stride, mask=mask, other=0.0)

    tl.store(out_m_ptr + offs * seq_stride, m + pm, mask=mask)
    tl.store(out_s_ptr + offs * seq_stride, s + ps, mask=mask)
    tl.store(out_n_ptr + offs * seq_stride, n + pn, mask=mask)
    tl.store(out_Z_ptr + offs * seq_stride, Z + pZ, mask=mask)
    tl.store(out_g_ptr + offs * seq_stride, g + pg, mask=mask)

class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for associative scan with expand pre-processing."""
    @staticmethod
    def forward(ctx, m, s, n, Z, g):
        # Input shapes: m/s/g [B, D, L], n [B, D, L, H], Z [B, D, L, H, H]
        L = m.shape[-1]
        try:
            H = n.shape[-1]
        except IndexError:
            raise ValueError("Tensor n must have shape (..., L, H)")

        # Validate shapes
        assert m.ndim >= 2 and s.ndim >= 2 and g.ndim >= 2, "m/s/g need at least 2 dims (..., L)"
        assert n.ndim >= 3, "n needs at least 3 dims (..., L, H)"
        assert Z.ndim >= 4, "Z needs at least 4 dims (..., L, H, H)"
        assert s.shape[-1] == L, f"s L dim ({s.shape[-1]}) != m ({L})"
        assert g.shape[-1] == L, f"g L dim ({g.shape[-1]}) != m ({L})"
        assert n.shape[-2] == L, f"n L dim ({n.shape[-2]}) != m ({L})"
        assert Z.shape[-3] == L, f"Z L dim ({Z.shape[-3]}) != m ({L})"
        assert Z.shape[-2] == H, f"Z first H dim ({Z.shape[-2]}) != n's H ({H})"
        assert Z.shape[-1] == H, f"Z second H dim ({Z.shape[-1]}) != n's H ({H})"
        assert m.shape[:-1] == s.shape[:-1] == g.shape[:-1], "m/s/g base shapes mismatch"
        assert m.shape[:-1] == n.shape[:-2], "n base shape mismatch"
        assert m.shape[:-1] == Z.shape[:-3], "Z base shape mismatch"

        input_shape = m.shape
        h2 = H * H
        batch_dims = input_shape[:-1]
        batch_size = torch.prod(torch.tensor(batch_dims)).item()

        # Flatten batch/feature dims
        m_flat = m.reshape(batch_size, L)
        s_flat = s.reshape(batch_size, L)
        g_flat = g.reshape(batch_size, L)
        n_flat = n.reshape(batch_size, L, H)
        Z_flat = Z.reshape(batch_size, L, H, H)

        # Expand features to match Z's H*H
        m_exp = m_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        s_exp = s_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        g_exp = g_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        n_exp = n_flat.unsqueeze(-1).expand(-1, -1, -1, H)
        Z_exp = Z_flat

        # Reshape to [Batch', h2, L]
        m_ker = m_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        s_ker = s_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        n_ker = n_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        Z_ker = Z_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        g_ker = g_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()

        # Move to CUDA
        m_ker, s_ker, n_ker, Z_ker, g_ker = [t.cuda() for t in [m_ker, s_ker, n_ker, Z_ker, g_ker]]

        # Prepare outputs
        out_m_ker = torch.empty_like(m_ker)
        out_s_ker = torch.empty_like(s_ker)
        out_n_ker = torch.empty_like(n_ker)
        out_Z_ker = torch.empty_like(Z_ker)
        out_g_ker = torch.empty_like(g_ker)

        # Dynamic dispatch
        triton_dtype = dtype_map.get(m_ker.dtype, tl.float32)
        if L <= SMALL_SEQ_THRESHOLD:
            # Single scan for small sequences
            grid_single = (batch_size, h2)
            single_scan[grid_single](
                m_ker, out_m_ker,
                s_ker, out_s_ker,
                n_ker, out_n_ker,
                Z_ker, out_Z_ker,
                g_ker, out_g_ker,
                batch_size, h2, L,
                m_ker.stride(0), m_ker.stride(1), m_ker.stride(2),
                DTYPE=triton_dtype,
            )
        else:
            # Hierarchical scan for long sequences
            n_tiles = math.ceil(L / TILE)
            seq_stride = m_ker.stride(2)
            tile_stride = n_tiles
            streams = batch_size * h2

            # Allocate carry and prefix buffers
            carry_m = torch.empty((streams, n_tiles), dtype=m_ker.dtype, device=m_ker.device)
            carry_s = torch.empty_like(carry_m)
            carry_n = torch.empty_like(carry_m)
            carry_Z = torch.empty_like(carry_m)
            carry_g = torch.empty_like(carry_m)
            prefix_m = torch.empty_like(carry_m)
            prefix_s = torch.empty_like(carry_m)
            prefix_n = torch.empty_like(carry_m)
            prefix_Z = torch.empty_like(carry_m)
            prefix_g = torch.empty_like(carry_m)

            # Pass 1: Scan each tile
            grid_tile = (streams, n_tiles)
            fwd_tile_scan[grid_tile](
                m_ker, s_ker, n_ker, Z_ker, g_ker,
                out_m_ker, out_s_ker, out_n_ker, out_Z_ker, out_g_ker,
                carry_m, carry_s, carry_n, carry_Z, carry_g,
                tile_len=TILE,
                seq_stride=seq_stride,
                tile_stride=tile_stride,
                feature_size_h2=h2,
            )

            # Pass 2: Scan carries
            grid_stream = (streams,)
            carry_scan[grid_stream](
                carry_m, carry_s, carry_n, carry_Z, carry_g,
                prefix_m, prefix_s, prefix_n, prefix_Z, prefix_g,
                n_tiles=n_tiles,
                tile_stride=tile_stride,
                feature_size_h2=h2,
            )

            # Pass 3: Add prefixes
            fwd_add_carry[grid_tile](
                out_m_ker, out_s_ker, out_n_ker, out_Z_ker, out_g_ker,
                prefix_m, prefix_s, prefix_n, prefix_Z, prefix_g,
                tile_len=TILE,
                seq_stride=seq_stride,
                tile_stride=tile_stride,
                feature_size_h2=h2,
            )

        # Post-process outputs
        out_m_flat = out_m_ker.transpose(1, 2)
        out_s_flat = out_s_ker.transpose(1, 2)
        out_n_flat = out_n_ker.transpose(1, 2)
        out_Z_flat = out_Z_ker.transpose(1, 2)
        out_g_flat = out_g_ker.transpose(1, 2)

        out_m_reshaped = out_m_flat.reshape(*input_shape[:-1], L, H, H)
        out_s_reshaped = out_s_flat.reshape(*input_shape[:-1], L, H, H)
        out_n_reshaped = out_n_flat.reshape(*input_shape[:-1], L, H, H)
        out_Z_reshaped = out_Z_flat.reshape(*input_shape[:-1], L, H, H)
        out_g_reshaped = out_g_flat.reshape(*input_shape[:-1], L, H, H)

        m_final = out_m_reshaped[..., 0, 0]
        s_final = out_s_reshaped[..., 0, 0]
        n_final = out_n_reshaped[..., :, 0]
        Z_final = out_Z_reshaped
        g_final = out_g_reshaped[..., 0, 0]

        # Save for backward
        ctx.save_for_backward(m, s, n, Z, g)
        ctx.input_shape = input_shape
        ctx.H = H
        ctx.triton_dtype = triton_dtype

        return m_final, s_final, n_final, Z_final, g_final

    @staticmethod
    def backward(ctx, grad_m, grad_s, grad_n, grad_Z, grad_g):
        pass

    @staticmethod
    def _move_bdim_to_front(x, bdim):
        if bdim is None:
            return x, None
        return x.movedim(bdim, 0), 0

    @staticmethod
    def _flatten_leading_dims(x, keep=1):
        batch_shape = x.shape[:-keep]
        flat = x.reshape(-1, *x.shape[-keep:]) if batch_shape else x
        return flat, batch_shape


def associative_scan(m, s, n, Z, g):
    """Performs associative scan with expand pre-processing."""
    return AssociativeScan.apply(m, s, n, Z, g)


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
        torch.ones_like(qk_slice),
        v_slice,
        Z_slice,
        g_slice,
    )
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
    B, H = sim.shape[0], sim.shape[1]
    sim_flat = sim.flatten(0, 1)
    v_flat = v.flatten(0, 1)
    gated_Z_flat = gated_Z.flatten(0, 1)
    gates_z_flat = gates_z.flatten(0, 1)

    scan_all = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    result = scan_all(sim_flat, v_flat, gated_Z_flat, gates_z_flat)

    return tuple(t.reshape(B, H, *t.shape[1:]) for t in result)


@torch.no_grad()
def do_bench(fn, warmup=25, rep=100):
    """Benchmark a function's execution time."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / rep


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = "cuda"

    # Correctness Check
    print("--- Correctness Check ---")
    B_test, D_test, L_test, H_test = 2, 1, 64, 4

    m = torch.randn(B_test, D_test, L_test, dtype=torch.float64, device=device, requires_grad=True)
    s = torch.ones_like(m, requires_grad=True)
    n = torch.randn(B_test, D_test, L_test, H_test, dtype=torch.float64, device=device, requires_grad=True)
    Z = torch.randn(B_test, D_test, L_test, H_test, H_test, dtype=torch.float64, device=device, requires_grad=True)
    g = torch.randn(B_test, D_test, L_test, dtype=torch.float64, device=device, requires_grad=True)

    print("Input Shapes (Test):")
    print(f"  m: {m.shape}")
    print(f"  s: {s.shape}")
    print(f"  n: {n.shape}")
    print(f"  Z: {Z.shape}")
    print(f"  g: {g.shape}")

    print("\nRunning Triton forward pass (test)...")
    m_out, s_out, n_out, Z_out, g_out = associative_scan(m, s, n, Z, g)
    print("Triton forward pass completed.")

    print("\nRunning reference forward pass...")
    ref_m_out, ref_s_out, ref_n_out, ref_Z_out, ref_g_out = batched_scan_fn(m, n, Z, g)
    print("Reference forward pass completed.")

    print("\nComparing forward pass outputs...")
    try:
        triton.testing.assert_close(m_out, ref_m_out, rtol=1e-5, atol=1e-5)
        triton.testing.assert_close(s_out, ref_s_out, rtol=1e-5, atol=1e-5)
        triton.testing.assert_close(n_out, ref_n_out, rtol=1e-5, atol=1e-5)
        triton.testing.assert_close(Z_out, ref_Z_out, rtol=1e-5, atol=1e-5)
        triton.testing.assert_close(g_out, ref_g_out, rtol=1e-5, atol=1e-5)
        print("Forward outputs MATCH!")
    except AssertionError as e:
        print("Forward outputs DO NOT MATCH:")
        print(e)

    # ====================================
    #          Benchmarking
    # ====================================

    print("\n--- Benchmarking Forward Pass ---")
    B_bench, D_bench, L_bench, H_bench = 4, 128, 8192, 2
    bench_dtype = torch.float32

    print(f"Benchmarking with: B={B_bench}, D={D_bench}, L={L_bench}, H={H_bench}, dtype={bench_dtype}")

    m_bench = torch.randn(B_bench, D_bench, L_bench, dtype=bench_dtype, device=device)
    s_bench = torch.ones_like(m_bench)
    n_bench = torch.randn(B_bench, D_bench, L_bench, H_bench, dtype=bench_dtype, device=device)
    Z_bench = torch.randn(B_bench, D_bench, L_bench, H_bench, H_bench, dtype=bench_dtype, device=device)
    g_bench = torch.randn(B_bench, D_bench, L_bench, dtype=bench_dtype, device=device)

    bench_fn = lambda: associative_scan(m_bench, s_bench, n_bench, Z_bench, g_bench)
    bench_fn_torch = lambda: batched_scan_fn(m_bench, n_bench, Z_bench, g_bench)

    print("Running initial call for compilation/autotuning...")
    _ = bench_fn()
    _ = bench_fn_torch()  # Warm up PyTorch version too
    print("Starting benchmark runs...")
    time_triton = do_bench(bench_fn)
    time_torch = do_bench(bench_fn_torch)
    print(f"Time Triton Forward: {time_triton:.4f} ms")
    print(f"Time Torch Forward: {time_torch:.4f} ms")

    bytes_input = sum(t.numel() * t.element_size() for t in [m_bench, s_bench, n_bench, Z_bench, g_bench])
    bytes_output = bytes_input
    total_bytes = bytes_input + bytes_output
    gb_transferred = total_bytes / (1024**3)
    print(f"Total memory transfer: {gb_transferred:.2f} GB")

    h100_bw_gbs = 3350
    ideal_time_ms = (total_bytes / (h100_bw_gbs * (1024**3))) * 1000
    print(f"Ideal time on H100 ({h100_bw_gbs} GB/s): {ideal_time_ms:.4f} ms")
    print(f"Bandwidth util of Triton Forward: {ideal_time_ms / time_triton * 100:.2f} %")
    print(f"Bandwidth util of Torch Forward: {ideal_time_ms / time_torch * 100:.2f} %")
