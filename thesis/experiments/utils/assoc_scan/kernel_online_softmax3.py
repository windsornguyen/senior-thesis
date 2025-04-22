"""autograd integration

Do spam broadcasts so this is extremely inefficient right now.
"""

import torch
import triton
import triton.language as tl

from typing import Tuple
from triton.testing import do_bench


dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def softmax_combine_fn(m_x, d_x, n_x, Z_x, g_x, m_y, d_y, n_y, Z_y, g_y):
    # Note: Inputs are assumed to be pre-broadcasted to the same shape
    # by the caller if used with tl.associative_scan.
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y

    # Removed [:, None] as inputs m_x/m_y and n_x/n_y are pre-broadcasted
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
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def fwd_online_softmax_kernel(
    x_ptr,
    v_ptr,
    gated_Z_ptr,
    gates_z_ptr,
    out_ptr,
    m_leaf_ptr,
    s_leaf_ptr,
    n_leaf_ptr,
    Z_leaf_ptr,
    g_leaf_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    x_stride_b: int,
    x_stride_h: int,
    x_stride_l: int,
    v_stride_b: int,
    v_stride_h: int,
    v_stride_l: int,
    v_stride_h2: int,
    Z_stride_b: int,
    Z_stride_h: int,
    Z_stride_l: int,
    Z_stride_h1: int,
    Z_stride_h2: int,
    gates_z_stride_b: int,
    gates_z_stride_h: int,
    gates_z_stride_l: int,
    out_stride_b: int,
    out_stride_h: int,
    out_stride_l: int,
    m_leaf_stride_b: int,
    m_leaf_stride_h: int,
    m_leaf_stride_l: int,
    s_leaf_stride_b: int,
    s_leaf_stride_h: int,
    s_leaf_stride_l: int,
    n_leaf_stride_b: int,
    n_leaf_stride_h: int,
    n_leaf_stride_l: int,
    n_leaf_stride_h2: int,
    Z_leaf_stride_b: int,
    Z_leaf_stride_h: int,
    Z_leaf_stride_l: int,
    Z_leaf_stride_h1: int,
    Z_leaf_stride_h2: int,
    g_leaf_stride_b: int,
    g_leaf_stride_h: int,
    g_leaf_stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
    H_CONST: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * x_stride_b + pid_feature * x_stride_h

    # Initialize global leaves
    m_i = float("-inf")
    d_i = 0.0
    n_i = tl.zeros([H_CONST], dtype=DTYPE)
    Z_i = tl.zeros([H_CONST, H_CONST], dtype=DTYPE)
    g_i = 0.0

    # First pass: scan to compute m_N, s_N, n_N, Z_N, g_N
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len
        boundary_mask = offs < seq_len - start_col

        # Calculate sequence offsets with broadcasting dimension
        seq_offs = start_offset + (start_col + offs) * x_stride_l
        seq_offs_v = start_offset + (start_col + offs) * v_stride_l
        seq_offs_Z = start_offset + (start_col + offs) * Z_stride_l
        seq_offs_gates = start_offset + (start_col + offs) * gates_z_stride_l

        # Calculate head offsets
        head_offs_v = tl.arange(0, H_CONST) * v_stride_h2
        head_offs_Z = tl.arange(0, H_CONST)[:, None] * Z_stride_h1 + tl.arange(0, H_CONST)[None, :] * Z_stride_h2

        # Calculate full offsets using broadcasting
        x_offsets = seq_offs
        v_offsets = seq_offs_v[:, None] + head_offs_v[None, :]
        Z_offsets = seq_offs_Z[:, None, None] + head_offs_Z[None, :, :]
        gates_z_offsets = seq_offs_gates

        x = tl.load(x_ptr + x_offsets, mask=boundary_mask & mask, other=float("-inf"))
        m0 = x
        d0 = tl.where(mask, 1.0, 0.0)
        n0 = tl.load(v_ptr + v_offsets, mask=boundary_mask[:, None] & mask[:, None], other=0.0)
        Z0 = tl.load(gated_Z_ptr + Z_offsets, mask=boundary_mask[:, None, None] & mask[:, None, None], other=0.0)
        g0 = tl.load(gates_z_ptr + gates_z_offsets, mask=boundary_mask & mask, other=0.0)

        # --- Pre-broadcast leaves to satisfy tl.associative_scan shape constraint ---
        # This is inefficient (memory/compute) but preserves O(log n) scan.
        # Future optimization: custom kernel avoiding this broadcast.
        m0_b = tl.broadcast_to(m0[:, None, None], [BLOCK_SIZE, H_CONST, H_CONST])
        d0_b = tl.broadcast_to(d0[:, None, None], [BLOCK_SIZE, H_CONST, H_CONST])
        # Broadcast n0 from [BLOCK_SIZE, H_CONST] to [BLOCK_SIZE, H_CONST, H_CONST]
        # Add a singleton dim for H2, then broadcast H1
        n0_b = tl.broadcast_to(n0[:, :, None], [BLOCK_SIZE, H_CONST, H_CONST])
        Z0_b = Z0  # Z0 already has the shape [BLOCK_SIZE, H_CONST, H_CONST]
        g0_b = tl.broadcast_to(g0[:, None, None], [BLOCK_SIZE, H_CONST, H_CONST])
        # -----------------------------------------------------------------------------

        # Scan within chunk using pre-broadcasted inputs
        m_out, d_out, n_out, Z_out, g_out = tl.associative_scan(
            (m0_b, d0_b, n0_b, Z0_b, g0_b), axis=0, combine_fn=softmax_combine_fn, reverse=False
        )

        # Select last valid element in chunk
        chunk_size = tl.minimum(BLOCK_SIZE, seq_len - start_col)
        last_idx = chunk_size - 1
        last_mask = offs == last_idx
        # Broadcast last_mask to match the shape of scan outputs [BLOCK_SIZE, H_CONST, H_CONST]
        last_mask_b = last_mask[:, None, None]

        m_local = tl.max(tl.where(last_mask_b, m_out, float("-inf")), axis=0)
        d_local = tl.sum(tl.where(last_mask_b, d_out, 0.0), axis=0)

        # The axis=0 sum effectively handles the reduction after selecting the last element via mask
        n_local = tl.sum(tl.where(last_mask_b, n_out, 0.0), axis=0)
        Z_local = tl.sum(tl.where(last_mask_b, Z_out, 0.0), axis=0)
        g_local = tl.sum(tl.where(last_mask_b, g_out, 0.0), axis=0)

        # --- Reduce local results to match accumulator shapes ---
        # Exploit redundancy: max for m, sum for d/n/g. Z is already correct.
        m_local_reduced = tl.max(tl.max(m_local, axis=1), axis=0)  # H,H -> scalar
        d_local_reduced = tl.sum(tl.sum(d_local, axis=1), axis=0)  # H,H -> scalar (use sum for d)
        n_local_reduced = tl.sum(n_local, axis=1)  # H,H -> H (use sum for n, reduce last dim)
        Z_local_reduced = Z_local  # H,H -> H,H (no reduction needed)
        g_local_reduced = tl.sum(tl.sum(g_local, axis=1), axis=0)  # H,H -> scalar (use sum for g)
        # -----------------------------------------------------------

        # Update global leaves using reduced local results
        old_m_i = m_i  # old scalar m_i
        m_new = tl.maximum(m_i, m_local_reduced)  # scalar vs scalar -> scalar

        exp_acc = tl.exp(m_i - m_new)  # scalar
        exp_local = tl.exp(m_local_reduced - m_new)  # scalar

        # Update d_i (scalar)
        d_i = d_i * exp_acc + d_local_reduced * exp_local

        # Update n_i (vector [H]) - broadcast scalar exps
        n_i = n_i * exp_acc + n_local_reduced * exp_local

        # Update Z_i (matrix [H, H])
        Z_i = Z_i + Z_local_reduced

        # Update g_i (scalar)
        g_i = g_i + g_local_reduced

        # Update m_i for next iteration
        m_i = m_new  # Assign scalar back to m_i

        # Store leaf outputs for this chunk
        m_leaf_seq_offs = start_offset + (start_col + offs) * m_leaf_stride_l
        s_leaf_seq_offs = start_offset + (start_col + offs) * s_leaf_stride_l
        n_leaf_seq_offs = start_offset + (start_col + offs) * n_leaf_stride_l
        Z_leaf_seq_offs = start_offset + (start_col + offs) * Z_leaf_stride_l
        g_leaf_seq_offs = start_offset + (start_col + offs) * g_leaf_stride_l

        # Reduce the results from the bloated shape [BLOCK_SIZE, H, H]
        # back to their intended conceptual shapes using tl.max, exploiting the
        # redundancy introduced by pre-broadcasting.
        m_out_to_store = tl.max(tl.max(m_out, axis=2), axis=1)
        d_out_to_store = tl.max(tl.max(d_out, axis=2), axis=1)
        n_out_to_store = tl.max(n_out, axis=2)  # Reduce only the last H dim
        Z_out_to_store = Z_out  # Z keeps full shape
        g_out_to_store = tl.max(tl.max(g_out, axis=2), axis=1)

        tl.store(m_leaf_ptr + m_leaf_seq_offs, m_out_to_store, mask=boundary_mask & mask)
        tl.store(s_leaf_ptr + s_leaf_seq_offs, d_out_to_store, mask=boundary_mask & mask)
        tl.store(
            n_leaf_ptr + n_leaf_seq_offs[:, None] + tl.arange(0, H_CONST)[None, :] * n_leaf_stride_h2,
            n_out_to_store,  # Use reduced n_out
            mask=boundary_mask[:, None] & mask[:, None],
        )
        tl.store(
            Z_leaf_ptr
            + Z_leaf_seq_offs[:, None, None]
            + (tl.arange(0, H_CONST)[:, None] * Z_leaf_stride_h1 + tl.arange(0, H_CONST)[None, :] * Z_leaf_stride_h2)[
                None, :, :
            ],
            Z_out_to_store,  # Use original Z_out
            mask=boundary_mask[:, None, None] & mask[:, None, None],
        )
        tl.store(g_leaf_ptr + g_leaf_seq_offs, g_out_to_store, mask=boundary_mask & mask)

    # Second pass: compute softmax output exp(x - m_N) / s_N
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len
        boundary_mask = offs < seq_len - start_col
        out_seq_offs = start_offset + (start_col + offs) * out_stride_l
        x_seq_offs = start_offset + (start_col + offs) * x_stride_l  # Recalculate x_offset for clarity
        x = tl.load(x_ptr + x_seq_offs, mask=boundary_mask & mask, other=float("-inf"))
        softmax_out = tl.exp(x - m_i) / (d_i + 1e-6)
        tl.store(out_ptr + out_seq_offs, softmax_out, mask=boundary_mask & mask)


class OnlineSoftmaxScan(torch.autograd.Function):
    """PyTorch autograd wrapper for an online softmax scan using Triton."""

    @staticmethod
    def forward(x: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
        batch_size, feature_size, seq_len = x.shape
        h = v.shape[-1]
        x = x.contiguous().cuda()
        v = v.contiguous().cuda()
        gated_Z = gated_Z.contiguous().cuda()
        gates_z = gates_z.contiguous().cuda()
        out = torch.empty_like(x)
        m_leaf = torch.empty_like(x)
        s_leaf = torch.empty_like(x)
        n_leaf = torch.empty_like(v)
        Z_leaf = torch.empty_like(gated_Z)
        g_leaf = torch.empty_like(gates_z)

        grid = (batch_size, feature_size, 1)
        triton_dtype = dtype_map.get(x.dtype, tl.float32)

        fwd_online_softmax_kernel[grid](
            x,
            v,
            gated_Z,
            gates_z,
            out,
            m_leaf,
            s_leaf,
            n_leaf,
            Z_leaf,
            g_leaf,
            batch_size,
            feature_size,
            seq_len,
            x.stride(0),
            x.stride(1),
            x.stride(2),
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
            out.stride(0),
            out.stride(1),
            out.stride(2),
            m_leaf.stride(0),
            m_leaf.stride(1),
            m_leaf.stride(2),
            s_leaf.stride(0),
            s_leaf.stride(1),
            s_leaf.stride(2),
            n_leaf.stride(0),
            n_leaf.stride(1),
            n_leaf.stride(2),
            n_leaf.stride(3),
            Z_leaf.stride(0),
            Z_leaf.stride(1),
            Z_leaf.stride(2),
            Z_leaf.stride(3),
            Z_leaf.stride(4),
            g_leaf.stride(0),
            g_leaf.stride(1),
            g_leaf.stride(2),
            H_CONST=h,
            DTYPE=triton_dtype,
        )

        return out, m_leaf, s_leaf, n_leaf, Z_leaf, g_leaf

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, v, gated_Z, gates_z = inputs
        ctx.save_for_backward(x, v, gated_Z, gates_z)
        ctx.triton_dtype = dtype_map.get(x.dtype, tl.float32)

    @staticmethod
    def backward(ctx, grad_out, grad_m_leaf, grad_s_leaf, grad_n_leaf, grad_Z_leaf, grad_g_leaf):
        # Placeholder for backward pass, to be implemented later
        pass


def online_softmax_scan(x: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
    """Executes an online softmax scan on the provided tensors."""
    return OnlineSoftmaxScan.apply(x, v, gated_Z, gates_z)


# Takes combine_fn, xs, dim
from thesis.experiments.utils.assoc_scan.kernel import associative_scan


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
    return associative_scan(combine_fn=combine_fn_ref, xs=leaves, dim=0, combine_mode="generic")


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

    """
    ----
    qk_slice : [L]          similarity logits for this stream
    v_slice  : [L, h]       L2‑normalised V (first‑order numerator)
    Z_slice  : [L, h, h]    gated outer‑product accumulator
    g_slice  : [L]          scalar gate sequence
    """
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

    # Compare outputs
    print(f"Max absolute error (softmax): {(out - torch.softmax(sim, dim=-1)).abs().max().item()}")
    print(f"Max absolute error (m_leaf): {(m_leaf - ref_m).abs().max().item()}")
    print(f"Max absolute error (s_leaf): {(s_leaf - ref_s).abs().max().item()}")
    print(f"Max absolute error (n_leaf): {(n_leaf - ref_n).abs().max().item()}")
    print(f"Max absolute error (Z_leaf): {(Z_leaf - ref_Z).abs().max().item()}")
    print(f"Max absolute error (g_leaf): {(g_leaf - ref_g).abs().max().item()}")

    # Performance
    t1 = do_bench(lambda: online_softmax_scan(sim, v, gated_Z, gates_z))
    t2 = do_bench(lambda: torch.softmax(sim, dim=-1))
    print(f"Triton: {t1:.3f} ms, Torch: {t2:.3f} ms")
    num_bytes = sim.numel() * sim.element_size() + out.numel() * out.element_size()
    ideal_time_ms = num_bytes / 3.35e12 * 1e3  # On H100
    print(f"Ideal time on H100: {ideal_time_ms:.3f} ms")
    print(f"Bandwidth util of Triton: {min(ideal_time_ms / t1 * 100, 100):.2f} %")
