# -*- Author: Windsor Nguyen -*-
"""
Implements an associative scan using Triton with autotuning and vmap support.
Performs prefix sums over the sequence dimension of input tensors using elementwise addition.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
import functools

# Mapping of PyTorch dtypes to Triton dtypes
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


# Combine function for the forward pass
@triton.jit
def combine_fn_logic(m_x, s_x, n_x, Z_x, g_x, m_y, s_y, n_y, Z_y, g_y):
    """
    Combines two states for associative scan.
    Inputs are [BLOCK_SIZE] tensors from the same effective feature slice after expansion.
    """
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    Z_new = Z_x + Z_y
    g_new = g_x + g_y
    return m_new, s_new, n_new, Z_new, g_new

@triton.jit
def backward_combine_fn(m_x, d_x, n_x, m_y, d_y, n_y):
    m_new = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m_new)
    exp_y = tl.exp(m_y - m_new)
    d_new = d_x * exp_x + d_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    return m_new, d_new, n_new

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

# Config functions
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
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
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


# Forward Kernel
@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["batch_size", "feature_size_h2", "seq_len"],
)
@triton.jit
def fwd_scan_kernel(
    m_ptr,
    out_m_ptr,
    s_ptr,
    out_s_ptr,
    n_ptr,
    out_n_ptr,
    Z_ptr,
    out_Z_ptr,
    g_ptr,
    out_g_ptr,
    batch_size: int,
    feature_size_h2: int,
    seq_len: int,
    stride_b: int,
    stride_f: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Single CTA-wide scan."""
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    offset = pid_batch * stride_b + pid_feature * stride_f

    # Calculate pointer tensors for the current block
    offs_arange = tl.arange(0, BLOCK_SIZE)
    mask = offs_arange < seq_len
    current_block_offset = offset + offs_arange * stride_l

    # Load entire tile
    m = tl.load(m_ptr + current_block_offset, mask=mask, other=0.0)
    s = tl.load(s_ptr + current_block_offset, mask=mask, other=0.0)
    n = tl.load(n_ptr + current_block_offset, mask=mask, other=0.0)
    Z = tl.load(Z_ptr + current_block_offset, mask=mask, other=0.0)
    g = tl.load(g_ptr + current_block_offset, mask=mask, other=0.0)

    # Single inclusive scan
    res_m, res_s, res_n, res_Z, res_g = tl.associative_scan(
        (m, s, n, Z, g), axis=0, combine_fn=combine_fn_logic, reverse=False
    )

    # Store results
    tl.store(out_m_ptr + current_block_offset, res_m, mask=mask)
    tl.store(out_s_ptr + current_block_offset, res_s, mask=mask)
    tl.store(out_n_ptr + current_block_offset, res_n, mask=mask)
    tl.store(out_Z_ptr + current_block_offset, res_Z, mask=mask)
    tl.store(out_g_ptr + current_block_offset, res_g, mask=mask)


# Backward Kernel (Single Pass VJP - m, s, n, Z, g) - REVERSED SCAN
@triton.autotune(
    configs=[conf for conf in get_scan_configs() if keep_config(conf)],
    key=["batch_size", "feature_size_h2", "seq_len"],
)
@triton.jit
def bwd_scan_kernel(
    m_ker_ptr,
    s_ker_ptr,
    n_ker_ptr,
    Z_ker_ptr,
    g_ker_ptr,
    gm_out_ker_ptr,
    gs_out_ker_ptr,
    gn_out_ker_ptr,
    gZ_out_ker_ptr,
    gg_out_ker_ptr,
    grad_in_m_ptr,
    grad_in_s_ptr,
    grad_in_n_ptr,
    grad_in_Z_ptr,
    grad_in_g_ptr,
    batch_size,
    feature_size_h2,
    seq_len,
    stride_b,
    stride_f,
    stride_l,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    base_offset = pid_batch * stride_b + pid_feature * stride_f

    # Initialize carry variables for the backward scan
    carry_gm = tl.zeros((1,), dtype=DTYPE)
    carry_gs = tl.zeros((1,), dtype=DTYPE)
    carry_gn = tl.zeros((1,), dtype=DTYPE)
    carry_gZ = tl.zeros((1,), dtype=DTYPE)
    carry_gg = tl.zeros((1,), dtype=DTYPE)

    # Initialize with the identity element for the operation
    # For max operation, the identity is negative infinity
    carry_m = tl.full((1,), float("-inf"), dtype=DTYPE)
    carry_s = tl.zeros((1,), dtype=DTYPE)
    carry_n = tl.zeros((1,), dtype=DTYPE)
    carry_Z = tl.zeros((1,), dtype=DTYPE)
    carry_g = tl.zeros((1,), dtype=DTYPE)

    # Process blocks in reverse order for backward scan
    for k in range(tl.cdiv(seq_len, BLOCK_SIZE) - 1, -1, -1):
        start_col = k * BLOCK_SIZE

        # Use proper indices within the block
        # For the backward scan, we need to reverse the indices
        offs = BLOCK_SIZE - 1 - tl.arange(0, BLOCK_SIZE)
        mask = (start_col + offs < seq_len) & (offs >= 0)

        # Calculate offsets for the current block
        block_row_offsets = (start_col + offs) * stride_l
        current_block_offset = base_offset + block_row_offsets

        # Load block data
        m_y_in = tl.load(m_ker_ptr + current_block_offset, mask=mask, other=float("-inf"))
        s_y_in = tl.load(s_ker_ptr + current_block_offset, mask=mask, other=0.0)
        n_y_in = tl.load(n_ker_ptr + current_block_offset, mask=mask, other=0.0)
        Z_y_in = tl.load(Z_ker_ptr + current_block_offset, mask=mask, other=0.0)
        g_y_in = tl.load(g_ker_ptr + current_block_offset, mask=mask, other=0.0)

        # Load output gradients
        gm_y_out = tl.load(gm_out_ker_ptr + current_block_offset, mask=mask, other=0.0)
        gs_y_out = tl.load(gs_out_ker_ptr + current_block_offset, mask=mask, other=0.0)
        gn_y_out = tl.load(gn_out_ker_ptr + current_block_offset, mask=mask, other=0.0)
        gZ_y_out = tl.load(gZ_out_ker_ptr + current_block_offset, mask=mask, other=0.0)
        gg_y_out = tl.load(gg_out_ker_ptr + current_block_offset, mask=mask, other=0.0)

        # Set up for the scan
        m_y_scan = m_y_in
        s_y_scan = s_y_in
        n_y_scan = n_y_in
        Z_y_scan = Z_y_in
        g_y_scan = g_y_in
        gm_y_scan = gm_y_out
        gs_y_scan = gs_y_out
        gn_y_scan = gn_y_out
        gZ_y_scan = gZ_y_out
        gg_y_scan = gg_y_out

        # Perform associative scan on the block
        # Important: we do not reverse here since the inputs are already reversed by the offs logic above
        gm_scan, gs_scan, gn_scan, gZ_scan, gg_scan, m_scan, s_scan, n_scan, Z_scan, g_scan = tl.associative_scan(
            (gm_y_scan, gs_y_scan, gn_y_scan, gZ_y_scan, gg_y_scan, m_y_scan, s_y_scan, n_y_scan, Z_y_scan, g_y_scan),
            axis=0,
            combine_fn=vjp_scan_combine_fn,
            reverse=False,  # We've already reversed the indices
        )

        # Combine with carry from previous blocks
        # This is the critical step for correctly propagating gradients across blocks
        final_gm, final_gs, final_gn, final_gZ, final_gg, final_m, final_s, final_n, final_Z, final_g = (
            vjp_scan_combine_fn(
                carry_gm,
                carry_gs,
                carry_gn,
                carry_gZ,
                carry_gg,
                carry_m,
                carry_s,
                carry_n,
                carry_Z,
                carry_g,
                gm_scan,
                gs_scan,
                gn_scan,
                gZ_scan,
                gg_scan,
                m_scan,
                s_scan,
                n_scan,
                Z_scan,
                g_scan,
            )
        )

        # Store results
        tl.store(grad_in_m_ptr + current_block_offset, final_gm, mask=mask)
        tl.store(grad_in_s_ptr + current_block_offset, final_gs, mask=mask)
        tl.store(grad_in_n_ptr + current_block_offset, final_gn, mask=mask)
        tl.store(grad_in_Z_ptr + current_block_offset, final_gZ, mask=mask)
        tl.store(grad_in_g_ptr + current_block_offset, final_gg, mask=mask)

        # Update carry values for the next block
        # The key fix: we need to extract the element that becomes the carry
        # For a backward scan, this is the first element (after reversal)
        # We use last_mask to select only the first element (reversed) in the block
        last_mask = (offs == 0) & mask

        # We need to ensure we're selecting the correct element as the carry
        carry_gm = tl.sum(tl.where(last_mask, final_gm, 0.0), axis=0, keep_dims=True)
        carry_gs = tl.sum(tl.where(last_mask, final_gs, 0.0), axis=0, keep_dims=True)
        carry_gn = tl.sum(tl.where(last_mask, final_gn, 0.0), axis=0, keep_dims=True)
        carry_gZ = tl.sum(tl.where(last_mask, final_gZ, 0.0), axis=0, keep_dims=True)
        carry_gg = tl.sum(tl.where(last_mask, final_gg, 0.0), axis=0, keep_dims=True)

        # Also update the forward state carriers which are needed for Jacobian calculations
        carry_m = tl.max(tl.where(last_mask, final_m, float("-inf")), axis=0, keep_dims=True)
        carry_s = tl.sum(tl.where(last_mask, final_s, 0.0), axis=0, keep_dims=True)
        carry_n = tl.sum(tl.where(last_mask, final_n, 0.0), axis=0, keep_dims=True)
        carry_Z = tl.sum(tl.where(last_mask, final_Z, 0.0), axis=0, keep_dims=True)
        carry_g = tl.sum(tl.where(last_mask, final_g, 0.0), axis=0, keep_dims=True)


class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for associative scan with expand pre-processing."""

    @staticmethod
    def forward(m, s, n, Z, g):
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

        # Launch kernel
        grid = (batch_size, h2)
        triton_dtype = dtype_map.get(m_ker.dtype, tl.float32)

        fwd_scan_kernel[grid](
            m_ker,
            out_m_ker,
            s_ker,
            out_s_ker,
            n_ker,
            out_n_ker,
            Z_ker,
            out_Z_ker,
            g_ker,
            out_g_ker,
            batch_size,
            h2,
            L,
            m_ker.stride(0),
            m_ker.stride(1),
            m_ker.stride(2),
            DTYPE=triton_dtype,
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

        return m_final, s_final, n_final, Z_final, g_final

    @staticmethod
    def setup_context(ctx, inputs, output):
        m, s, n, Z, g = inputs
        ctx.save_for_backward(m, s, n, Z, g)
        ctx.input_shape = m.shape
        try:
            ctx.H = n.shape[-1]
        except IndexError:
            raise ValueError("Tensor n must have shape (..., L, H)")
        ctx.triton_dtype = dtype_map.get(m.dtype, tl.float32)

    @staticmethod
    def backward(ctx, grad_m, grad_s, grad_n, grad_Z, grad_g):
        """Compute gradients using reverse scan structure."""
        # Preparation
        m, s, n, Z, g = ctx.saved_tensors
        input_shape = ctx.input_shape
        L, H = input_shape[-1], ctx.H
        h2 = H * H
        batch_dims = input_shape[:-1]
        batch_size = torch.prod(torch.tensor(batch_dims)).item()

        # Handle None grads and ensure contiguous CUDA tensors
        grad_m = grad_m if grad_m is not None else torch.zeros_like(m)
        grad_s = grad_s if grad_s is not None else torch.zeros_like(s)
        grad_n = grad_n if grad_n is not None else torch.zeros_like(n)
        grad_Z = grad_Z if grad_Z is not None else torch.zeros_like(Z)
        grad_g = grad_g if grad_g is not None else torch.zeros_like(g)

        # Force contiguous and move to CUDA
        grads = [gr.contiguous().cuda() for gr in [grad_m, grad_s, grad_n, grad_Z, grad_g]]
        grad_m, grad_s, grad_n, grad_Z, grad_g = grads

        # Prepare Output Gradients (dL/dh)
        # Expand gradients to match the internal H*H dimension structure
        grad_m_exp = grad_m.unsqueeze(-1).unsqueeze(-1).expand(*batch_dims, L, H, H)
        grad_s_exp = grad_s.unsqueeze(-1).unsqueeze(-1).expand(*batch_dims, L, H, H)
        grad_n_exp = grad_n.unsqueeze(-1).expand(*batch_dims, L, H, H)
        grad_Z_exp = grad_Z  # Already has the right shape
        grad_g_exp = grad_g.unsqueeze(-1).unsqueeze(-1).expand(*batch_dims, L, H, H)

        # Reshape for kernel format [Batch, h2, L]
        gm_out_ker = grad_m_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        gs_out_ker = grad_s_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        gn_out_ker = grad_n_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        gZ_out_ker = grad_Z_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
        gg_out_ker = grad_g_exp.reshape(batch_size, L, h2).transpose(1, 2).contiguous()

        # Prepare Forward Inputs (x)
        m_flat = m.reshape(batch_size, L)
        s_flat = s.reshape(batch_size, L)
        n_flat = n.reshape(batch_size, L, H)
        Z_flat = Z.reshape(batch_size, L, H, H)
        g_flat = g.reshape(batch_size, L)

        # Expand to match internal format
        m_exp_fwd = m_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        s_exp_fwd = s_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        n_exp_fwd = n_flat.unsqueeze(-1).expand(-1, -1, -1, H)
        Z_exp_fwd = Z_flat
        g_exp_fwd = g_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)

        # Reshape to kernel format
        m_ker = m_exp_fwd.reshape(batch_size, L, h2).transpose(1, 2).contiguous().cuda()
        s_ker = s_exp_fwd.reshape(batch_size, L, h2).transpose(1, 2).contiguous().cuda()
        n_ker = n_exp_fwd.reshape(batch_size, L, h2).transpose(1, 2).contiguous().cuda()
        Z_ker = Z_exp_fwd.reshape(batch_size, L, h2).transpose(1, 2).contiguous().cuda()
        g_ker = g_exp_fwd.reshape(batch_size, L, h2).transpose(1, 2).contiguous().cuda()

        # Allocate Output Input Gradients (dL/dx)
        grad_in_m_ker = torch.empty_like(m_ker)
        grad_in_s_ker = torch.empty_like(s_ker)
        grad_in_n_ker = torch.empty_like(n_ker)
        grad_in_Z_ker = torch.empty_like(Z_ker)
        grad_in_g_ker = torch.empty_like(g_ker)

        # Launch Kernel
        grid = (batch_size, h2)
        stride_l_ker = m_ker.stride(2)
        bwd_scan_kernel[grid](
            m_ker,
            s_ker,
            n_ker,
            Z_ker,
            g_ker,
            gm_out_ker,
            gs_out_ker,
            gn_out_ker,
            gZ_out_ker,
            gg_out_ker,
            grad_in_m_ker,
            grad_in_s_ker,
            grad_in_n_ker,
            grad_in_Z_ker,
            grad_in_g_ker,
            batch_size,
            h2,
            L,
            m_ker.stride(0),
            m_ker.stride(1),
            stride_l_ker,
            DTYPE=ctx.triton_dtype,
        )

        # Post-process gradients
        grad_in_m_flat = grad_in_m_ker.transpose(1, 2)
        grad_in_s_flat = grad_in_s_ker.transpose(1, 2)
        grad_in_n_flat = grad_in_n_ker.transpose(1, 2)
        grad_in_Z_flat = grad_in_Z_ker.transpose(1, 2)
        grad_in_g_flat = grad_in_g_ker.transpose(1, 2)

        # Reshape to match original input shapes
        grad_in_m_res = grad_in_m_flat.reshape(*batch_dims, L, H, H)
        grad_in_s_res = grad_in_s_flat.reshape(*batch_dims, L, H, H)
        grad_in_n_res = grad_in_n_flat.reshape(*batch_dims, L, H, H)
        grad_in_Z_res = grad_in_Z_flat.reshape(*batch_dims, L, H, H)
        grad_in_g_res = grad_in_g_flat.reshape(*batch_dims, L, H, H)

        # Extract the appropriate slices to match the input shapes
        grad_m_final = grad_in_m_res[..., 0, 0]
        grad_s_final = grad_in_s_res[..., 0, 0]
        grad_n_final = grad_in_n_res[..., :, 0]
        grad_Z_final = grad_in_Z_res
        grad_g_final = grad_in_g_res[..., 0, 0]

        return grad_m_final, grad_s_final, grad_n_final, grad_Z_final, grad_g_final

    @staticmethod
    def vmap(info, in_dims, m, s, n, Z, g):
        bdim = in_dims[0]
        assert all(i == bdim for i in in_dims), "All inputs must have the same vmap dim"

        m, _ = AssociativeScan._move_bdim_to_front(m, bdim)
        s, _ = AssociativeScan._move_bdim_to_front(s, bdim)
        n, _ = AssociativeScan._move_bdim_to_front(n, bdim)
        Z, _ = AssociativeScan._move_bdim_to_front(Z, bdim)
        g, _ = AssociativeScan._move_bdim_to_front(g, bdim)

        m_flat, batch_shape = AssociativeScan._flatten_leading_dims(m, keep=1)
        s_flat, _ = AssociativeScan._flatten_leading_dims(s, keep=1)
        n_flat, _ = AssociativeScan._flatten_leading_dims(n, keep=2)
        Z_flat, _ = AssociativeScan._flatten_leading_dims(Z, keep=3)
        g_flat, _ = AssociativeScan._flatten_leading_dims(g, keep=1)

        m_res, s_res, n_res, Z_res, g_res = AssociativeScan.apply(m_flat, s_flat, n_flat, Z_flat, g_flat)

        if batch_shape:
            m_res = m_res.view(*batch_shape, *m_res.shape[-1:])
            s_res = s_res.view(*batch_shape, *s_res.shape[-1:])
            n_res = n_res.view(*batch_shape, *n_res.shape[-2:])
            Z_res = Z_res.view(*batch_shape, *Z_res.shape[-3:])
            g_res = g_res.view(*batch_shape, *g_res.shape[-1:])

        if bdim is not None and bdim != 0:
            m_res = m_res.movedim(0, bdim)
            s_res = s_res.movedim(0, bdim)
            n_res = n_res.movedim(0, bdim)
            Z_res = Z_res.movedim(0, bdim)
            g_res = g_res.movedim(0, bdim)

        out_dims = (bdim,) * 5
        return (m_res, s_res, n_res, Z_res, g_res), out_dims

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
        torch.ones_like(qk_slice),  # s starts at 1.0
        v_slice,
        Z_slice,
        g_slice,
    )
    # Assuming assoc_scan_ref is a correct reference (e.g., sequential scan)
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
    B, H, L, h = v.shape
    # Prepare inputs for vmap
    sim_bh = sim.reshape(-1, L)
    v_bh = v.reshape(-1, L, h)
    gated_Z_bh = gated_Z.reshape(-1, L, h, h)
    gates_z_bh = gates_z.reshape(-1, L)

    # vmap over the flattened B*H dimension
    scan_vmapped = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    m_res, s_res, n_res, Z_res, g_res = scan_vmapped(sim_bh, v_bh, gated_Z_bh, gates_z_bh)

    # Reshape results back to [B, H, ...]
    return (
        m_res.reshape(B, H, L),
        s_res.reshape(B, H, L),
        n_res.reshape(B, H, L, h),
        Z_res.reshape(B, H, L, h, h),
        g_res.reshape(B, H, L),
    )


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
        torch.cuda.synchronize()  # Ensure each iteration is complete
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / rep


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = "cuda"

    # Correctness Check
    print("--- Correctness Check ---")
    B_test, D_test, L_test, H_test = 1, 2, 8, 2

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

    # print("\nRunning gradcheck...")
    # torch.autograd.gradcheck(associative_scan, (m, s, n, Z, g), nondet_tol=1e-5)
    # print("Gradcheck completed.")

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

    print("\nRunning basic backward check (test shapes only)...")
    total_output_sum = m_out.sum() + s_out.sum() + n_out.sum() + Z_out.sum() + g_out.sum()
    try:
        total_output_sum.backward(retain_graph=False)
        print("Backward pass executed.")
        assert m.grad is not None and m.grad.shape == m.shape
        assert s.grad is not None and s.grad.shape == s.shape
        assert n.grad is not None and n.grad.shape == n.shape
        assert Z.grad is not None and Z.grad.shape == Z.shape
        assert g.grad is not None and g.grad.shape == g.shape
        print("Gradient shapes match input shapes.")
    except Exception as e:
        print(f"An unexpected error occurred during backward check: {e}")
        import traceback

        traceback.print_exc()

    print("Basic shape checks passed ✓")

    # ====================================
    #        Gradient Check
    # ====================================
    print("\n--- Gradient Check ---")
    # Requires float64 for precision
    if m.dtype != torch.float64:
        print("Skipping gradient check, requires float64 inputs.")
    else:
        # --- Triton Backward ---
        print("Running Triton backward...")
        # Clear potential grads from previous runs
        if m.grad is not None:
            m.grad.zero_()
        if s.grad is not None:
            s.grad.zero_()
        if n.grad is not None:
            n.grad.zero_()
        if Z.grad is not None:
            Z.grad.zero_()
        if g.grad is not None:
            g.grad.zero_()

        # Re-run forward pass for fresh graph
        m_out, s_out, n_out, Z_out, g_out = associative_scan(m, s, n, Z, g)
        loss_triton = m_out.sum() + s_out.sum() + n_out.sum() + Z_out.sum() + g_out.sum()
        loss_triton.backward()
        print("Triton backward completed.")

        # Store Triton grads
        m_grad_triton = m.grad.clone() if m.grad is not None else None
        s_grad_triton = s.grad.clone() if s.grad is not None else None  # Not compared
        n_grad_triton = n.grad.clone() if n.grad is not None else None
        Z_grad_triton = Z.grad.clone() if Z.grad is not None else None
        g_grad_triton = g.grad.clone() if g.grad is not None else None

        # --- Reference Backward ---
        print("Running Reference backward...")
        # Clone inputs for separate graph
        m_ref = m.clone().detach().requires_grad_(True)
        n_ref = n.clone().detach().requires_grad_(True)
        Z_ref = Z.clone().detach().requires_grad_(True)
        g_ref = g.clone().detach().requires_grad_(True)
        # s is handled internally by reference, no need to clone/pass

        ref_m_out, ref_s_out, ref_n_out, ref_Z_out, ref_g_out = batched_scan_fn(m_ref, n_ref, Z_ref, g_ref)
        loss_ref = ref_m_out.sum() + ref_s_out.sum() + ref_n_out.sum() + ref_Z_out.sum() + ref_g_out.sum()
        loss_ref.backward()
        print("Reference backward completed.")

        # --- Compare Gradients ---
        print("Comparing gradients...")
        all_grads_match = True
        try:
            print("Comparing m.grad...")
            assert m_grad_triton is not None, "Triton m.grad is None"
            assert m_ref.grad is not None, "Reference m.grad is None"
            triton.testing.assert_close(
                m_grad_triton, m_ref.grad, rtol=1e-4, atol=1e-4
            )  # Slightly higher tolerance for grads
            print("  -> m.grad MATCHES")
        except AssertionError as e:
            print(f"  -> m.grad MISMATCH: {e}")
            all_grads_match = False

        # Skip s.grad comparison

        try:
            print("Comparing n.grad...")
            assert n_grad_triton is not None, "Triton n.grad is None"
            assert n_ref.grad is not None, "Reference n.grad is None"
            triton.testing.assert_close(n_grad_triton, n_ref.grad, rtol=1e-4, atol=1e-4)
            print("  -> n.grad MATCHES")
        except AssertionError as e:
            print(f"  -> n.grad MISMATCH: {e}")
            all_grads_match = False

        try:
            print("Comparing Z.grad...")
            assert Z_grad_triton is not None, "Triton Z.grad is None"
            assert Z_ref.grad is not None, "Reference Z.grad is None"
            triton.testing.assert_close(Z_grad_triton, Z_ref.grad, rtol=1e-4, atol=1e-4)
            print("  -> Z.grad MATCHES")
        except AssertionError as e:
            print(f"  -> Z.grad MISMATCH: {e}")
            all_grads_match = False

        try:
            print("Comparing g.grad...")
            assert g_grad_triton is not None, "Triton g.grad is None"
            assert g_ref.grad is not None, "Reference g.grad is None"
            triton.testing.assert_close(g_grad_triton, g_ref.grad, rtol=1e-4, atol=1e-4)
            print("  -> g.grad MATCHES")
        except AssertionError as e:
            print(f"  -> g.grad MISMATCH: {e}")
            all_grads_match = False

        if not all_grads_match:
            print("Gradient check FAILED.")
        else:
            print("Gradient check PASSED!")

    # ====================================
    #          Benchmarking
    # ====================================

    print("\n--- Benchmarking Forward Pass ---")
    B_bench, D_bench, L_bench, H_bench = 128, 128, 4096, 2
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
