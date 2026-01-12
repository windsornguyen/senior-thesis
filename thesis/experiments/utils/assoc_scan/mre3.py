# -*- Author: Windsor Nguyen -*-
import torch
import triton
import triton.language as tl
from typing import Tuple
from triton.testing import do_bench

# Mapping for Triton data types
DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


# --- Configuration for Triton Autotuning ---
def get_configs():
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


def filter_config(conf, seq_len=None):
    block_size = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if block_size >= 512 and num_warps < 8:
        return False
    if block_size < 128 and num_warps > 4:
        return False
    if seq_len is not None and block_size > seq_len:
        return False
    return True


# --- Softmax Combine Functions ---
@triton.jit
def softmax_combine(
    max_x: float,
    sum_x: float,
    num_x: float,
    z_x: float,
    gate_x: float,
    max_y: float,
    sum_y: float,
    num_y: float,
    z_y: float,
    gate_y: float,
) -> Tuple[float, float, float, float, float]:
    """Combines two states for forward associative scan."""
    max_new = tl.maximum(max_x, max_y)
    exp_x = tl.exp(max_x - max_new)
    exp_y = tl.exp(max_y - max_new)
    sum_new = sum_x * exp_x + sum_y * exp_y
    num_new = num_x * exp_x + num_y * exp_y
    z_new = z_x + z_y
    gate_new = gate_x + gate_y
    return max_new, sum_new, num_new, z_new, gate_new


# --- Analytical Derivative Kernels ---
@triton.jit
def max_indicator(x: float, y: float) -> float:
    """Returns 1.0 if x >= y, else 0.0."""
    return tl.where(x >= y, 1.0, 0.0)


@triton.jit
def compute_exps(max_x: float, max_y: float) -> Tuple[float, float, float]:
    """Computes exponentials and new max for stability."""
    max_new = tl.maximum(max_x, max_y)
    diff_x = max_x - max_new
    diff_y = max_y - max_new
    return tl.exp(diff_x), tl.exp(diff_y), max_new


@triton.jit
def vjp_combine(
    # -------- left tuple (x) --------
    gm_x: float,
    gs_x: float,
    gn_x: float,
    gz_x: float,
    gg_x: float,
    m_x: float,
    s_x: float,
    n_x: float,
    z_x: float,
    g_x: float,
    # -------- right tuple (y) -------
    gm_y: float,
    gs_y: float,
    gn_y: float,
    gz_y: float,
    gg_y: float,
    m_y: float,
    s_y: float,
    n_y: float,
    z_y: float,
    g_y: float,
    H: tl.constexpr,
    H2: tl.constexpr,
    DTYPE: tl.constexpr,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    # --- Debug Print --- #
    # Check what values are actually received for H, H2, DTYPE
    # Note: device_print works inside @triton.jit, prints to console when kernel runs
    # Only print from the first thread/instance to avoid flooding
    if (
        tl.program_id(0) == 0 and tl.program_id(1) == 0 and tl.program_id(2) == 0
    ):  # Adjust based on kernel grid dims if needed
        tl.device_print("vjp_combine received H: ", H)
        tl.device_print("vjp_combine received H2: ", H2)
        # tl.device_print("vjp_combine received DTYPE: ", DTYPE) # Printing dtype might be tricky/verbose

    # Recompute forward combine components for Jacobian calculation
    m_new = tl.maximum(m_x, m_y)
    exp_x, exp_y, _ = compute_exps(m_x, m_y)
    s_new = s_x * exp_x + s_y * exp_y
    n_new = n_x * exp_x + n_y * exp_y
    z_new = z_x + z_y
    g_new = g_x + g_y

    # ---------- Jacobian of (m,s,n) wrt left operand x ----------
    I = max_indicator(m_x, m_y)  # 1 if m_x ≥ m_y else 0
    dm_dmx = I
    ds_dmx = s_x * exp_x * (1.0 - I) - s_y * exp_y * I
    dn_dmx = n_x * exp_x * (1.0 - I) - n_y * exp_y * I
    ds_dsx = exp_x
    dn_dnx = exp_x
    # ---------- Jᵀ · g_y  added to existing g_x ----------
    gm_x += gm_y * dm_dmx + gs_y * ds_dmx + gn_y * dn_dmx
    gs_x += gs_y * ds_dsx
    gn_x += gn_y * dn_dnx
    gz_x += gz_y  # ∂z_new/∂z_x = 1
    gg_x += gg_y  # ∂g_new/∂g_x = 1

    # ---------- return new gradient tuple  +  new forward state ----------
    return (
        gm_x,
        gs_x,
        gn_x,
        gz_x,
        gg_x,  # updated grads for x
        m_new,
        s_new,
        n_new,
        z_new,
        g_new,
    )


# --- Triton Kernel ---
@triton.autotune(
    configs=[conf for conf in get_configs() if filter_config(conf)],
    key=["batch_size", "feature_size_h2", "seq_len"],
)
@triton.jit
def forward_scan_kernel(
    max_ptr,
    sum_ptr,
    num_ptr,
    z_ptr,
    gate_ptr,
    out_max_ptr,
    out_sum_ptr,
    out_num_ptr,
    out_z_ptr,
    out_gate_ptr,
    batch_size: int,
    feature_size_h2: int,
    seq_len: int,
    stride_b: int,
    stride_f_max: int,
    stride_f_sum: int,
    stride_f_num: int,
    stride_f_z: int,
    stride_f_gate: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < seq_len
    offset_l = indices * stride_l

    # Each tensor's offset
    off_max = pid_batch * stride_b + pid_feature * stride_f_max + offset_l
    off_sum = pid_batch * stride_b + pid_feature * stride_f_sum + offset_l
    off_num = pid_batch * stride_b + pid_feature * stride_f_num + offset_l
    off_z = pid_batch * stride_b + pid_feature * stride_f_z + offset_l
    off_gate = pid_batch * stride_b + pid_feature * stride_f_gate + offset_l

    max_val = tl.load(max_ptr + off_max, mask=mask, other=float("-inf"))
    sum_val = tl.load(sum_ptr + off_sum, mask=mask, other=0.0)
    num_val = tl.load(num_ptr + off_num, mask=mask, other=0.0)
    z_val = tl.load(z_ptr + off_z, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + off_gate, mask=mask, other=0.0)

    res_max, res_sum, res_num, res_z, res_gate = tl.associative_scan(
        (max_val, sum_val, num_val, z_val, gate_val), axis=0, combine_fn=softmax_combine
    )

    tl.store(out_max_ptr + off_max, res_max, mask=mask)
    tl.store(out_sum_ptr + off_sum, res_sum, mask=mask)
    tl.store(out_num_ptr + off_num, res_num, mask=mask)
    tl.store(out_z_ptr + off_z, res_z, mask=mask)
    tl.store(out_gate_ptr + off_gate, res_gate, mask=mask)


@triton.autotune(
    configs=[conf for conf in get_configs() if filter_config(conf)],
    key=["batch_size", "feature_size_h2", "seq_len"],
)
@triton.jit
def backward_scan_kernel(
    max_ptr,
    sum_ptr,
    num_ptr,
    z_ptr,
    gate_ptr,
    grad_out_max_ptr,
    grad_out_sum_ptr,
    grad_out_num_ptr,
    grad_out_z_ptr,
    grad_out_gate_ptr,
    grad_in_max_ptr,
    grad_in_sum_ptr,
    grad_in_num_ptr,
    grad_in_z_ptr,
    grad_in_gate_ptr,
    batch_size: int,
    feature_size_h2: int,
    seq_len: tl.constexpr,
    stride_b: int,
    stride_f: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Performs backward associative scan to compute gradients."""
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    base_offset = pid_batch * stride_b + pid_feature * stride_f
    indices = tl.arange(0, seq_len)
    rev_indices = seq_len - 1 - indices
    mask = indices < seq_len
    offset = base_offset + rev_indices * stride_l

    # Load inputs in reverse order
    max_val = tl.load(max_ptr + offset, mask=mask, other=float("-inf"))
    sum_val = tl.load(sum_ptr + offset, mask=mask, other=0.0)
    num_val = tl.load(num_ptr + offset, mask=mask, other=0.0)
    z_val = tl.load(z_ptr + offset, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offset, mask=mask, other=0.0)

    # Load output gradients in reverse order
    grad_max_out = tl.load(grad_out_max_ptr + offset, mask=mask, other=0.0)
    grad_sum_out = tl.load(grad_out_sum_ptr + offset, mask=mask, other=0.0)
    grad_num_out = tl.load(grad_out_num_ptr + offset, mask=mask, other=0.0)
    grad_z_out = tl.load(grad_out_z_ptr + offset, mask=mask, other=0.0)
    grad_gate_out = tl.load(grad_out_gate_ptr + offset, mask=mask, other=0.0)

    # Perform VJP scan
    grad_max, grad_sum, grad_num, grad_z, grad_gate, _, _, _, _, _ = tl.associative_scan(
        (
            grad_max_out,
            grad_sum_out,
            grad_num_out,
            grad_z_out,
            grad_gate_out,
            max_val,
            sum_val,
            num_val,
            z_val,
            gate_val,
        ),
        axis=0,
        combine_fn=vjp_combine,
        reverse=False,
    )

    # Store input gradients
    tl.store(grad_in_max_ptr + offset, grad_max, mask=mask)
    tl.store(grad_in_sum_ptr + offset, grad_sum, mask=mask)
    tl.store(grad_in_num_ptr + offset, grad_num, mask=mask)
    tl.store(grad_in_z_ptr + offset, grad_z, mask=mask)
    tl.store(grad_in_gate_ptr + offset, grad_gate, mask=mask)


# --- Associative Scan Autograd Function ---
class AssociativeScan(torch.autograd.Function):
    @staticmethod
    def forward(m, s, n, Z, g):
        L = m.shape[-1]
        H = n.shape[-1]
        h2 = H * H

        assert all(x.shape[-1] == L for x in [m, s, g])
        assert n.shape[-2] == L and n.shape[-1] == H
        assert Z.shape[-3:] == (L, H, H)

        batch_shape = m.shape[:-1]
        batch_size = int(torch.tensor(batch_shape).prod())

        m_flat = m.reshape(batch_size, L)
        s_flat = s.reshape(batch_size, L)
        g_flat = g.reshape(batch_size, L)
        n_flat = n.reshape(batch_size, L, H)
        Z_flat = Z.reshape(batch_size, L, H, H)

        # Broadcast using stride-0 where possible
        m_exp = m_flat.unsqueeze(1).expand(-1, h2, -1)
        s_exp = s_flat.unsqueeze(1).expand_as(m_exp)
        g_exp = g_flat.unsqueeze(1).expand_as(m_exp)

        n_exp = n_flat.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, H, -1).reshape(batch_size, h2, L)
        Z_exp = Z_flat.permute(0, 2, 3, 1).reshape(batch_size, h2, L)

        ker_tensors = [t.contiguous().cuda() for t in (m_exp, s_exp, n_exp, Z_exp, g_exp)]
        m_ker, s_ker, n_ker, Z_ker, g_ker = ker_tensors

        out_tensors = [torch.empty_like(t) for t in ker_tensors]
        out_m_ker, out_s_ker, out_n_ker, out_Z_ker, out_g_ker = out_tensors

        triton_dtype = DTYPE_MAP.get(m_ker.dtype, tl.float32)

        forward_scan_kernel[(batch_size, h2)](
            m_ker,
            s_ker,
            n_ker,
            Z_ker,
            g_ker,
            out_m_ker,
            out_s_ker,
            out_n_ker,
            out_Z_ker,
            out_g_ker,
            batch_size,
            h2,
            L,
            m_ker.stride(0),
            m_ker.stride(1),
            s_ker.stride(1),
            n_ker.stride(1),
            Z_ker.stride(1),
            g_ker.stride(1),
            m_ker.stride(2),
            DTYPE=triton_dtype,
        )

        out_m, out_s, out_n, out_Z, out_g = [t.transpose(1, 2).reshape(*batch_shape, L, H, H) for t in out_tensors]

        return (out_m[..., 0, 0], out_s[..., 0, 0], out_n[..., :, 0], out_Z, out_g[..., 0, 0])

    @staticmethod
    def setup_context(ctx, inputs, output):
        m, s, n, Z, g = inputs
        ctx.save_for_backward(m, s, n, Z, g)
        ctx.input_shape = m.shape
        try:
            ctx.H = n.shape[-1]
        except IndexError as e:
            raise ValueError(f"IndexError: {e} Tensor n must have shape (..., L, H)") from e
        ctx.triton_dtype = DTYPE_MAP.get(m.dtype, tl.float32)

    @staticmethod
    def backward(ctx, grad_m, grad_s, grad_n, grad_Z, grad_g):
        m, s, n, Z, g = ctx.saved_tensors
        input_shape = ctx.input_shape
        L, H = input_shape[-1], ctx.H
        h2 = H * H
        batch_dims = input_shape[:-1]
        batch_size = torch.prod(torch.tensor(batch_dims)).item()

        # Handle None gradients
        grads = [
            g if g is not None else torch.zeros_like(t)
            for g, t in zip([grad_m, grad_s, grad_n, grad_Z, grad_g], [m, s, n, Z, g])
        ]
        grad_m, grad_s, grad_n, grad_Z, grad_g = [g.contiguous().cuda() for g in grads]

        # Expand output gradients
        grad_m_exp = grad_m.unsqueeze(-1).unsqueeze(-1).expand(*batch_dims, L, H, H)
        grad_s_exp = grad_s.unsqueeze(-1).unsqueeze(-1).expand(*batch_dims, L, H, H)
        grad_n_exp = grad_n.unsqueeze(-1).expand(*batch_dims, L, H, H)
        grad_Z_exp = grad_Z
        grad_g_exp = grad_g.unsqueeze(-1).unsqueeze(-1).expand(*batch_dims, L, H, H)

        # Reshape output gradients
        grad_out_ker = [
            t.reshape(batch_size, L, h2).transpose(1, 2).contiguous()
            for t in [grad_m_exp, grad_s_exp, grad_n_exp, grad_Z_exp, grad_g_exp]
        ]
        gm_out_ker, gs_out_ker, gn_out_ker, gZ_out_ker, gg_out_ker = grad_out_ker

        # Prepare forward inputs
        m_flat = m.reshape(batch_size, L)
        s_flat = s.reshape(batch_size, L)
        n_flat = n.reshape(batch_size, L, H)
        Z_flat = Z.reshape(batch_size, L, H, H)
        g_flat = g.reshape(batch_size, L)

        m_exp = m_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        s_exp = s_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)
        n_exp = n_flat.unsqueeze(-1).expand(-1, -1, -1, H)
        Z_exp = Z_flat
        g_exp = g_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, H)

        # Reshape forward inputs
        ker_tensors = [
            t.reshape(batch_size, L, h2).transpose(1, 2).contiguous().cuda()
            for t in [m_exp, s_exp, n_exp, Z_exp, g_exp]
        ]
        m_ker, s_ker, n_ker, Z_ker, g_ker = ker_tensors

        # Allocate input gradients
        grad_in_ker = [torch.empty_like(t) for t in ker_tensors]
        grad_in_m_ker, grad_in_s_ker, grad_in_n_ker, grad_in_Z_ker, grad_in_g_ker = grad_in_ker

        # Launch backward kernel
        grid = (batch_size, h2)
        backward_scan_kernel[grid](
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
            m_ker.stride(2),
            DTYPE=ctx.triton_dtype,
        )

        # Post-process gradients
        grad_in_flat = [t.transpose(1, 2).reshape(*batch_dims, L, H, H) for t in grad_in_ker]
        grad_in_m, grad_in_s, grad_in_n, grad_in_Z, grad_in_g = grad_in_flat

        # Extract final gradients
        grad_m_final = grad_in_m[..., 0, 0]
        grad_s_final = grad_in_s[..., 0, 0]
        grad_n_final = grad_in_n[..., :, 0]
        grad_Z_final = grad_in_Z
        grad_g_final = grad_in_g[..., 0, 0]

        # Validate shapes
        assert grad_m_final.shape == m.shape
        assert grad_s_final.shape == s.shape
        assert grad_n_final.shape == n.shape
        assert grad_Z_final.shape == Z.shape
        assert grad_g_final.shape == g.shape

        return grad_m_final, grad_s_final, grad_n_final, grad_Z_final, grad_g_final


def associative_scan(m, s, n, Z, g):
    """Performs associative scan with preprocessing."""
    return AssociativeScan.apply(m, s, n, Z, g)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    leaves = (qk_slice, torch.ones_like(qk_slice), v_slice, Z_slice, g_slice)
    return assoc_scan_ref(combine_fn=combine_fn_ref, xs=leaves, dim=0, combine_mode="generic")


def batched_scan_fn(
    sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, L, h = v.shape
    sim_bh = sim.reshape(-1, L)
    v_bh = v.reshape(-1, L, h)
    gated_Z_bh = gated_Z.reshape(-1, L, h, h)
    gates_z_bh = gates_z.reshape(-1, L)
    scan_vmapped = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    m_res, s_res, n_res, Z_res, g_res = scan_vmapped(sim_bh, v_bh, gated_Z_bh, gates_z_bh)
    return (
        m_res.reshape(B, H, L),
        s_res.reshape(B, H, L),
        n_res.reshape(B, H, L, h),
        Z_res.reshape(B, H, L, h, h),
        g_res.reshape(B, H, L),
    )


# --- Main Function (Unchanged) ---
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = "cuda"

    # Correctness Check
    print("\n--- Correctness Check ---")
    B_test, D_test, L_test, H_test = 1, 2, 4, 2

    # Create non-trivial inputs with carefully selected values
    # m: logits with clear max values
    m = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],  # clear max at end
                [4.0, 3.0, 2.0, 1.0],
            ]  # clear max at start
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    # s: ones (as required)
    s = torch.ones_like(m, requires_grad=True)

    # n: vectors with non-trivial values
    n = torch.tensor(
        [
            [
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],  # increasing
                [[5.0, 4.0], [4.0, 3.0], [3.0, 2.0], [2.0, 1.0]],
            ]  # decreasing
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    # Z: matrices with non-trivial values
    Z = torch.tensor(
        [
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],  # increasing
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[3.0, 4.0], [5.0, 6.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ],
                [
                    [[7.0, 6.0], [5.0, 4.0]],  # decreasing
                    [[6.0, 5.0], [4.0, 3.0]],
                    [[5.0, 4.0], [3.0, 2.0]],
                    [[4.0, 3.0], [2.0, 1.0]],
                ],
            ]
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    # g: gates with alternating values
    g = torch.tensor(
        [
            [
                [1.0, -1.0, 1.0, -1.0],  # alternating
                [-1.0, 1.0, -1.0, 1.0],
            ]  # opposite alternating
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    # Clone inputs for reference implementation
    m_ref_input = m.clone().detach().requires_grad_(True)
    n_ref_input = n.clone().detach().requires_grad_(True)
    Z_ref_input = Z.clone().detach().requires_grad_(True)
    g_ref_input = g.clone().detach().requires_grad_(True)

    print("\nInput shapes:")
    print(f"m: {m.shape}")  # (1, 2, 4)
    print(f"s: {s.shape}")  # (1, 2, 4)
    print(f"n: {n.shape}")  # (1, 2, 4, 2)
    print(f"Z: {Z.shape}")  # (1, 2, 4, 2, 2)
    print(f"g: {g.shape}")  # (1, 2, 4)

    print("\nInput values:")
    print("m (logits):")
    print(m.squeeze().tolist())
    print("\nn (vectors):")
    print(n.squeeze().tolist())
    print("\nZ (matrices):")
    print(Z.squeeze().tolist())
    print("\ng (gates):")
    print(g.squeeze().tolist())

    print("\nRunning Triton forward pass (test)...")
    m_out, s_out, n_out, Z_out, g_out = associative_scan(m, s, n, Z, g)
    print("Triton forward pass completed.")

    print("\nRunning reference forward pass...")
    ref_m_out, ref_s_out, ref_n_out, ref_Z_out, ref_g_out = batched_scan_fn(
        m_ref_input, n_ref_input, Z_ref_input, g_ref_input
    )
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
        print("Triton Outputs:")
        print(f"  m: {m_out.squeeze().tolist()}")
        print(f"  s: {s_out.squeeze().tolist()}")
        print(f"  n: {n_out.squeeze().tolist()}")
        print(f"  Z: {Z_out.squeeze().tolist()}")
        print(f"  g: {g_out.squeeze().tolist()}")
        print("Reference Outputs:")
        print(f"  m: {ref_m_out.squeeze().tolist()}")
        print(f"  s: {ref_s_out.squeeze().tolist()}")
        print(f"  n: {ref_n_out.squeeze().tolist()}")
        print(f"  Z: {ref_Z_out.squeeze().tolist()}")
        print(f"  g: {ref_g_out.squeeze().tolist()}")

    # Gradient Check
    print("\n--- Gradient Check ---")
    if m.dtype != torch.float64:
        print("Skipping gradient check, requires float64 inputs.")
    else:
        all_grads_match = True
        components = [
            ("m", lambda x: x.sum()),
            ("s", lambda x: x.sum()),
            ("n", lambda x: x.sum()),
            ("Z", lambda x: x.sum()),
            ("g", lambda x: x.sum()),
        ]

        for comp_name, loss_fn in components:
            print(f"\nTesting {comp_name} gradients...")

            # Triton backward
            print("Running Triton backward...")
            m_triton = m.clone().detach().requires_grad_(True)
            s_triton = s.clone().detach().requires_grad_(True)
            n_triton = n.clone().detach().requires_grad_(True)
            Z_triton = Z.clone().detach().requires_grad_(True)
            g_triton = g.clone().detach().requires_grad_(True)

            m_out, s_out, n_out, Z_out, g_out = associative_scan(m_triton, s_triton, n_triton, Z_triton, g_triton)

if m.dtype != torch.float64:
    print("Skipping gradient check, requires float64 inputs.")
else:
    all_grads_match = True
    components = [
        ("m", lambda x: x.sum()),
        ("s", lambda x: x.sum()),
        ("n", lambda x: x.sum()),
        ("Z", lambda x: x.sum()),
        ("g", lambda x: x.sum()),
    ]

    for comp_name, loss_fn in components:
        print(f"\nTesting {comp_name} gradients...")

        # Triton backward
        print("Running Triton backward...")
        m_triton = m.clone().detach().requires_grad_(True)
        s_triton = s.clone().detach().requires_grad_(True)
        n_triton = n.clone().detach().requires_grad_(True)
        Z_triton = Z.clone().detach().requires_grad_(True)
        g_triton = g.clone().detach().requires_grad_(True)

        m_out, s_out, n_out, Z_out, g_out = associative_scan(m_triton, s_triton, n_triton, Z_triton, g_triton)

        output_map = {"m": m_out, "s": s_out, "n": n_out, "Z": Z_out, "g": g_out}
        loss_triton = loss_fn(output_map[comp_name])
        loss_triton.backward()
        print("Triton backward completed.")

        grad_map = {
            "m": m_triton.grad,
            "s": s_triton.grad,
            "n": n_triton.grad,
            "Z": Z_triton.grad,
            "g": g_triton.grad,
        }

        # Reference backward
        print("Running Reference backward...")
        m_ref = m.clone().detach().requires_grad_(True)
        s_ref = s.clone().detach().requires_grad_(True)
        n_ref = n.clone().detach().requires_grad_(True)
        Z_ref = Z.clone().detach().requires_grad_(True)
        g_ref = g.clone().detach().requires_grad_(True)

        ref_m_out, ref_s_out, ref_n_out, ref_Z_out, ref_g_out = batched_scan_fn(m_ref, n_ref, Z_ref, g_ref)

        ref_output_map = {
            "m": ref_m_out,
            "s": ref_s_out,
            "n": ref_n_out,
            "Z": ref_Z_out,
            "g": ref_g_out,
        }

        loss_ref = loss_fn(ref_output_map[comp_name])
        loss_ref.backward()
        print("Reference backward completed.")

        # Compare gradients
        print(f"Comparing gradients for {comp_name}...")
        try:
            triton_grad = grad_map[comp_name]
            ref_grad = {
                "m": m_ref.grad,
                "s": s_ref.grad,
                "n": n_ref.grad,
                "Z": Z_ref.grad,
                "g": g_ref.grad,
            }[comp_name]

            assert triton_grad is not None, f"Triton {comp_name}.grad is None"
            assert ref_grad is not None, f"Reference {comp_name}.grad is None"

            triton.testing.assert_close(triton_grad, ref_grad, rtol=1e-4, atol=1e-4)
            print(f"  -> {comp_name}.grad MATCHES")
        except AssertionError as e:
            print(f"  -> {comp_name}.grad MISMATCH: {e}")
            print(f"     Triton: {triton_grad.squeeze().tolist() if triton_grad is not None else None}")
            print(f"     Ref   : {ref_grad.squeeze().tolist() if ref_grad is not None else None}")
            all_grads_match = False

    if all_grads_match:
        print("\nGradient check PASSED (Individual Components)!")
    else:
        print("\nGradient check FAILED (Individual Components).")

    # Benchmarking
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
    _ = bench_fn_torch()
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
