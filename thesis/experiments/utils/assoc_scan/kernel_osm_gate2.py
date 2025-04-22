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
def _flat_broadcast(x, repeat: tl.constexpr, B_const: tl.constexpr):
    """
    Repeat the last dimension `repeat` times and flatten.
       [B]           ->  [B*repeat]
       [B,repeat]    ->  [B*repeat]  (noop if already there)
    """
    if repeat == 1:
        return x
    x_rep = tl.broadcast_to(x[:, None], (B_const, repeat))
    return tl.reshape(x_rep, (B_const * repeat,))


@triton.jit
def combine_softmax(m_x, s_x, n_x, m_y, s_y, n_y):
    m = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m)
    exp_y = tl.exp(m_y - m)
    s = s_x * exp_x + s_y * exp_y
    n = n_x * exp_x + n_y * exp_y  # shapes already match (1‑D flat)
    return m, s, n


@triton.jit
def combine_Zg(Z_x, g_x, Z_y, g_y):
    return Z_x + Z_y, g_x + g_y


@triton.jit
def combine_fn(x, y):
    # Numerically stable combine function for associative scan
    # Inputs: x, y are tuples (m_x, s_x, n_x, Z_x, g_x), (m_y, s_y, n_y, Z_y, g_y)
    # m: log-max accumulator [BLOCK_SIZE]
    # s: normalization denominator [BLOCK_SIZE]
    # n: first-order numerator [BLOCK_SIZE, h]
    # Z: second-order accumulator [BLOCK_SIZE, h, h]
    # g: gate accumulator [BLOCK_SIZE]
    m_x, s_x, n_x, Z_x, g_x = x
    m_y, s_y, n_y, Z_y, g_y = y
    m = tl.maximum(m_x, m_y)
    exp_x = tl.exp(m_x - m)
    exp_y = tl.exp(m_y - m)
    s = s_x * exp_x + s_y * exp_y
    n = n_x * exp_x[:, None] + n_y * exp_y[:, None]
    Z = Z_x + Z_y
    g = g_x + g_y
    return m, s, n, Z, g


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
def fused_scan_kernel(
    # ──── inputs ────────────────────────────────────────────────────────────
    sim_ptr,
    v_ptr,
    gated_Z_ptr,
    gates_ptr,
    # ──── outputs ───────────────────────────────────────────────────────────
    max_cumul_ptr,
    norm_cumul_ptr,
    v_cumul_ptr,
    Z_cumul_ptr,
    gate_cumul_ptr,
    # ──── global tensor metadata (shapes & strides) ────────────────────────
    sim_shape_0,
    sim_shape_1,
    sim_strides_0,
    sim_strides_1,
    v_shape_0,
    v_shape_1,
    v_shape_2,
    v_strides_0,
    v_strides_1,
    v_strides_2,
    gated_Z_shape_0,
    gated_Z_shape_1,
    gated_Z_shape_2,
    gated_Z_shape_3,
    gated_Z_strides_0,
    gated_Z_strides_1,
    gated_Z_strides_2,
    gated_Z_strides_3,
    gates_shape_0,
    gates_shape_1,
    gates_strides_0,
    gates_strides_1,
    max_cumul_shape_0,
    max_cumul_shape_1,
    max_cumul_strides_0,
    max_cumul_strides_1,
    norm_cumul_shape_0,
    norm_cumul_shape_1,
    norm_cumul_strides_0,
    norm_cumul_strides_1,
    v_cumul_shape_0,
    v_cumul_shape_1,
    v_cumul_shape_2,
    v_cumul_strides_0,
    v_cumul_strides_1,
    v_cumul_strides_2,
    Z_cumul_shape_0,
    Z_cumul_shape_1,
    Z_cumul_shape_2,
    Z_cumul_shape_3,
    Z_cumul_strides_0,
    Z_cumul_strides_1,
    Z_cumul_strides_2,
    Z_cumul_strides_3,
    gate_cumul_shape_0,
    gate_cumul_shape_1,
    gate_cumul_strides_0,
    gate_cumul_strides_1,
    # ──── problem sizes / compile‑time constants ───────────────────────────
    feature_size: int,  # = B · H  (one program per feature)
    seq_len: int,
    H_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Map kernel instance to one (batch, head) feature
    pid_feature = tl.program_id(0)

    # Pre-advance base pointers (unchanged)
    sim_base = sim_ptr + pid_feature * sim_strides_0
    v_base = v_ptr + pid_feature * v_strides_0
    gated_Z_base = gated_Z_ptr + pid_feature * gated_Z_strides_0
    gates_base = gates_ptr + pid_feature * gates_strides_0
    max_base = max_cumul_ptr + pid_feature * max_cumul_strides_0
    norm_base = norm_cumul_ptr + pid_feature * norm_cumul_strides_0
    v_out_base = v_cumul_ptr + pid_feature * v_cumul_strides_0
    Z_out_base = Z_cumul_ptr + pid_feature * Z_cumul_strides_0
    gate_out_base = gate_cumul_ptr + pid_feature * gate_cumul_strides_0

    # Scan over sequence in BLOCK_SIZE tiles
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        start_col = k * BLOCK_SIZE

        # Build block pointers (unchanged)
        sim_block_ptr = tl.make_block_ptr(
            base=sim_base, shape=(sim_shape_1,), strides=(sim_strides_1,),
            offsets=(start_col,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_base, shape=(v_shape_1, v_shape_2), strides=(v_strides_1, v_strides_2),
            offsets=(start_col, 0), block_shape=(BLOCK_SIZE, H_CONST), order=(1, 0),
        )
        gated_Z_block_ptr = tl.make_block_ptr(
            base=gated_Z_base, shape=(gated_Z_shape_1, gated_Z_shape_2, gated_Z_shape_3),
            strides=(gated_Z_strides_1, gated_Z_strides_2, gated_Z_strides_3),
            offsets=(start_col, 0, 0), block_shape=(BLOCK_SIZE, H_CONST, H_CONST), order=(2, 1, 0),
        )
        gates_block_ptr = tl.make_block_ptr(
            base=gates_base, shape=(gates_shape_1,), strides=(gates_strides_1,),
            offsets=(start_col,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        max_block_ptr = tl.make_block_ptr(
            base=max_base, shape=(max_cumul_shape_1,), strides=(max_cumul_strides_1,),
            offsets=(start_col,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        norm_block_ptr = tl.make_block_ptr(
            base=norm_base, shape=(norm_cumul_shape_1,), strides=(norm_cumul_strides_1,),
            offsets=(start_col,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        v_out_block_ptr = tl.make_block_ptr(
            base=v_out_base, shape=(v_cumul_shape_1, v_cumul_shape_2), strides=(v_cumul_strides_1, v_cumul_strides_2),
            offsets=(start_col, 0), block_shape=(BLOCK_SIZE, H_CONST), order=(1, 0),
        )
        Z_out_block_ptr = tl.make_block_ptr(
            base=Z_out_base, shape=(Z_cumul_shape_1, Z_cumul_shape_2, Z_cumul_shape_3),
            strides=(Z_cumul_strides_1, Z_cumul_strides_2, Z_cumul_strides_3),
            offsets=(start_col, 0, 0), block_shape=(BLOCK_SIZE, H_CONST, H_CONST), order=(2, 1, 0),
        )
        gate_out_block_ptr = tl.make_block_ptr(
            base=gate_out_base, shape=(gate_cumul_shape_1,), strides=(gate_cumul_strides_1,),
            offsets=(start_col,), block_shape=(BLOCK_SIZE,), order=(0,),
        )

        # Load tiles
        sim = tl.load(sim_block_ptr, boundary_check=(0,), padding_option="zero")  # [BLOCK_SIZE]
        v = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")  # [BLOCK_SIZE, H_CONST]
        gated_Z = tl.load(gated_Z_block_ptr, boundary_check=(0,), padding_option="zero")  # [BLOCK_SIZE, H_CONST, H_CONST]
        gates = tl.load(gates_block_ptr, boundary_check=(0,), padding_option="zero")  # [BLOCK_SIZE]

        # Initialize accumulators
        m_acc = tl.zeros([1], dtype=DTYPE) - float("inf")  # Scalar log-max
        s_acc = tl.zeros([1], dtype=DTYPE)  # Scalar normalization denominator
        n_acc = tl.zeros([H_CONST], dtype=DTYPE)  # Vector numerator [h]
        Z_acc = tl.zeros([H_CONST, H_CONST], dtype=DTYPE)  # Matrix accumulator [h, h]
        g_acc = tl.zeros([1], dtype=DTYPE)  # Scalar gate accumulator

        # Allocate output buffers
        max_out = tl.zeros([BLOCK_SIZE], dtype=DTYPE)
        norm_out = tl.zeros([BLOCK_SIZE], dtype=DTYPE)
        v_out = tl.zeros([BLOCK_SIZE, H_CONST], dtype=DTYPE)
        Z_out = tl.zeros([BLOCK_SIZE, H_CONST, H_CONST], dtype=DTYPE)
        gate_out = tl.zeros([BLOCK_SIZE], dtype=DTYPE)

        # Sequential scan within block
        for i in tl.static_range(BLOCK_SIZE):
            # Current position inputs
            sim_i = sim[i]  # Scalar
            v_i = v[i, :]  # [h]
            Z_i = gated_Z[i, :, :]  # [h, h]
            g_i = gates[i]  # Scalar

            # Update max accumulator
            m_new = tl.maximum(m_acc, sim_i)
            
            # Compute scaling factors
            exp_old = tl.exp(m_acc - m_new)
            exp_new = tl.exp(sim_i - m_new)
            
            # Update accumulators
            s_acc = s_acc * exp_old + exp_new
            n_acc = n_acc * exp_old + v_i * exp_new
            Z_acc = Z_acc + Z_i
            g_acc = g_acc + g_i
            m_acc = m_new

            # Store per-position results
            max_out[i] = m_acc
            norm_out[i] = s_acc
            v_out[i, :] = n_acc
            Z_out[i, :, :] = Z_acc
            gate_out[i] = g_acc

        # Store results back to global memory
        tl.store(max_block_ptr, max_out, boundary_check=(0,))
        tl.store(norm_block_ptr, norm_out, boundary_check=(0,))
        tl.store(v_out_block_ptr, v_out, boundary_check=(0,))
        tl.store(Z_out_block_ptr, Z_out, boundary_check=(0,))
        tl.store(gate_out_block_ptr, gate_out, boundary_check=(0,))

class FusedScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
        feature_size, seq_len = sim.shape
        head_dim = v.shape[-1]
        sim = sim.contiguous().cuda()
        v = v.contiguous().cuda()
        gated_Z = gated_Z.contiguous().cuda()
        gates_z = gates_z.contiguous().cuda()
        max_cumul = torch.empty_like(sim)
        norm_cumul = torch.empty_like(sim)
        v_cumul = torch.empty_like(v)
        Z_cumul = torch.empty_like(gated_Z)
        gate_cumul = torch.empty_like(gates_z)

        grid = (feature_size,)
        triton_dtype = dtype_map.get(sim.dtype, tl.float32)

        fused_scan_kernel[grid](
            sim,
            v,
            gated_Z,
            gates_z,
            max_cumul,
            norm_cumul,
            v_cumul,
            Z_cumul,
            gate_cumul,
            feature_size,
            seq_len,
            sim.shape[0],
            sim.shape[1],
            sim.stride(0),
            sim.stride(1),
            v.shape[0],
            v.shape[1],
            v.shape[2],
            v.stride(0),
            v.stride(1),
            v.stride(2),
            gated_Z.shape[0],
            gated_Z.shape[1],
            gated_Z.shape[2],
            gated_Z.shape[3],
            gated_Z.stride(0),
            gated_Z.stride(1),
            gated_Z.stride(2),
            gated_Z.stride(3),
            gates_z.shape[0],
            gates_z.shape[1],
            gates_z.stride(0),
            gates_z.stride(1),
            max_cumul.shape[0],
            max_cumul.shape[1],
            max_cumul.stride(0),
            max_cumul.stride(1),
            norm_cumul.shape[0],
            norm_cumul.shape[1],
            norm_cumul.stride(0),
            norm_cumul.stride(1),
            v_cumul.shape[0],
            v_cumul.shape[1],
            v_cumul.shape[2],
            v_cumul.stride(0),
            v_cumul.stride(1),
            v_cumul.stride(2),
            Z_cumul.shape[0],
            Z_cumul.shape[1],
            Z_cumul.shape[2],
            Z_cumul.shape[3],
            Z_cumul.stride(0),
            Z_cumul.stride(1),
            Z_cumul.stride(2),
            Z_cumul.stride(3),
            gate_cumul.shape[0],
            gate_cumul.shape[1],
            gate_cumul.stride(0),
            gate_cumul.stride(1),
            H_CONST=head_dim,
            DTYPE=triton_dtype,
        )

        ctx.save_for_backward(sim, v, gated_Z, gates_z, max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul)
        ctx.triton_dtype = triton_dtype
        ctx.shape = sim.shape
        ctx.head_dim = head_dim
        return max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul


def fused_scan(sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
    return FusedScan.apply(sim, v, gated_Z, gates_z)


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


def scan_fn(qk_slice: torch.Tensor, v_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor):
    leaves = (
        qk_slice,
        torch.ones_like(qk_slice),
        v_slice,
        Z_slice,
        g_slice,
    )
    m = torch.cummax(leaves[0], dim=0)[0]
    s = torch.cumsum(torch.exp(leaves[0] - m), dim=0)
    n = torch.cumsum(leaves[2] * torch.exp(leaves[0] - m).unsqueeze(-1), dim=0)
    Z = torch.cumsum(leaves[3], dim=0)
    g = torch.cumsum(leaves[4], dim=0)
    return m, s, n, Z, g


def batched_scan_fn(sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor):
    B, H = sim.shape[0], sim.shape[1]
    sim_flat = sim.flatten(0, 1)
    v_flat = v.flatten(0, 1)
    gated_Z_flat = gated_Z.flatten(0, 1)
    gates_z_flat = gates_z.flatten(0, 1)
    scan_all = torch.vmap(scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
    result = scan_all(sim_flat, v_flat, gated_Z_flat, gates_z_flat)
    return tuple(t.reshape(B, H, *t.shape[1:]) for t in result)


if __name__ == "__main__":
    B, H, L, h = 4, 2, 128, 16
    torch.manual_seed(1746)

    # Test data
    print("Generating data...")
    sim = torch.randn(B, H, L, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, L, h, dtype=torch.float32, device="cuda")
    gated_Z = torch.randn(B, H, L, h, h, dtype=torch.float32, device="cuda")
    gates_z = torch.randn(B, H, L, dtype=torch.float32, device="cuda")
    print("Data generated.")

    # Run fused scan
    sim_flat = sim.flatten(0, 1)
    v_flat = v.flatten(0, 1)
    gated_Z_flat = gated_Z.flatten(0, 1)
    gates_z_flat = gates_z.flatten(0, 1)
    print("Running fused_scan...")
    max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = fused_scan(sim_flat, v_flat, gated_Z_flat, gates_z_flat)
    print("Fused scan finished.")

    # Run reference scan
    print("Running reference scan...")
    ref_m, ref_s, ref_n, ref_Z, ref_g = batched_scan_fn(sim, v, gated_Z, gates_z)
    print("Reference scan finished.")

    # Check shapes
    print("Checking output shapes:")
    print(f"  max_cumul: {max_cumul.shape}")
    print(f"  norm_cumul: {norm_cumul.shape}")
    print(f"  v_cumul: {v_cumul.shape}")
    print(f"  Z_cumul: {Z_cumul.shape}")
    print(f"  gate_cumul: {gate_cumul.shape}")

    # Compare with reference
    print("Comparing Triton results with reference scan...")
    assert torch.allclose(max_cumul, ref_m, atol=1e-5), "max_cumul mismatch with reference"
    print("max_cumul check passed.")
    assert torch.allclose(norm_cumul, ref_s, atol=1e-5), "norm_cumul mismatch with reference"
    print("norm_cumul check passed.")
    assert torch.allclose(v_cumul, ref_n, atol=1e-5), "v_cumul mismatch with reference"
    print("v_cumul check passed.")
    assert torch.allclose(Z_cumul, ref_Z, atol=1e-5), "Z_cumul mismatch with reference"
    print("Z_cumul check passed.")
    assert torch.allclose(gate_cumul, ref_g, atol=1e-5), "gate_cumul mismatch with reference"
    print("gate_cumul check passed.")
    print("All checks passed.")

    # Performance
    t_fused = do_bench(lambda: fused_scan(sim_flat, v_flat, gated_Z_flat, gates_z_flat))
    print(f"FusedScan Time: {t_fused:.3f} ms")
