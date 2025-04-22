import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
from typing import Tuple
from torchaudio.functional import convolve

dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def combine_fn(carry_gated_Z, carry_gates, next_gated_Z, next_gates):
    """Combine function for the associative scan: elementwise addition."""
    return (carry_gated_Z + next_gated_Z, carry_gates + next_gates)


# Define autotune configurations, inspired by 05-layer-norm.py and 06-fused-attention.py
def get_scan_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024]
        for s in [2, 3, 4]  # Vary pipeline stages for latency hiding
        for w in [4, 8]  # Common warp counts for CUDA
    ]
    # Add HIP-specific configs for AMD GPUs, as in 06-fused-attention.py
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
        )
    return configs


# Filter configs to avoid bad combinations, as in 06-fused-attention.py
def keep_config(conf, seq_len=None):
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    # Ensure larger block sizes use sufficient warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    # Filter out configs with too small block sizes for large warps
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    # Ensure BLOCK_SIZE <= seq_len
    if seq_len is not None and BLOCK_SIZE > seq_len:
        return False
    return True


@triton.autotune(
    configs=list(filter(lambda conf: keep_config(conf), get_scan_configs())),
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def fwd_scan_kernel(
    gated_Z_ptr,
    gates_ptr,
    out_gated_Z_ptr,
    out_gates_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_d: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Forward kernel for cumulative associative scan with tiling."""

    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d

    # Initialize cumulative state for the scan
    cumulative_gated_Z = tl.zeros([1], dtype=DTYPE)
    cumulative_gates = tl.zeros([1], dtype=DTYPE)

    # Process sequence in chunks
    for start_idx in tl.range(0, seq_len, BLOCK_SIZE):
        indices = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = indices < seq_len
        safe_indices = tl.where(mask, indices, seq_len - 1)
        offsets = start_offset + safe_indices * stride_l

        # Load input chunk
        gated_Z = tl.load(gated_Z_ptr + offsets, mask=mask, other=0.0)
        gates = tl.load(gates_ptr + offsets, mask=mask, other=0.0)

        # Perform associative scan within the chunk
        result_gated_Z, result_gates = tl.associative_scan(
            (gated_Z, gates), axis=0, combine_fn=combine_fn, reverse=False
        )

        # Add cumulative state from previous chunks
        result_gated_Z += cumulative_gated_Z
        result_gates += cumulative_gates

        # Store results
        tl.store(out_gated_Z_ptr + offsets, result_gated_Z, mask=mask)
        tl.store(out_gates_ptr + offsets, result_gates, mask=mask)

        # Update cumulative state with the last valid element
        last_valid_idx = tl.minimum(BLOCK_SIZE - 1, seq_len - start_idx - 1)
        if last_valid_idx >= 0:
            last_valid_mask = tl.arange(0, BLOCK_SIZE) == last_valid_idx
            cumulative_gated_Z = tl.sum(tl.where(last_valid_mask, result_gated_Z, 0.0), axis=0, keep_dims=True)
            cumulative_gates = tl.sum(tl.where(last_valid_mask, result_gates, 0.0), axis=0, keep_dims=True)


@triton.autotune(
    configs=list(filter(keep_config, get_scan_configs())),
    key=["batch_size", "feature_size", "seq_len"],
)
@triton.jit
def bwd_scan_kernel(
    grad_cumul_Z_ptr,
    grad_cumul_g_ptr,
    grad_Z_ptr,
    grad_g_ptr,
    batch_size: int,
    feature_size: int,
    seq_len: int,
    stride_b: int,
    stride_d: int,
    stride_l: int,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Right‑to‑left prefix‑sum of gradients (reverse scan)."""

    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    base = pid_batch * stride_b + pid_feature * stride_d

    # running carry (Z, g) coming from the *right* of the current block
    carry_Z = tl.zeros([1], dtype=DTYPE)
    carry_g = tl.zeros([1], dtype=DTYPE)

    # iterate over chunks from *left* to *right*, but address memory reversed
    for start in tl.range(0, seq_len, BLOCK_SIZE):
        # logical (left‑to‑right) indices of this chunk
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < seq_len
        # map to physical indices counted from the end
        rev_idx = seq_len - 1 - idx
        safe_idx = tl.where(mask, rev_idx, 0)  # clamp for OOB safety
        offs = base + safe_idx * stride_l

        # --- load reversed gradients -----------------------------------
        gZ = tl.load(grad_cumul_Z_ptr + offs, mask=mask, other=0.0)
        gg = tl.load(grad_cumul_g_ptr + offs, mask=mask, other=0.0)

        # --- intra‑block forward scan (because we already reversed order)
        res_Z, res_g = tl.associative_scan((gZ, gg), axis=0, combine_fn=combine_fn, reverse=False)

        # --- add carry from the right ----------------------------------
        res_Z += carry_Z
        res_g += carry_g

        # --- store back to the same reversed locations -----------------
        tl.store(grad_Z_ptr + offs, res_Z, mask=mask)
        tl.store(grad_g_ptr + offs, res_g, mask=mask)

        # --- update carry with the *last* valid element in this chunk --
        last_valid = tl.minimum(BLOCK_SIZE - 1, seq_len - start - 1)
        if last_valid >= 0:
            lv_mask = tl.arange(0, BLOCK_SIZE) == last_valid
            carry_Z = tl.sum(tl.where(lv_mask, res_Z, 0), axis=0, keep_dims=True)
            carry_g = tl.sum(tl.where(lv_mask, res_g, 0), axis=0, keep_dims=True)


class AssociativeScan(torch.autograd.Function):
    """PyTorch autograd wrapper for an associative scan using Triton."""

    @staticmethod
    def forward(gated_Z: torch.Tensor, gates: torch.Tensor):
        batch_size, feature_size, seq_len = gated_Z.shape
        gated_Z = gated_Z.contiguous().cuda()
        gates = gates.contiguous().cuda()
        cumulative_gated_Z = torch.empty_like(gated_Z)
        cumulative_gates = torch.empty_like(gates)

        grid = (batch_size, feature_size, 1)
        triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)

        fwd_scan_kernel[grid](
            gated_Z,
            gates,
            cumulative_gated_Z,
            cumulative_gates,
            batch_size,
            feature_size,
            seq_len,
            gated_Z.stride(0),
            gated_Z.stride(1),
            gated_Z.stride(2),
            DTYPE=triton_dtype,
            enable_fp_fusion=False,  # Avoid fusion issues
        )

        return cumulative_gated_Z, cumulative_gates

    @staticmethod
    def setup_context(ctx, inputs, output):
        gated_Z, gates = inputs
        cumulative_gated_Z, cumulative_gates = output
        ctx.save_for_backward(gated_Z, gates)
        ctx.triton_dtype = dtype_map.get(gated_Z.dtype, tl.float32)

    @staticmethod
    def backward(ctx, grad_cumulative_gated_Z: torch.Tensor, grad_cumulative_gates: torch.Tensor):
        gated_Z, gates = ctx.saved_tensors
        batch_size, feature_size, seq_len = gated_Z.shape
        grad_cumulative_gated_Z = grad_cumulative_gated_Z.contiguous().cuda()
        grad_cumulative_gates = grad_cumulative_gates.contiguous().cuda()
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates = torch.empty_like(gates)

        grid = (batch_size, feature_size, 1)

        bwd_scan_kernel[grid](
            grad_cumulative_gated_Z,
            grad_cumulative_gates,
            grad_gated_Z,
            grad_gates,
            batch_size,
            feature_size,
            seq_len,
            grad_cumulative_gated_Z.stride(0),
            grad_cumulative_gated_Z.stride(1),
            grad_cumulative_gated_Z.stride(2),
            DTYPE=ctx.triton_dtype,
            enable_fp_fusion=False,
        )

        return grad_gated_Z, grad_gates

    @staticmethod
    def _move_bdim_to_front(x, bdim):
        """Move the vmap dim to dim‑0; return (tensor, new_bdim)."""
        if bdim is None:
            return x, None
        return x.movedim(bdim, 0), 0

    @staticmethod
    def _flatten_leading_dims(x, keep=2):
        """
        Collapse all dims *except* the last `keep` dims into one.
        Returns (flat, original_batch_shape).
        """
        batch_shape = x.shape[:-keep]
        if batch_shape:  # non‑empty ⇒ need flatten
            flat = x.reshape(-1, *x.shape[-keep:])
        else:  # already rank‑`keep`
            flat = x
        return flat, batch_shape

    @staticmethod
    def vmap(info, in_dims, gated_Z, gates):
        """Batched rule: works for torch.vmap / torch.func.vmap."""
        bdim = in_dims[0]
        assert bdim == in_dims[1], "`gated_Z` and `gates` must share the same vmap dim"

        # 1. put the vmap dim in front (if any)
        gated_Z, _ = AssociativeScan._move_bdim_to_front(gated_Z, bdim)
        gates, _ = AssociativeScan._move_bdim_to_front(gates, bdim)

        # 2. collapse *all* leading batch axes into one
        gated_Z, batch_shape = AssociativeScan._flatten_leading_dims(gated_Z, keep=2)
        gates, _ = AssociativeScan._flatten_leading_dims(gates, keep=2)

        # 3. run the (non‑batched) kernel
        Z_cumul, g_cumul = AssociativeScan.apply(gated_Z, gates)

        # 4. un‑flatten to original batch shape and restore dim order
        if batch_shape:
            Z_cumul = Z_cumul.view(*batch_shape, *Z_cumul.shape[-2:])
            g_cumul = g_cumul.view(*batch_shape, *g_cumul.shape[-2:])

        if bdim is not None and bdim != 0:
            Z_cumul = Z_cumul.movedim(0, bdim)
            g_cumul = g_cumul.movedim(0, bdim)

        # out_dims must mirror the positions we put the batch dim back to
        return (Z_cumul, g_cumul), (bdim, bdim)


def associative_scan(gated_Z: torch.Tensor, gates: torch.Tensor):
    """Executes an associative scan on the provided tensors."""
    return AssociativeScan.apply(gated_Z, gates)


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return torch.from_numpy(Z)


def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False, device=None, dtype=torch.float32
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).numpy()
    sigma, phi = np.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k = phi_k * sigma_k**0.25
    return torch.from_numpy(phi_k).to(device=device, dtype=dtype)


def lrelu2(x: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
    out = torch.where(condition=x > 0, input=x, other=alpha * x)
    return out * out


class SpectralAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        spectral_basis: torch.Tensor,
        use_tensordot: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.use_tensordot = use_tensordot
        self.eps = eps
        self.register_buffer("spectral_basis", spectral_basis, persistent=True)
        self.r = self.head_dim * 4
        self.k = spectral_basis.shape[1]

        self.M_phi = nn.Linear(self.k * self.num_heads * self.head_dim, dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.wo.SCALE_INIT = 1

        if self.use_tensordot:
            self.lora_lo = nn.Linear(self.k, self.r, bias=False)
            self.lora_hi = nn.Linear(self.r, dim, bias=False)

        self.wg_z = nn.Linear(self.head_dim**2, 1)
        self.wg = nn.Linear(1, self.head_dim)
        self.register_parameter("kv_norm_scale", nn.Parameter(torch.empty(1, 1, 1, self.head_dim, self.head_dim)))
        self.register_parameter("qk_norm_scale", nn.Parameter(torch.empty(1, self.num_heads, 1)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, *, debug: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim
        bhld = x.view(B, L, H, h).transpose(1, 2)
        x_tilde = self.stu_conv(self.spectral_basis, bhld)
        x_tilde = x_tilde.permute(0, 2, 3, 1, 4).reshape(B, L, self.k * H * h)
        x_tilde = self.M_phi(x_tilde)
        q = self.wq(x_tilde).view(B, L, H, h).transpose(1, 2)
        k = self.wk(x_tilde).view(B, L, H, h).transpose(1, 2)
        v = self.wv(x_tilde).view(B, L, H, h).transpose(1, 2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        Z = torch.einsum("bhsn,bhsp->bhspn", k, v)
        gate_input_z = Z.reshape(B, H, L, -1)
        gates_logits_z = self.wg_z(gate_input_z)
        gates_z = lrelu2(gates_logits_z) + self.eps
        gates_z = gates_z.squeeze(-1)
        gated_Z = Z * gates_z.unsqueeze(-1).unsqueeze(-1)
        print(f"SpectralAttention: gated_Z shape={gated_Z.shape}, gates_z shape={gates_z.shape}")
        Z_cumul, gate_cumul = self.batched_scan_fn(gated_Z, gates_z)
        H = Z_cumul / (gate_cumul.unsqueeze(-1).unsqueeze(-1) + self.eps)
        Y = torch.einsum("bhtp,bhtpn->bhtn", q, H)
        Y_attn = Y.permute(0, 2, 1, 3).reshape(B, L, D)
        Y_attn = F.normalize(Y_attn, dim=-1)
        out = self.wo(Y_attn)
        return out

    def stu_conv(self, filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if filters.dim() != 2:
            raise ValueError("filters must be 2D tensor of shape [K_len, K_num]")
        if inputs.dim() != 4:
            raise ValueError("inputs must be 4D tensor of shape [B, H, L, h]")
        B, H, L, h = inputs.shape
        K_len, K_num = filters.shape
        inputs_flat = inputs.permute(0, 1, 3, 2).reshape(B * H * h, L)
        causal_base = lambda sig, ker: convolve(sig, ker, mode="full")[..., : sig.shape[-1]]
        apply_all_filters = torch.vmap(causal_base, in_dims=(None, 1), out_dims=0)
        vmap_flat_batch = torch.vmap(apply_all_filters, in_dims=(0, None), out_dims=0)
        y_flat = vmap_flat_batch(inputs_flat, filters)
        y_reshaped = y_flat.reshape(B, H, h, K_num, L)
        return y_reshaped.permute(0, 1, 4, 3, 2)

    def combine_fn(
        self, x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Z_x, g_x = x
        Z_y, g_y = y
        Z_new = Z_x + Z_y
        g_new = g_x + g_y
        return (Z_new, g_new)

    def scan_fn(self, Z_slice: torch.Tensor, g_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        leaves = (Z_slice, g_slice)
        return associative_scan(*leaves)

    @torch._dynamo.disable
    def batched_scan_fn(self, gated_Z: torch.Tensor, gates_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, L, h, _ = gated_Z.shape
        assert gated_Z.shape == (B, H, L, h, h), f"Expected [B,H,L,h,h], got {gated_Z.shape}"
        assert gates_z.shape == (B, H, L), f"Expected [B,H,L], got {gates_z.shape}"
        # print(f"batched_scan_fn: gated_Z shape={gated_Z.shape}, strides={gated_Z.stride()}, is_contiguous={gated_Z.is_contiguous()}")
        # print(f"batched_scan_fn: gates_z shape={gates_z.shape}, strides={gates_z.stride()}, is_contiguous={gates_z.is_contiguous()}")

        # Reshape gated_Z: [B, H, L, h, h] -> [B*H, L, h*h] -> [B*H, h*h, L]
        gated_Z_flat = gated_Z.reshape(B * H, L, h * h).permute(0, 2, 1).contiguous()

        # Reshape and Expand gates_z: [B, H, L] -> [B*H, L] -> [B*H, 1, L] -> [B*H, h*h, L]
        gates_z_flat = gates_z.reshape(B * H, L).unsqueeze(1).expand(-1, h * h, -1).contiguous()

        # print(f"batched_scan_fn: gated_Z_flat shape={gated_Z_flat.shape}, strides={gated_Z_flat.stride()}, is_contiguous={gated_Z_flat.is_contiguous()}")
        # print(f"batched_scan_fn: gates_z_flat shape={gates_z_flat.shape}, strides={gates_z_flat.stride()}, is_contiguous={gates_z_flat.is_contiguous()}")

        # Apply associative_scan. The static vmap method in AssociativeScan handles batching correctly now.
        # We just need to call the function directly.
        Z_cumul_flat, gate_cumul_flat = associative_scan(gated_Z_flat, gates_z_flat)

        # Reshape Z_cumul back: [B*H, h*h, L] -> [B*H, L, h*h] -> [B, H, L, h, h]
        Z_cumul = Z_cumul_flat.permute(0, 2, 1).reshape(B, H, L, h, h)

        # Reshape gate_cumul back: [B*H, h*h, L] -> select first feature -> [B*H, L] -> [B, H, L]
        # Since gates were identical across the feature dim, the cumulative sum is also identical.
        # We only need to take the result from one feature slice.
        gate_cumul = gate_cumul_flat[:, 0, :].reshape(B, H, L)

        # print(f"batched_scan_fn: Z_cumul shape={Z_cumul.shape}, gate_cumul shape={gate_cumul.shape}")
        return Z_cumul, gate_cumul

    def reset_parameters(self):
        with torch.no_grad():
            L = float(self.seq_len)
            g0 = math.log2(L * L - L)
            self.qk_norm_scale.fill_(g0)
            self.kv_norm_scale.fill_(g0)


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = "cuda"

    # Parameters matching reported shapes: gated_Z [128, 16384, 256], gates [128, 1, 256]
    batch_size = 2
    num_heads = 1
    head_dim = 16  # h, so h*h = 16384 (128*128)
    dim = num_heads * head_dim  # D = 8*128 = 1024
    seq_len = 32  # L
    K = 24  # Number of spectral filters

    # Create spectral basis
    spectral_basis = get_spectral_filters(seq_len, K, use_hankel_L=False, device=device, dtype=torch.float32)

    # Initialize model
    model = SpectralAttention(
        dim=dim, num_heads=num_heads, seq_len=seq_len, spectral_basis=spectral_basis, use_tensordot=True, eps=1e-5
    ).to(device)

    # Input tensor
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32, device=device)

    # Forward pass with debugging
    print("Starting forward pass")
    output = model(x, debug=True)
    print("Forward pass completed, output shape:", output.shape)
