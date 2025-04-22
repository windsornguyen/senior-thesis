import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SpectralAttention(nn.Module):
    """Associative attention mechanism with spectral filtering and gating."""

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

        # Layers default to float32
        self.M_phi = nn.Linear(self.k * self.num_heads * self.head_dim, dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.wo.SCALE_INIT = 1

        # Causal Conv1d layer with learnable kernel
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim * self.k,
            kernel_size=3,
            padding=2,  # Pad to maintain length after causal convolution
            groups=dim
        )

        if self.use_tensordot:
            self.lora_lo = nn.Linear(self.k, self.r, bias=False)  # M_i^1
            self.lora_hi = nn.Linear(self.r, dim, bias=False)  # M_i^2

        # Gates
        self.wg_z = nn.Linear(self.head_dim**2, 1)
        self.wg = nn.Linear(1, self.head_dim)

        self.register_parameter("kv_norm_scale", nn.Parameter(torch.empty(1, 1, 1, self.head_dim, self.head_dim)))
        self.register_parameter("qk_norm_scale", nn.Parameter(torch.empty(1, self.num_heads, 1)))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, *, debug: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim

        # Branch 1: Causal convolution
        x_conv = x.transpose(1, 2)  # [B, D, L]
        x_conv = self.conv1d(x_conv)[..., :L]  # [B, D*k, L], trim to original length
        x_conv = x_conv.transpose(1, 2)  # [B, L, D*k]
        x_tilde = x_conv.view(B, L, H, h, self.k).permute(0, 2, 1, 4, 3)  # [B, H, L, k, h]

        # Merge head/filter dims
        x_tilde = x_tilde.permute(0, 2, 3, 1, 4).reshape(B, L, self.k * H * h)
        x_tilde = self.M_phi(x_tilde)

        # Branch 2: Compute multihead linear attention
        q = self.wq(x_tilde).view(B, L, H, h).transpose(1, 2)  # (B, H, L, h)
        k = self.wk(x_tilde).view(B, L, H, h).transpose(1, 2)  # (B, H, L, h)
        v = self.wv(x_tilde).view(B, L, H, h).transpose(1, 2)  # (B, H, L, h)
        q, k, v = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        sim = torch.einsum("bhld,bhld->bhl", q, k) * self.qk_norm_scale
        Z = torch.einsum("bhsn,bhsp->bhspn", k, v) * self.kv_norm_scale

        gate_input_z = Z.reshape(B, H, L, -1)
        gates_logits_z = self.wg_z(gate_input_z)
        gates_z = F.leaky_relu(gates_logits_z) + self.eps
        gates_z = gates_z.squeeze(-1)  # [B,H,L]

        gated_Z = Z * gates_z.unsqueeze(-1).unsqueeze(-1)
        max_cumul, norm_cumul, v_cumul, Z_cumul, gate_cumul = self.batched_scan_fn(sim, v, gated_Z, gates_z)

        linear_attn = v_cumul / (norm_cumul[..., None] + self.eps)  # [B,H,L,h]
        weights = torch.exp(sim - max_cumul) / (norm_cumul + self.eps)  # [B,H,L]

        H = Z_cumul / (gate_cumul.unsqueeze(-1).unsqueeze(-1) + self.eps)  # [B,H,L,h,h]

        # ── memory‑neutral interpolation (no gigantic H_prime tensor) ─────────
        Y_base = torch.einsum("bhtp,bhtpn->bhtn", q, H)  # [B,H,L,h]
        Y_lin = torch.einsum("bhtp,bhtpn->bhtn", q, linear_attn.unsqueeze(-2))  # [B,H,L,h]
        Y = Y_base + (Y_lin - Y_base) * weights[..., None]  # [B,H,L,h]

        Y_attn = Y.permute(0, 2, 1, 3).reshape(B, L, D)  # (B, T, d)
        Y_attn = F.normalize(Y_attn, dim=-1)
        out = self.wo(Y_attn)

        return out

    def combine_fn(self, x: Tuple, y: Tuple) -> Tuple:
        m_x, s_x, n_x, Z_x, g_x = x
        m_y, s_y, n_y, Z_y, g_y = y

        m = torch.maximum(m_x, m_y)
        exp_x, exp_y = torch.exp(m_x - m), torch.exp(m_y - m)

        s = s_x * exp_x + s_y * exp_y
        n = n_x * exp_x[..., None] + n_y * exp_y[..., None]

        Z = Z_x + Z_y
        g = g_x + g_y

        return m, s, n, Z, g

    def scan_fn(
        self, qk_slice: torch.Tensor, v_slice: torch.Tensor, Z_slice: torch.Tensor, g_slice: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        leaves = (
            qk_slice,
            torch.ones_like(qk_slice),
            v_slice,
            Z_slice,
            g_slice,
        )
        return associative_scan(combine_fn=self.combine_fn, xs=leaves, dim=0, combine_mode="generic")

    def batched_scan_fn(
        self, sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H = sim.shape[0], sim.shape[1]
        sim_flat = sim.flatten(0, 1)
        v_flat = v.flatten(0, 1)
        gated_Z_flat = gated_Z.flatten(0, 1)
        gates_z_flat = gates_z.flatten(0, 1)

        scan_all = torch.vmap(self.scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
        result = scan_all(sim_flat, v_flat, gated_Z_flat, gates_z_flat)

        return tuple(t.reshape(B, H, *t.shape[1:]) for t in result)

    def reset_parameters(self):
        with torch.no_grad():
            L = float(self.seq_len)
            g0 = math.log2(L * L - L)
            self.qk_norm_scale.fill_(g0)
            self.kv_norm_scale.fill_(g0)
