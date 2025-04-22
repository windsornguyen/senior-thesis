# Does this also hold with Spectron QK-norm scaling?
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
from thesis.experiments.utils.assoc_scan.kernel import associative_scan
from torchaudio.functional import convolve

def lrelu2(x: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
    out = torch.where(
        condition=x > 0,
        input=x,
        other=alpha * x,
    )
    return out * out


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
        # Removed dtype parameter
    ):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.use_tensordot = use_tensordot
        self.eps = eps

        # self.register_buffer("cheby_coeffs", get_cheby_coeffs(seq_len, dtype=torch.float32), persistent=True)
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

        if self.use_tensordot:
            self.lora_lo = nn.Linear(self.k, self.r, bias=False)  # M_i^1
            self.lora_hi = nn.Linear(self.r, dim, bias=False)  # M_i^2

        # Gates
        self.wg_z = nn.Linear(self.head_dim**2, 1)
        # self.wg_z = nn.Linear(2 * self.head_dim, 1)
        self.wg = nn.Linear(1, self.head_dim)
        # self.gate = nn.Linear(dim, dim)

        # self.register_parameter("kv_norm_scale", nn.Parameter(torch.empty(1, 1, 1, self.head_dim, self.head_dim)))
        # Parameter defaults to float32
        self.register_parameter("qk_norm_scale", nn.Parameter(torch.empty(1, self.num_heads, 1)))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, *, debug: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim

        # Branch 1: Compute STU features
        bhld = x.view(B, L, H, h).transpose(1, 2)  # [B, H, L, h]
        x_tilde = self.stu_conv(self.spectral_basis, bhld)  # [B, H, L, K, h]

        # Merge head/filter dims
        x_tilde = x_tilde.permute(0, 2, 3, 1, 4).reshape(B, L, self.k * H * h)
        x_tilde = self.M_phi(x_tilde)

        # Branch 2: Compute multihead linear attention
        q = self.wq(x_tilde).view(B, L, H, h).transpose(1, 2)  # (B, H, L, h)
        k = self.wk(x_tilde).view(B, L, H, h).transpose(1, 2)  # (B, H, L, h)
        v = self.wv(x_tilde).view(B, L, H, h).transpose(1, 2)  # (B, H, L, h)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        sim = torch.einsum("bhld,bhld->bhl", q, k) * self.qk_norm_scale
        Z = torch.einsum("bhsn,bhsp->bhspn", k, v)

        gate_input_z = Z.reshape(B, H, L, -1)
        gates_logits_z = self.wg_z(gate_input_z)
        gates_z = lrelu2(gates_logits_z) + self.eps
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

    def cheby_conv(self, coeffs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r"""Applies a single shared filter causally using a batched vmap.

        Equivalent to :func:`cheby_conv`, but implemented by reshaping the
        ``B, H, D`` dimensions into a single batch dimension and applying
        a single `torch.vmap` call.

        See ``test_conv.py`` for vmap and batched versions.

        Args:
            coeffs (torch.Tensor): Filter coefficients. Shape: ``[K_len]``.
            inputs (torch.Tensor): Input sequences. Shape: ``[B, H, L, D]``.

        Returns:
            torch.Tensor: Output tensor after convolution. Shape: ``[B, H, L, D]``.
        """
        if coeffs.dim() != 1:
            raise ValueError("coeffs must be 1D tensor of shape [K_len]")
        if inputs.dim() != 4:
            raise ValueError("inputs must be 4D tensor of shape [B, H, L, D]")

        B, H, L, D = inputs.shape
        K_len = coeffs.shape[0]

        # Flatten B, H, D dims into one batch dim
        inputs_flat = inputs.permute(0, 1, 3, 2).reshape(B * H * D, L)  # [BHD, L]
        causal = lambda sig, ker: convolve(sig, ker, mode="full")[..., : sig.shape[-1]]

        # vmap over the flattened batch dimension
        vmap_causal = torch.vmap(causal, in_dims=(0, None), out_dims=0)
        y_flat = vmap_causal(inputs_flat, coeffs)  # [BHD, L]

        # Reshape back
        y_perm = y_flat.reshape(B, H, D, L)  # [B, H, D, L]
        return y_perm.permute(0, 1, 3, 2)  # [B, H, L, D]

    def tensordot_conv(self, filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r"""Applies channel-specific filters causally using a batched vmap approach.

        Equivalent to :func:`tensordot_conv`, but implemented using tensor
        permutations and nested `torch.vmap` optimized for batching.
        Assumes input's last dimension `h` is pre-projected.

        See ``test_conv.py`` for vmap and batched versions.

        Args:
            filters (torch.Tensor): Bank of channel-specific filters.
                Shape: ``[K_len, h]``.
            inputs (torch.Tensor): Pre-projected input sequences.
                Shape: ``[B, H, L, h]``.

        Returns:
            torch.Tensor: Output tensor after convolution. Shape: ``[B, H, L, h]``.
        """
        if filters.dim() != 2:
            raise ValueError("filters must be 2D tensor of shape [K_len, h]")
        if inputs.dim() != 4 or inputs.shape[-1] != filters.shape[-1]:
            raise ValueError("inputs must be 4D tensor of shape [B, H, L, h] with h matching filters")

        B, H, L, h = inputs.shape
        K_len, _ = filters.shape

        # This one is trickier with a single vmap due to channel-specific filters.
        # We use the approach of mapping over 'h' first.
        # Permute inputs to [h, B, H, L] then flatten B,H -> [h, BH, L]
        inputs_perm = inputs.permute(3, 0, 1, 2).reshape(h, B * H, L)
        # Permute filters to [h, K_len]
        filters_perm = filters.permute(1, 0)

        causal_base = lambda sig, ker: convolve(sig, ker, mode="full")[..., : sig.shape[-1]]
        # Inner vmap operates on the flattened BH dimension
        causal_bh_map = torch.vmap(causal_base, in_dims=(0, None), out_dims=0)

        # Outer vmap maps over h dimension for both inputs and filters
        vmap_h = torch.vmap(causal_bh_map, in_dims=(0, 0), out_dims=0)

        y_perm_h = vmap_h(inputs_perm, filters_perm)  # [h, BH, L]

        # Reshape back
        y_reshaped = y_perm_h.reshape(h, B, H, L)  # [h, B, H, L]
        return y_reshaped.permute(1, 2, 3, 0)  # [B, H, L, h]

    def stu_conv(self, filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r"""Applies K shared filters causally to all channels using a batched vmap.

        Equivalent to :func:`stu_conv`, but implemented by reshaping the
        ``B, H, h`` dimensions into a single batch dimension and using nested
        `torch.vmap` optimized for batching. Results in an expanded output
        dimension ``K_num``.

        See ``test_conv.py`` for vmap and batched versions.

        Args:
            filters (torch.Tensor): Bank of shared filters.
                Shape: ``[K_len, K_num]``.
            inputs (torch.Tensor): Input sequences (typically pre-projected).
                Shape: ``[B, H, L, h]``.

        Returns:
            torch.Tensor: Output tensor after convolution. Shape: ``[B, H, L, K_num, h]``.
        """
        if filters.dim() != 2:
            raise ValueError("filters must be 2D tensor of shape [K_len, K_num]")
        if inputs.dim() != 4:
            raise ValueError("inputs must be 4D tensor of shape [B, H, L, h]")

        B, H, L, h = inputs.shape
        K_len, K_num = filters.shape

        # Flatten B, H, h dims into one batch dim
        inputs_flat = inputs.permute(0, 1, 3, 2).reshape(B * H * h, L)  # [BHh, L]

        causal_base = lambda sig, ker: convolve(sig, ker, mode="full")[..., : sig.shape[-1]]

        # Inner function applies all K_num filters to a single signal
        apply_all_filters = torch.vmap(
            causal_base, in_dims=(None, 1), out_dims=0
        )  # sig[L], filters[K_len, K_num] -> y[K_num, L]

        # Outer vmap maps this inner function over the flattened input batch
        vmap_flat_batch = torch.vmap(apply_all_filters, in_dims=(0, None), out_dims=0)

        y_flat = vmap_flat_batch(inputs_flat, filters)  # [BHh, K_num, L]

        # Reshape back
        y_reshaped = y_flat.reshape(B, H, h, K_num, L)  # [B, H, h, K_num, L]
        return y_reshaped.permute(0, 1, 4, 3, 2)  # [B, H, L, K_num, h]

    def bld_stu_conv(self, filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r"""Applies K shared filters causally to all channels using nested vmap.

        Performs 1D causal convolution, applying *each* of the `K_num filters
        provided in filters independently to *every* channel (h) of the input,
        across all batches (B) and heads (H). This implementation uses nested
        torch.vmap calls and results in an expanded output dimension `K_num.

        Args:
            filters (torch.Tensor): Bank of shared filters. Shape: `[K_len, K_num],
                where `K_len is the kernel length and K_num is the number
                of distinct filters.
            inputs (torch.Tensor): Input sequences (typically pre-projected). Shape:
                `[B, H, L, h], where L is sequence length.

        Returns:
            torch.Tensor: Output tensor after convolution. Shape: `[B, H, L, K_num, h].

        Example::

            >>> B, H, L, h, K_len, K_num = 2, 3, 16, 4, 5, 7
            >>> filters = torch.randn(K_len, K_num)
            >>> inputs = torch.randn(B, H, L, h)
            >>> output = stu_conv(filters, inputs)
            >>> output.shape
            torch.Size([2, 3, 16, 7, 4])
        """
        if filters.dim() != 2:
            raise ValueError("filters must be 2D tensor of shape [K_len, K_num]")
        if inputs.dim() != 4:
            raise ValueError("inputs must be 4D tensor of shape [B, H, L, h]")

        inputs_perm = inputs.permute(0, 1, 3, 2)  # [B, H, h, L]
        causal = lambda sig, ker: convolve(sig, ker, mode="full")[..., : sig.shape[-1]]

        kmap = torch.vmap(causal, in_dims=(None, 1), out_dims=0)  # Map over K_num filters
        cmap = torch.vmap(kmap, in_dims=(0, None), out_dims=1)  # Map over h signal, add K dim after h
        hmap = torch.vmap(cmap, in_dims=(0, None), out_dims=1)  # Map over H signal, add K dim after H
        bmap = torch.vmap(hmap, in_dims=(0, None), out_dims=0)  # Map over B signal, add K dim before H

        y = bmap(inputs_perm, filters)  # [B, K_num, H, h, L]
        return y.permute(0, 2, 4, 1, 3)  # [B, H, L, K_num, h]

    def combine_fn(self, x: Tuple, y: Tuple) -> Tuple:
        """Numerically‑stable combine for associative scan over sequence.

        State per timestep
        ------------------
        m : log‑max accumulator             shape [L]
        s : normalization denominator       shape [L]
        n : first‑order numerator           shape [L, h]
        Z : second‑order accumulator        shape [L, h, h]
        g : gate accumulator                shape [L]

        """
        m_x, s_x, n_x, Z_x, g_x = x
        m_y, s_y, n_y, Z_y, g_y = y

        m = torch.maximum(m_x, m_y)  # new log‑max
        exp_x, exp_y = torch.exp(m_x - m), torch.exp(m_y - m)

        # n_x is always [L, h], exp_x is [L]. Broadcast exp_x to [L, 1].
        s = s_x * exp_x + s_y * exp_y  # scalar denom
        n = n_x * exp_x[..., None] + n_y * exp_y[..., None]  # [L,h]

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
        return associative_scan(combine_fn=self.combine_fn, xs=leaves, dim=0, combine_mode="generic")

    def batched_scan_fn(
        self, sim: torch.Tensor, v: torch.Tensor, gated_Z: torch.Tensor, gates_z: torch.Tensor
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
        scan_all = torch.vmap(self.scan_fn, in_dims=(0, 0, 0, 0), out_dims=0)
        result = scan_all(sim_flat, v_flat, gated_Z_flat, gates_z_flat)  # Tuple of 5 tensors

        # Reshape each output tensor back to (B, H, ...)
        return tuple(t.reshape(B, H, *t.shape[1:]) for t in result)

    def reset_parameters(self):
        with torch.no_grad():
            # self.kv_norm_scale.fill_(self.head_dim**-0.5)
            self.qk_norm_scale.fill_(self.head_dim**-0.5)

# Self-contained cell: SpectralAttention gradient vs head dim
import torch, math
torch.manual_seed(0)

# --- hyper‑params ---
n_tok, d_k = 3000, 64
spectral_basis = torch.randn(16, 4)            # dummy, not used in this probe

# --- model (1 head, proj matrices are random) ---
sa = SpectralAttention(dim=d_k, num_heads=1,
                       seq_len=1, spectral_basis=spectral_basis).eval()

# freeze proj layers to isolate pure geometry
for p in sa.parameters(): p.requires_grad_(False)

# --- synthetic tokens with length bias ---
scale  = torch.exp(torch.randn(n_tok))           # log‑normal spread
tokens = torch.randn(n_tok, d_k) * scale[:,None] # shape [N, d_k]

# --- feed through W_q, W_k and L2‑norm exactly as forward does ---
with torch.no_grad():
    q = sa.wq(tokens).view(n_tok, 1, d_k)        # [N,1,h]
    k = sa.wk(tokens).view(n_tok, 1, d_k)
    qn, kn = torch.nn.functional.normalize(q, -1), torch.nn.functional.normalize(k, -1)
    sim = (qn*kn).sum(-1).squeeze() * sa.qk_norm_scale.squeeze()   # scalar per token

# --- correlation ---
def corr(a,b):
    a,b = a-a.mean(), b-b.mean()
    return (a*b).mean() / (a.std()*b.std())

print("corr(norm, QK‑norm score) =", corr(scale, sim).item())
