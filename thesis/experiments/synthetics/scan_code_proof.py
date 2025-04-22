import torch
import torch.nn.functional as F
from thesis.experiments.utils.assoc_scan.kernel import associative_scan

device = torch.device("cuda")

# shapes & data as before
B, H, T, S, N, P = 8, 16, 1000, 1000, 32, 32
Q = torch.randn(B, T, H, N // H, device=device).transpose(1, 2)  # [B,H,T,n]
K = torch.randn(B, S, H, N // H, device=device).transpose(1, 2)  # [B,H,S,n]
V = torch.randn(B, S, H, P // H, device=device).transpose(1, 2)  # [B,H,S,p]

# 1) Quadratic baseline
Y_standard = ((Q @ K.transpose(-2, -1)) * torch.tril(torch.ones(T, S, device=device))[None, None]) @ V
Y_standard = Y_standard.transpose(1, 2).reshape(B, T, N)

# 2) Build the Z tensor once
Z = torch.einsum("bhsp,bhsn->bhspn", V, K)  # [B,H,S,p,n]


# 3) Prefix‐sum via associative_scan instead of cumsum
def sum_combine(a, b):
    # simple elementwise add is associative
    return a + b


# scan along the "source" axis (dim=2 = S)
Z_scan = associative_scan(combine_fn=sum_combine, xs=Z, dim=2, combine_mode="generic")  # → [B,H,S,p,n]

# 4) Finish the linear‐attention contraction
Y_scan = torch.einsum("bhtn,bhtpn->bhtp", Q, Z_scan)  # [B,H,T,p]
Y_scan = Y_scan.transpose(1, 2).reshape(B, T, N)

# 5) Verify bit‐for‐bit match
print("assoc_scan vs quadratic:", torch.allclose(Y_scan, Y_standard, atol=1e-3))

# 6) Online‑normalized linear attention in one scan
# -------------------------------------------------

# build numerator leaves: outer‑product of V with phi_K
Z_norm = torch.einsum("bhsp,bhsn->bhspn", V, K)  # [B,H,S,p,n]


# combine_fn that accumulates both H and s together
def norm_combine(a, b):
    H_x, s_x = a
    H_y, s_y = b
    return (H_x + H_y, s_x + s_y)


# run the fused scan over the source axis
# leaves is a pair (H_leaf, s_leaf)
leaves = (Z_norm, K)
(H_pref, s_pref) = associative_scan(
    combine_fn=norm_combine,
    xs=leaves,
    dim=2,  # scan over S dim
    combine_mode="generic",
)
# H_pref: [B,H,S,p,n],  s_pref: [B,H,S,n]

# final contraction + normalization
Y_norm = torch.einsum("bhtn,bhtpn->bhtp", Q, H_pref)  # [B,H,T,p]
Y_norm = Y_norm.transpose(1, 2).reshape(B, T, N)  # [B,T,N]

print("normalized linear attention shape:", Y_norm.shape)

# -----------------------------------------------------------------------------
# 6b) Exact causal‑softmax via one associative_scan *per query row*
#     (scores are pre‑computed, so scan depth is O(log S))
# -----------------------------------------------------------------------------
pass
# -----------------------------------------------------------------------------
# 7b) Precompute per‑row similarities
# -----------------------------------------------------------------------------
# sim_full[b,h,t,j] = Q[b,h,t]·K[b,h,j]
sim_full = torch.einsum("bhtn,bhsn->bhts", Q, K)  # [B,H,T,S]


# -----------------------------------------------------------------------------
# 7c) Define the online‑softmax combine monoid
# -----------------------------------------------------------------------------
def softmax_combine(a, b):
    m_x, s_x, n_x = a
    m_y, s_y, n_y = b
    m = torch.maximum(m_x, m_y)
    expx = torch.exp(m_x - m)
    expy = torch.exp(m_y - m)
    s = s_x * expx + s_y * expy
    n = n_x * expx[..., None] + n_y * expy[..., None]
    return (m, s, n)


def scan_softmax_row(sim_row: torch.Tensor, vals: torch.Tensor):
    leaves = (
        sim_row,  # logits [S]
        torch.ones_like(sim_row),  # denom starts at 1
        vals,  # values [S,p]
    )
    (m_pref, s_pref, n_pref) = associative_scan(
        softmax_combine,
        xs=leaves,
        dim=0,  # scan over the length‑S dimension
        combine_mode="generic",
    )
    # final softmax output for that row
    return n_pref / (s_pref[..., None] + 1e-6)  # [S,p]


# 4) nest vmap in *relative* dims

# A) map over the sequence‐axis (T) of one (B,H) block:
#    sim_full[b,h] is [T,S], V[b,h] is [S,p]
#    we want to call scan_softmax_row(sim_full[b,h,t,:], V[b,h]) for each t
scan_over_t = torch.vmap(
    scan_softmax_row,
    in_dims=(0, None),  # vectorize over dim‐0 of sim_bh ([T,S]) but keep vals fixed
    out_dims=0,  # output will have its first dim be the scan axis T
)

# B) now map that over heads (H)
#    sim_full[b] is [H,T,S], V[b] is [S,p]
scan_over_h = torch.vmap(
    scan_over_t,
    in_dims=(0, 0),  # vectorize over dim‐0 of sim_full[b] (H) and V[b]
    out_dims=0,  # output's H‐axis is created here
)

# C) now map over batch (B)
Y_scan_full = torch.vmap(  # B‑map
    scan_over_h, in_dims=(0, 0), out_dims=0
)(sim_full, V)  # <--  sim_full here!
# shape [B,H,T,S,p]

# ❷ take the causal diagonal, make it contiguous, then reshape
Y_scan = Y_scan_full.diagonal(dim1=2, dim2=3).contiguous()  # [B,H,T,p]
Y_scan = Y_scan.transpose(1, 2).reshape(B, T, N)  # [B,T,N]

# 6) verify
print("vmap‐scan vs reference:", torch.allclose(Y_scan, Y_ref, atol=1e-6))
# Calculate difference metrics
diff = torch.abs(Y_scan - Y_ref)
max_diff = torch.max(diff)
mean_diff = torch.mean(diff)
num_not_close = torch.sum(diff > 1e-5)
total_elements = Y_ref.numel()

print(f"  Max abs difference: {max_diff.item()}")
print(f"  Mean abs difference: {mean_diff.item()}")
print(
    f"  Elements failing atol=1e-5: {num_not_close.item()} / {total_elements} ({num_not_close.item() / total_elements:.2%})"
)

# Optional: Verify intermediate shapes if needed
# print("Shapes:", sim_flat.shape, V_flat.shape, n_flat_pref.shape, s_flat_pref.shape, n_selected.shape, s_selected.shape, Y_flat.shape, Y_fast.shape)
