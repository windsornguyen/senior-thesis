import torch
import torch.nn.functional as F

from thesis.experiments.utils.assoc_scan.kernel import associative_scan

# =============================================================================
# Configuration
# =============================================================================
# --- Dimensions ---
B, H, T, S, n, p = 1, 1, 8, 8, 16, 16  # Keep reasonably small for verification
N = H * p  # Total output dim derived
device = "cuda"  # "cuda" also valid

# --- Reproducibility ---
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# =============================================================================
# Setup - Random Inputs
# =============================================================================
Q = torch.randn(B, T, H, n, device=device).transpose(1, 2)  # [B,H,T,n]
K = torch.randn(B, S, H, n, device=device).transpose(1, 2)  # [B,H,S,n]
V = torch.randn(B, S, H, p, device=device).transpose(1, 2)  # [B,H,S,p]

print(f"--- Configuration ---")
print(f" B={B}, H={H}, T={T}, S={S}, n={n}, p={p}, N={N}, device='{device}'")
print("-" * 20)

# =============================================================================
# 1) Reference: Quadratic Causal Softmax Attention
# =============================================================================
print("1) Calculating Reference (Quadratic Causal Softmax)...")
# Calculate similarity scores
S_full = torch.einsum("bhtn,bhsn->bhts", Q, K)  # [B,H,T,S]

# Apply causal mask
mask = torch.tril(torch.ones(T, S, device=device))[None, None]  # Use T for mask size matching Q dim
S_full = S_full.masked_fill(mask == 0, -float("inf"))

# Apply softmax
A_full = F.softmax(S_full, dim=-1)  # [B,H,T,S]

# Apply attention weights to values
Y_ref = torch.einsum("bhts,bhsp->bhtp", A_full, V)  # [B,H,T,p]

# Reshape to final output format
Y_ref = Y_ref.transpose(1, 2).reshape(B, T, N)  # [B,T,N]
print("   Reference Y_ref shape:", Y_ref.shape)


# =============================================================================
# 2) Tree-Scan Implementation
# =============================================================================
print("\n2) Calculating Tree-Scan Implementation...")


# -- associative combine exactly as before -----------------------------------
def softmax_combine(a, b):
    m_x, s_x, n_x = a
    m_y, s_y, n_y = b
    m    = torch.maximum(m_x, m_y)
    expx = torch.exp(m_x - m)
    expy = torch.exp(m_y - m)
    s    = s_x * expx + s_y * expy
    n    = n_x * expx[..., None] + n_y * expy[..., None]
    return (m, s, n)

# -- scan ONE row ------------------------------------------------------------
# q_row : [n]          (query for this t)
# K_row : [S,n]        (all keys for one head)
# V_row : [S,p]        (all values for one head)
def scan_one_row(q_row, K_row, V_row):
    sim_row = (q_row * K_row).sum(-1)             # build logits [S]
    leaves   = (sim_row,
                torch.ones_like(sim_row),         # denom=1
                V_row)                            # values
    (m_pref, s_pref, n_pref) = associative_scan(
        softmax_combine, leaves, dim=0, combine_mode="generic"
    )
    return n_pref / (s_pref[..., None] + 1e-6)    # [S,p]

# -- vmap over sequence length T ---------------------------------------------
# For a fixed (b,h) block:
#   Q_bh: [T,n] ,  K_bh: [S,n] ,  V_bh: [S,p]
vmap_t = torch.vmap(
    scan_one_row,
    in_dims=(0, None, None),    # q_row varies (dim‑0), K & V are broadcast
    out_dims=0                  # stack T results
)                               # → [T,S,p]

# -- vmap over heads, then batch ---------------------------------------------
vmap_h = torch.vmap(vmap_t, in_dims=(0, 0, 0), out_dims=0)    # [H,T,S,p]
vmap_b = torch.vmap(vmap_h, in_dims=(0, 0, 0), out_dims=0)    # [B,H,T,S,p]

# Apply to all batches/heads
Y_scan_full = vmap_b(Q, K, V)               # NEVER builds [B,H,T,S] scores

# -- extract causal diagonal j=t ---------------------------------------------
Y_diag = Y_scan_full.diagonal(offset=0, dim1=2, dim2=3)  # [B,H,p,T]
Y_diag = Y_diag.permute(0, 1, 3, 2).contiguous()         # [B,H,T,p]

# -- reshape to [B,T,N] -------------------------------------------------------
N = H * p
Y_scan = Y_diag.transpose(1, 2).reshape(B, T, N)
print("   Tree‑scan Y_scan shape:", Y_scan.shape)

# ---------------------------------------------------------------------------
# 3) Verification
# ---------------------------------------------------------------------------
tolerance = 1e-6
match = torch.allclose(Y_scan, Y_ref, atol=tolerance)
print(f"   Scan result matches reference (atol={tolerance}): {match}")

if not match:
    diff = torch.abs(Y_scan - Y_ref)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    num_not_close = torch.sum(diff > tolerance)
    total_elements = Y_ref.numel()
    print(f"     Max abs difference: {max_diff.item():.4g}")
    print(f"     Mean abs difference: {mean_diff.item():.4g}")
    print(
        f"     Elements failing tolerance: {num_not_close.item()} / {total_elements} ({num_not_close.item() / total_elements:.2%})"
    )

print("-" * 20)
print("Verification complete.")
