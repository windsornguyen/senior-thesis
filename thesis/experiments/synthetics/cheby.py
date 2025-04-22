import math, torch, matplotlib.pyplot as plt
from numpy import polynomial as npt

# ————————————————————————————
# 1.  Helpers for n and θ_max
# ————————————————————————————
def get_opt_degree(T: int) -> int:
    return math.ceil((7/6) * math.log2(T))           # Thm 1.4

def get_theta_max(n: int) -> float:
    return math.asin(1.0 / (64.0 * n**2))            # β = 1/(64 n²)

# ————————————————————————————
# 2.  Monic Chebyshev coefficients
# ————————————————————————————
def monic_cheby_coeffs(n: int, *, device=None, dtype=torch.float32):
    Tn = npt.Chebyshev.basis(n).convert(kind=npt.Polynomial)
    coef = torch.tensor(Tn.coef, device=device, dtype=dtype)
    coef = coef.flip(dims=[0]).contiguous()
    return coef / (2 ** (n - 1))     # leading = 1

# ————————————————————————————
# 3.  Analytic Hankel (full broadcast)
# ————————————————————————————
def build_hankel_chebys(T: int, *, device="cpu"):
    n   = get_opt_degree(T)
    th  = get_theta_max(n)                 # θ_max
    k   = T - n                            # Hankel dim
    c   = monic_cheby_coeffs(n, device=device)  # (n+1,)

    # indices
    i  = torch.arange(k, device=device).view(k, 1, 1, 1)   # (k,1,1,1)
    j  = torch.arange(k, device=device).view(1, k, 1, 1)   # (1,k,1,1)
    ii = torch.arange(n+1, device=device).view(1, 1, n+1, 1)
    jj = torch.arange(n+1, device=device).view(1, 1, 1, n+1)

    A = i + ii                       # (k,k,n+1,n+1)
    B = j + jj
    sum_exp  = A + B                 # j+k-a-b
    diff_exp = A - B                 # (j-a)-(k-b)

    denom = (sum_exp + 2).float()    # radial integral
    diff_f = diff_exp.float()

    ang = torch.where(
        diff_exp == 0,
        2.0 * th,
        2.0 * torch.sin(diff_f * th) / diff_f
    )

    parity = 1 + (-1)**diff_exp      # 0 for odd diffs

    P = c.view(1,1,-1,1) * c.view(1,1,1,-1)   # c_a c_b

    Z = (parity * ang / denom * P).sum(dim=(-2, -1))   # (k,k)
    Z = (Z + Z.conj().T) * 0.5                         # Hermitian
    return Z

# ————————————————————————————
# 4.  Filters  φ_j  (σ^{¼} scaling)
# ————————————————————————————
def cheby_filters(T: int, K: int, *, device="cpu"):
    Z = build_hankel_chebys(T, device=device)      # analytic Hankel
    _, psi = torch.linalg.eigh(Z)                  # ψ[:,j] orthonormal
    phi_k = psi[:, -K:]                            # take top‑K as‑is
    return Z, phi_k, None                          # σ not returned/needed


# ————————————————————————————
# 5.  Agarwal Hankel (for comparison)
# ————————————————————————————
def hankel_agarwal(T: int, *, device=None):
    t = torch.arange(1, T+1, dtype=torch.float32, device=device)
    s = t.view(-1,1) + t.view(1,-1)
    return 2.0 / (s**3 - s)

# ————————————————————————————
# 6.  Demo + invariants
# ————————————————————————————
T, K = 512, 24
device = "cpu"

# baseline
Z_ag   = hankel_agarwal(T, device=device)
eig_ag = torch.linalg.eigvalsh(Z_ag)

# Chebyshev spectral filtering
Z_csf, phi_csf, sigma_csf = cheby_filters(T, K, device=device)
eig_csf = torch.linalg.eigvalsh(Z_csf)

# plot
plt.semilogy(eig_ag.flip(0),  label="Agarwal Hankel")
plt.semilogy(eig_csf.flip(0), label="Chebyshev Hankel")
plt.title(f"Scree plot  (T={T})"); plt.legend(); plt.show()

# — invariants —
n = get_opt_degree(T)
assert monic_cheby_coeffs(n)[0] == 1.0, "Chebyshev not monic"
assert torch.allclose(Z_csf, Z_csf.conj().T, atol=1e-6)
assert torch.linalg.eigvalsh(Z_csf).min() > -1e-7

# whitened orthonormality
sigma_k = sigma_csf[-K:]
psi_k   = phi_csf * sigma_k.pow(-0.25).unsqueeze(0)
I_K = torch.eye(K, device=device)
assert torch.allclose(psi_k.T @ psi_k, I_K, atol=1e-5)

# handle real/complex imag check safely
if torch.is_complex(phi_csf):
    assert phi_csf.imag.abs().max() < 1e-6

# spectral‑slope
drop = (eig_csf.flip(0)[n] / eig_csf.flip(0)[min(100, T-n-1)]).log10().item()
assert drop > 3.0, f"Slope only {drop:.2f} decades (should be >3)"

print("✅  All invariants passed")
