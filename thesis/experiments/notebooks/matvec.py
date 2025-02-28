#!/usr/bin/env python3

import math
import time
import torch


################################################################################
# 1) Build the Hankel matrix from partial fractions & the reversal matrix J
################################################################################

def build_hankel_matrix(n: int) -> torch.Tensor:
    """
    Z[i,j] = 1/(i+j-1) - 2/(i+j) + 1/(i+j+1),  (1-based i,j).
    We'll store it in 0-based indexing with i,j in [0..n-1].
    """
    Z = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            ip, jp = i + 1, j + 1
            x = ip + jp
            # partial fraction
            val = 1.0/(x - 1.0) - 2.0/x + 1.0/(x + 1.0)
            Z[i, j] = val
    return Z

def build_reversal_matrix(n: int) -> torch.Tensor:
    """
    J is the n x n reversal matrix: J[i, j] = 1 if j == n-i-1, else 0
    """
    J = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        J[i, n - i - 1] = 1.0
    return J


################################################################################
# 2) Compute displacement D = Z - JZJ and factor D ~ P Q^T via truncated SVD
################################################################################

def low_rank_factor_via_svd(D: torch.Tensor, r: int):
    """
    Compute a rank-r factorization D ~ P Q^T via the truncated SVD of D.
      D = U * S * V^T,  then
      P = U[:, :r] * sqrt(S[:r])
      Q = V[:, :r] * sqrt(S[:r])   (but we transpose to get shape (n,r))
    """
    U, S, Vt = torch.linalg.svd(D, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]  # shape (r, n)
    # define P, Q
    P = U_r * torch.sqrt(S_r)  # shape (n, r)
    Q = Vt_r.t() * torch.sqrt(S_r)  # shape (n, r)
    return P, Q

def test_displacement_factor(n=256, rank=3):
    Z = build_hankel_matrix(n)
    J = build_reversal_matrix(n)
    D = Z - J @ Z @ J
    # factor D
    P, Q = low_rank_factor_via_svd(D, rank)
    # check reconstruction
    D_approx = P @ Q.t()
    err = (D_approx - D).norm() / D.norm()
    print(f"[Info] Displacement factor: rel error = {err.item():.2e}")
    return Z, D, P, Q


################################################################################
# 3) Find a single "global 3-term recurrence" for rows (P) or columns (Q).
################################################################################

def find_global_3term_recurrence(M: torch.Tensor):
    """
    Fit a single triple (alpha, beta, gamma) that best satisfies
      M[i+3] ~ alpha*M[i+2] + beta*M[i+1] + gamma*M[i],  for i in [0..n-4].
    We'll just do one big least-squares system across all i.

    Returns alpha, beta, gamma (floats).
    """
    n, r = M.shape
    if n < 4:
        # trivial fallback
        return (0.0, 0.0, 0.0)

    big_blocks = []
    big_target = []
    for i in range(n - 3):
        # local system: M[i+3] = A_i * c, where A_i = [M[i+2], M[i+1], M[i]] as columns
        block = torch.stack([M[i+2], M[i+1], M[i]], dim=1)  # shape (r,3)
        big_blocks.append(block)
        big_target.append(M[i+3].unsqueeze(-1))             # shape (r,1)

    A = torch.cat(big_blocks, dim=0)     # shape ((n-3)*r, 3)
    b = torch.cat(big_target, dim=0)     # shape ((n-3)*r, 1)

    c, _, _, _ = torch.linalg.lstsq(A, b)  # shape (3,1)
    alpha, beta, gamma = c.squeeze()       # shape (3,)
    return (alpha.item(), beta.item(), gamma.item())

def measure_3term_recurrence(M: torch.Tensor, alpha: float, beta: float, gamma: float):
    """
    Quick residual check for M[i+3] vs alpha*M[i+2] + beta*M[i+1] + gamma*M[i].
    """
    n, r = M.shape
    local_diffs = []
    for i in range(n - 3):
        lhs = M[i + 3]
        rhs = alpha * M[i + 2] + beta * M[i + 1] + gamma * M[i]
        local_diffs.append((lhs - rhs).norm().item())
    diffs = torch.tensor(local_diffs, dtype=torch.float64)
    print(f"  [3-term] max diff = {diffs.max():.2e}, mean diff = {diffs.mean():.2e}")

################################################################################
# 4) Converting row & column recurrences => semiseparable generators
#    The code snippet below is partial: if you do a single global triple for P
#    and Q, you have alpha_P,beta_P,gamma_P and alpha_Q,beta_Q,gamma_Q. Then
#    you'd unify them into the typical 2x2 block form. For now, let's just show
#    a direct "Theorem 2.23 style" formula if you had alpha_k, beta_k, gamma_k,
#    delta_k, theta_k for a 2-term.
################################################################################

def convert_to_semiseparable_generators(alpha, beta, gamma, delta, theta):
    """
    Suppose we have the "Szegö-type" or "[EGO05]-type" recurrences in the form
       [ alpha_k   beta_k ]
       [ gamma_k   delta_k*x + theta_k ]
    for k=1..n-1, with a single global triple or something. Then Theorem 2.23 says:

      p_{k+1} = 1
      q_k = 1 / delta_k
      d_k = - theta_k / delta_k
      g_k = beta_k
      b_k = alpha_k
      h_k = - gamma_k / delta_k

    We'll just do a trivial example that returns them as scalars. In practice you'd do
    this "k" times. This snippet is purely a placeholder for demonstration.
    """
    p = 1.0
    q = 1.0 / delta
    d = -theta / delta
    g = beta
    b = alpha
    h = -gamma / delta
    return (p, q, d, g, b, h)


################################################################################
# 5) Implement a fast rank-1 semiseparable matvec in O(n) time
#
#    If your matrix is rank=r, you sum up r rank-1 components. Shown here for r=1.
################################################################################

def fast_semiseparable_matvec(p: torch.Tensor, q: torch.Tensor, d: torch.Tensor,
                              g: torch.Tensor, b: torch.Tensor, h: torch.Tensor,
                              x: torch.Tensor) -> torch.Tensor:
    """
    Suppose we have an n x n rank-1 semiseparable matrix with:
      diag => d[i],
      upper => for j>i => (prod_{k=i+1..j} p[k]) * q[i],
      lower => for i>j => g[i]* (prod_{k=j+1..i-1} b[k]) * h[j].

    We'll do it in O(n) using prefix sums & prefix products (no big python for-loops).
    EXACT formula for the rank=1 case:
      y[i] = d[i]*x[i]
              + sum_{j < i} [g[i]* bprod(j+1..i-1) * h[j] * x[j]]
              + sum_{j > i} [pprod(i+1..j)* q[i] * x[j]].

    We do:
      # subdiagonal part:
        y[i] += g[i]* b_cum[i-1] * sum_{j=0..i-1}[ (h[j]/b_cum[j]) * x[j] ]
      # diagonal:
        y[i] += d[i]* x[i]
      # upper part:
        y[i] += q[i]/ p_cum[i] * sum_{j=i+1..n-1}[ p_cum[j]* x[j] ]

    with safe handling of i=0 for subdiagonal, etc.
    """
    n = x.size(0)
    device = x.device
    dtype = x.dtype
    y = torch.zeros_like(x)

    # handle diagonal in one shot
    y = d * x

    # prefix product for p: p_cum[k] = p[0]*...p[k], define p_cum[-1] = 1
    # but we only need it from p[1], or so. We'll define:
    #   p_cum[0] = 1, p_cum[k] = p_cum[k-1]*p[k] for k >= 1
    # Then the product from i+1..j is p_cum[j]/ p_cum[i].
    p_cum = torch.cumprod(p, dim=0)  # shape (n,) if p has length n
    # but we might define a shift or unify. We'll just assume p[0],...p[n-1].
    # to handle i-based indexing carefully, let's define a small helper:
    # we'll define extended arrays with length n+1, so p_cum[k+1] = p_cum[k]*p[k]
    p_extended = torch.ones(n+1, dtype=dtype, device=device)
    p_extended[1:] = p
    p_cum = torch.cumprod(p_extended, dim=0)  # shape (n+1,)

    # define R[j] = p_cum[j+1]* x[j], (since product from 0..j is p_cum[j+1] if indexing)
    # Then sum_{j=i+1..n-1} product(...) = (1/p_cum[i+1]) * sum_{j=i+1..n-1} R[j].
    # We'll build R_j as p_cum[j+1]*x[j], do a cumsum, etc.
    R = p_cum[1:] * x  # shape (n,)
    S = torch.cumsum(R, dim=0)  # S[j] = sum_{u=0..j} R[u].

    # For the lower part, we do something similar with b:
    b_extended = torch.ones(n+1, dtype=dtype, device=device)
    b_extended[1:] = b
    b_cum = torch.cumprod(b_extended, dim=0)  # shape (n+1,)

    # define R'[j] = (h[j]/ b_cum[j]) * x[j]. Then prefix sum.
    # We'll define h_extended to match indexing if h is length n
    # We'll assume for simplicity: h[0..n-1] => h[j].
    # Then R'[j] = h[j]* x[j]/ b_cum[j], but we need j+1 or so.
    # We'll define b_cum[j] for j in [0..n], with b_cum[0]=1, b_cum[1] = b[0], etc.
    idxs = torch.arange(n, device=device)
    # h, g, d are shape (n,). We'll just do direct elementwise
    Rprime = h / b_cum[:n] * x  # shape (n,)
    Sprime = torch.cumsum(Rprime, dim=0)  # shape (n,)

    # now build y in O(n) by formula
    # y[i] already has d[i]*x[i].
    # add upper part => y[i] += q[i]/ p_cum[i] * [S[n-1] - S[i]]  for i < n
    # for i in python: y[i] += q[i]/p_cum[i+1] * (S[n-1] - S[i])
    # watch indexing carefully
    idx = torch.arange(n, device=device)
    # define something for sub / upper

    # subdiagonal => i>0 => y[i] += g[i]* b_cum[i] * Sprime[i-1]
    # note i=0 => subdiagonal sum is empty => 0
    # let's do them vectorized. We'll define a mask i>0
    mask_i_gt0 = (idx > 0)
    sub_term = torch.zeros(n, dtype=dtype, device=device)
    sub_term[mask_i_gt0] = g[mask_i_gt0] * b_cum[mask_i_gt0] * Sprime[mask_i_gt0 - 1]
    # add
    y += sub_term

    # upper => y[i] += q[i]/ p_cum[i] * sum_{j=i+1..n-1} R[j] = q[i]/p_cum[i+1]*(S[n-1]-S[i])
    # do it in 2 lines:
    denom = p_cum[idx + 1]  # shape(n,)
    upper_term = q/denom * (S[-1] - S)
    y += upper_term

    return y

################################################################################
# 6) Testing & Benchmarking on CPU or CUDA
################################################################################

def naive_matvec(Z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Naive O(n^2) multiply Z x. Just do a matmul for reference.
    """
    return Z @ x

def main():
    torch.set_printoptions(precision=4, sci_mode=True)

    # Basic config
    n = 256
    rank = 3

    print(f"=== 1) Build Z, compute displacement, factor rank={rank}")
    Z, D, P, Q = test_displacement_factor(n, rank=rank)

    print(f"=== 2) Build 3-term recurrences for rows (P) and columns (Q) ===")
    alphaP, betaP, gammaP = find_global_3term_recurrence(P)
    print(f"Row side: alpha={alphaP:.4f}, beta={betaP:.4f}, gamma={gammaP:.4f}")
    measure_3term_recurrence(P, alphaP, betaP, gammaP)

    alphaQ, betaQ, gammaQ = find_global_3term_recurrence(Q)
    print(f"Col side: alpha={alphaQ:.4f}, beta={betaQ:.4f}, gamma={gammaQ:.4f}")
    measure_3term_recurrence(Q, alphaQ, betaQ, gammaQ)

    # If we truly want the final (H,1) semiseparable form, we'd unify those row/column recurrences,
    # get the (delta_k x + theta_k) block, then do Theorem 2.23. For demonstration, let's
    # pretend we have a single set of (alpha,beta,gamma,delta,theta) for rank=1. We'll just pick some dummy:
    # (In practice, you'd do the actual 2x2 block method. This is purely to illustrate final usage.)
    alpha_demo = 0.5
    beta_demo  = -0.3
    gamma_demo = 0.1
    delta_demo = 1.2
    theta_demo = -0.8
    p_s, q_s, d_s, g_s, b_s, h_s = convert_to_semiseparable_generators(
        alpha_demo, beta_demo, gamma_demo, delta_demo, theta_demo
    )
    print(f"Dummy example semiseparable scalars:\n  p={p_s:.3g}, q={q_s:.3g}, d={d_s:.3g}, "
          f"g={g_s:.3g}, b={b_s:.3g}, h={h_s:.3g}")
    # Typically, each index k has its own alpha_k,... but let's skip that detail.

    # Now let's define p,q,d,g,b,h as length-n (rank-1) to test the fast matvec
    # We'll do a random fill for them that "looks" like each entry is constant-ish
    # In a real code path, you'd fill them from the actual recurrences. This is just to show the HPC approach.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving data to device={device} for a speed test.")
    n_big = 1 << 14  # 16384
    p_vec = torch.ones(n_big, dtype=torch.float32, device=device)
    q_vec = torch.ones(n_big, dtype=torch.float32, device=device) * 0.5
    d_vec = torch.ones(n_big, dtype=torch.float32, device=device) * 0.1
    g_vec = torch.ones(n_big, dtype=torch.float32, device=device) * 0.8
    b_vec = torch.ones(n_big, dtype=torch.float32, device=device) * 1.01
    h_vec = torch.ones(n_big, dtype=torch.float32, device=device) * -0.3

    # We'll also build a naive rank-1 semiseparable matrix on the GPU to compare times.
    # That matrix is M = diag(d) + strictly lower + strictly upper, shape (n_big,n_big).
    # But that's O(n^2) memory => 256MB for n=16384 if float32 => be careful.
    # We'll do it for a smaller n_test to show correctness. Then we'll time on large n only with the fast approach.

    n_test = 2048
    p_test = p_vec[:n_test].clone()
    q_test = q_vec[:n_test].clone()
    d_test = d_vec[:n_test].clone()
    g_test = g_vec[:n_test].clone()
    b_test = b_vec[:n_test].clone()
    h_test = h_vec[:n_test].clone()

    # build naive semiseparable matrix M_test to confirm correctness
    # M[i,i] = d[i],
    # M[i,j] = g[i]*(prod_{k=j+1..i-1} b[k])*h[j], for i>j
    # M[i,j] = (prod_{k=i+1..j} p[k]) * q[i], for i<j
    # We'll do it with a double-loop for demonstration. O(n^2) but n=2048 is borderline feasible.
    def build_naive_semiseparable(p, q, d, g, b, h):
        n_ = len(d)
        M_ = torch.zeros((n_, n_), dtype=d.dtype, device=d.device)
        # diagonal
        idx_ = torch.arange(n_, device=d.device)
        M_[idx_, idx_] = d
        # upper
        #   for each i in 0..n-1:
        #       partial product p_{i+1..j}
        for i_ in range(n_):
            # prefix so we don't keep re-multiplying
            # but let's just do a loop for clarity
            pprod = 1.0
            for j_ in range(i_+1, n_):
                pprod *= p[j_]
                M_[i_, j_] = pprod * q[i_]
            # done
        # lower
        for i_ in range(n_):
            bprod = 1.0
            for j_ in range(i_-1, -1, -1):
                # j_ < i_
                bprod *= b[j_+1] if (j_+1 < i_) else 1.0
                M_[i_, j_] = g[i_]* bprod* h[j_]
        return M_

    M_test = build_naive_semiseparable(p_test, q_test, d_test, g_test, b_test, h_test)
    # quick spot check with random x
    x_ = torch.randn(n_test, dtype=d_test.dtype, device=d_test.device)
    naive_out = M_test @ x_
    fast_out = fast_semiseparable_matvec(p_test, q_test, d_test, g_test, b_test, h_test, x_)
    diff_ = (naive_out - fast_out).norm().item() / naive_out.norm().item()
    print(f"[Correctness check on n={n_test}] rel diff = {diff_:.2e}")

    # Now let's do a big HPC benchmark. We won't build the big matrix because it's too large.
    # We'll just compare naive matmul on a synthetic NxN vs. the fast semiseparable approach:
    #   naive => we can't do NxN fully, but let's do an M @ x ignoring structure => O(n^2)
    #   fast => O(n) approach
    print("=== HPC Benchmark on large semiseparable matvec vs. naive O(n^2) approach ===")
    # We'll do a smaller "naive" dimension because 16384^2 = 268M floats which is somewhat large but might be possible on a big GPU
    # If you're truly "allergic to explicit loops," we can do expansions but that would blow up memory. We'll do the standard matmul approach.

    n_small = 4096
    # build a random NxN (like a normal matrix) for naive approach
    M_small = torch.randn((n_small, n_small), device=device, dtype=torch.float32)
    x_small = torch.randn(n_small, device=device, dtype=torch.float32)

    # warmup
    for _ in range(3):
        _ = M_small @ x_small

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(3):
        y_naive = M_small @ x_small
    torch.cuda.synchronize() if device == "cuda" else None
    naive_time = (time.time() - t0)/3
    print(f"[naive big matmul] n={n_small}, time={naive_time*1e3:.2f} ms => ~{2*n_small**2/naive_time/1e9:.3f} GF/s")

    # Now do the fast semiseparable approach on n_big. That is rank=1, so it's truly O(n).
    x_big = torch.randn(n_big, device=device, dtype=torch.float32)

    # warmup
    for _ in range(3):
        _ = fast_semiseparable_matvec(p_vec, q_vec, d_vec, g_vec, b_vec, h_vec, x_big)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(10):
        y_big = fast_semiseparable_matvec(p_vec, q_vec, d_vec, g_vec, b_vec, h_vec, x_big)
    torch.cuda.synchronize() if device == "cuda" else None
    semisep_time = (time.time() - t0)/10
    print(f"[fast semiseparable matvec] n={n_big}, rank=1 => time={semisep_time*1e3:.2f} ms => "
          f"~{(n_big/semisep_time)/1e6:.3f} M el/s")


if __name__ == "__main__":
    main()
