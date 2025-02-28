import math
import time

import numpy as np
import torch


def poly_mul_x(poly):
    # Multiply polynomial by x: shift coefficients right by one index.
    return [0] + poly


def poly_scale(poly, factor):
    # Scale polynomial coefficients by factor.
    return [coef * factor for coef in poly]


def poly_sub(poly1, poly2):
    # Subtract poly2 from poly1; extend with zeros if necessary.
    length = max(len(poly1), len(poly2))
    result = []
    for i in range(length):
        coef1 = poly1[i] if i < len(poly1) else 0
        coef2 = poly2[i] if i < len(poly2) else 0
        result.append(coef1 - coef2)
    return result


def chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x)
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    T_nm2 = [1]  # T_0(x)
    T_nm1 = [0, 1]  # T_1(x)
    for _ in range(2, n + 1):
        # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        term = poly_mul_x(T_nm1)
        term = poly_scale(term, 2)
        T_n = poly_sub(term, T_nm2)
        T_nm2, T_nm1 = T_nm1, T_n
    return T_n


def normalized_chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x) normalized by 2**(n-1).
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    coeff = chebyshev_coeff(n)
    leading_term = coeff[-1]
    return [c / leading_term for c in coeff]


def integral(a, b, beta):
    # compute the integral of z^a * z_conj^b over the sector
    # r <= 1 and |arg(z)| <= beta
    if a == b:
        return 2 * beta / (a + b + 2)
    return 2 * np.sin((a - b) * beta) / ((a - b) * (a + b + 2))

def vectorized_integral(a, b, beta):
    # a and b are numpy arrays of the same shape.
    # Compute the integral in a vectorized way.
    diff = a - b
    denom = a + b + 2
    # Use np.where to handle the case when diff==0
    return np.where(diff == 0, 2 * beta / denom, 2 * np.sin(diff * beta) / (diff * denom))


def Z(n, beta, t):
    matrix_size = t - n
    poly_coeff = normalized_chebyshev_coeff(n)

    # poly_coeff = chebyshev_coeff(n)
    def compute_entry(i, j, n, beta):
        ans = 0
        for ii in range(n + 1):
            for jj in range(n + 1):
                if poly_coeff[ii] == 0 or poly_coeff[jj] == 0:
                    continue
                ans += poly_coeff[ii] * poly_coeff[jj] * integral(i + ii, j + jj, beta)
        return ans

    results = np.zeros((matrix_size, matrix_size), dtype=complex)
    for i in range(matrix_size):
        for j in range(matrix_size):
            results[i, j] = compute_entry(i, j, n, beta)
    return results


def Z_vectorized(n, beta, t):
    matrix_size = t - n
    poly = np.array(normalized_chebyshev_coeff(n))  # shape (n+1,)

    # Create index arrays for the (i, j) entries
    I = np.arange(matrix_size).reshape(matrix_size, 1, 1, 1)
    J = np.arange(matrix_size).reshape(1, matrix_size, 1, 1)

    # Create arrays for the summation indices over the polynomial coefficients
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    ii = ii.reshape(1, 1, n + 1, n + 1)
    jj = jj.reshape(1, 1, n + 1, n + 1)

    # For each matrix entry (i, j), we want to compute the integral at indices (i+ii, j+jj)
    A = I + ii  # shape: (matrix_size, matrix_size, n+1, n+1)
    B = J + jj  # shape: (matrix_size, matrix_size, n+1, n+1)

    int_vals = vectorized_integral(A, B, beta)  # shape: (matrix_size, matrix_size, n+1, n+1)

    # Compute the outer product of the Chebyshev coefficients.
    P = poly.reshape(n + 1, 1) * poly.reshape(1, n + 1)  # shape: (n+1, n+1)

    # Multiply elementwise and sum over the last two axes to get the (i, j) entry.
    Z = np.sum(int_vals * P, axis=(2, 3))
    return Z

# --- PyTorch vectorized & chunked implementation ---
def integral_torch(a, b, beta):
    """
    Compute the integral of z^a * z̄^b over the polar wedge:
      r ∈ [0, 1], θ ∈ [-beta, beta],
    in closed form:
      if a==b: 2*beta/(a+b+2)
      else:   2*sin((a-b)*beta)/((a-b)*(a+b+2))
    Here a and b are tensors (floats).
    """
    diff = a - b
    denom = a + b + 2
    # Use torch.where to handle the diff==0 case
    return torch.where(diff == 0, 2 * beta / denom, 2 * torch.sin(diff * beta) / (diff * denom))


def Z_pytorch(n, beta, t, chunk_size=256, device="cuda"):
    """
    Compute the matrix Z as in your code, using PyTorch.
    - n: degree of the Chebyshev polynomial.
    - beta: the imaginary-part bound (typically set to 1/(64*n*n)).
    - t: sequence length.
    - chunk_size: controls how many rows/columns are computed at once.
    - device: 'cuda' (or 'cpu').
    """
    matrix_size = t - n
    # Compute Chebyshev coefficients and convert to a torch tensor.
    poly_coeff = normalized_chebyshev_coeff(n)
    poly = torch.tensor(poly_coeff, dtype=torch.float32, device=device)  # shape: (n+1,)
    # Outer product of polynomial coefficients.
    P = poly.unsqueeze(1) * poly.unsqueeze(0)  # shape: (n+1, n+1)
    P = P.unsqueeze(0).unsqueeze(0)  # reshape to (1, 1, n+1, n+1) for broadcasting

    # Precompute the index arrays for the summation indices (ii, jj)
    ii = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    jj = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    ii, jj = torch.meshgrid(ii, jj, indexing="ij")  # shape: (n+1, n+1)
    ii = ii.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, n+1, n+1)
    jj = jj.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, n+1, n+1)

    # Allocate the result matrix
    Z = torch.empty((matrix_size, matrix_size), dtype=torch.complex64, device=device)

    # Process in chunks to save memory.
    for i_start in range(0, matrix_size, chunk_size):
        i_end = min(i_start + chunk_size, matrix_size)
        # Create i indices (as floats)
        i_vals = torch.arange(i_start, i_end, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        for j_start in range(0, matrix_size, chunk_size):
            j_end = min(j_start + chunk_size, matrix_size)
            j_vals = torch.arange(j_start, j_end, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            # Compute A and B for the chunk: shape (chunk_i, chunk_j, n+1, n+1)
            A = i_vals + ii  # effectively computes i + ii for each chunk element
            B = j_vals + jj
            # Compute the closed-form integral for each (i+ii, j+jj)
            int_vals = integral_torch(A, B, beta)
            # Multiply by P and sum over the polynomial indices to yield the (i,j) entry.
            chunk_Z = torch.sum(int_vals * P, dim=(2, 3))
            # Write the chunk to the result matrix.
            Z[i_start:i_end, j_start:j_end] = chunk_Z.to(torch.complex64)
    return Z


def benchmark_Z(t_values):
    results = []
    for t in t_values:
        n = math.ceil(math.log(t))
        beta = 1 / (64 * n * n)

        # warm up gpu
        _ = Z_pytorch(n, beta, t)
        torch.cuda.synchronize()

        # time numpy implementation
        start = time.perf_counter()
        res_numpy = Z(n, beta, t)
        numpy_time = time.perf_counter() - start

        # time pytorch implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        res_pytorch = Z_pytorch(n, beta, t)
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start

        # verify results match
        res_torch_from_numpy = torch.from_numpy(res_numpy).to(dtype=torch.complex64, device="cuda")
        max_diff = torch.max(torch.abs(res_torch_from_numpy - res_pytorch)).item()

        results.append(
            {
                "t": t,
                "n": n,
                "numpy_time": numpy_time,
                "pytorch_time": pytorch_time,
                "speedup": numpy_time / pytorch_time,
                "max_diff": max_diff,
            }
        )

        print(f"\nDimensions: t={t}, n={n}")
        print(f"NumPy time:  {numpy_time:.3f}s")
        print(f"PyTorch time: {pytorch_time:.3f}s")
        print(f"Speedup: {numpy_time/pytorch_time:.2f}x")
        print(f"Max difference: {max_diff:.2e}")

    return results


if __name__ == "__main__":
    # test increasing sizes
    Ts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = benchmark_Z(Ts)

    # print summary table
    print("\nSummary:")
    print("t\tn\tNumPy(s)\tPyTorch(s)\tSpeedup\tMax Diff")
    print("-" * 60)
    for r in results:
        print(
            f"{r['t']}\t{r['n']}\t{r['numpy_time']:.3f}\t{r['pytorch_time']:.3f}\t{r['speedup']:.2f}x\t{r['max_diff']:.2e}"
        )
