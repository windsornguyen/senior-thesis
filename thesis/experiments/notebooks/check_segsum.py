import torch
import time


def local_block_sum(Z_block: torch.Tensor, L_block: torch.Tensor) -> torch.Tensor:
    """
    Computes, for each time index t in a block of length L:
      out[t] = sum_{j=0}^{t} L_block[t, j] * Z_block[j]
    where:
      Z_block: (B, L, ...)   (extra dims are allowed)
      L_block: (B, L, L)     lower-triangular block of weights for the chunk.
    """
    B, L, *rest = Z_block.shape
    out = Z_block.new_zeros(Z_block.shape)
    for t in range(L):
        # For each t, sum over j=0...t
        out[:, t] = (Z_block[:, : t + 1] * L_block[:, t, : t + 1].unsqueeze(-1)).sum(dim=1)
    return out


def chunked_lower_tri_sum(
    Z: torch.Tensor,  # (B, T, ...); our signal to sum over.
    L: torch.Tensor,  # (T, T) lower-triangular mask with weights, or (B, T, T)
    block_len: int,
) -> torch.Tensor:
    """
    Computes H where for each time index i:
         H[i] = sum_{j <= i} L[i,j] * Z[j]
    using a chunking strategy.

    We assume T is divisible by block_len.

    The approach:
      1. Reshape T into (C, block_len) so that we process C chunks.
      2. For each chunk i:
         - Compute the local (diagonal) cumulative sum using the block
           L_chunks[:, i, :, i] and Z_chunks[:, i].
         - For each previous chunk c (< i), add the off-diagonal contribution:
           For each time t in chunk i, add:
             sum_{s in chunk c} L[i_global, c*block_len+s] * Z[c, s]
           This is done via an einsum over the weight block L_chunks[:, i, :, c, :]
           and the full Z from chunk c.
      3. Concatenate all chunks back into (B, T, ...).
    """
    B, T, *extra = Z.shape
    C = T // block_len  # number of chunks

    # Reshape Z to (B, C, block_len, ...)
    Z_chunks = Z.reshape(B, C, block_len, *extra)
    # L: if 2D, add batch dimension.
    if L.dim() == 2:
        L = L.unsqueeze(0).expand(B, -1, -1)
    # Reshape L to (B, C, block_len, C, block_len)
    L_chunks = L.reshape(B, C, block_len, C, block_len)

    H_chunks = []  # To collect per-chunk cumulative sums.

    for i in range(C):
        # (a) Diagonal block: within-chunk cumulative sum.
        local_L = L_chunks[:, i, :, i]  # (B, block_len, block_len)
        local_Z = Z_chunks[:, i]  # (B, block_len, ...)
        H_local = local_block_sum(local_Z, local_L)  # (B, block_len, ...)

        # (b) Off-diagonal: contributions from all previous chunks.
        off_diag = 0
        if i > 0:
            for c in range(i):
                # weight block from chunk c (source) to chunk i (target)
                # shape: (B, block_len, block_len) where for each row t in chunk i,
                # it gives weights for each time s in chunk c.
                weight = L_chunks[:, i, :, c, :]  # (B, block_len, block_len)
                # Get the entire previous chunk's Z, shape: (B, block_len, ...)
                Z_prev = Z_chunks[:, c]  # (B, block_len, ...)
                # Compute contribution: for each t, sum_{s} weight[t,s] * Z_prev[s]
                # Using einsum: "bts, bse -> bte" (with extra dims 'e').
                off_diag = off_diag + torch.einsum("bts,bse->bte", weight, Z_prev)
        H_i = H_local + off_diag
        H_chunks.append(H_i)
    # Concatenate along the chunk (time) dimension.
    H = torch.cat(H_chunks, dim=1)
    return H


def block_bmm(L_block: torch.Tensor, Z_block: torch.Tensor) -> torch.Tensor:
    """
    Multiply a (B, block_len, block_len) weight block by (B, block_len, D).
    We can just use torch.bmm; the 'lower triangular' structure doesn't matter
    correctness-wise. It's just extra zeros in L_block's upper part if it's truly triangular.
    """
    # L_block: (B, block_len, block_len)
    # Z_block: (B, block_len, D)
    # returns: (B, block_len, D)
    return torch.bmm(L_block, Z_block)


def chunked_lower_tri_sum_vectorized(
    Z: torch.Tensor,  # shape (B, T, D)
    L: torch.Tensor,  # shape (T, T) or (B, T, T), lower-triangular
    block_len: int,
) -> torch.Tensor:
    """
    Computes H where for each time index i:
        H[i] = sum_{j <= i} L[i, j] * Z[j]
    using chunking *and* batched block matrix multiplies (rather than a per-timestep loop).

    Steps:
      1) Split T -> C x block_len, so we have C chunks.
      2) For chunk i, the diagonal sub-block is lower triangular, but we can still
         multiply it by the chunk of Z in one bmm call (extra zeros in L_block's upper part
         won't affect correctness).
      3) For each chunk c < i, the mask sub-block is fully used (i.e. it's not triangular
         because we know i > c). So that's a standard block bmm as well.
      4) Accumulate all those partial block multiplications.

    Complexity is still O(C^2 * block_len^3), but it's fully vectorized inside each chunk-block
    operation—and often significantly faster on GPU than a per-timestep loop.
    """
    B, T, D = Z.shape
    C = T // block_len
    # (B, C, block_len, D)
    Z_chunks = Z.reshape(B, C, block_len, D)

    # If L is 2D, broadcast a batch dim.
    if L.dim() == 2:
        L = L.unsqueeze(0).expand(B, -1, -1)
    # (B, C, block_len, C, block_len)
    L_chunks = L.reshape(B, C, block_len, C, block_len)

    H_chunks = []
    for i in range(C):
        # 1) Sum over all previous chunks c < i
        off_diag = 0
        for c in range(i):
            L_block = L_chunks[:, i, :, c, :]  # (B, block_len, block_len)
            Z_block = Z_chunks[:, c]  # (B, block_len, D)
            off_diag += block_bmm(L_block, Z_block)

        # 2) Diagonal block
        L_diag = L_chunks[:, i, :, i, :]  # (B, block_len, block_len)
        Z_diag = Z_chunks[:, i]  # (B, block_len, D)
        diag_part = block_bmm(L_diag, Z_diag)

        # 3) Combine
        H_chunks.append(off_diag + diag_part)

    H = torch.cat(H_chunks, dim=1)  # shape (B, T, D)
    return H


def naive_lower_tri_sum(Z: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Naively computes H[i] = sum_{j<=i} L[i,j] * Z[j]
    for Z: (B, T, ...) and L: (T, T) or (B, T, T).
    """
    B, T, *extra = Z.shape
    if L.dim() == 2:
        L = L.unsqueeze(0).expand(B, -1, -1)
    H = Z.new_zeros(Z.shape)
    for i in range(T):
        H[:, i] = (Z[:, : i + 1] * L[:, i, : i + 1].unsqueeze(-1)).sum(dim=1)
    return H


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Parameters for the test:
    B = 1  # Batch size
    T = 8  # Sequence length
    block_len = 2  # Block size (assume T % block_len == 0)
    E = 1  # Feature dimension

    # Create a simple input Z of shape (B, T, E) on the selected device.
    Z = torch.arange(T, dtype=torch.float32, device=device).reshape(B, T, E)

    # Create a simple lower-triangular mask L on the device.
    L_full = torch.zeros(T, T, dtype=torch.float32, device=device)
    for i in range(T):
        for j in range(i + 1):
            L_full[i, j] = float(i - j + 1)
    L_full = torch.tril(L_full)

    # Compute H using all three approaches
    H_naive = naive_lower_tri_sum(Z, L_full)
    H_chunked = chunked_lower_tri_sum(Z, L_full, block_len)
    H_vectorized = chunked_lower_tri_sum_vectorized(Z, L_full, block_len)

    # Print out the results
    print("Input Z:")
    print(Z.squeeze(-1))
    print("\nLower-Triangular Mask L:")
    print(L_full)
    print("\nNaively computed H (cumulative sums):")
    print(H_naive.squeeze(-1))
    print("\nChunked computed H (cumulative sums):")
    print(H_chunked.squeeze(-1))
    print("\nVectorized computed H (cumulative sums):")
    print(H_vectorized.squeeze(-1))
    print("\nDifferences:")
    print("Naive - Chunked:")
    print(H_naive.squeeze(-1) - H_chunked.squeeze(-1))
    print("Naive - Vectorized:")
    print(H_naive.squeeze(-1) - H_vectorized.squeeze(-1))
    print("Chunked - Vectorized:")
    print(H_chunked.squeeze(-1) - H_vectorized.squeeze(-1))

    # Check if all implementations give the same results within numerical precision
    naive_chunked_close = torch.allclose(H_naive, H_chunked, rtol=1e-5, atol=1e-5)
    naive_vectorized_close = torch.allclose(H_naive, H_vectorized, rtol=1e-5, atol=1e-5)
    chunked_vectorized_close = torch.allclose(H_chunked, H_vectorized, rtol=1e-5, atol=1e-5)

    print("\nAll implementations match?")
    print(f"Naive vs Chunked: {naive_chunked_close}")
    print(f"Naive vs Vectorized: {naive_vectorized_close}")
    print(f"Chunked vs Vectorized: {chunked_vectorized_close}")

    # Benchmarking the three approaches on larger input sizes.
    print("\nStarting benchmarks on larger input sizes...")

    B_bench = 16
    T_bench = 1024
    D_bench = 64
    block_len_bench = 64  # T_bench should be divisible by block_len_bench

    # Create random benchmark inputs on the device
    Z_bench = torch.rand(B_bench, T_bench, D_bench, device=device)
    L_bench = torch.tril(torch.rand(T_bench, T_bench, device=device))

    def benchmark_function(fn, name, warmups=10, iterations=50):
        # Warmup runs
        for _ in range(warmups):
            fn()
        time.sleep(1)  # power throttling
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"{name}: {avg_time_ms:.3f} ms per iteration")
        time.sleep(1)  # more throttling

    print("\nBenchmarks:")
    benchmark_function(lambda: naive_lower_tri_sum(Z_bench, L_bench), "Naive approach")
    benchmark_function(lambda: chunked_lower_tri_sum(Z_bench, L_bench, block_len_bench), "Chunked approach")
    benchmark_function(
        lambda: chunked_lower_tri_sum_vectorized(Z_bench, L_bench, block_len_bench), "Vectorized approach"
    )
