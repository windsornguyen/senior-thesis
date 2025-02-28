import torch


def get_hankel(
    seq_len: int, use_hankel_L: bool = False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    """
    Constructs a Hankel matrix where the (i, j)-th entry is defined as:
      Z[i, j] = 2 / ((i+j+1)^3 - (i+j+1))
    for 0-indexed i, j.

    """
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64, device=device)  # Changed to float64
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def compute_full_rank(matrix: torch.Tensor, atol: float = 1e-13) -> int:  # Tightened tolerance
    """Computes the full numerical rank of a matrix using SVD."""
    # Get singular values
    s = torch.linalg.svdvals(matrix)
    # More robust rank computation
    thresh = atol * s[0].item()  # Reference against largest singular value
    return torch.sum(s > thresh).item()


def compute_semiseparable_rank(matrix: torch.Tensor, atol: float = 1e-13) -> int:  # Tightened tolerance
    """
    Computes the semiseparable rank with CUDA optimization and improved numerical stability.
    """
    n = matrix.shape[0]
    max_off_diag_rank = 0
    for k in range(1, n):
        off_diag_block = matrix[k:, :k]
        # Get singular values
        s = torch.linalg.svdvals(off_diag_block)
        if len(s) > 0:  # Check if block is non-empty
            thresh = atol * s[0].item()  # Reference against largest singular value
            block_rank = torch.sum(s > thresh).item()
            max_off_diag_rank = max(max_off_diag_rank, block_rank)
    return max_off_diag_rank


if __name__ == "__main__":
    # Set default device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_dtype(torch.float64)  # Set default dtype to float64

    widths = [32, 64, 128, 256, 512, 1024, 2048]
    print("\nWidth | Full Rank | Semiseparable Rank")
    print("---------------------------------------")

    # Warm up GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for n in widths:
        H = get_hankel(n, device=device)
        full_rank = compute_full_rank(H)
        semi_rank = compute_semiseparable_rank(H)
        print(f"{n:5d} | {full_rank:9d} | {semi_rank:18d}")

    # Ensure all GPU ops are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
