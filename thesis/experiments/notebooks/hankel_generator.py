import torch

def build_hankel_generator(seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build generator lookup tables for a Hankel matrix via partial fraction decomposition.

    Given a sequence length, this function constructs three 1D lookup tables (a, b, c)
    corresponding to the partial fraction representation of the Hankel matrix defined as:

        Z[i, j] = 2 / ((i+j)^3 - (i+j))
                = 1/(i+j-1) - 2/(i+j) + 1/(i+j+1)

    Here, i and j are treated as 1-indexed. The lookup tables are indexed by k = i+j and are
    defined as:
        - a[k] = 1/(k-1)
        - b[k] = -2/k
        - c[k] = 1/(k+1)

    We allocate arrays with indices up to 2*seq_len+1 to ensure all needed values are covered.

    Parameters:
        seq_len (int): The length of the sequence, determining that the Hankel matrix will be 
                       of size (seq_len x seq_len).

    Returns:
        tuple: A tuple (a, b, c) of torch.Tensor objects that serve as the generator lookup tables.
    """
    max_index = 2 * seq_len + 2  # Allocate indices up to 2*seq_len+1
    a = torch.zeros(max_index, dtype=torch.float32)
    b = torch.zeros(max_index, dtype=torch.float32)
    c = torch.zeros(max_index, dtype=torch.float32)
    
    # Populate the lookup tables for indices k in [2, 2*seq_len]
    for k in range(2, 2 * seq_len + 1):
        a[k] = 1.0 / (k - 1)
        b[k] = -2.0 / k
        c[k] = 1.0 / (k + 1)
    
    return a, b, c


def hankel_entry(i: int, j: int, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    """
    Compute the (i, j)-th entry of the Hankel matrix using generator lookup tables.

    This function uses the lookup tables derived from the partial fraction decomposition:
        Z[i, j] = 1/(i+j-1) - 2/(i+j) + 1/(i+j+1)
    where i and j are 1-indexed. The entry is obtained by summing the appropriate values
    from the tables a, b, and c at index k = i + j.

    Parameters:
        i (int): The 1-indexed row number.
        j (int): The 1-indexed column number.
        a (torch.Tensor): Lookup table for the term 1/(k-1).
        b (torch.Tensor): Lookup table for the term -2/k.
        c (torch.Tensor): Lookup table for the term 1/(k+1).

    Returns:
        float: The computed value for the Hankel matrix entry Z[i, j].
    """
    k = i + j  # k is the sum index for the lookup tables
    return a[k] + b[k] + c[k]


def build_hankel_matrix(seq_len: int, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Build the full Hankel matrix from the generator lookup tables.

    Constructs a (seq_len x seq_len) Hankel matrix where each entry is computed using the
    formula:
        Z[i, j] = a[i+j] + b[i+j] + c[i+j]
    with i and j treated as 1-indexed. This function iterates over all (i, j) pairs and
    uses the hankel_entry function to compute each element.

    Parameters:
        seq_len (int): The number of rows (and columns) of the Hankel matrix.
        a (torch.Tensor): Lookup table for the term 1/(i+j-1).
        b (torch.Tensor): Lookup table for the term -2/(i+j).
        c (torch.Tensor): Lookup table for the term 1/(i+j+1).

    Returns:
        torch.Tensor: A (seq_len x seq_len) tensor representing the Hankel matrix.
    """
    Z = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    for i in range(1, seq_len + 1):
        for j in range(1, seq_len + 1):
            Z[i - 1, j - 1] = hankel_entry(i, j, a, b, c)
    return Z
