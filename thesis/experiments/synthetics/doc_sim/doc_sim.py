import numpy as np

from typing import Tuple


def generate_document_similarity(
    num_examples: int = 256,
    num_documents: int = 1024,
    *,
    seed: int = 1746,
    dtype: str | np.dtype = "float32",
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Generate ``num_examples`` independent datasets, each containing
    ``num_documents`` binary vectors.
    
    Exactly _one_ pair in each dataset is orthogonal; all other inner products
    are strictly positive. This mirrors the worst‑case inputs used in
    Alman & Yu (2024) to prove quadratic‑time lower bounds.

    Args:
        num_examples (int): Number of independent datasets to create.
        num_documents (int): Number of documents (vectors) per dataset (must be ≥ 2).
        seed (int, optional): Seed for reproducibility. Defaults to 1337.
        dtype (str | np.dtype, optional): Output dtype for the document tensor
            (0/1 cast to this type). Defaults to "float32".

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - docs (np.ndarray): Binary documents (0/1) cast to dtype. 
                Shape: (num_examples, num_documents, dim)
            - pairs (np.ndarray): Sorted indices of the unique orthogonal 
                pair in each example. Shape: (num_examples, 2), dtype: int64
    """
    if num_documents < 2:
        raise ValueError("`num_documents` must be at least 2.")
    if num_examples < 1:
        raise ValueError("`num_examples` must be positive.")

    rng = np.random.default_rng(seed)
    dim = (num_documents - 1).bit_length()           # dim = ⌈log₂ n⌉

    # Allocate binary docs in uint8 for bitwise ops
    docs  = rng.integers(0, 2, size=(num_examples, num_documents, dim),
                         dtype=np.uint8)
    pairs = np.empty((num_examples, 2), dtype=np.int64)

    for b in range(num_examples):
        # Choose i ≠ j and make j the bitwise complement of i
        i = rng.integers(num_documents)
        j = (i + rng.integers(1, num_documents)) % num_documents
        docs[b, j] = 1 - docs[b, i]
        pairs[b]   = np.sort([i, j])

        # Ensure uniqueness of the orthogonal pair
        while True:
            prod = (docs[b] & docs[b][:, None]).sum(-1)   # Hamming inner product
            np.fill_diagonal(prod, 1)
            prod[pairs[b, 0], pairs[b, 1]] = prod[pairs[b, 1], pairs[b, 0]] = 1
            if (prod == 0).any():                         # accidental extra pair
                docs[b] = rng.integers(0, 2,
                                        size=(num_documents, dim),
                                        dtype=np.uint8)
                docs[b, j] = 1 - docs[b, i]               # re‑impose complement
            else:
                break

    return docs.astype(dtype), pairs
