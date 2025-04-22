import torch
from torch.utils.data import TensorDataset

# ───────────────────────────────── reference (slow & simple) ─────────────────
def make_orthogonal_pair_dataset(B:int, n:int, *, dtype=torch.float32, seed:int=0) -> TensorDataset:
    """Naive O(B·n²) implementation – easy to follow, no clever tricks."""
    torch.manual_seed(seed)
    d = torch.ceil(torch.log2(torch.tensor(n, dtype=torch.float64))).int().item()

    docs  = torch.randint(0, 2, (B, n, d), dtype=dtype)
    pairs = torch.empty((B, 2), dtype=torch.long)

    for b in range(B):
        # pick a random (i,j) and force j = ¬i
        i = torch.randint(n, ())
        j = (i + torch.randint(1, n, ())) % n
        docs[b, j] = 1.0 - docs[b, i]
        pairs[b]   = torch.sort(torch.tensor([i, j]))[0]  # store (min,max)

        # resample if accidental extra orthogonal pair
        while True:
            prod = docs[b] @ docs[b].T      # [n,n]
            prod.fill_diagonal_(1.0)        # ignore diagonal
            prod[pairs[b,0], pairs[b,1]] = 1.0
            prod[pairs[b,1], pairs[b,0]] = 1.0
            if (prod == 0).any():
                docs[b] = torch.randint(0, 2, (n, d), dtype=dtype)
                docs[b, pairs[b,1]] = 1.0 - docs[b, pairs[b,0]]
                continue
            break

    return TensorDataset(docs, pairs)
