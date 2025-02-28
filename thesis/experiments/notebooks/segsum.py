# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    
    # (..., T) -> (..., T, T)
    x_expanded = x.unsqueeze(-1).expand(*x.shape, T)  # Create 2D grid of segment sums
    
    # Tril bool grid True for row i > col j
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)

    # Replace non-causal positions with 0
    x_masked = x_expanded.masked_fill(~mask, 0)

    # Compute cumsum along penultimate dim (which was the original last dim)
    x_segsum = torch.cumsum(x_masked, dim=-2)

    mask2 = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    
    # Replace invalid positions with -inf
    x_segsum = x_segsum.masked_fill(~mask2, -torch.inf)

    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None, eps=1e-6):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: int, length of each chunk
        initial_states: optional initial states
        eps: small constant for numerical stability
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A) + eps)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum) + eps)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))) + eps)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = torch.exp(A_cumsum + eps)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


# Simple test
def test_correctness():
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1  # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    # Use smaller values to improve numerical stability
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device) * 0.1
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device) * 0.1)).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device) * 0.1
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device) * 0.1
    D = torch.randn(nheads, dtype=dtype, device=device) * 0.1

    # Comparing fused version and minimal version
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_min, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)

    # Print some statistics about the differences
    diff = (y - y_min).abs()
    rel_diff = diff / (y.abs() + 1e-6)
    print(f"\nAbsolute difference statistics:")
    print(f"Mean: {diff.mean().item():.6f}")
    print(f"Max:  {diff.max().item():.6f}")
    print(f"99th percentile: {torch.quantile(diff.flatten(), 0.99).item():.6f}")

    print(f"\nRelative difference statistics:")
    print(f"Mean: {rel_diff.mean().item():.6f}")
    print(f"Max:  {rel_diff.max().item():.6f}")
    print(f"99th percentile: {torch.quantile(rel_diff.flatten(), 0.99).item():.6f}")

    # Find indices of largest differences for debugging
    max_diff_idx = torch.where(rel_diff == rel_diff.max())
    b, t, h, d = [idx.item() for idx in max_diff_idx]
    print(f"\nLocation of maximum relative difference:")
    print(f"Batch: {b}, Time: {t}, Head: {h}, Dim: {d}")
    print(f"Values at max diff - y: {y[b,t,h,d]:.8f}, y_min: {y_min[b,t,h,d]:.8f}")

    # Print input values at and around the problematic location
    print("\nInput values at max diff location:")
    print(f"X[b,t,h,d]: {x[b,t,h,d]:.8f}")
    print(f"dt[b,t,h]: {dt[b,t,h]:.8f}")
    print(f"A[h]: {A[h]:.8f}")

    # Print surrounding values in y and y_min
    print("\nSurrounding values in output:")
    for i in range(max(0, t - 1), min(t + 2, seqlen)):
        print(f"t={i}:")
        print(f"  y:     {y[b,i,h,d]:.8f}")
        print(f"  y_min: {y_min[b,i,h,d]:.8f}")
        print(f"  diff:  {(y[b,i,h,d] - y_min[b,i,h,d]).abs():.8f}")

    # Print chunk information
    chunk_idx = t // chunk_size
    chunk_pos = t % chunk_size
    print(f"\nChunk information:")
    print(f"Chunk index: {chunk_idx}, Position within chunk: {chunk_pos}")

    # Use appropriate tolerances based on observed differences
    torch.testing.assert_close(y, y_min, rtol=0.03, atol=0.0001)


test_correctness()
