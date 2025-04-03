import torch
from torch.nn.attention.flex_attention import _mask_mod_signature

def causal_mask(
    batch_size: int,
    num_heads: int,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor
) -> torch.Tensor:
    """
    Returns a boolean tensor indicating which positions in the attention matrix
    are valid for causal (autoregressive) attention. By default, it's True for
    positions (i, j) where i >= j.

    Args:
        batch_size (int): Batch size (unused here).
        num_heads (int): Number of heads (unused here).
        q_idx (torch.Tensor): Tensor indexing the query positions.
        kv_idx (torch.Tensor): Tensor indexing the key/value positions.

    Returns:
        torch.Tensor: A boolean tensor where True indicates that the query at
        position i can attend to the key at position j, respecting i >= j.
    """
    return q_idx >= kv_idx


def generate_sliding_window_mask(window_size: int, causal: bool = True) -> _mask_mod_signature:
    """
    Creates a sliding window mask function.

    If `causal=True`, each query token at position i can attend only to tokens j 
    in [i - window_size, i].  
    If `causal=False`, each query token i can attend to any token j in 
    [i - window_size, i + window_size], i.e. a symmetric window of size `window_size`.

    Args:
        window_size (int): The maximum distance from i that i can attend to.
        causal (bool): Whether to enforce causal ordering (i >= j). Defaults to True.

    Returns:
        _mask_mod_signature: A callable mask function that takes 
        (batch_size, num_heads, q_idx, kv_idx) and returns a boolean tensor
        indicating allowed attention connections.
    """
    def sliding_window_mask(
        batch_size: int,
        num_heads: int,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        If causal is True:
            within_window = (q_idx - kv_idx) <= window_size, and q_idx >= kv_idx.
        If causal is False:
            within_window = abs(q_idx - kv_idx) <= window_size.
        """
        if causal:
            # standard “look back” window
            distance = q_idx - kv_idx
            within_window = (distance >= 0) & (distance <= window_size)
        else:
            # symmetrical window around i
            distance = (q_idx - kv_idx).abs()
            within_window = distance <= window_size
        return within_window

    name_ext = "causal" if causal else "noncausal"
    sliding_window_mask.__name__ = f"sliding_window_{window_size}_{name_ext}"
    return sliding_window_mask


def generate_dilated_sliding_window_mask(
    window_size: int,
    dilation: int,
    causal: bool = True
) -> _mask_mod_signature:
    """
    Creates a dilated sliding window mask function. 

    If `causal=True`, each query token i can attend tokens j in [i - window_size, i]
    such that (i - j) % dilation == 0.  
    If `causal=False`, each query token i can attend tokens j in [i - window_size, 
    i + window_size] for which |i - j| % dilation == 0.

    Args:
        window_size (int): The maximum distance from i to j (backwards if causal=True, 
                           otherwise symmetric around i).
        dilation (int): The stride for skipping positions.
        causal (bool): Whether to enforce causal ordering (i >= j). Defaults to True.

    Returns:
        _mask_mod_signature: A callable mask function that takes
        (batch_size, num_heads, q_idx, kv_idx) and returns a boolean tensor
        indicating allowed attention connections.
    """
    def dilated_sliding_window_mask(
        batch_size: int,
        num_heads: int,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        If causal is True:
            distance = q_idx - kv_idx
            0 <= distance <= window_size and distance % dilation == 0.
        If causal is False:
            distance = (q_idx - kv_idx).abs()
            distance <= window_size and distance % dilation == 0.
        """
        if causal:
            distance = q_idx - kv_idx
            within_window = (distance >= 0) & (distance <= window_size)
        else:
            distance = (q_idx - kv_idx).abs()
            within_window = distance <= window_size

        meets_dilation = (distance % dilation) == 0
        mask = within_window & meets_dilation
        return mask

    mode_str = "causal" if causal else "noncausal"
    dilated_sliding_window_mask.__name__ = (
        f"dilated_sliding_window_{window_size}_dilation_{dilation}_{mode_str}"
    )
    return dilated_sliding_window_mask

def generate_top_k_mask(k: int, sim: torch.Tensor, causal: bool = True) -> _mask_mod_signature:
    # TODO: Make this block-wise for better memory access patterns, per Native Sparse Attention paper?

    """
    Creates a top-k mask function that selects, for each query, the top k key indices
    based on the provided similarity scores.

    Args:
        k (int): Number of top entries to select per query.
        sim (torch.Tensor): A tensor of shape [B, H, Q_LEN, KV_LEN, D] containing similarity scores.
        causal (bool): Whether to enforce causal masking (Defaults to True).

    Returns:
        _mask_mod_signature: A callable mask function that takes (batch_idx, head_idx, q_idx, kv_idx)
        and returns a boolean tensor of shape [Q_LEN, KV_LEN] (which attn_gym will combine to [B, H, Q_LEN, KV_LEN]).
    """
    # Precompute top-k indices along the key dimension.
    # Shape: [B, H, Q_LEN, k]
    topk_indices = sim.topk(k, dim=-1)[1]
    B, H, Q_LEN, KV_LEN = sim.shape

    def top_k_mask(b, h, q_idx, kv_idx):
        # Here, q_idx and kv_idx are expected to be of shape [B, H, Q_LEN, KV_LEN].
        # Expand kv_idx to [B, H, Q_LEN, KV_LEN, 1] and topk_indices to [B, H, Q_LEN, 1, k]
        mask = (kv_idx.unsqueeze(-1) == topk_indices.unsqueeze(3)).any(dim=-1)
        if causal:
            mask = mask & (q_idx >= kv_idx)
        return mask

    top_k_mask.__name__ = f"top_k_mask_{k}"
    return top_k_mask


def main():
    B, H = 1, 1
    Q_LEN, KV_LEN = 8, 8

    # Build coordinate grids of shape [B, H, Q_LEN, KV_LEN]:
    dummy_q_idx = torch.arange(Q_LEN, dtype=torch.int).view(1, 1, Q_LEN, 1).expand(B, H, Q_LEN, KV_LEN)
    dummy_kv_idx = torch.arange(KV_LEN, dtype=torch.int).view(1, 1, 1, KV_LEN).expand(B, H, Q_LEN, KV_LEN)

    print("= Causal Mask =")
    c_mask = causal_mask(B, H, dummy_q_idx, dummy_kv_idx)
    print(c_mask.int(), "\n")

    print("= Sliding Window (window_size=2, causal=True) =")
    sw_causal_fn = generate_sliding_window_mask(window_size=2, causal=True)
    sw_causal = sw_causal_fn(B, H, dummy_q_idx, dummy_kv_idx)
    print(sw_causal.int(), "\n")

    print("= Sliding Window (window_size=2, causal=False) =")
    sw_noncausal_fn = generate_sliding_window_mask(window_size=2, causal=False)
    sw_noncausal = sw_noncausal_fn(B, H, dummy_q_idx, dummy_kv_idx)
    print(sw_noncausal.int(), "\n")

    print("= Dilated Sliding Window (window_size=4, dilation=2, causal=True) =")
    ds_causal_fn = generate_dilated_sliding_window_mask(window_size=4, dilation=2, causal=True)
    ds_causal = ds_causal_fn(B, H, dummy_q_idx, dummy_kv_idx)
    print(ds_causal.int(), "\n")

    print("= Dilated Sliding Window (window_size=4, dilation=2, causal=False) =")
    ds_noncausal_fn = generate_dilated_sliding_window_mask(window_size=4, dilation=2, causal=False)
    ds_noncausal = ds_noncausal_fn(B, H, dummy_q_idx, dummy_kv_idx)
    print(ds_noncausal.int(), "\n")

    # Quick checks...
    assert torch.all(c_mask == (dummy_q_idx >= dummy_kv_idx)), "Causal mask mismatch!"

    # === Top-k Mask Tests ===
    # For top-k, our API expects coordinate grids of shape [B, H, Q_LEN, KV_LEN].
    # We'll create those same dummy grids.
    print("= Top-k Mask (k=3, non-causal) =")
    k = 3
    # Create similarity scores with values increasing along the key axis:
    # For each query, the top 3 keys (ignoring causality) will be the last 3 indices.
    importance_scores = torch.arange(KV_LEN, dtype=torch.float).unsqueeze(0).unsqueeze(0) \
                         .expand(B, H, Q_LEN, KV_LEN)
    topk_fn_noncausal = generate_top_k_mask(k, importance_scores, causal=False)
    topk_mask_noncausal = topk_fn_noncausal(0, 0, dummy_q_idx, dummy_kv_idx)
    print(topk_mask_noncausal.int(), "\n")

    for i in range(Q_LEN):
        expected = torch.zeros(KV_LEN, dtype=torch.bool)
        expected[-k:] = True
        assert torch.all(topk_mask_noncausal[0, 0, i] == expected), f"Non-causal top-k mask mismatch in query {i}"

    print("= Top-k Mask (k=3, causal) =")
    # For causal, build a similarity tensor where for each query i, only keys with j <= i get high scores.
    importance_scores_causal = torch.empty(Q_LEN, KV_LEN)
    for i in range(Q_LEN):
        for j in range(KV_LEN):
            importance_scores_causal[i, j] = float(j) if j <= i else -1000.0
    importance_scores_causal = importance_scores_causal.unsqueeze(0).unsqueeze(0)  # [B, H, Q_LEN, KV_LEN]
    topk_fn_causal = generate_top_k_mask(k, importance_scores_causal, causal=True)
    topk_mask_causal = topk_fn_causal(0, 0, dummy_q_idx, dummy_kv_idx)
    print(topk_mask_causal.int(), "\n")

    for i in range(Q_LEN):
        expected = torch.zeros(KV_LEN, dtype=torch.bool)
        if i < k:
            expected[: i+1] = True
        else:
            expected[i - k + 1: i+1] = True
        assert torch.all(topk_mask_causal[0, 0, i] == expected), f"Causal top-k mask mismatch in query {i}"

    print("All tests passed.")

if __name__ == "__main__":
    main()
