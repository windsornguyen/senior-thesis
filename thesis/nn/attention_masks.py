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
        return within_window & meets_dilation

    mode_str = "causal" if causal else "noncausal"
    dilated_sliding_window_mask.__name__ = (
        f"dilated_sliding_window_{window_size}_dilation_{dilation}_{mode_str}"
    )
    return dilated_sliding_window_mask


def main():
    """
    Demonstrates usage of each mask by printing attention grids. We include a few
    basic checks to ensure the masks behave as expected. We show both the causal
    and non-causal versions for the sliding window and dilated masks.
    """
    B, H = 1, 1
    Q_LEN, KV_LEN = 8, 8

    # coordinate grids
    q_idx = torch.arange(Q_LEN).unsqueeze(-1).expand(Q_LEN, KV_LEN)
    kv_idx = torch.arange(KV_LEN).unsqueeze(0).expand(Q_LEN, KV_LEN)

    print("= Causal Mask =")
    c_mask = causal_mask(B, H, q_idx, kv_idx)
    print(c_mask.int(), "\n")

    print("= Sliding Window (window_size=2, causal=True) =")
    sw_causal_fn = generate_sliding_window_mask(window_size=2, causal=True)
    sw_causal = sw_causal_fn(B, H, q_idx, kv_idx)
    print(sw_causal.int(), "\n")

    print("= Sliding Window (window_size=2, causal=False) =")
    sw_noncausal_fn = generate_sliding_window_mask(window_size=2, causal=False)
    sw_noncausal = sw_noncausal_fn(B, H, q_idx, kv_idx)
    print(sw_noncausal.int(), "\n")

    print("= Dilated Sliding Window (window_size=4, dilation=2, causal=True) =")
    ds_causal_fn = generate_dilated_sliding_window_mask(window_size=4, dilation=2, causal=True)
    ds_causal = ds_causal_fn(B, H, q_idx, kv_idx)
    print(ds_causal.int(), "\n")

    print("= Dilated Sliding Window (window_size=4, dilation=2, causal=False) =")
    ds_noncausal_fn = generate_dilated_sliding_window_mask(window_size=4, dilation=2, causal=False)
    ds_noncausal = ds_noncausal_fn(B, H, q_idx, kv_idx)
    print(ds_noncausal.int(), "\n")

    # Quick checks:
    # (1) Causal means no i < j
    assert torch.all(c_mask == (q_idx >= kv_idx)), "Causal mask mismatch!"
    # (2) For windowed masks with causal=True, check a random row
    i = 5
    row_sw = sw_causal[i]
    allowed_js = torch.where(row_sw)[0]
    if len(allowed_js) > 0:
        # difference i-j <= 2
        assert (i - allowed_js.min()) <= 2, "Window mismatch for sliding_window_mask(causal=True)."

    # (3) Dilated mask with causal=True should skip every other position if dilation=2
    i = 6
    row_ds = ds_causal[i]
    allowed_js = torch.where(row_ds)[0]
    for j in allowed_js:
        diff = i - j
        assert diff % 2 == 0, f"Dilation mismatch: got diff={diff}."

    print("All checks passed.")

if __name__ == "__main__":
    main()
