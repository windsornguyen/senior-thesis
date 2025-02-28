import torch
from torch import nn
from typing import Tuple, Optional
from attention import Attention
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attention_masks import causal_mask


class KVCache(nn.Module):
    """
    Efficient key/value cache for transformer inference with support for two modes:
    - Standard KV caching: Stores full key/value tensors for attention computation
    - Latent-compressed caching: Stores compressed representations (for memory efficiency)
      with optional rotary positional embedding components

    Inspired by DeepSeek-V2's latent caching approach while maintaining compatibility
    with standard attention implementations.

    Example:
    >>> # Standard caching for 2 sequences with 8 attention heads
    >>> cache = KVCache(
    ...     batch_size=2,
    ...     max_seq_len=2048,
    ...     num_kv_heads=8,
    ...     head_dim=64,
    ...     dtype=torch.float16
    ... )

    >>> # Compressed caching with latent representations
    >>> compressed_cache = KVCache(
    ...     batch_size=2,
    ...     max_seq_len=2048,
    ...     num_kv_heads=8,
    ...     head_dim=64,
    ...     dtype=torch.float16,
    ...     compressed=True,
    ...     latent_dim=32,  # 4x compression
    ...     rope_dim=16
    ... )
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        compressed: bool = False,
        latent_dim: int = 0,
        rope_dim: int = 0,
    ) -> None:
        """
        Initialize empty cache buffers.

        Args:
            batch_size: Maximum batch size the cache can handle
            max_seq_len: Maximum sequence length the cache can store
            num_kv_heads: Number of key/value attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for cache storage
            compressed: Enable memory-optimized latent caching
            latent_dim: Dimension of compressed representations (compressed=True only)
            rope_dim: Dimension for rotary positional embeddings (compressed=True only)
        """
        super().__init__()

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.compressed = compressed
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim

        # Single integer tracks write position (more efficient than array-based approaches)
        self.register_buffer("_pos_buffer", torch.zeros(1, dtype=torch.long), persistent=False)

        # Standard KV buffers (always allocated but may remain empty in compressed mode)
        self.register_buffer(
            "k_cache", torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype), persistent=False
        )

        # Compressed storage buffers (conditionally allocated)
        if self.compressed:
            self.register_buffer(
                "latent_cache",
                torch.zeros(batch_size, num_kv_heads, max_seq_len, latent_dim, dtype=dtype),
                persistent=False,
            )
            if self.rope_dim > 0:
                self.register_buffer(
                    "rope_cache",
                    torch.zeros(batch_size, num_kv_heads, max_seq_len, rope_dim, dtype=dtype),
                    persistent=False,
                )

    def reset(self) -> None:
        """
        Reset cache to initial empty state in-place. O(1) operation.

        This:
        - Zeros all storage buffers (preserving pre-allocated memory)
        - Resets write position to start of cache
        - Maintains dtype/device properties
        """
        with torch.no_grad():
            self.k_cache.zero_()
            self.v_cache.zero_()
            if self.compressed:
                self.latent_cache.zero_()
                if self.rope_dim > 0:
                    self.rope_cache.zero_()
            self._pos_buffer.zero_()

    @property
    def size(self) -> int:
        """
        Current number of tokens stored in cache. This is the offset where the next
        write will begin (0 <= size <= max_seq_len).
        """
        return self._pos_buffer.item()

    def update(
        self,
        k_val: Optional[torch.Tensor] = None,
        v_val: Optional[torch.Tensor] = None,
        latent_val: Optional[torch.Tensor] = None,
        rope_val: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Update cache with new token(s) in-place. Operation is O(1) per parameter updated.

        Args:
            k_val: Full key tensor [B, H, S, D] (standard mode required)
            v_val: Full value tensor [B, H, S, D] (standard mode required)
            latent_val: Compressed latent tensor [B, H, S, L] (compressed mode)
            rope_val: Rotary positional components [B, H, S, R] (compressed mode)

        Returns:
            (k_out, v_out):
                - In standard mode: References to full k_cache/v_cache
                - In compressed mode: (None, None) unless k_val/v_val were provided

        Raises:
            ValueError: On cache overflow or missing required arguments for mode
        """
        seq_len = 0
        if k_val is not None:
            seq_len = k_val.shape[2]
        if latent_val is not None:
            seq_len = max(seq_len, latent_val.shape[2])

        if seq_len == 0:
            raise ValueError("Must provide at least one valid input tensor")

        write_pos = self.size
        if (write_pos + seq_len) > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: {write_pos} + {seq_len} > {self.max_seq_len}. "
                "Consider increasing max_seq_len or truncating inputs."
            )

        with torch.no_grad():
            # Standard KV path (requires full tensors)
            if not self.compressed:
                if k_val is None or v_val is None:
                    raise ValueError("Standard mode requires both k_val and v_val")
                self.k_cache[:, :, write_pos : write_pos + seq_len] = k_val
                self.v_cache[:, :, write_pos : write_pos + seq_len] = v_val

            # Compressed path (accepts latent/rope tensors, optionally standard too)
            else:
                if latent_val is not None:
                    self.latent_cache[:, :, write_pos : write_pos + seq_len] = latent_val
                if self.rope_dim > 0 and rope_val is not None:
                    self.rope_cache[:, :, write_pos : write_pos + seq_len] = rope_val

                # Allow hybrid storage (compressed + standard)
                if k_val is not None:
                    self.k_cache[:, :, write_pos : write_pos + seq_len] = k_val
                if v_val is not None:
                    self.v_cache[:, :, write_pos : write_pos + seq_len] = v_val

            self._pos_buffer.add_(seq_len)

        # Only return standard cache refs if they were updated
        return (self.k_cache, self.v_cache) if (k_val is not None and v_val is not None) else (None, None)


def test_kv_cache_correctness():
    """
    We'll do a side-by-side test:
    1) use the standard Attention module in a naive (no cache) manner
       by feeding the entire sequence at once.
    2) use the same module but feed tokens one by one, storing them in
       a KVCache. Then do partial attention steps with a proper causal mask.

    If the KVCache is correct, by the time we've processed the entire
    sequence, the final output should match the naive approach's output.
    """
    print("=== Testing KVCache correctness ===")

    batch_size = 1
    seq_len = 8
    d_model = 16
    n_heads = 4

    # seed for repeatability
    torch.manual_seed(1746)

    # define a random input
    x = torch.randn(batch_size, seq_len, d_model)
    # define our attention block
    attn = Attention(d_model, n_heads, mask_mod=causal_mask)

    # 1) naive forward over the entire sequence
    naive_out = attn(x)  # shape [B, T, d_model]

    # 2) forward one token at a time with the KVCache
    #    the output after the last token is processed should match naive_out
    attn_cached = Attention(d_model, n_heads, mask_mod=causal_mask)
    # copy weights from attn -> attn_cached so they match exactly
    attn_cached.load_state_dict(attn.state_dict())

    # create a KVCache for storing the keys and values
    # for this demonstration, we do normal caching (compressed=False).
    kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=seq_len,
        num_kv_heads=n_heads,
        head_dim=(d_model // n_heads),
        dtype=x.dtype,
        compressed=False,
    )

    # run a loop: for each token, we'll treat that as the 'new' token
    # and do attention over [0..t]. We'll store the output for the current token.
    out_cached = []
    for t in range(seq_len):
        # slice the current token
        cur_x = x[:, t : t + 1, :]

        # produce K/V for just this token
        k_t = attn_cached.wk(cur_x).reshape(batch_size, n_heads, 1, d_model // n_heads)
        v_t = attn_cached.wv(cur_x).reshape(batch_size, n_heads, 1, d_model // n_heads)
        kv_cache.update(k_val=k_t, v_val=v_t)

        # produce Q for this token
        q_t = attn_cached.wq(cur_x).reshape(batch_size, n_heads, 1, d_model // n_heads)

        # retrieve all K/V from the cache up to position t
        K_full = kv_cache.k_cache[:, :, : (t + 1)]  # shape [B, H, t+1, head_dim]
        V_full = kv_cache.v_cache[:, :, : (t + 1)]

        # let's replicate the same "flex_attention" call, but for Q_LEN=1, KV_LEN=(t+1)
        block_mask = create_block_mask(
            mask_mod=causal_mask, B=batch_size, H=n_heads, Q_LEN=1, KV_LEN=t + 1, device=x.device
        )

        # flex_attention expects the 4D Q, K, V
        out_t = flex_attention(q_t, K_full, V_full, block_mask=block_mask)
        # out_t shape: [B, H, 1, head_dim]
        out_t = out_t.reshape(batch_size, 1, d_model)

        # reproject
        out_t = attn_cached.o_proj(out_t)  # [B, 1, d_model]

        out_cached.append(out_t)

    out_cached = torch.cat(out_cached, dim=1)  # [B, T, d_model]

    # compare naive_out vs out_cached
    diff = (naive_out - out_cached).abs().max()
    print(f"Max difference = {diff.item():.6f}")

    # they won't be 0 due to floating point differences, but should be very close
    # if caching is correct
    assert diff < 1e-6, "Cached attention output does not match naive attention output!"
    print("KVCache correctness test passed!\n")


def test_kv_cache_memory():
    """
    We'll measure memory usage of:
    1) normal KV caching
    2) compressed KV caching

    We won't do a big training run, but just allocate for a large seq_len
    and compare memory footprints for demonstration.
    """
    print("=== Testing KVCache memory usage ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 2
    seq_len = 1024
    d_model = 64
    n_heads = 4

    def measure_cache_usage(compressed):
        torch.cuda.reset_peak_memory_stats(device)
        cache = KVCache(
            batch_size=batch_size,
            max_seq_len=seq_len,
            num_kv_heads=n_heads,
            head_dim=(d_model // n_heads),
            dtype=torch.float16,
            compressed=compressed,
            latent_dim=16 if compressed else 0,
            rope_dim=16 if compressed else 0,
        ).to(device)
        # do a fake update for, say, 256 tokens
        single_token = torch.randn(batch_size, d_model, dtype=torch.float16, device=device)
        k_val = single_token.reshape(batch_size, n_heads, 1, d_model // n_heads)
        v_val = k_val.clone()
        for t in range(256):
            cache.update(k_val=k_val, v_val=v_val)
        mem_usage = torch.cuda.max_memory_allocated(device) / (1024**2)
        return mem_usage

    if device == "cpu":
        print("No CUDA device available; skipping memory usage test.")
        return

    mem_no_comp = measure_cache_usage(compressed=False)
    mem_comp = measure_cache_usage(compressed=True)
    print(f"[No compression] Peak GPU usage: {mem_no_comp:.2f} MB")
    print(f"[Compressed]    Peak GPU usage: {mem_comp:.2f} MB")
    print("Memory test complete!\n")


def main():
    test_kv_cache_correctness()
    test_kv_cache_memory()


if __name__ == "__main__":
    main()
