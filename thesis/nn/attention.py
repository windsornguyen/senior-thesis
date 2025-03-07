import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from thesis.nn.attention_masks import causal_mask


class Attention(nn.Module):
    """
    Generalized Multihead Attention and supports various attention masks.
    """
    def __init__(self, dim, num_heads, mask_mod=causal_mask):
        """
        Initializes the Attention class.

        Args:
            dim (int): Embedding size.
            num_heads (int): Number of heads.
            mask_mod (Callable): Mask to modify attention scores, e.g. causal.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible num_heads ({num_heads})"
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)
        self.mask_mod = mask_mod

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None
    ) -> torch.Tensor:
        if x is not None:
            q = k = v = x
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")

        bsz, q_len, _ = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape

        Q = self.wq(q).reshape(bsz, self.num_heads, q_len, self.head_dim)
        K = self.wk(k).reshape(bsz, self.num_heads, k_len, self.head_dim)
        V = self.wv(v).reshape(bsz, self.num_heads, v_len, self.head_dim)

        block_mask = create_block_mask(
            mask_mod=self.mask_mod,
            B=bsz,
            H=self.num_heads,
            Q_LEN=q_len,
            KV_LEN=k_len,
            device=q.device,
        )

        output = flex_attention(Q, K, V, block_mask=block_mask)
        output = output.reshape(bsz, q_len, self.dim)
        output = self.o_proj(output)
        return output
