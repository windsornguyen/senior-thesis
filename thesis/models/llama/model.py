import torch
import torch.nn as nn

from transformers import PreTrainedModel

from attn import Attention
from attn_layer import AttentionLayer
from config import TransformerConfig
from utils import nearest_power_of_two


import torch

from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    model_type = "Transformer"

    def __init__(
        self,
        bsz: int = 1,
        n_embd: int = 1536,
        n_heads: int = 8,
        num_layers: int = 26,
        seq_len: int = 8192,
        window_size: int = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        softcap: float = 50.0,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = n_embd
        self.intermediate_size = n_embd * mlp_scale
        self.hidden_act = "swish"
        self.bias = bias
        self.dropout = dropout
        self.softcap = softcap
        self.dtype = dtype


class Transformer(PreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config) -> None:
        super(Transformer, self).__init__(config)
        self.num_layers = config.num_layers
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)

        # TODO: Use RoPE or ALiBi eventually
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, dtype=config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(AttentionLayer(config))

        self.norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias, dtype=config.torch_dtype)

        # Tie weights
        self.tok_emb.weight = self.lm_head.weight

        self.std = (config.n_embd) ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        y_hat = self.lm_head(x)

        return y_hat

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        if self.tok_emb.weight is not self.lm_head.weight:
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)


import torch
import torch.nn as nn

from attn import Attention
from mlp import MLP


class AttentionLayer(nn.Module):
    def __init__(self, n_embd, n_heads, dtype=torch.float32) -> None:
        super(AttentionLayer, self).__init__()
        self.attn = Attention(n_embd, n_heads)
        self.mlp = MLP(config, dtype=dtype)
        self.attn_norm = nn.RMSNorm(n_embd, dtype=dtype)
        self.mlp_norm = nn.RMSNorm(n_embd, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


import torch.nn as nn

from torch.nn import functional as F


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

def get_num_flops_per_tok(num_params: int, model_config, seq_len) -> int:
    sl, h, q, t = model_config.num_layers, model_config.n_heads, model_config.dim // model_config.n_heads, seq_len

    """
    Reason why we have factor of 12 for self-attention part of formula:
    (1). Each self-attention has 2 matmuls in the fwd and 4 in the bwd (+6).
    (2). Flash Attention does 1 more matmul recomputation in the bwd,
         but we shouldn't count recomputation in calculating MFU. (+0)
    (3). Each matmul performs 1 multiplication and 1 addition (*2).
    (4). We follow the convention and do not account for sparsity in causal attention.
    """
    flops_per_tok = 6 * num_params + 12 * sl * h * q * t
    return flops_per_tok

def reshape_for_broadcast(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Reshape a frequency tensor to broadcast with a target tensor.

    This function slices the given frequency tensor and reshapes it to align
    with the sequence length and last dimension of the target tensor 'x',
    enabling proper broadcasting during element-wise operations.

    The input 'freqs_cis' must have shape (max_seq_len, dim), and 'dim' must
    match the last dimension of 'x'. Only the first 'seq_len' entries are used.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor of shape (max_seq_len, dim).
        x (torch.Tensor): Target tensor for which broadcasting is needed.

    Returns:
        torch.Tensor: The frequency tensor reshaped for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "Input tensor 'x' must have at least 2 dimensions."

    seq_len = x.shape[1]
    freqs_cis = freqs_cis[:seq_len]
    assert freqs_cis.shape == (
        seq_len,
        x.shape[-1],
    ), f"freqs_cis shape {freqs_cis.shape} does not match (seq_len={seq_len}, dim={x.shape[-1]})."

    shape = [dim if i == 1 or i == ndim - 1 else 1 for i, dim in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to input tensors
    using the given frequency tensor.

    Args:
        xq (torch.Tensor): Query tensor of shape (bsz_q, seq_len_q, dim_q).
        xk (torch.Tensor): Key tensor of shape (bsz_k, seq_len_k, dim_k).
        freqs_cis (torch.Tensor): Frequency tensor used for rotary embedding.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing the transformed
        query and key tensors with rotary embeddings applied.
    """
    bsz_q, seq_len_q, _ = xq.shape
    bsz_k, seq_len_k, _ = xk.shape

    # Reshape inputs as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(bsz_q, seq_len_q, -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(bsz_k, seq_len_k, -1, 2))

    # Reshape frequency tensor for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Compute the complex Hadamard products, then convert back to real
    xq_prod = torch.view_as_real(xq_ * freqs_cis)
    xk_prod = torch.view_as_real(xk_ * freqs_cis)

    # Flatten to original shape: (bsz, seq_len, dim/2, 2) -> (bsz, seq_len, dim)
    xq_out = xq_prod.flatten(3)
    xk_out = xk_prod.flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
