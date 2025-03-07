import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    _score_mod_signature,
    flex_attention,
    create_block_mask,
    BlockMask,
)

from thesis.nn.attention_masks import causal_mask

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(f"Unable to import Triton-based flash attention: {e}. No alternative currently available.")


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    # For half the dimensions, build the scale factor:
    freq_seq = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (theta**freq_seq)

    # Outer product with positions
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)

    # Build a complex exponential e^{i * theta}
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    x is [B, n_heads, seq_len, head_dim_as_complex],
    so we want to broadcast freqs_cis from [max_seq_len, half_dim]
    to [1, 1, seq_len, half_dim].
    """
    seq_len = x.shape[2]
    freqs_cis = freqs_cis[:seq_len]  # slice down to current seq_len
    return freqs_cis.view(1, 1, seq_len, -1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert real -> complex by grouping last dim in pairs
    # shape => [B, n_heads, seq_len, head_dim//2, 2] => complex => [B, n_heads, seq_len, head_dim//2]
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast the frequencies to match [B, n_heads, seq_len, head_dim//2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    # Multiply => apply rotation
    xq_complex = xq_complex * freqs_cis
    xk_complex = xk_complex * freqs_cis

    # Convert back to real => shape [B, n_heads, seq_len, head_dim]
    xq_out = torch.view_as_real(xq_complex).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_complex).reshape(*xk.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    This module implements the SwiGLU function defined as:
    FFN_SwiGLU(x, W, V, W2) = (Swish_{1}(xW) ⊙ (xV))W2
    where ⊙ denotes the Hadamard product and Swish_{1} is the Swish function with β=1.

    See more: https://arxiv.org/pdf/2002.05202

    Note: The Swish function with β=1 is equivalent to PyTorch's SiLU function.

    Args:
        dim (int): Input and output dimension.
        h_dim (int): Hidden dimension.
        bias (bool, optional): If false, additive biases will not be learned.

    Attributes:
         v (nn.Module): Additional linear layer for feature transformation.
         w (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
    """

    def __init__(self, dim: int, h_dim: int, bias: bool = False, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.w = nn.Linear(dim, h_dim, bias=bias, dtype=dtype)
        self.v = nn.Linear(dim, h_dim, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(h_dim, dim, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.w2(F.gelu(self.w(x), approximate="tanh") * self.v(x))


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.c_proj.SCALE_INIT = 1

        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if config.use_alibi else None
        self.window_size = config.window_size
        self.softcap = config.softcap

        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

    def _generate_slopes(self, n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, num_heads: int, interpolation_factor: float = 0.25):
        # If n_heads is a power of 2, generate slopes directly
        if math.log2(num_heads).is_integer():
            slopes = self._generate_slopes(num_heads)
        else:
            # Get slopes for the nearest power of two
            n = nearest_power_of_two(num_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)

            # Generate extra slopes
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: num_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=torch.device("cuda"))
        slopes = slopes * interpolation_factor  # https://arxiv.org/pdf/2310.13017
        return slopes

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
    ) -> torch.Tensor:
        if x is not None:
            q = k = v = x
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")

        bsz, q_len, dim = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape

        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim)
        v = v.view(bsz, v_len, self.num_heads, self.head_dim)

        if self.alibi_slopes is None:  # Use either ALiBi or RoPE
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        y = flash_attn_func(  # https://arxiv.org/pdf/2307.08691
            q=q,
            k=k,
            v=v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),  # Set to config.seq_len if full attention
            alibi_slopes=self.alibi_slopes,  # https://arxiv.org/pdf/2108.12409
            softcap=self.softcap,  # https://arxiv.org/pdf/2408.00118
        )

        y = y.contiguous().view(bsz, q_len, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FlexAttention(nn.Module):
    """
    Generalized Multihead Attention and supports various attention masks.
    Supports Rotary Positional Embeddings.
    """

    def __init__(self, config, mask_mod=causal_mask, score_mod=None):
        """
        Initializes the Attention class.

        Args:
            dim (int): Embedding size.
            num_heads (int): Number of heads.
            mask_mod (Callable): Mask to modify attention scores, e.g. causal.
        """
        super().__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.mask_mod = mask_mod

        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)

        self.c_proj = nn.Linear(config.dim, config.dim)

    # TODO: Is this useful?
    def _get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
        def _mask_mod(b, h, q, kv):
            return mask_mod(b, h, q + offset, kv)

        return _mask_mod

    def _validate_inputs(self, x, q, k, v):
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        block_mask: BlockMask = None,
        score_mod: _score_mod_signature = None,
    ) -> torch.Tensor:
        if x is not None:
            q = k = v = x

        self._validate_inputs(x, q, k, v)

        bsz, q_len, _ = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape

        Q = self.wq(q).reshape(bsz, self.num_heads, q_len, self.head_dim)
        K = self.wk(k).reshape(bsz, self.num_heads, k_len, self.head_dim)
        V = self.wv(v).reshape(bsz, self.num_heads, v_len, self.head_dim)

        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis)

        output = flex_attention(Q, K, V, block_mask=block_mask, score_mod=score_mod)
        output = output.reshape(bsz, q_len, -1)
        output = self.c_proj(output)
        return output

class AttentionLayer(nn.Module):
    def __init__(self, config, mask_mod=causal_mask) -> None:
        super(AttentionLayer, self).__init__()
        self.mask_mod = mask_mod
        self.score_mod = None  # TODO: Every layer computes its own score_mod
        self.attn_norm = nn.RMSNorm(config.dim)
        # self.attn = FlexAttention(config=config, mask_mod=self.mask_mod, score_mod=self.score_mod)
        self.attn = Attention(config=config)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_scale * config.dim, config.bias, config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(  # Residual + (optional) dropout
            self.attn(
                x=self.attn_norm(x),  # Pre-norm
                freqs_cis=freqs_cis,  # Rotary position embeddings
                # block_mask=block_mask,  # Current input position (for the BlockMask)
            )
        )
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))  # Residual + MLP + (optional) dropout
        return x


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        bsz: int = 1,
        dim: int = 896,
        d_in: int = 4,  # input dimension
        d_out: int = 4,  # output dimension
        num_heads: int = 8,
        num_layers: int = 12,
        seq_len: int = 8192,
        window_size: int = 8192,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        softcap: float = 50.0,
        theta: float = 10_000.0,
        use_alibi: bool = False,  # Default to RoPE
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.dim = dim
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.window_size = window_size
        self.hidden_size = dim
        self.mlp_scale = mlp_scale
        self.intermediate_size = self.dim * self.mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.softcap = softcap
        self.theta = theta
        self.use_alibi = use_alibi
        self.torch_dtype = torch_dtype
        self.device = device


class Transformer(PreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config, mask_mod=causal_mask) -> None:
        super(Transformer, self).__init__(config)
        self.num_layers = config.num_layers
        assert (
            config.dim % config.num_heads == 0
        ), f"dim ({config.dim}) must be divisible by num_heads ({config.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.mask_mod = mask_mod

        # Initialize the block mask (can be overridden externally/dynamically)
        self.block_mask = create_block_mask(
            mask_mod=self.mask_mod,
            B=None,  # Broadcasted
            H=None,  # Broadcasted
            Q_LEN=config.seq_len,
            KV_LEN=config.seq_len,
            device=config.device,
        )

        # RoPE position embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                head_dim=self.head_dim,
                max_seq_len=config.seq_len,
                theta=config.theta,
            ),
            persistent=True,
        )

        self.in_proj = nn.Linear(config.d_in, config.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = AttentionLayer(config=config, mask_mod=self.mask_mod)
            self.layers.append(layer)

        self.norm = nn.RMSNorm(config.dim)
        self.out_proj = nn.Linear(config.dim, config.d_out, bias=config.bias)

        self.std = config.dim**-0.5
        self.apply(self._init_weights)  # TODO: Move this out for meta device logic?
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        block_mask: BlockMask | None = None,
        score_mod: _score_mod_signature | None = None,
    ) -> torch.Tensor:
        x = self.dropout(self.in_proj(x))

        if block_mask is None:
            block_mask = self.block_mask

        for layer in self.layers:
            x = layer(x=x, freqs_cis=self.freqs_cis, block_mask=block_mask)

        x = self.norm(x)
        y_hat = self.out_proj(x)

        return y_hat

    def _get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)


if __name__ == "__main__":
    config_path = "config.json"

    with open(config_path, "r") as f:
        config_data = json.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    torch_dtype = getattr(torch, config_data["torch_dtype"])
    print("Torch dtype:", torch_dtype)

    configs = TransformerConfig(
        bsz=config_data["bsz"],
        dim=config_data["dim"],
        num_heads=config_data["num_heads"],
        num_layers=config_data["num_layers"],
        seq_len=config_data["seq_len"],
        window_size=config_data["window_size"],
        mlp_scale=config_data["mlp_scale"],
        bias=config_data["bias"],
        dropout=config_data["dropout"],
        softcap=config_data["softcap"],
        theta=config_data["theta"],
        use_alibi=config_data["use_alibi"],
        torch_dtype=torch_dtype,
    )

    print("Configs:")
    for key, value in vars(configs).items():
        print(f"  {key}: {value}")

    model = Transformer(configs).to(device=device, dtype=torch_dtype)
    x = torch.randn(configs.bsz, configs.seq_len, configs.d_in, device=device, dtype=torch_dtype)
    outputs = model(x)

    print("Output shape:", outputs.shape)
    print("Sample output:", outputs[0, 0, :10])
