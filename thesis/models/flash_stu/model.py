import math

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig

from thesis.nn.stu import STU
from thesis.nn.mlp import MLP
from thesis.utils.modules.attn import Attention
from thesis.utils import nearest_power_of_two


def get_hankel(n: int, use_hankel_L: bool = False) -> torch.Tensor:
    """
    Generates a Hankel matrix Z, as defined in Equation (3) of 2312.06837.

    Args:
        n (int): Size of the square Hankel matrix.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    entries = torch.arange(1, n + 1)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k**0.25
    filters = phi_k.to(device)
    return filters


def compute_dimensions(n: int) -> tuple[int, int, int]:
    if n <= 2:
        raise ValueError("n must be greater than 2")

    T_prime = (math.ceil(math.sqrt(n - 2))) ** 2 + 2
    sqrt_T_prime = math.ceil(math.sqrt(T_prime - 2))
    k_max = sqrt_T_prime
    return T_prime, sqrt_T_prime, k_max


def get_tensorized_spectral_filters(
    n: int = 8192,
    k: int = 24,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute tensorized spectral filters for given sequence length and filter count.

    Args:
        n: Sequence length
        k: Number of filters
        use_hankel_L: Hankel_main ⊗ Hankel_L? Default is Hankel_main ⊗ Hankel_main.
        device: Computation device
        dtype: Computation dtype
    """
    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    Z = get_hankel(sqrt_T_prime)
    sigma, phi = torch.linalg.eigh(Z)
    phi_i = phi[:, -k:] * sigma[-k:] ** 0.25
    phi_j = phi_i

    filters = torch.kron(phi_i, phi_j)
    return filters.to(device=device, dtype=dtype)


class FlashSTUConfig(PretrainedConfig):
    model_type = "FlashSTU"

    def __init__(
        self,
        bsz: int = 1,
        n_embd: int = 1536,
        n_heads: int = 8,
        num_layers: int = 26,
        seq_len: int = 8192,
        window_size: int = 1024,
        vocab_size: int = 200064,  # De facto d_in/d_out
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        num_eigh: int = 24,
        k_u: int = 3,
        use_hankel_L: bool = False,
        use_flash_fft: bool = True,
        use_approx: bool = True,
        use_attn: bool = True,
        softcap: float = 50.0,
        torch_dtype: torch.dtype = torch.bfloat16,
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
        self.num_eigh = num_eigh
        self.k_u = k_u  # TODO: Get rid of these
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_approx = use_approx
        self.use_attn = use_attn
        self.softcap = softcap
        self.torch_dtype = torch_dtype


class FlashSTU(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config: FlashSTUConfig, phi: torch.Tensor) -> None:
        super(FlashSTU, self).__init__(config)
        self.num_layers = config.num_layers
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.phi = phi
        self.use_tensordot = config.use_tensordot

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, dtype=config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STUBlock(config, self.phi, self.n))
            else:
                self.layers.append(AttentionBlock(config) if config.use_attn else STUBlock(config, self.phi, self.n))

        self.norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)

        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias, dtype=config.torch_dtype)
        self.tok_emb.weight = self.output.weight

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


class STUBlock(nn.Module):
    """
    STU block combining attention and feed-forward layers.

    Attributes:
        stu (nn.Module): Spectral Transform Unit layer (STU).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        stu_norm (nn.Module): Layer normalization for STU.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, config, phi, n):
        super(STUBlock, self).__init__()
        self.stu = STU(config, phi, n)
        self.ffn = MLP(config, dtype=config.torch_dtype)
        self.stu_norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.ffn_norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.stu_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class AttentionBlock(nn.Module):
    """
    Attention block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Spectral Transform Unit layer (STU).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for STU.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, config) -> None:
        super(AttentionBlock, self).__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)
        self.ffn = MLP(config, dtype=config.torch_dtype)
        self.attn = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
