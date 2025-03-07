import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature
from torch._inductor.lowering import make_pointwise, register_lowering

# Some internal torch.compile details
from torch._inductor.virtualized import ops
from functools import partial


@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: Tensor) -> Tensor:
    return torch.tanh(inp)


@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


def _tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


class _TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

    @staticmethod
    def vmap(info, in_dims, x):
        return torch.tanh(x), 0


_tanh_approx = _TanhApprox.apply


def generate_tanh_softcap(soft_cap: int, approx: bool = False) -> _score_mod_signature:
    """Returns an tanh bias score_mod given the number of heads H

    Args:
        soft_cap: The soft cap value to use for normalizing logits
        approx: Whether to use the `tanh.approx.` ptx instruction

    Returns:
        tanh_softcap: score_mod
    """
    tanh = _tanh_approx if approx else torch.tanh

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return soft_cap * tanh(score / soft_cap)

    prefix = "tanh_softcap_approx" if approx else "tanh_softcap"
    tanh_softcap.__name__ = f"{prefix}_{soft_cap}"

    return tanh_softcap

def generate_alibi_bias(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        return score + bias

    return alibi_mod


def generate_tanh_softcap_alibi(H: int, soft_cap: float, approx: bool = False) -> _score_mod_signature:
    """Returns a combined ALiBi and tanh softcapping score_mod.

    Args:
        H (int): number of heads for ALiBi scaling
        soft_cap (float): the soft cap value for normalizing/logit clipping
        approx (bool): Whether to use the 'tanh.approx' PTX-based approximation

    Returns:
        A combined score_mod function that first applies ALiBi,
        then performs softcap + tanh (optionally approximate).
    """
    tanh_func = _tanh_approx if approx else torch.tanh

    def alibi_tanh_softcap(score, b, h, q_idx, kv_idx):
        # Compute ALiBi bias
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        score = score + bias

        # Apply softcap
        score = score / soft_cap

        # Apply tanh
        score = tanh_func(score)

        # Rescale by soft_cap
        score = score * soft_cap
        return score

    # Give the score_mod a unique name:
    if approx:
        alibi_tanh_softcap.__name__ = f"tanh_softcap_alibi_approx_{soft_cap}"
    else:
        alibi_tanh_softcap.__name__ = f"tanh_softcap_alibi_{soft_cap}"

    return alibi_tanh_softcap
