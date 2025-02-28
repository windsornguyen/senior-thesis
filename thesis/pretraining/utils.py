import math

import torch

from collections.abc import Iterable
from itertools import chain

from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from thesis.utils.logger import logger


def torch_version_ge(version: str) -> bool:
    """
    Check if torch version is greater than or equal to the given version.

    Args:
        version (str): The torch version to compare against

    Returns:
        bool: True if torch version is greater than or equal to the given version.

    Example:
        >>> print(torch.__version__)
        2.4.0
        >>> torch_version_ge("2.0")
        True
    """
    return version in torch.__version__ or torch.__version__ >= version


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


@torch.no_grad()
def check_model_value_range(model: torch.nn.Module, range: float = 1e3, std: float = 1e3):
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        # Handle DTensors if necessary
        if isinstance(param, DTensor):
            param = param.to_local()

        if param.numel() == 0:
            print(f"Model parameter {name} is empty, probably because of FSDP sharding")
            continue

        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Model parameter {name} contains NaN or Inf")

        param_range = param.max() - param.min()
        param_std = param.std()

        # Check for large range
        if param_range > range:
            print(
                f"Model parameter {name} has a suspiciously large range ({param_range}): "
                "please check initialization and ensure init_weights is defined and called"
            )

        # Check for large std
        if param_std > std:
            print(
                f"Model parameter {name} has a suspiciously large standard deviation ({param_std}): "
                "please check initialization and ensure init_weights is defined and called"
            )

        # TODO: Check this condition.
        # Check if all zeros. It's okay for biases to be zero at init.
        if (param == 0).all():
            if "bias" in name:
                # Zero-initialized biases are common practice and generally not an issue.
                pass
            else:
                logger.warning(
                    f"Model parameter {name} is all zeros: it might be because of a missing initialization. "
                    "Ensure that this is desired behavior."
                )

@torch.no_grad()
def clip_grad_norm_(
    params: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dims.
    We will need to manually reduce the gradient norms across PP stages.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach)
    return total_norm
