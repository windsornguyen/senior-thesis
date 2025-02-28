# TODO: Where do we use this file?

import os
import torch
import torch.nn as nn

from utils import torch_version_ge
from logging import init_logger, logger
from flash_stu.layers.attention_layer import AttentionLayer
from flash_stu.layers.stu_layer import STULayer
from losses.chunked_cross_entropy import ChunkedCrossEntropyLoss

init_logger()


def compile_model(
    model: nn.Module,
    verbose: bool = True,
) -> None:
    """
    Compiles a model in-place.

    On PyTorch nightlies, we use per-layer compiling to reduce compilation time.

    Args:
        model (nn.Module): The model to compile.
            Can be a TransformerDecoder or DeepFusionModel; in the latter case only
            the model's decoder will be compiled.
        verbose (bool): Whether to log compile info. Default: True
    Returns:
        None
    """
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    if torch_version_ge("2.5.0"):
        if verbose:
            logger.info("Compiling model layers with torch.compile...")
        for module in reversed(list(model.modules())):
            if isinstance(module, AttentionLayer) or isinstance(module, STULayer):
                module.compile(backend=backend)
    else:
        if verbose:
            logger.info(
                """
                Compiling full model with torch.compile...
                For faster compile times via per-layer compile, please run on PyTorch nightlies.
                """
            )
        model.compile(backend=backend)

def compile_loss(loss: nn.Module, verbose: bool = True) -> None:
    """
    Utility to compile and return loss function. If the loss function is chunked cross-entropy,
    we only compile the upcast + cross-entropy calculation, not the chunking. For other losses
    we compile the entire loss function.

    Args:
        loss (nn.Module): A loss function to compile.
        verbose (bool): Whether to log compile info. Default: True
    Returns:
        loss (nn.Module): loss with either entire module compiled or (in the case of
            CEWithChunkedOutputLoss) only the upcast and cross-entropy calculation compiled.
    """
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    if verbose:
        logger.info("Compiling loss with torch.compile...")
    if isinstance(loss, ChunkedCrossEntropyLoss):
        loss.compute_cross_entropy = torch.compile(loss.compute_cross_entropy, backend=backend)
    else:
        loss = torch.compile(loss, backend=backend)
    return loss
