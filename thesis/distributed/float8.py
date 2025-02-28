"""
Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/float8.py

Note: Float8 experimental is intended to be ran under `torch.compile` for competitive performance
"""

import torch
import torch.nn as nn

from thesis.utils.pretraining_config import JobConfig
from thesis.distributed.parallelisms import ParallelDims
from thesis.utils.logger import logger


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

class Float8Handler:
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = False

        float8_config = job_config.distributed.float8
        if not float8_config.get("enable_float8_linear"):
            logger.info("Float8 linear is disabled.")
            return
        if not _is_sm89_or_later():
            logger.warning("Float8 is only supported on SM89 or later (H100+ GPUs).")
            return

        try:
            from torchao.float8 import Float8LinearConfig
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        # Mutate model in-place, replacing instances of torch.nn.Linear with Float8Linear
        enable_fsdp_float8_all_gather = (
            parallel_dims.dp_shard_enabled and float8_config.distributed.get("enable_fsdp_float8_all_gather")
        )
        self.config = Float8LinearConfig(enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather)
        self.enabled = True

        # For precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = (
            enable_fsdp_float8_all_gather
            and float8_config.precompute_float8_dynamic_scale_for_fsdp
        )
        
        logger.info("Float8 training enabled!")

    def convert_to_float8_training(self, model: nn.Module) -> None:
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that as of 01/22/2025, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return
        
        from torchao.float8 import convert_to_float8_training
        
        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=lambda mod, fqn: fqn != "output", # Replace all nn.Linear's except the output layer
        )
        print(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def maybe_precompute_float8_dynamic_scale_for_fsdp(self, model: nn.Module | list[nn.Module]) -> None:
        if not self.enabled or not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for model in models:
            precompute_float8_dynamic_scale_for_fsdp(model)
