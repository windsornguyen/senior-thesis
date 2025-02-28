import copy

from collections.abc import Callable

import torch
import torch.nn as nn

from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from thesis.distributed.parallelisms import ParallelDims
from thesis.utils.logger import logger
from thesis.utils.pretraining_config import JobConfig
from thesis.models.mamba import MambaConfig
from thesis.distributed.parallelisms.pipelining_utils import (
    build_pipeline_schedule,
    generate_split_points,
    stage_ids_this_rank,
)

DeviceType = int | str | torch.device

def pipeline_mamba(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: MambaConfig,
    loss_fn: Callable[..., torch.Tensor],
) -> None:
    """
    Applies pipeline parallelism to the Mamba model.
    """
    stages, models = pipeline_mamba_manual_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        parallel_dims=parallel_dims,
        job_config=job_config,
        device=device,
        model_config=model_config,
        loss_fn=loss_fn,
    )
    pp_schedule = build_pipeline_schedule(job_config, stages, models)
    return pp_schedule, models

def pipeline_mamba_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: MambaConfig,
    loss_fn: Callable[..., torch.Tensor],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API extracts one torch.nn.Module object for the part of the model configured
    to run inside this stage.
    
    It wraps the model chunk in a ManualPipelineStage object and returns both the
    stage and model objects.
    
    The stage object is used to create a pipeline schedule, and the model object
    can be used for applying SPMD (Single Program, Multiple Data) parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    splits = job_config.distributed.parallelism.get("pipeline_parallel_split_points") or generate_split_points(job_config, parallel_dims.pp, model_config)

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.tok_emb = None

        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # We keep the layers in a contiguous region between [start, stop)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]
        
        if not is_last:
            model.norm = None
            model.output = None
        
        
        stage = PipelineStage(
            submodule=model,
            stage_idx=stage_idx,
            num_stages=num_stages,
            device=device,
            group=pp_mesh.get_group("pp")
        )
        return stage, model
    
    num_stages = len(splits) + 1
    stage_idx = pp_rank
    
    stages, models = [], []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx == 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx=stage_idx,
            start_layer=start_layer,
            stop_layer=stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"Pipeline parallelism rank {pp_rank} is building stage index {stage_idx} "
            "with start_layer {start_layer}, stop_layer {stop_layer}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models
