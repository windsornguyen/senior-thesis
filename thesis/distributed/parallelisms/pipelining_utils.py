import os

from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from thesis.utils.logger import logger
from thesis.utils.pretraining_config import JobConfig

# TODO: Implement the DeepSeek-V3 DualPipe algorithm

def generate_split_points(job_config: JobConfig, pp_dim: int, model_config) -> list[str]:
    """
    Generate a default split point based on the number of layers and
    pipeline parallel dimension.
    """
    schedule_class = get_schedule_class(job_config.distributed.parallelism.get("pipeline_parallel_schedule"))
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but we'll set
        # set it as the default if no pipeline split is specified.
        num_stages_per_rank = 2
    else:
        raise ValueError(f"Unsupported pipeline schedule: {job_config.distributed.parallelism.get("pipeline_parallel_schedule")}")
    
    total_stages = pp_dim * num_stages_per_rank
    num_layers = model_config.num_layers
    if total_stages > num_layers:
        raise ValueError(
            f"Total number of pipelining stages cannot be greater than the number of layers. "
            f"Got {total_stages} > num_layers {num_layers}"
        )
    
    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages
    
    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # To minimize pipeline bubble times, middle stages get
            # an extra layer if there are any remaining.
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))

    logger.warning(
        f"No 'pipeline_parallel_split_points' provided, so the generated splits are {splits}. "
        "This may be sub-optimal as the number of layers per stage may be imbalanced."
    )
    return splits

def build_pipeline_schedule(job_config, stages, loss_fn):
    pp_schedule_csv = job_config.distributed.parallelism.get("pipeline_parallel_schedule_csv")

    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(
            job_config.distributed.parallelism.get("pipeline_parallel_schedule")
        )
    
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    logger.info(f'Using pipeline schedule {job_config.distributed.parallelism.get("pipeline_parallel_schedule")}')

    n_microbatches = job_config.distributed.parallelism.get("pipeline_parallel_microbatches")

    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.distributed.parallelism.get("pipeline_parallel_degree") * len(stages)

    if n_microbatches is None:
        n_microbatches = num_total_stages
    elif n_microbatches < num_total_stages:
        logger.warning(f"Number of microbatches ({n_microbatches}) is less than the total number "
                       f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )
    
    pp_schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    
    logger.info(f"Using pipeline schedule {job_config.distributed.parallelism.get("pipeline_parallel_schedule")} "
                f"with {n_microbatches} and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in (
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ), "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multi-stage), "
        "and _PipelineScheduleRuntime support CSV schedules."
        pp_schedule._load_csv(pp_schedule_csv)
    
    return pp_schedule

def stage_ids_this_rank(pp_rank: int, pp_size: int, num_stages: int, style: str = "loop") -> tuple[int]:
    """
    Compute the stage ids for the stages that will run on this pipeline parallel rank
    for either a looped or V-style schedule.
    """
    assert num_stages % pp_size == 0, f"num_stages ({num_stages}) must be divisible by pp_size ({pp_size})"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert stages_per_rank == 2, f"V-style schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1), strict=True))
    return stage_v_pairs[pp_rank]
