from thesis.distributed.parallelisms.parallel_dims import ParallelDims
# from thesis.distributed.parallelisms.parallelize_flash_stu import parallelize_flash_stu
from thesis.distributed.parallelisms.parallelize_mamba import parallelize_mamba
# from thesis.distributed.parallelisms.pipeline_flash_stu import pipeline_flash_stu
from thesis.distributed.parallelisms.pipeline_mamba import pipeline_mamba


__all__ = [
    "models_parallelize_fns",
    "models_pipelining_fns",
    "ParallelDims",
]

models_parallelize_fns = {
    # "flash_stu": parallelize_flash_stu,
    "mamba": parallelize_mamba,
}

models_pipelining_fns = {
    # "flash_stu": pipeline_flash_stu,
    "mamba": pipeline_mamba,
}
