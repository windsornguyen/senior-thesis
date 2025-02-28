"""Adapted from https://github.com/facebookresearch/lingua/blob/main/lingua/distributed.py"""

import atexit
import contextlib
import json
import os
import random
import shutil
import tempfile
import signal
import socket
import subprocess
import sys

import numpy as np
import psutil
import torch
import torch.multiprocessing as mp

from collections.abc import Generator, Callable
from dataclasses import asdict, dataclass
from functools import lru_cache, reduce
from itertools import chain

from torch import distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

from thesis.utils.logger import logger
from thesis.utils.pretraining_config import JobConfig

from datetime import timedelta
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn

# TODO: Do we want this file as standalone functions or as a `DistributedTrainingManager` class?

# ----------------------------------------------------------------------------------------
# 1. Dataclasses and global config objects
# ----------------------------------------------------------------------------------------


# TODO: Do we use this anywhere? How would we?
@dataclass
class EnvironmentArgs:
    ENABLE_INTRA_NODE_COMM: str = "1"  # Faster intra-node collectives
    TORCH_NCCL_AVOID_RECORD_STREAMS: str = "1"  # Avoid OOM w/ long contexts
    NCCL_IB_TIMEOUT: str = "22"  # Increase NCCL timeout, 22 gives 16s timeout
    NCCL_DEBUG: str = "WARN"  # NCCL debugging
    TORCH_NCCL_ASYNC_ERROR_HANDLING: str = "1"  # Enable async error handling
    TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
    TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"


# ----------------------------------------------------------------------------------------
# 2. Basic environment / device introspection
# ----------------------------------------------------------------------------------------


@lru_cache()
def is_distributed() -> bool:
    """
    Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization

    Checks the following conditions:

    * torch.distributed is available
    * master port and master address environment variables are set
    * world size is >1
    * rank environment variable is set

    Returns:
        bool: True if all of the above conditions hold, False otherwise.
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


@lru_cache()
def is_slurm_job() -> bool:
    """
    Checks if the current process is running as part of a SLURM-managed job
    and ensures it is not a torchrun job.
    """
    return "SLURM_JOB_ID" in os.environ and not is_distributed()


@lru_cache()
def get_global_rank() -> int:
    """
    Retrieves the global rank of the current process in the distributed system.

    Returns:
        int: The global rank. Defaults to 0 if neither torchrun nor SLURM is detected.
    """
    if is_distributed():
        return int(os.environ["RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    """
    Retrieves the local rank of the current process within a node.

    Returns:
        int: The local rank. Defaults to 0 if neither torchrun nor SLURM is detected.
    """
    if is_distributed():
        return int(os.environ["LOCAL_RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


# TODO: Deprecated, consider removing in favor of dist.get_world_size
@lru_cache()
def _get_world_size() -> int:
    """
    Retrieves the total number of processes participating in the distributed job.

    Returns:
        int: The total number of processes. Defaults to 1 if neither torchrun nor SLURM is detected.
    """
    if is_distributed():
        return int(os.environ["WORLD_SIZE"])
    elif is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def is_master_process() -> bool:
    """
    Determines if the current process is the master process (global rank 0).

    Returns:
        bool: True if the process is the master, otherwise False.
    """
    return get_global_rank() == 0


def is_port_available(port: int) -> bool:
    """
    Checks if a port is available for use.

    Args:
        port (int): Port number to check.

    Returns:
        bool: True if the port is available, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


@lru_cache()
def get_master_port(job_id: int, min_port: int = 20000, max_port: int = 60000) -> int:
    """
    Determines the master port for distributed training.

    Args:
        job_id (int): The job ID, used as a seed for reproducible port allocation.
        min_port (int): Minimum port number for random allocation (default: 20000).
        max_port (int): Maximum port number for random allocation (default: 60000).

    Returns:
        int: The port to use as the master port.
    """
    if is_distributed():
        # Ensure MASTER_PORT is set and valid
        master_port = os.environ.get("MASTER_PORT")
        if master_port is None:
            raise RuntimeError("MASTER_PORT is not set in the environment.")
        try:
            return int(master_port)
        except ValueError as e:
            raise ValueError(f"Invalid MASTER_PORT value: {master_port}") from e
    else:
        # Use random port generation based on job_id for reproducibility
        rng = random.Random(job_id)
        port = rng.randint(min_port, max_port)

        # Verify port availability
        if not is_port_available(port):
            logger.warning(f"Port {port} is unavailable, retrying.")
            for _ in range(10):  # Retry up to 10 times
                port = rng.randint(min_port, max_port)
                if is_port_available(port):
                    break
            else:
                raise RuntimeError("Failed to find an available port after 10 attempts.")

        return port


@lru_cache()
def get_master_addr() -> str:
    """
    Determines the master address for distributed training.

    Returns:
        str: The master address as a string.
    """
    if is_distributed():
        # Ensure MASTER_ADDR is set in the environment
        master_addr = os.environ.get("MASTER_ADDR")
        if not master_addr:
            raise RuntimeError("MASTER_ADDR is not set in the environment for torchrun.")
        return master_addr

    elif is_slurm_job():
        try:
            # Get the first hostname from the SLURM job node list
            hostnames = subprocess.check_output(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                text=True,
            ).splitlines()
            if not hostnames:
                raise RuntimeError("No hostnames found in SLURM_JOB_NODELIST.")
            return hostnames[0]
        except subprocess.CalledProcessError as e:
            logger.error("Failed to retrieve master address from SLURM: %s", e)
            raise RuntimeError("Could not determine master address in SLURM setup.") from e
        except KeyError as e:
            logger.error("SLURM_JOB_NODELIST environment variable is not set.")
            raise RuntimeError("SLURM_JOB_NODELIST is missing from the environment.") from e

    else:
        # Fallback for local or non-distributed setups
        logger.info("Using default master address 127.0.0.1 for local setup.")
        return "127.0.0.1"


@lru_cache()
def get_user_id() -> str:
    """Fetch the current user ID from the environment."""
    return os.environ.get("USER") or os.environ.get("LOGNAME") or os.environ.get("SLURM_JOB_USER", "unknown_user")


# ----------------------------------------------------------------------------------------
# 3. Parallel factor checks and distributed collectives
# ----------------------------------------------------------------------------------------


def validate_parallel_factors(
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    pp: int,
    auto_infer_shard: bool = False,
) -> int:
    """
    Validates that dp_replicate, dp_shard, cp, tp, and pp multiply to world_size.
    - If auto_infer_shard=True and dp_shard == -1, it will compute dp_shard for you.
    - Returns a finalized dp_shard, which might be updated if auto_infer_shard is True.
    - Raises ValueError if the configuration is invalid.
    """
    world_size = dist.get_world_size()

    # Check base constraints
    if dp_replicate < 1:
        raise ValueError(f"dp_replicate must be >= 1, got {dp_replicate}")
    if cp < 1:
        raise ValueError(f"cp must be >= 1, got {cp}")
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    if pp < 1:
        raise ValueError(f"pp must be >= 1, got {pp}")

    # Handle dp_shard special rules
    if dp_shard == -1:
        if not auto_infer_shard:
            raise ValueError("dp_shard == -1, but auto_infer_shard=False.")
        # Auto-infer dp_shard
        base = dp_replicate * cp * tp * pp
        if base == 0 or world_size % base != 0:
            raise ValueError(
                f"Cannot auto-infer dp_shard because world_size={world_size} "
                f"is not divisible by dp_replicate * cp * tp * pp = {base}."
            )
        dp_shard = world_size // base
    else:
        if dp_shard < 1:
            raise ValueError(f"dp_shard must be >=1 or -1, got {dp_shard}")

    # Final product check
    product = dp_replicate * dp_shard * cp * tp * pp
    if product != world_size:
        raise ValueError(
            f"Invalid parallel factor product: "
            f"{dp_replicate} * {dp_shard} * {cp} * {tp} * {pp} = {product}, "
            f"which does not match world_size={world_size}."
        )

    return dp_shard


def dist_reduce(x: torch.Tensor, reduceOp: str, mesh: DeviceMesh) -> float:
    """
    Perform a distributed reduce operation on a scalar tensor across the given device mesh.

    This function uses functional collectives to reduce a single-element tensor `x` according
    to the provided `reduceOp`. If `x` is a DTensor, it is converted to a regular tensor
    first. The result of the reduction is returned as a Python float.

    Args:
        x (torch.Tensor): A single-element tensor (scalar).
        reduceOp (str): The name of the reduction operation (e.g., "SUM", "MAX", "AVG").
        mesh (DeviceMesh): The device mesh (group of ranks) over which to perform the reduction.

    Returns:
        float: The reduced scalar value.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()
    assert x.numel() == 1, "Input tensor must contain exactly one element."
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(x: torch.Tensor, mesh: DeviceMesh) -> float:
    """
    Compute the maximum value of a scalar tensor across all ranks in the given device mesh.

    Internally calls `dist_reduce` with a MAX reduction operation. Expects `x` to be a
    single-element tensor (scalar).

    Args:
        x (torch.Tensor): A single-element tensor (scalar).
        mesh (DeviceMesh): The device mesh (group of ranks) over which to compute the max.

    Returns:
        float: The maximum value across all ranks.
    """
    return dist_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_mean(x: torch.Tensor, mesh: DeviceMesh) -> float:
    """
    Compute the mean (average) of a scalar tensor across all ranks in the given device mesh.

    Internally calls `dist_reduce` with an AVG reduction operation. Expects `x` to be a
    single-element tensor (scalar). The mean is calculated by summing values across all
    ranks and dividing by the number of ranks.

    Args:
        x (torch.Tensor): A single-element tensor (scalar).
        mesh (DeviceMesh): The device mesh (group of ranks) over which to compute the mean.

    Returns:
        float: The mean value across all ranks.
    """
    return dist_reduce(x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh)


# ----------------------------------------------------------------------------------------
# 4. Environment setup, seeding, system info logging
# ----------------------------------------------------------------------------------------


def setup_env(env_args: EnvironmentArgs) -> None:
    """
    Sets up the environment for distributed training, optimizing for HPC clusters
    with SLURM, shared filesystems, and tools like Triton.

    Environment Variables Set:
    - `TRITON_CACHE_DIR`: Temporary, per-process cache directory for Triton kernels.
      Prevents "Stale file handle" errors when backed by NFS.
    - `TMP_DIR`: Redirects temporary files to a shared, high-performance scratch directory
      if running under SLURM. Falls back to `/scratch/{user_id}` if `/scratch/gpfs` or
      other SLURM-specific paths are unavailable.

    Args:
        env_args: A dataclass object containing key-value pairs of environment variables.

    Notes:
        - Triton, used for custom kernels, defaults to `~/.triton/cache`. This can fail
          when backed by NFS. A temporary, process-specific cache avoids this issue.
        - SLURM jobs benefit from using a scratch directory for TMP_DIR to prevent
          tmpfs overflows and performance degradation.

    Returns:
        None
    """
    env_vars = asdict(env_args)

    set_torch_num_threads()

    # Set up a per-process Triton cache directory
    triton_cache_dir = tempfile.mkdtemp(prefix="triton_cache_")
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    env_vars["TRITON_CACHE_DIR"] = triton_cache_dir
    logger.info(f"Triton cache directory set to: {triton_cache_dir}")

    # Determine a suitable scratch directory for TMP_DIR
    if is_slurm_job():
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        user_id = get_user_id()

        # Check for `/scratch/gpfs` first, then fallback
        potential_scratch_dirs = [
            f"/scratch/gpfs/{user_id}/{slurm_job_id}",  # Preferred for Princeton-like setups
            f"/scratch/{user_id}/{slurm_job_id}",  # Generic fallback
        ]
        scratch_dir = next((d for d in potential_scratch_dirs if os.path.exists(d)), None)

        if scratch_dir:
            env_vars["TMP_DIR"] = scratch_dir
            logger.info(f"Temporary directory set to: {scratch_dir}")
        else:
            logger.warning(
                f"Neither '/scratch/gpfs/{user_id}/{slurm_job_id}' nor '/scratch/{user_id}/{slurm_job_id}' exist. "
                f"Using system default TMP_DIR."
            )

    # Apply environment variables
    for name, value in env_vars.items():
        current_value = os.environ.get(name)
        if current_value != str(value):
            os.environ[name] = str(value)
            logger.info(f"Environment variable {name} set to: {value}")


def set_seeds(seed: int, cuda_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seeds set to {seed}")


def log_system_info(world_size: int, rank: int):
    local_gpu_count = torch.cuda.device_count()
    system_info = {
        "Rank": rank,
        "CPU count": psutil.cpu_count(),
        "Total RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "Available RAM": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "Local GPU count": local_gpu_count,
        "Total GPU count across all nodes": world_size * local_gpu_count,
    }

    logger.info("System info (NON-DEBUG mode):")
    for key, val in sorted(system_info.items()):
        logger.info(f"  {key} = {val}")

    # Log specific GPU properties for the local GPUs
    for i in range(local_gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"  GPU {i} name = {gpu_name}")
        logger.info(f"  GPU {i} memory = {gpu_mem:.2f} GB")


# ----------------------------------------------------------------------------------------
# 5. Distributed setup and teardown
# ----------------------------------------------------------------------------------------


def setup_distributed(job_config: JobConfig):
    """
    Configures the environment for distributed training:
    - Sets up flight recorder / NCCL environment variables.
    - Initializes the process group.
    - Returns useful distributed and device info.

    Args:
        job_config (JobConfig): Configuration dataclass containing distributed,
            training, and kernel settings.

    Returns:
        device (torch.device): The device assigned to the current process.
        rank (int): The global rank of the current process.
        world_size (int): Total number of processes in the distributed job.
        master_process (bool): True if the current process is the main (rank 0).
    """
    if not dist.is_available():
        raise RuntimeError("Distributed package is not available!")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training!")

    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    mp.set_start_method(job_config.training.spawn_method, force=True)

    local_rank = get_local_rank()
    os.environ["MASTER_ADDR"] = get_master_addr()
    # If using SLURM, get_master_port might read the SLURM_JOB_ID
    os.environ["MASTER_PORT"] = str(get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1))))

    # Overwrite environment variable for async error handling to skip cleanup
    _warn_overwrite_env(EnvironmentArgs.ASYNC_ERROR_HANDLING, EnvironmentArgs.SKIP_CLEANUP)

    # Enable PyTorch NCCL flight recorder in a mode that will dump files if a timeout is detected
    trace_buf_size = getattr(job_config.distributed, "comm.trace_buf_size")
    _warn_overwrite_env(EnvironmentArgs.TRACE_BUFFER_SIZE, trace_buf_size)

    if trace_buf_size > 0:
        # If trace buffer is enabled, dump on timeout by default
        _warn_overwrite_env(EnvironmentArgs.DUMP_ON_TIMEOUT, "1")
        dump_dir = f"{job_config.job.dump_folder}/comm_trace"
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(EnvironmentArgs.TRACE_FILE, f"{dump_dir}/rank_")

    # Mitigate memory issue in collectives (e.g., async_op=True holding memory)
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # Designate a distributed backend
    dist.init_process_group(
        backend=_get_distributed_backend(job_config),
        init_method="env://",
        timeout=timedelta(seconds=getattr(job_config.distributed, "comm.init_timeout_seconds")),
    )

    # Basic distributed info + seed setup
    rank = get_global_rank()
    world_size = dist.get_world_size()
    master_process = is_master_process()
    device = torch.device(f"cuda:{local_rank}")
    set_seeds(job_config.training.seed + rank)

    # Precision / kernel configurations
    if job_config.kernel.matmul_allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.warning(
            "WARNING: TF32 is enabled for matrix multiplication. "
            "This may improve performance on Ampere or newer GPUs but could reduce precision."
        )

    if job_config.kernel.set_float32_matmul_precision:
        torch.set_float32_matmul_precision(job_config.kernel.set_float32_matmul_precision)
        logger.info(
            f"Float32 matrix multiplication precision set to '{job_config.kernel.set_float32_matmul_precision}'. "
            "Options: 'highest' (full float32), 'high' (may use TF32 or bfloat16), 'medium' (may use bfloat16)."
        )

    if job_config.kernel.allow_bf16_reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        logger.warning(
            "WARNING: BF16 reduced precision is enabled for matrix multiplication. "
            "This may improve performance but could reduce precision."
        )

    # Debugging / Anomaly Detection
    torch.autograd.set_detect_anomaly(job_config.kernel.detect_anomaly)

    # Confirm the distributed setup environment
    if master_process:
        logger.info(f"Main process initialized on {socket.gethostname()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"World info: size={world_size}, rank={rank}, local_rank={local_rank}")
        log_system_info(world_size, rank)

        if debug_mode:
            # Pretty-print all environment variables
            pretty_env_str = json.dumps(dict(os.environ), indent=2, sort_keys=True)
            logger.debug("Environment variables (DEBUG mode):\n%s", pretty_env_str)
        else:
            # Only print those relevant for distributed
            relevant_prefixes = ("MASTER_", "TORCH_", "NCCL_", "CUDA_", "RANK", "WORLD_SIZE")
            filtered_env = {k: v for k, v in os.environ.items() if k.startswith(relevant_prefixes)}
            logger.info("Environment variables (NON-DEBUG mode):")
            for k, v in sorted(filtered_env.items()):
                logger.info(f"  {k} = {v}")

    return device, rank, world_size, master_process


def cleanup_distributed(rank: int):
    if dist.is_initialized():
        logger.info(f"[Rank {rank}]: Finished training.")
        logger.info(f"[Rank {rank}]: Waiting for other processes to finish...")
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------------------------------------------------------------------
# 6. Module inspection and FSDP grouping helpers
# ----------------------------------------------------------------------------------------


def get_module(module: any, access_string: str) -> any:
    """
    Retrieves an attribute from a nested module using a dot-separated access string.

    Args:
        module (Any): The root module or object.
        access_string (str): Dot-separated string representing the attribute hierarchy (e.g., "submodule.layer.weight").

    Returns:
        Any: The retrieved attribute.

    Example:
        >>> get_module(model, "encoder.layer.0.attention")
    """
    return reduce(getattr, access_string.split("."), module)


def set_module(module: any, access_string: str, value: any) -> None:
    """
    Sets the value of an attribute in a nested module using a dot-separated access string.

    Args:
        module (Any): The root module or object.
        access_string (str): Dot-separated string representing the attribute hierarchy (e.g., "submodule.layer.weight").
        value (Any): The value to set.

    Example:
        >>> set_module(model, "encoder.layer.0.attention.weight", new_weight_tensor)
    """
    names = access_string.split(".")
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], value)


def default_fsdp_grouping_plan(num_layers: int) -> list[tuple[str, bool]]:
    """
    Creates a default Fully Sharded Data Parallel (FSDP) grouping plan for layers.

    Args:
        num_layers (int): The total number of layers to include in the grouping plan.

    Returns:
        list[tuple[str, bool]]: A list of tuples where the first element is the layer's name
        and the second element is a boolean indicating whether to recompute it.
    """
    return [(f"layers.{i}", i < num_layers - 1) for i in range(num_layers)]


@torch.no_grad()
def check_model_value_range(model: torch.nn.Module, range_threshold: float = 1e3, std_threshold: float = 1e3) -> None:
    """
    Checks the value range and standard deviation of model parameters and buffers.

    Args:
        model (torch.nn.Module): The model to check.
        range_threshold (float): Maximum acceptable range (difference between max and min values).
        std_threshold (float): Maximum acceptable standard deviation.

    Notes:
        - Logs warnings if any parameter or buffer has extreme values, NaN/Inf, or is uninitialized.
    """
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(param, DTensor):
            param = param.to_local()

        if param.numel() == 0:
            logger.warning(f"Model parameter {name} is empty, possibly due to FSDP sharding.")
            continue

        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"Model parameter {name} contains NaN or Inf.")

        param_range = param.max() - param.min()
        param_std = param.std()

        if param_range > range_threshold:
            logger.warning(f"Model parameter {name} has a large range ({param_range}). Check initialization.")
        if param_std > std_threshold:
            logger.warning(
                f"Model parameter {name} has a large standard deviation ({param_std}). Check initialization."
            )
        if (param == 0).all():
            logger.warning(f"Model parameter {name} is all zeros. Initialization may be missing.")


def validate_params_on_meta_device(model: nn.Module) -> None:
    """
    Utility to validate that all parameters and buffers in the model
    are on the meta device. If any parameter or buffer is not on meta,
    an error indicating the param/buffer name will be raised.

    Args:
        model (nn.Module): model to check

    Raises:
        RuntimeError: If any params or buffers are found off the meta device
    """
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if not param.is_meta:
            raise RuntimeError(
                f"Expected parameter/buffer '{name}' to be on meta device, but found device: {param.device}"
            )


def validate_no_params_on_meta_device(model: nn.Module) -> None:
    """
    Utility to validate that the model has no parameters or buffers on the meta device.
    If any parameter or buffer is on meta, an error indicating the param/buffer name will be raised.

    Args:
        model (nn.Module): model to check

    Raises:
        RuntimeError: If any params or buffers are found on the meta device
    """
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if param.is_meta:
            raise RuntimeError(f"Unexpected parameter/buffer '{name}' on meta device.")


# ----------------------------------------------------------------------------------------
# 7. Signal handling and SLURM requeue
# ----------------------------------------------------------------------------------------


def init_signal_handler(handler: Callable[[int, any], None]) -> None:
    """
    Installs a custom signal handler to handle SLURM preemption signals (SIGUSR2).

    Args:
        handler (Callable): The function to call when SIGUSR2 is received.

    Notes:
        - Useful for saving checkpoints or triggering job requeueing during preemption.
    """
    signal.signal(signal.SIGUSR2, handler)
    logger.warning("Signal handler for SIGUSR2 installed.")


def requeue_slurm_job() -> None:
    """
    Requeues the current SLURM job to maintain its position in the job queue upon preemption.

    Notes:
        - Only the master process (global rank 0) requeues the job.
        - Exits the process after requeueing.
    """
    prod_id = int(os.environ.get("SLURM_PROCID", "0"))
    logger.warning(f"Host: {socket.gethostname()} - Global rank: {prod_id}")

    if prod_id == 0 and os.environ.get("LAUNCH_WITH", "") != "DORA":
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if job_id:
            logger.warning(f"Requeuing job {job_id}.")
            os.system(f"scontrol requeue {job_id}")
        else:
            logger.error("SLURM_JOB_ID is not set. Cannot requeue job.")
    else:
        logger.info("Not the master process, skipping job requeue.")

    sys.exit(0)


# ----------------------------------------------------------------------------------------
# 8. Temporary environment cleanup for subprocesses
# ----------------------------------------------------------------------------------------


@contextlib.contextmanager
def clean_env() -> Generator[None, None, None]:
    """
    A context manager to temporarily clean up the environment by removing
    distributed and cluster-specific variables.

    This is useful for creating a "clean slate" environment in cases where
    distributed or cluster-specific settings might interfere with subprocesses or
    other operations.

    Temporarily removes:
    - Distributed environment variables (`MASTER_ADDR`, `MASTER_PORT`, etc.).
    - SLURM-related environment variables (`SLURM_*`, `SLURMD_*`, etc.).
    - Submitit-related variables (`SUBMITIT_*`).
    - WANDB-related variables (`WANDB_*`).

    After the context manager's block is exited, the environment is restored to
    its original state.

    Yields:
        None: Control is returned to the context's code block.

    Example:
        >>> with clean_env():
        >>>     subprocess.run(["python", "some_script.py"])
        # Environment is restored here.
    """
    # The distributed and cluster-related environment variables to clean
    distrib_vars = {
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    }

    # Collect and remove relevant environment variables
    cluster_env = {
        var: os.environ.pop(var)
        for var in list(os.environ)  # Use list to avoid modifying dict during iteration
        if var.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_")) or var in distrib_vars
    }

    try:
        yield  # Provide a clean environment for the context block
    finally:
        # Restore the original environment variables
        os.environ.update(cluster_env)


# ----------------------------------------------------------------------------------------
# 9. Determinism, collectives, and context parallel
# ----------------------------------------------------------------------------------------


def _warn_overwrite_env(env: str, val: str | int | float | bool) -> None:
    """
    Set (or overwrite) an environment variable and log a warning if it was already set.

    If `env` is present in the process environment, a warning will be issued indicating
    that the old value is being overridden. The `val` parameter will be converted to a
    string before assignment.

    Args:
        env (str):
            The name of the environment variable (e.g., "TORCH_NCCL_TRACE_BUFFER_SIZE").
        val (Union[str, int, float, bool]):
            The value to set for the environment variable. If not a string, it will be
            cast to a string before assignment.

    Example:
        >>> _warn_overwrite_env("MY_ENV_VAR", 1746)
        WARNING:__main__:ENV[MY_ENV_VAR] will be overridden to 1746 based on job config.
        # os.environ["MY_ENV_VAR"] is now "1746"
    """
    if env in os.environ:
        logger.warning(f"ENV[{env}] will be overridden to {val} based on job config.")
    os.environ[env] = str(val)


def set_determinism(
    world_mesh: DeviceMesh | None,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = False,
) -> None:
    """
    Sets determinism flags for increased reproducibility at the expense of performance.

    We set the same DTensor manual seed for ALL ranks within the same DTensor SPMD group,
    but different seeds across Pipeline Parallel groups (if applicable).

    NOTE: Currently, we do not set seeds for the CUDA RNG since we always use DTensor
    for SPMD parallelisms, and DTensor manages its own RNG tracker; however, we
    could extend to support both if needed.
    """
    if deterministic:
        logger.warning("Deterministic algorithm enabled (expected performance degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # This is an environment variable for deterministic CuBLAS, see more:
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    """
    To ensure we can control which ranks have the same or different seeds, all ranks will agree
    on a starting seed.
    
    If the user provides one, we will use theirs. Otherwise, the master process (rank 0) will
    roll the dice and everyone else uses that.
    """
    if seed is None:
        # Extract the seed for PyTorch's main generator on rank 0
        seed_tensor = torch.get_rng_state()[:8].to(device)

        # Standardize on this to build seeds for unique SPMD groups
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.view(torch.uint64).item()

    """
    For Pipeline Parallelism + SPMD cases, we want to separate the world into the
    SPMD mesh and the PP mesh.
    """
    if c10d.get_world_size() > 1 and "pp" in world_mesh.mesh_dim_names:
        pp_mesh = world_mesh["pp"]
        seed += pp_mesh.get_local_rank()
        seed %= 2**64

        logger.debug(f"PP rank {pp_mesh.get_local_rank()}, Global rank {c10d.get_rank()} using seed: {seed}.")
        spmd_mesh_dims = list(filter(lambda name: name != "pp", world_mesh.mesh_dim_names))
        spmd_mesh = world_mesh[spmd_mesh_dims] if len(spmd_mesh_dims) else None
    else:
        spmd_mesh = world_mesh
        logger.debug(f"Global rank {c10d.get_rank()} using seed: {seed}.")

    """
    The native RNGs and Python RNGs may not be important, except for the 1-D PP case,
    but we will seed them anyway for consistency.
    """
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    if spmd_mesh and spmd_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, spmd_mesh)


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all process groups in the provided mesh and the default (world) group.

    NOTE: Synchronizes via a barrier before changing the timeout.

    This is important because otherwise we may face a race condition where the
    slow rank has not reached the timeout reduction point yet due to slow
    operations permitted under the old timeout value, while faster ranks may start
    issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(f"Synchronizing and adjusting timeout for all process groups to {timeout}.")

    # Ensure all ranks have reached the point of setting the new timeout
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]
    # `None` represents the "default" process group, not part of the mesh
    groups.append(None)

    for group in groups:
        c10d._set_pg_timeout(timeout, group)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
) -> None:
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        logger.error(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(job_config: JobConfig):
    @contextlib.contextmanager
    def context(nested_ctx: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            # 1. Loss parallel context
            if not job_config.distributed.parallelism.get("disable_loss_parallel"):
                stack.enter_context(dist.tensor.parallel.loss_parallel())

            # 2. Compiled autograd context
            if job_config.kernel.enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            # 3. Activation offloading context
            if job_config.distributed.activation_offloading.get("enabled"):
                from thesis.distributed.activation_offloading import OffloadActivations, OffloadConfig

                offload_cfg = OffloadConfig(
                    use_pin_memory=job_config.distributed.activation_offloading.get("use_pin_memory"),
                    use_streams=job_config.distributed.activation_offloading.get("use_streams"),
                    max_fwd_stash_size=job_config.distributed.activation_offloading.get("max_fwd_stash_size"),
                    min_offload_size=job_config.distributed.activation_offloading.get("min_offload_size"),
                    virtual_memory_safe_pct=job_config.distributed.activation_offloading.get(
                        "virtual_memory_safe_pct"
                    ),
                )
                stack.enter_context(OffloadActivations(config=offload_cfg))
            else:
                from thesis.distributed.activation_offloading import NoOpManager

                stack.enter_context(NoOpManager())

            # 4. Context/sequence parallel context
            if nested_ctx is not None:
                from torch.nn.attention import sdpa_kernel, SDPBackend

                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(nested_ctx)

            yield

    return context


# ----------------------------------------------------------------------------------------
# 10. Low-level initialization and concurrency
# ----------------------------------------------------------------------------------------


def set_torch_num_threads() -> None:
    """
    Sets the number of threads used by torch to utilize all physical CPU
    cores for intra-op parallelism.

    Currently, this function sets num_threads to be the number of physical
    CPU cores divided by the number of GPUs as we use one process per GPU,
    and this avoids CPU oversubscription.

    Note that this is currently a rough approximation, and doesn't take into
    account environments where things like CPU affinity is set.
    """
    num_threads = os.cpu_count() // (dist.get_world_size() if dist.is_initialized() else 1)
    torch.set_num_threads(num_threads)
    logger.info(f"Set intra op parallelism no. of threads to {num_threads}")


def _get_distributed_backend(job_config):
    backend = "nccl"

    if device_type in dist.Backend.default_device_backend_map.keys():
        backend = dist.Backend.default_device_backend_map.get(device_type)

    if getattr(job_config.distributed, "parallelism.enable_cpu_offload"):
        backend = f"{device_type}:{backend},cpu:gloo"

    return backend
