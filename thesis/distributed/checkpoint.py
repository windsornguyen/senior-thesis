"""Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py"""

import enum
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import os
import re
import shutil
import socket
import time

from multiprocessing import Queue, get_context
from io import BytesIO
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)

from thesis.utils.logger import logger
from thesis.utils.pretraining_config import JobConfig, TORCH_DTYPE_MAP
from thesis.utils.optimizer import Optimizers
from thesis.utils.scheduler import Schedulers

TrainStateValue = torch.Tensor | BytesIO
TrainStateDict = dict[str, TrainStateValue]
ModelStateDict = dict[str, torch.Tensor]

class IntervalType(enum.Enum):
    """
    InteralType is an enum that defines the various interval types for checkpointing
    in a more human-readable format.
    
    SECONDS:
        Interval in seconds.

    STEPS:
        Number of steps.
    """
    SECONDS = enum.auto()
    STEPS = enum.auto()

class AsyncMode(str, enum.Enum):
    """
    AsyncMode is an enum that defines the various async checkpointing modes
    in a more human-readable format.
    
    DISABLED:
        Async checkpointing is turned off.

    ASYNC:
        Async checkpointing happens in the background w/o blocking the main training loop.

    ASYNC_WITH_PINNED_MEMORY:
        Optimized async checkpointing w/ pinned memory to increase data transfer throughput.
    """
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEMORY = "async_with_pinned_memory"


# TODO: Deprecated in favor of TrainingState in experiments/training/state.py
@dataclass
class TrainState(Stateful):
    """
    TrainState is a data container class that inherits from DCP's Stateful class.
    It stores various metrics for the current training step.

    We use BytesIO for a number of reasons:
    (1) It minimizes I/O operations by keeping the data in memory
        while it's being processed.

    (2) It complement async checkpointing since the serialized data can be quickly 
        passed off to another thread/process without blocking the main training loop.
    """
    # Current training step
    step: int = 0

    # Current gradient accumulating step
    accum_step: int = 0

    # Tracks average loss values globally over all processes
    global_avg_losses: list[float] = field(default_factory=list)
    
    # Tracks maximum loss values globally over all processes; used for anomaly detection
    global_max_losses: list[float] = field(default_factory=list)

    # Keeps a record of the steps at which we have logged the loss values
    log_steps: list[int] = field(default_factory=list)

    def state_dict(self) -> TrainStateDict:
        """
        Saves the current state of the TrainState dataclass to a BytesIO object.
        Checkpoints only every `log_frequency` to avoid async overhead.

        Args:
            None

        Returns:
            dict[str, BytesIO]: A dictionary mapping state keys to BytesIO objects.
        """
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)

        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)

        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)

        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "accum_step": torch.tensor(self.accum_step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict: TrainStateDict) -> None:
        """
        Loads the state of the TrainState dataclass from a dictionary.

        Args:
            state_dict TrainStateDict): A dictionary mapping state keys to values.
        
        Returns:
            None
        """
        self.step = state_dict["step"].item()
        self.accum_step = state_dict["accum_step"].item()

        # NOTE: We seek(0) to reset the internal pointer of the BytesIO object
        # to the beginning of the stream before reading it.
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(state_dict["global_avg_losses"], weights_only=False)

        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(state_dict["global_max_losses"], weights_only=False)

        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


class StatefulModel(Stateful):
    """
    The StatefulModel class wraps around PyTorch modules to make them compatible
    with DCP's Stateful class.
    """
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        """
        Initializes a StatefulModel instance with a list of PyTorch modules.
        
        Args:
            model (nn.Module | list[nn.Module]): A PyTorch module or a list of PyTorch modules.
        
        Returns:
            None
        """
        self.models = (
            {"model_0": model} if isinstance(model, nn.Module)
            else {f"model_{i}": m for i, m in enumerate(model)}
        )

    def state_dict(self) -> dict[str, ModelStateDict]:
        """
        Returns a dictionary of state dicts for each module in the model.
        
        Args:
            None
        
        Returns:
            dict[str, ModelStateDict]: A dictionary mapping state keys to values.
        """
        return {
            k: v 
            for model_name, model in self.models.items() 
            for k, v in get_model_state_dict(model).items()
        }

    def load_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        """
        Loads a state dictionary into one or more models managed by the StatefulModel.

        Args:
            state_dict (dict[str, dict[str, torch.Tensor]]): A dictionary mapping state keys to values.
        
        Returns:
            None
        """
        for model_name, model in self.models.items():
            if model_name in state_dict:
                set_model_state_dict(
                    model=model,
                    model_state_dict=state_dict[model_name],
                    options=StateDictOptions(strict=False),
                )

class Terminate:
    """
    A control class used to signal the termination of the checkpoint background process.
    
    This signal is sent through the multiprocessing queue to inform the background process
    that it should gracefully exit.
    """
    ...


class SaveDone:
    """
    A control class used to signal that the checkpointing operation has completed.
    
    This signal is sent through the multiprocessing queue to:
    (1) Indicate that the background checkpointing process has finished saving the state.
    (2) Synchronize the main process with the background process so that subsequent operations
        do not proceed until the checkpointing process is confirmed to be completed.
    """
    ...


def find_available_port() -> int:
    """
    Finds and returns an available port on the system.

    This function uses the socket library to bind to port 0, which allows the OS to automatically
    select an unused port. The chosen port is then returned to be used in a distributed system.

    Returns:
        int: An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to port 0 to let the OS assign an available port
        port = s.getsockname()[1]  # Retrieve the port number assigned by the OS
    return port


def checkpoint_mp(recv: Queue, send: Queue) -> None:
    """
    Background process responsible for checkpointing the training state by
    using the `recv` and `send` channels to interact with the main process.
    
    Args:
        recv (multiprocessing.Queue): A queue used to receive control signals from the main process.
        send (multiprocessing.Queue): A queue used to send control signals to the main process.
    
    Returns:
        None
    """
    os.environ["MASTER_PORT"] = str(find_available_port())
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Checkpoint background process is done.")
            send.put(SaveDone())
            logger.debug("Waiting for the new state_dict...")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the checkpoint background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, checkpoint_id = obj
            dcp.save(state, checkpoint_id=checkpoint_id)
            logger.info(
                "Finish saving the checkpoint in the background process in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
    finally:
        logger.info("Destroying the process group...")
        dist.destroy_process_group()

class CheckpointManager:
    def __init__(
        self,
        dataloader: DataLoader, # TODO: Change to our nd dataloader?
        model_parts: list[nn.Module],
        optimizers: Optimizers,
        lr_schedulers: Schedulers,
        states: dict[str, Stateful],
        job_config: JobConfig,
    ) -> None:
        checkpoint_config = job_config.distributed.checkpoint
        self.enable_checkpoint = checkpoint_config.get("enable_checkpoint")
        self.keep_latest_k = checkpoint_config.get("keep_latest_k")
        if not self.enable_checkpoint:
            return

        """
        NOTE: Pipeline Parallelism and Virtual Stages cases to consider:
        
        1. Even for simple PP schedules, each PP rank has its own separate optimizer.

            Example:
            - Rank 0 has param_group[0] which refers to layers.0 from original model
            - Rank 1 would _also_ have param_group[0] since it's index-based, but
            it should actually be referring to layers.1
            
            When saving, these collide and one of them is lost. When we reload, only
            one stage can restore its optimizer states and the other will error.
            
            We can solve this via optimizer flattening; just pass in `flatten_optimizer_state_dict`
            kwarg to DCP functions called in StatefulOptimizer.
        
        2. With complex PP schedules, we have multiple model chunks per PP rank. 
            This compounds challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

            We solve this in the Model and Optimizer wrapper classes by flattening the state dicts from each object
            into one state dict before saving/loading. We assume that individual state_dicts do not collide, and
            should be careful that this assumption holds.

        3. LR schedulers also index model states like optimizers and would need to be flattened properly to support
        resharding.  Unfortunately, the implementations of different lr_schedulers do not follow a clear pattern like
        optimizers do, so it's hard to write a generic 'flattener' utility.
            TODO: This is currently unsolved and needs a fix.
        """
        self.states = states
        
        self.states.update(
            {
                "model": StatefulModel(model_parts),
                "optimizer": optimizers,
                "dataloader": dataloader, # We store the dataloader state too
                "lr_scheduler": lr_schedulers,
            }
        )

        # Init general setup
        self.folder = os.path.join(job_config.job.dump_folder, checkpoint_config.folder)
        self.interval_type = (
            IntervalType.SECONDS
            if checkpoint_config.interval_type == "seconds"
            else IntervalType.STEPS
        )

        # Init async checkpointing setup
        self.mp = None
        async_mode = checkpoint_config.async_mode.lower()
        if async_mode == AsyncMode.ASYNC or self.interval_type == IntervalType.SECONDS:
            self.process_group = dist.new_group(backend="gloo")

        self.interval = checkpoint_config.interval
        self.begin_time = 0
        self.time_sync_work = None
        self.time_sync_result = None
        self.model_weights_only = checkpoint_config.model_weights_only
        self.export_dtype = TORCH_DTYPE_MAP[checkpoint_config.export_dtype]

        # Disable async checkpointing
        if async_mode == AsyncMode.DISABLED: 
            self.async_mode = AsyncMode.DISABLED

        # Enable async checkpointing (using async_save)
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
            self.async_future = None

        # Enable async checkpointing w/ pinned memory
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEMORY:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEMORY

            # Create non-blocking queues
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue() # Channel for checkpointing process
            self.mp_queue_recv = ctx.Queue() # Channel for main process
            self.mp = ctx.Process(
                target=checkpoint_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None # Hold model weights after moved to CPU
            self.staging = False # Checkpoint currently staging?
            self.staging_id = None # Which checkpoint currently being processed?
            self.staging_stream = torch.cuda.Stream() # Open up add'l stream to not block default stream during checkpointing
        else:
            raise ValueError(f"Unknown checkpoint async mode: {checkpoint_config.async_mode}")
        
        logger.info(f"Checkpointing activated. The checkpoints will be loaded from and save to {self.folder}.")

    def __del__(self):
        if self.enable_checkpoint and self.mp and self.mp.is_alive():
            self.mp_queue_send.put(Terminate())
            self.mp.join()

    def reset(self) -> None:
        self.begin_time = time.monotonic()
    
    def _create_checkpoint_id(self, step: int) -> str:
        return os.path.join(self.folder, f"step-{step}")
    
    def _save_last_step(self, curr_step: int) -> None:
        """
        We only save weights at the end of training.
        
        We only allow dtype conversion when:
        (1) We are saving checkpoint model weights
        (2) The current dtype is not the same as the export dtype at the end of training.
        
        Args:
            curr_step (int): Current step.
        """
        train_state = self.states.get("train_state")
        if train_state and hasattr(train_state, "accum_step"):
            # Log warning if we're stopping mid-accumulation
            if train_state.accum_step != 0:
                logger.warning(
                    f"Saving checkpoint at step {curr_step} with incomplete gradient "
                    f"accumulation (accum_step={train_state.accum_step}). This may "
                    "affect training resumption."
                )

        if self.model_weights_only:
            self.states = self.states["model"].state_dict()
            # TODO: Do we want this?
            # For now, we will manually pop the freqs_cis buffer, as we made this permanent
            # temporarily and we don't want to include it in the exported state_dict.
            # Context: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py#L348
            # self.states.pop("freqs_cis")
            
            if self.export_dtype != torch.float32:
                self.states = {
                    k: v.to(self.export_dtype) for k, v in self.states.items()
                }
            logger.info(
                f"Saving a model-weights-only checkpoint with type {self.export_dtype} "
                f"at last step {curr_step}."
            )
        else:
            logger.info(
                f"Saving a full checkpoint at last step {curr_step}."
            )
        
        dcp.save(self.states, checkpoint_id=self._create_checkpoint_id(curr_step))
        self.reset()
    
    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False
        
        if not force:
            if self.interval_type == IntervalType.STEPS and not curr_step % self.interval == 0:
                return False
        
            if self.interval_type == IntervalType.SECONDS:
                time_sync_results = (time.monotonic() - self.begin_time) >= self.interval
                self.time_sync_result = torch.tensor(int(time_sync_results))
                if self.time_sync_work is None:
                    self.time_sync_work = dist.all_reduce(
                        tensor=self.time_sync_result,
                        group=self.process_group,
                        async_op=True,
                    )
                    return False
                elif curr_step % 5 == 4: # what does this condition do?
                    self.time_sync_work.wait()
                    self.time_sync_work = None
                    time_sync_result = self.time_sync_result.item()
                    self.time_sync_result = None
                    if time_sync_result == 0:
                        return False
                else:
                    return False
        
        if self.time_sync_work:
            self.time_sync_work.wait()
            self.time_sync_work = None
            self.time_sync_result = None
        
        return True

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEMORY:
            logger.debug(f"Waiting for the background process to finish, {time.monotonic()=}.:.2f")
            if not self.mp.is_alive():
                raise RuntimeError("The checkpoint background process is dead.")
            _ = self.mp_queue_recv.get()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()
        
    def _async_with_pinned_memory(self, checkpoint_id: str) -> None:
        try:
            from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        
        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict=state_dict,
                pin_memory=True,
                share_memory=True,
            )
        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")

        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict=state_dict,
                copy_state_dict=self.cpu_offload_state_dict,
                non_block=True,
            )
            self.staging = True
            self.staging_id = checkpoint_id
    
    def _sync_func(self):
        self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_id))

    def _purge_stale_checkpoints(self) -> None:
        if self.keep_latest_k > 0:
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                path = os.path.join(self.folder, filename)
                discovered_checkpoints.append((int(match.group(1)), path))
            
            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[:-self.keep_latest_k]
            
            for _, path in to_delete:
                logger.warning(f"Deleting old checkpoint {path}...")
                shutil.rmtree(path, ignore_errors=True)

    def save(self, curr_step: int, force: bool = False) -> None:
        """
        force=True will force the checkpoint to be saved,
        even if the interval has not been reached.
        
        This only happens when train_state.step == job_config.training.steps,
        or for the initial seed checkpoint.
        
        Args:
            curr_step (int): Current step.
            force (bool, optional): Whether to force the checkpoint to be saved. Defaults to False.
        
        Returns:
            None
        """
        if not self._should_save(curr_step, force):
            return
        
        begin = time.monotonic()
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._async_wait()
        
        if force:
            self._save_last_step(curr_step)
        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEMORY:
            self._async_with_pinned_memory(checkpoint_id)
        elif self.async_mode == AsyncMode.ASYNC:
            self.async_future = dcp.async_save(
                state_dict=self.states,
                checkpoint_id=checkpoint_id,
                process_group=self.process_group,
            )
        else:
            dcp.save(
                state_dict=self.states,
                checkpoint_id=checkpoint_id,
            )
        self.reset()
        self._purge_stale_checkpoints()
        
        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled)"
            f"in {time.monotonic() - begin:.2f} seconds."
        )

    def sync_staging(self) -> None:
        if (
            self.enable_checkpoint
            and self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEMORY
            and self.staging
        ):
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            """
            TODO: We eventually want zero-overhead checkpointing staging, i.e. running the
            syncing logic in a separate thread with target=sync_func, so
            something like: self.my_thread = threading.Thread(target=func).start()
            """
            self.sync_func()
            self.staging = False
    
    def load(self, step: int = -1) -> bool:
        """
        Loads a checkpoint from disk, handling gradient accumulation state.
        
        When loading a checkpoint that was saved mid-accumulation, training will
        resume from that exact accumulation step. This ensures consistent gradient
        accumulation across training restarts.
        
        Args:
            step (int, optional): Specific step to load. Defaults to -1 (latest).
        
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise.
        """
        if not self.enable_checkpoint:
            logger.info("Tried to load the model but checkpointing is disabled. Returning...")
            return False
        if not os.path.isdir(self.folder):
            logger.error(f"Checkpoint folder {self.folder} does not exist. Returning...")
            return False
        if step != -1 and not os.path.isdir(self._create_checkpoint_id(step)):
            logger.error(f"Checkpoint {self._create_checkpoint_id(step)} does not exist. Returning...")
            return False

        if step == -1:
            step_counts = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                metadata_probe = os.path.join(self.folder, filename, ".metadata")
                if match and os.path.isfile(metadata_probe):
                    step_counts.append(int(match.group(1)))
            if not step_counts:
                return False
            step = max(step_counts)

        states = { "model": self.states["model"]} if step == 0 else self.states
        original_stateful_states = {}
        for k, v in states.items():
            if isinstance(v, Stateful):
                original_stateful_states[k] = v

        logger.info(f"Loading the checkpoint at step {step}.")
        start = time.monotonic()
        dcp.load(
            state_dict=states,
            checkpoint_id=self._create_checkpoint_id(step),
        )
        
        # Log warning if resuming from mid-accumulation checkpoint
        train_state = states.get("train_state")
        if train_state and hasattr(train_state, "accum_step") and train_state.accum_step != 0:
            logger.warning(
                f"Resuming from checkpoint at step {step} with non-zero accumulation "
                f"step (accum_step={train_state.accum_step}). Will continue accumulation "
                "from this point."
            )
        
        logger.info(f"Finished loading the checkpoint in {time.monotonic() - start:.2f} seconds.")

        states.update(original_stateful_states)
        return True
