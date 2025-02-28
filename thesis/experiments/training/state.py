from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Tuple
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.device_mesh import DeviceMesh

from thesis.utils.optimizer import Optimizers
from thesis.utils.scheduler import Schedulers
from thesis.distributed.checkpoint import StatefulModel
from thesis.distributed import is_distributed, get_global_rank, _get_world_size, dist_mean, dist_max
from thesis.distributed.parallelisms.parallel_dims import ParallelDims
from thesis.pretraining.utils import clip_grad_norm_

TrainStateValue = Union[torch.Tensor, BytesIO]
TrainStateDict = Dict[str, TrainStateValue]

SYNTHETIC_TASKS = ["copy", "assoc_recall", "ind_heads", "lds", "mqar", "doc_sim", "hash_hop"]


@dataclass
class TrainState(Stateful):
    """A unified training state manager for both standalone and distributed training.

    TrainState is a data container class that inherits from DCP's Stateful class.

    This class combines functionality from multiple training state managers:
    - Basic training state management (model, optimizer, metrics)
    - Distributed training support via Stateful
    - Gradient accumulation
    - Model state dict handling via StatefulModel
    - Comprehensive metric tracking
    - Support for multiple optimizers/schedulers via Optimizers/Schedulers

    It can be used for:
    - Simple synthetic experiments
    - Single-GPU training
    - Multi-GPU distributed training
    - Large-scale distributed training
    - Pipeline parallel training

    We use BytesIO for a number of reasons:
    (1) It minimizes I/O operations by keeping the data in memory
        while it's being processed.

    (2) It complement async checkpointing since the serialized data can be quickly
        passed off to another thread/process without blocking the main training loop.
    """

    # Core training components
    model: nn.Module | List[nn.Module]  # Support single or multiple models
    optimizer: Optimizers  # Use our custom Optimizers class
    train_loader: DataLoader
    scheduler: Optional[Schedulers] = None  # Use our custom Schedulers class
    val_loader: Optional[DataLoader] = None  # Make validation optional
    criterion: Optional[nn.Module] = None  # Make criterion optional
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    config: Optional[DictConfig] = None

    # Training progress
    step: int = 0
    epoch: int = 0
    accum_step: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0

    # Training state
    training: bool = True  # Flag to indicate training mode

    # Metrics history (local)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)

    # Distributed training metrics
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    # Distributed training flags
    is_distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # State management
    _stateful_model: Optional[StatefulModel] = None
    _device_mesh: Optional[DeviceMesh] = None

    def __post_init__(self):
        """Initialize distributed training if needed."""
        self.is_distributed = is_distributed()
        if self.is_distributed:
            self.local_rank = get_global_rank()
            self.world_size = _get_world_size()

            # Initialize device mesh based on config
            if self.config and hasattr(self.config, "parallelism"):
                parallel_dims = ParallelDims(
                    dp_replicate=self.config.parallelism.dp_replicate,
                    dp_shard=self.config.parallelism.dp_shard,
                    cp=self.config.parallelism.cp,
                    tp=self.config.parallelism.tp,
                    pp=self.config.parallelism.pp,
                    world_size=self.world_size,
                    enable_loss_parallel=self.config.parallelism.get("enable_loss_parallel", False),
                )
                self._device_mesh = parallel_dims.build_mesh(device_type=self.device.type)
            else:
                # Fallback to simple DP mesh for backward compatibility
                self._device_mesh = DeviceMesh(
                    device_type=self.device.type, mesh_shape=[self.world_size], mesh_dim_names=["dp"]
                )
        else:
            # For standalone setups
            self.local_rank = 0
            self.world_size = 1
            self._device_mesh = None

        # Convert single model to list for unified handling
        if isinstance(self.model, nn.Module):
            self.model = [self.model]

        # Initialize StatefulModel for model state management
        self._stateful_model = StatefulModel(self.model)

        # Validate optimizer type
        if not isinstance(self.optimizer, Optimizers):
            raise TypeError(f"optimizer must be an instance of Optimizers, got {type(self.optimizer)}")

        # Validate scheduler type if present
        if self.scheduler is not None and not isinstance(self.scheduler, Schedulers):
            raise TypeError(f"scheduler must be an instance of Schedulers, got {type(self.scheduler)}")

    def _process_synthetic_batch(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process outputs and targets for synthetic tasks.

        Handles special processing for tasks like selective copy where we need to
        trim the outputs to match target length.
        Returns processed outputs and targets ready for loss computation.
        """
        # Reshape outputs if they're sequence outputs (B, L, V)
        if outputs.dim() == 3:
            batch_size, seq_len, vocab_size = outputs.shape
            target_len = targets.size(1)  # get target sequence length

            # For copy task, we need to trim the outputs to match target length
            if self.config.task.name == "copy":
                if self.config.task.params.selective:
                    # Remove BOS, Delimiter, and EOS tokens
                    outputs = outputs[:, 1:-2, :]  # remove first and last two tokens
                else:
                    # Remove BOS, Delimiter, EOS tokens
                    outputs = outputs[:, 1:-2, :]  # remove first and last two tokens

                # Now outputs and targets should have same sequence length
                assert (
                    outputs.size(1) == target_len
                ), f"Output seq_len {outputs.size(1)} != target seq_len {target_len}"

            # Flatten sequences
            outputs = outputs.reshape(-1, vocab_size)
            if self.config.task.params.one_hot:
                targets = targets.reshape(-1, vocab_size)
            else:
                targets = targets.reshape(-1)

        # Convert one-hot targets to indices if needed
        if self.config.task.params.one_hot:
            targets = targets.argmax(dim=-1)

        return outputs, targets

    def _process_llm_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process outputs and targets for LLM pretraining.

        Simple processing: just flatten the sequences and keep targets as indices.
        Returns processed outputs and targets ready for loss computation.
        """
        if outputs.dim() == 3:  # (B, L, V)
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
        return outputs, targets

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss after appropriate processing."""
        return F.cross_entropy(outputs, targets)

    def forward_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model[0](inputs)  # Use first model by default # TODO: Not scalable to distributed so pls fix.

        # Route to appropriate processing based on task type
        if self.config.task.name in SYNTHETIC_TASKS:
            outputs, targets = self._process_synthetic_batch(outputs, targets)
        else:  # LLM pretraining
            outputs, targets = self._process_llm_batch(outputs, targets)

        loss = self._compute_loss(outputs, targets)
        return outputs, targets, loss

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        outputs, targets, loss = self.forward_step(batch)
        if self.training:
            self.step_optimizer(loss)
        return loss.item()

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Helper to reduce loss in distributed setting."""
        if self.is_distributed and self._device_mesh is not None:
            return dist_mean(loss, self._device_mesh)
        return loss

    def _reduce_metrics(self, val_tensor: torch.Tensor) -> tuple[float, float]:
        """Helper to compute distributed metrics."""
        if self.is_distributed and self._device_mesh is not None:
            return (dist_mean(val_tensor, self._device_mesh), dist_max(val_tensor, self._device_mesh))
        return val_tensor.item(), val_tensor.item()

    def state_dict(self) -> TrainStateDict:
        """Returns a complete state dictionary for checkpointing.

        Includes:
        - Training progress
        - Model state(s) via StatefulModel
        - Optimizer state
        - Scheduler state
        - Metrics history
        - Distributed training state
        """
        # Get model states using StatefulModel
        model_states = self._stateful_model.state_dict()

        # Get optimizer state (already handles multiple optimizers)
        optimizer_bytes = BytesIO()
        torch.save(self.optimizer.state_dict(), optimizer_bytes)

        # Get scheduler state if it exists (already handles multiple schedulers)
        scheduler_bytes = None
        if self.scheduler is not None:
            scheduler_bytes = BytesIO()
            torch.save(self.scheduler.state_dict(), scheduler_bytes)

        # Serialize metrics lists
        train_losses_bytes = BytesIO()
        torch.save(self.train_losses, train_losses_bytes)

        val_losses_bytes = BytesIO()
        torch.save(self.val_losses, val_losses_bytes)

        val_accuracies_bytes = BytesIO()
        torch.save(self.val_accuracies, val_accuracies_bytes)

        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)

        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)

        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)

        state_dict = {
            # Training progress
            "step": torch.tensor(self.step, dtype=torch.int32),
            "epoch": torch.tensor(self.epoch, dtype=torch.int32),
            "accum_step": torch.tensor(self.accum_step, dtype=torch.int32),
            "best_val_loss": torch.tensor(self.best_val_loss),
            "best_val_acc": torch.tensor(self.best_val_acc),
            # States
            "optimizer_state": optimizer_bytes,
            # Metrics
            "train_losses": train_losses_bytes,
            "val_losses": val_losses_bytes,
            "val_accuracies": val_accuracies_bytes,
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

        # Add scheduler state if it exists
        if scheduler_bytes is not None:
            state_dict["scheduler_state"] = scheduler_bytes

        # Add model states
        state_dict.update(model_states)

        return state_dict

    def load_state_dict(self, state_dict: TrainStateDict) -> None:
        """Loads complete state from a checkpoint dictionary."""
        # Load training progress
        self.step = state_dict["step"].item()
        self.epoch = state_dict["epoch"].item()
        self.accum_step = state_dict["accum_step"].item()
        self.best_val_loss = state_dict["best_val_loss"].item()
        self.best_val_acc = state_dict["best_val_acc"].item()

        # Load model states using StatefulModel
        model_state = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(
                (
                    "step",
                    "epoch",
                    "accum_step",
                    "best_val",
                    "optimizer",
                    "scheduler",
                    "train_losses",
                    "val_losses",
                    "val_accuracies",
                    "global_avg_losses",
                    "global_max_losses",
                    "log_steps",
                )
            )
        }
        self._stateful_model.load_state_dict(model_state)

        # Load optimizer state (handles multiple optimizers)
        state_dict["optimizer_state"].seek(0)
        self.optimizer.load_state_dict(torch.load(state_dict["optimizer_state"]))

        # Load scheduler state if it exists (handles multiple schedulers)
        if "scheduler_state" in state_dict and self.scheduler is not None:
            state_dict["scheduler_state"].seek(0)
            self.scheduler.load_state_dict(torch.load(state_dict["scheduler_state"]))

        # Load metrics
        state_dict["train_losses"].seek(0)
        self.train_losses = torch.load(state_dict["train_losses"])

        state_dict["val_losses"].seek(0)
        self.val_losses = torch.load(state_dict["val_losses"])

        state_dict["val_accuracies"].seek(0)
        self.val_accuracies = torch.load(state_dict["val_accuracies"])

        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(state_dict["global_avg_losses"])

        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(state_dict["global_max_losses"])

        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"])

    def step_optimizer(self, loss: torch.Tensor) -> None:
        """Perform a single optimization step with optional gradient accumulation."""
        # Handle distributed loss reduction
        loss = self._reduce_loss(loss)

        # Determine gradient accumulation settings
        grad_accum_enabled = self.config.training.gradient_accumulation.enabled
        manual_steps = self.config.training.gradient_accumulation.get("steps")
        desired_batch = self.config.training.gradient_accumulation.get("desired_batch_size")
        desired_tokens = self.config.training.gradient_accumulation.get("desired_tokens_per_step")

        if grad_accum_enabled:
            if manual_steps:
                # Option 1: Manual accumulation steps
                grad_accum_steps = manual_steps
            elif desired_batch:
                # Option 2a: Based on desired global batch size
                actual_batch = self.config.training.batch_size * self.world_size
                grad_accum_steps = desired_batch // actual_batch
            elif desired_tokens:
                # Option 2b: Based on desired tokens per step (e.g. LLM pretraining)
                tokens_per_step = self.config.training.batch_size * self.config.training.seq_len * self.world_size
                grad_accum_steps = desired_tokens // tokens_per_step
            else:
                grad_accum_steps = 1  # fallback if enabled but no specific config

            # Validate computed steps
            if grad_accum_steps < 1:
                raise ValueError(
                    f"Invalid gradient accumulation config resulted in {grad_accum_steps} steps. "
                    "Check your batch sizes and token counts."
                )

            # Scale loss for accumulation
            loss = loss / grad_accum_steps
        else:
            grad_accum_steps = 1

        # Backward pass
        loss.backward()

        # Only step optimizer after accumulating enough gradients
        self.accum_step += 1
        if self.accum_step >= grad_accum_steps:
            # Clip gradients if max_norm is specified
            if self.config and self.config.training.get("max_norm", None):
                all_params = [param for model in self.model for param in model.parameters()]
                clip_grad_norm_(
                    all_params,
                    self.config.training.max_norm,
                    foreach=True,
                    pp_mesh=None,  # TODO: We don't have PP mesh in experiments yet, need to add if we wanna use this in pretrain/train.py
                )

            # Step optimizer and scheduler
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            self.accum_step = 0

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Run validation and return (val_loss, val_accuracy)."""
        if self.val_loader is None:
            raise ValueError("Validation loader not provided")

        # Set all models to eval mode and store previous training state
        prev_training = self.training
        self.training = False
        for model in self.model:
            model.eval()

        total_loss = 0.0
        correct_sequences = 0
        total_sequences = 0

        for batch in self.val_loader:
            outputs, targets, loss = self.forward_step(batch)
            total_loss += loss.item()

            # Reshape for sequence accuracy computation
            batch_size = len(batch[0])
            if outputs.dim() == 2:  # (B*L, V)
                vocab_size = outputs.size(-1)
                # Calculate seq_len based on total size and batch_size
                seq_len = outputs.size(0) // batch_size
                outputs = outputs.view(batch_size, seq_len, vocab_size)
                if hasattr(self.config.task.params, "one_hot") and self.config.task.params.one_hot:
                    targets = targets.view(batch_size, seq_len, -1)
                else:
                    targets = targets.view(batch_size, seq_len)

            # Get predictions
            _, predicted = torch.max(outputs, dim=-1)  # (batch_size, seq_len)
            if hasattr(self.config.task.params, "one_hot") and self.config.task.params.one_hot:
                _, target_classes = torch.max(targets, dim=-1)  # (batch_size, seq_len)
            else:
                target_classes = targets

            # Check if entire sequences match
            sequence_matches = (predicted == target_classes).all(dim=1)
            correct_sequences += sequence_matches.sum().item()
            total_sequences += len(sequence_matches)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct_sequences / total_sequences

        # Restore previous training state
        self.training = prev_training
        for model in self.model:
            model.train()

        return avg_loss, accuracy

    def update_best_metrics(self, val_loss: float, val_acc: float) -> Dict[str, Any]:
        """Update best metrics and return a dictionary of metrics including improvement flags."""
        improved = False
        best_loss = False
        best_acc = False

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improved = True
            best_loss = True

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            improved = True
            best_acc = True

        # Track global metrics if in distributed mode
        val_loss_tensor = torch.tensor(val_loss, device=self.device)
        avg_loss, max_loss = self._reduce_metrics(val_loss_tensor)

        if self.is_distributed:
            self.global_avg_losses.append(avg_loss)
            self.global_max_losses.append(max_loss)
            self.log_steps.append(self.step)

        return {
            "improved": improved,
            "best_val_loss": best_loss,
            "best_val_acc": best_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "global_avg_loss": avg_loss if self.is_distributed else val_loss,
            "global_max_loss": max_loss if self.is_distributed else val_loss,
        }

    def get_current_lr(self) -> float:
        """Get the current learning rate from the first param group of the first optimizer."""
        return self.optimizer.param_groups[0]["lr"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to a dictionary for logging."""
        state_dict = {
            "step": self.step,
            "epoch": self.epoch,
            "accum_step": self.accum_step,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "current_lr": self.get_current_lr(),
        }

        # Add distributed training info if applicable
        if self.is_distributed:
            state_dict.update(
                {
                    "rank": self.local_rank,
                    "world_size": self.world_size,
                    "global_avg_loss": self.global_avg_losses[-1] if self.global_avg_losses else None,
                    "global_max_loss": self.global_max_losses[-1] if self.global_max_losses else None,
                }
            )

        return state_dict
