import os

import orjson as oj
import torch
import torch.nn as nn
import wandb

from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Union

from thesis.utils.logger import logger
from thesis.distributed import is_master_process
from thesis.utils.pytorch import get_num_params


@dataclass
class WandbArgs:
    job_type: Optional[str] = None
    dir: Optional[str] = None
    project: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List] = None
    group: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    config_exclude_keys: Optional[List[str]] = None
    config_include_keys: Optional[List[str]] = None
    anonymous: Optional[str] = None
    mode: Optional[str] = None
    allow_val_change: Optional[bool] = None
    resume: Optional[Union[bool, str]] = None
    force: Optional[bool] = None
    tensorboard: Optional[bool] = None
    sync_tensorboard: Optional[bool] = None
    monitor_gym: Optional[bool] = None
    save_code: Optional[bool] = None
    id: Optional[str] = None
    fork_from: Optional[str] = None
    resume_from: Optional[str] = None


@dataclass
class LoggingArgs:
    freq: int = 10
    acc_freq: Optional[int] = None
    wandb: Optional[WandbArgs] = None


class BaseCallback:
    """
    Base class for callbacks. Implement any subset of these methods in subclasses:
    - on_train_start: called at the beginning of training
    - on_step_end: called at the end of a training step
    - on_validation_end: called after completing a validation run
    - on_epoch_end: called at the end of an epoch
    - on_train_end: called at the end of training
    """

    def on_train_start(self, config: DictConfig, model: Any) -> None:
        pass

    def on_step_end(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        pass

    def on_validation_end(self, step: int, metrics: Dict[str, float]) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_train_end(self) -> None:
        pass


class MetricLogger:
    """
    Logs metrics to a jsonl file and optionally to wandb.
    If wandb config is provided, it initializes and logs metrics there as well.
    """

    def __init__(self, outdir: str, args: Optional[Any] = None):
        self.outdir = outdir
        self.jsonl_writer = None
        self.args = args

    def open(self):
        if self.jsonl_writer is None:
            os.makedirs(os.path.dirname(self.outdir), exist_ok=True)
            self.jsonl_writer = open(self.outdir, "a")

        if self.args is not None and self.args.logging.wandb is not None and is_master_process() and wandb.run is None:
            wandb.init(
                config=asdict(self.args),
                **asdict(self.args.logging.wandb),
            )

    def log(self, metrics: Dict[str, Any]):
        # wandb logging
        if self.args is not None and self.args.logging.wandb is not None and (wandb.run is not None):
            step = metrics.get("global_step", None)
            wandb.log(metrics, step=step)

        # jsonl logging
        metrics.update({"created_at": datetime.now(timezone.utc).isoformat()})
        print(oj.dumps(metrics).decode("utf-8"), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


class MetricLoggerCallback(BaseCallback):
    """
    A callback that logs metrics using MetricLogger.
    Integrates console logging, jsonl logging, and wandb logging.
    """

    def __init__(self, outdir: str = "logs/metrics.jsonl", args: Optional[Any] = None):
        super().__init__()
        self.outdir = outdir
        self.args = args
        self.logger = MetricLogger(outdir, args)

    def on_train_start(self, config: DictConfig, model: nn.Module) -> None:
        self.logger.open()
        model_params = get_num_params(model)
        init_metrics = {"global_step": 0, "event": "train_start", "model_params": model_params}
        logger.info(f"Model Parameter Count: %.2fM\n" % (model_params / 1e6))
        self.logger.log(init_metrics)

    def on_step_end(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        # Only log at a given frequency if specified in args
        if self.args is not None and hasattr(self.args, "logging") and self.args.logging.freq:
            if step % self.args.logging.freq == 0:
                step_metrics = {"global_step": step, "loss": loss}
                step_metrics.update(metrics)
                self.logger.log(step_metrics)

    def on_validation_end(self, step: int, metrics: Dict[str, float]) -> None:
        val_metrics = {"global_step": step, "event": "validation_end"}
        val_metrics.update(metrics)
        logger.info(f"Validation at step {step}: {metrics}")
        self.logger.log(val_metrics)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        epoch_metrics = {"epoch": epoch, "event": "epoch_end"}
        epoch_metrics.update(metrics)
        self.logger.log(epoch_metrics)

    def on_train_end(self) -> None:
        self.logger.log({"event": "train_end"})
        self.logger.close()
        logger.info("Training ended.")


class GPUMemoryMonitor:
    """
    Monitors GPU memory usage and provides formatted statistics.

    Converts raw memory metrics into human-readable formats with units.
    """

    GIB_IN_BYTES = 1024 * 1024 * 1024

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self.device_name: str = torch.cuda.get_device_name(self.device)
        self.device_index: int = torch.cuda.current_device()
        self.device_capacity: int = torch.cuda.get_device_properties(self.device).total_memory
        self.device_capacity_gib: float = self._to_gib(self.device_capacity)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes: int) -> float:
        """Convert bytes to GiB."""
        return memory_in_bytes / self.GIB_IN_BYTES

    def _to_pct(self, memory: int) -> float:
        """Convert a memory value to a percentage of the total device capacity."""
        return 100 * memory / self.device_capacity

    def get_peak_stats(self) -> Dict[str, Union[float, int, None]]:
        """
        Retrieve peak memory statistics from CUDA.

        Returns:
            A dictionary with:
              - max_active_gib: Peak active memory in GiB.
              - max_active_pct: Peak active memory as a percentage.
              - max_reserved_gib: Peak reserved memory in GiB.
              - max_reserved_pct: Peak reserved memory as a percentage.
              - num_alloc_retries: Allocation retry count.
              - num_ooms: Out-of-memory error count.
              - power_draw: Current power draw in milliwatts (if available).
        """
        cuda_info = torch.cuda.memory_stats(self.device)
        max_active = cuda_info.get("active_bytes.all.peak", 0)
        max_reserved = cuda_info.get("reserved_bytes.all.peak", 0)
        num_retries = cuda_info.get("num_alloc_retries", 0)
        num_ooms = cuda_info.get("num_ooms", 0)

        power_draw = torch.cuda.power_draw() if hasattr(torch.cuda, "power_draw") else None

        return {
            "max_active_gib": self._to_gib(max_active),
            "max_active_pct": self._to_pct(max_active),
            "max_reserved_gib": self._to_gib(max_reserved),
            "max_reserved_pct": self._to_pct(max_reserved),
            "num_alloc_retries": num_retries,
            "num_ooms": num_ooms,
            "power_draw": power_draw,
        }

    def reset_peak_stats(self) -> None:
        """Reset the peak and accumulated memory statistics."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def format_stats(self, stats: Dict[str, Union[float, int, None]] = None) -> str:
        """
        Return a one-line formatted string of memory stats with units.

        The output is formatted like:
        Device: <device_name> (Device <index>) | Total Memory: <GiB> GiB | Peak Active: <GiB> GiB (<pct>%) |
        Peak Reserved: <GiB> GiB (<pct>%) | Retries: <n> | OOMs: <n> | Power Draw: <W>
        """
        if stats is None:
            stats = self.get_peak_stats()

        power_draw_val = stats.get("power_draw")
        power_draw_str = f"{power_draw_val / 1000:.2f} W" if power_draw_val is not None else "N/A"

        formatted = (
            f"Device: {self.device_name} (Device {self.device_index}) | "
            f"Total Memory: {self.device_capacity_gib:.2f} GiB | "
            f"Peak Active: {stats['max_active_gib']:.2f} GiB ({stats['max_active_pct']:.2f}%) | "
            f"Peak Reserved: {stats['max_reserved_gib']:.2f} GiB ({stats['max_reserved_pct']:.2f}%) | "
            f"Retries: {stats['num_alloc_retries']} | "
            f"OOMs: {stats['num_ooms']} | "
            f"Power Draw: {power_draw_str}"
        )
        return formatted

    def __str__(self) -> str:
        """Return a concise summary of GPU memory stats."""
        mem_stats = self.get_peak_stats()
        return (
            f"{self.device_name} (Device {self.device_index}): {self.device_capacity_gib:.2f} GiB total, "
            f"{mem_stats['max_reserved_gib']:.2f} GiB peak reserved ({mem_stats['max_reserved_pct']:.2f}%)"
        )


class GPUMemoryCallback(BaseCallback):
    """
    Logs GPU memory usage at regular intervals during training.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        log_interval: int = 1000,
        logger_callback: MetricLoggerCallback = None,
    ) -> None:
        super().__init__()
        self.monitor = GPUMemoryMonitor(device)
        self.log_interval = log_interval
        self.logger_callback = logger_callback

    def on_step_end(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        """
        Log formatted GPU memory stats every `log_interval` steps.
        """
        if step > 0 and step % self.log_interval == 0:
            mem_stats = self.monitor.get_peak_stats()
            mem_stats["global_step"] = step
            mem_stats["event"] = "gpu_mem"
            formatted_stats = self.monitor.format_stats(mem_stats)
            logger.info(f"GPU Memory Stats at step {step}: {formatted_stats}")
            if self.logger_callback:
                self.logger_callback.logger.log(mem_stats)
            self.monitor.reset_peak_stats()


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints when validation metrics improve.
    Also supports saving at fixed intervals if desired.
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "checkpoints",
        save_interval: Optional[int] = None,
        save_best_only: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_best_only = save_best_only
        os.makedirs(save_dir, exist_ok=True)

    def on_step_end(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        """Save a checkpoint at fixed intervals if enabled."""
        if not self.save_best_only and self.save_interval and step > 0 and step % self.save_interval == 0:
            self.save_checkpoint(f"step_{step}")

    def on_validation_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Save a checkpoint if validation metrics improved."""
        if metrics.get("improved", False):
            self.save_checkpoint(f"best_step_{step}")
            # Also save with a descriptive name if it's the best loss or accuracy
            if metrics.get("best_val_loss", False):
                self.save_checkpoint("best_loss")
            if metrics.get("best_val_acc", False):
                self.save_checkpoint("best_acc")

    def on_train_end(self) -> None:
        """Save a final checkpoint at the end of training."""
        self.save_checkpoint("final")

    def save_checkpoint(self, identifier: Union[int, str]) -> None:
        """
        Save the model's state dictionary to a file.

        Args:
            identifier: A step number or descriptive string to label the checkpoint.
        """
        ckpt_path = os.path.join(self.save_dir, f"checkpoint_{identifier}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")


class CallbackManager:
    """
    Manages multiple callbacks, propagating events to each.
    """

    def __init__(self, callbacks: List[BaseCallback]) -> None:
        self.callbacks = callbacks

    def on_train_start(self, config: DictConfig, model: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(config, model)

    def on_step_end(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(step, loss, metrics)

    def on_validation_end(self, step: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_validation_end(step, metrics)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end()


class PlottingCallback(BaseCallback):
    """
    A callback that generates plots at the end of training using the metrics from the jsonl file.
    """

    def __init__(self, outdir: str, task_type: str = "synthetic"):
        super().__init__()
        self.outdir = outdir
        self.task_type = task_type
        self.metrics = []

    def on_step_end(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        self.metrics.append({"step": step, "train_loss": loss, **metrics})

    def on_validation_end(self, step: int, metrics: Dict[str, float]) -> None:
        self.metrics.append({"step": step, **metrics})

    def on_train_end(self) -> None:
        """Generate plots at the end of training."""
        from thesis.experiments.utils.plotting import plot_experiment

        logger.info("Generating training plots...")
        plot_experiment(metrics=self.metrics, output_dir=self.outdir, task_type=self.task_type, window_size=5)
        logger.info(f"Plots saved to {self.outdir}")
