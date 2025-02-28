import hydra
import torch.nn as nn
import wandb

from omegaconf import DictConfig, OmegaConf
from wandb_osh.hooks import TriggerWandbSyncHook
from pathlib import Path

from thesis.experiments.utils.data import create_dataset, create_train_val_dataloaders
from thesis.utils.logger import logger
from thesis.experiments.utils.model import create_model
from thesis.experiments.training.state import TrainState
from thesis.utils.callbacks import (
    CallbackManager,
    MetricLoggerCallback,
    GPUMemoryCallback,
    CheckpointCallback,
    PlottingCallback,
)
from thesis.experiments.utils.config import (
    check_experiment_exists,
    get_experiment_dir,
    save_experiment_config,
)
from thesis.distributed import is_distributed
from thesis.utils.pytorch import get_device_info
from thesis.utils.optimizer import build_optimizers
from thesis.utils.scheduler import build_lr_schedulers


@hydra.main(version_base="1.3.2", config_path="pkg://thesis.experiments.conf", config_name="config")
def main(config: DictConfig) -> None:
    # Check if experiment already exists
    if existing_exp := check_experiment_exists(config):
        logger.warning(
            f"Experiment with this config already exists at {existing_exp}. "
            "Use --force to rerun or modify config to run a new experiment."
        )
        if not config.get("force", False):
            return

    # Ensure parallelism config exists for distributed training
    if is_distributed() and not hasattr(config, "parallelism"):
        config.parallelism = {
            "dp_replicate": 1,
            "dp_shard": -1,  # Will be auto-inferred
            "cp": 1,
            "tp": 1,
            "pp": 1,
            "enable_loss_parallel": False,
        }
        logger.info("Created default parallelism config for distributed training.")

    # Get experiment directory
    exp_dir = get_experiment_dir(config)

    # Convert config to dict for wandb
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Pretty print config
    logger.info("=" * 50)
    logger.info("Experiment Configuration:")
    logger.info("=" * 50)

    def pretty_print(d, indent=0):
        for key, value in d.items():
            # Skip parallelism config if not in distributed mode
            if key == "distributed" and not is_distributed():
                continue
            prefix = "  " * indent + "├─"
            if isinstance(value, dict):
                logger.info(f"{prefix} {key}:")
                pretty_print(value, indent + 1)
            else:
                logger.info(f"{prefix} {key}: {value}")

    pretty_print(config_dict)
    logger.info("=" * 50)

    # Ask for user confirmation
    response = input("\n[CONFIRM]: Are these configurations correct? (y/n): ").lower().strip()
    if response != "y":
        logger.info("Experiment cancelled by user.")
        return

    # Create experiment directory and save configs after confirmation
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, exp_dir)

    # Update config to use experiment directory for logging
    config.logging.log_dir = str(exp_dir / "logs")

    # Setup wandb in offline mode with sync hook
    trigger_sync = TriggerWandbSyncHook()
    wandb.init(
        project="thesis",
        config=config_dict,
        mode="offline",
        name=f"{config.task.name}_{config.model_type}",
        dir=str(exp_dir),
    )

    # Setup device
    device_type, device = get_device_info(return_type="type_device")
    logger.info(f"Using device: {device}")

    # Create dataset and dataloaders
    dataset_info = create_dataset(config)
    train_loader, val_loader, expected_shapes = create_train_val_dataloaders(
        dataset_info, batch_size=config.training.batch_size, pin_memory=device_type == "cuda"
    )

    # Create model with validation against dataset shapes
    model = create_model(config, expected_shapes, device)
    wandb.watch(model, log_freq=100)  # Log model gradients

    # Setup training components
    criterion = nn.CrossEntropyLoss()

    # Build optimizers and schedulers
    optimizer = build_optimizers([model], config)
    scheduler = build_lr_schedulers(optimizer.optimizers, config)

    # Create training state
    state = TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        config=config,
    )

    # Initialize callbacks
    callbacks = []
    callbacks.append(MetricLoggerCallback(outdir=str(Path(config.logging.log_dir) / "metrics.jsonl"), args=config))
    if device_type == "cuda":
        callbacks.append(GPUMemoryCallback(device=device, log_interval=100, logger_callback=callbacks[0]))
    callbacks.append(
        CheckpointCallback(model=model, save_dir=config.logging.log_dir, save_interval=config.logging.save_period)
    )
    # Add plotting callback
    callbacks.append(
        PlottingCallback(
            outdir=str(Path(config.logging.log_dir) / "plots"),
            task_type="llm" if config.task.name == "pretraining" else "synthetic",
        )
    )
    callback_manager = CallbackManager(callbacks)

    logger.info("Starting training...")
    # Log training schedule info
    logger.info(
        f"Training Schedule:\n"
        f"- Total Steps: {config.training.max_steps:,}\n"
        f"- Batch Size: {config.training.batch_size}\n"
        f"- Validation Every: {config.training.eval_period} steps\n"
        f"- Checkpoint Every: {config.logging.save_period} steps\n"
        f"- Gradient Accumulation: {'enabled' if config.training.gradient_accumulation.enabled else 'disabled'}"
    )
    callback_manager.on_train_start(config=config, model=model)

    try:
        while state.step < config.training.max_steps:
            for batch in state.train_loader:
                loss = state.train_step(batch)
                state.train_losses.append(loss)

                # Log training metrics
                metrics = {
                    "train/loss": loss,
                    "train/lr": state.get_current_lr(),
                }
                wandb.log(metrics, step=state.step)
                callback_manager.on_step_end(state.step, loss, metrics)

                # Run validation if we have a validation loader
                if state.val_loader is not None and state.step % config.training.eval_period == 0:
                    val_loss, val_acc = state.validate()
                    metrics = state.update_best_metrics(val_loss, val_acc)
                    callback_manager.on_validation_end(state.step, metrics)

                    logger.info(f"Step {state.step}: val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
                    trigger_sync()  # Trigger wandb sync after validation

                state.step += 1
                if state.step >= config.training.max_steps:
                    break

        # Final validation
        val_loss, val_acc = state.validate()
        logger.info(f"Training finished. Final val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        wandb.log({"val/final_loss": val_loss, "val/final_accuracy": val_acc}, step=state.step)
        trigger_sync()  # Final sync
        callback_manager.on_train_end()

        # Mark experiment as completed only if training finished without errors
        (exp_dir / "completed.flag").touch()

    except Exception as e:
        logger.exception("Runtime error occurred during training. Experiment will not be marked complete.")
        raise
    finally:
        # Ensure wandb sync/finish is always called
        wandb.finish()


if __name__ == "__main__":
    main()
