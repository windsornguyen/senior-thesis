import hydra
import torch
import torch.nn as nn
import wandb
import os

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm

from thesis.experiments.utils.data import create_dataset, create_train_val_dataloaders
from thesis.utils.logger import logger
from thesis.experiments.utils.model import create_model
from thesis.experiments.utils.config import get_experiment_dir, save_experiment_config


@hydra.main(version_base="1.3.2", config_path="pkg://thesis.experiments.conf", config_name="config")
def main(config: DictConfig) -> None:
    # Get experiment directory and save config
    exp_dir = get_experiment_dir(config)
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, exp_dir)

    # Setup wandb (not setting up for now)
    # wandb.init(
    #     project="thesis",
    #     config=OmegaConf.to_container(config, resolve=True),
    #     name=f"{config.task.name}_{config.model_type}",
    #     dir=str(exp_dir),
    # )

    # Setup device
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create dataset and dataloaders
    dataset_info = create_dataset(config)
    train_loader, val_loader, expected_shapes = create_train_val_dataloaders(
        dataset_info, batch_size=config.training.batch_size
    )

    # Create model
    model = create_model(config, expected_shapes, device)
    wandb.watch(model, log_freq=100)

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.max_steps)

    # Training loop
    best_val_loss = float("inf")
    step = 0

    logger.info("Starting training...")
    pbar = tqdm(total=config.training.max_steps, desc="Training")

    while step < config.training.max_steps:
        model.train()
        for batch in train_loader:
            if step >= config.training.max_steps:
                break

            # Move batch to device
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log training metrics
            wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Validation
            if step % config.training.eval_period == 0:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_inputs = val_batch[0].to(device)
                        val_targets = val_batch[1].to(device)

                        val_outputs = model(val_inputs)
                        val_loss += criterion(val_outputs.view(-1, val_outputs.size(-1)), val_targets.view(-1)).item()

                        predictions = val_outputs.argmax(dim=-1)
                        val_correct += (predictions == val_targets).sum().item()
                        val_total += val_targets.numel()

                val_loss /= len(val_loader)
                val_accuracy = 100 * val_correct / val_total

                # Log validation metrics
                wandb.log({"val/loss": val_loss, "val/accuracy": val_accuracy}, step=step)

                logger.info(f"Step {step}: val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                        },
                        os.path.join(exp_dir, "best_model.pt"),
                    )

            step += 1

    pbar.close()
    wandb.finish()

    # Save final model
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
        },
        os.path.join(exp_dir, "final_model.pt"),
    )

    # Mark experiment as completed
    (exp_dir / "completed.flag").touch()


if __name__ == "__main__":
    main()
