# -*- coding: utf-8 -*-
"""Trains a Spectron model on synthetic sequence tasks."""

import time
import math
import pandas as pd

# Torch is only needed for DataLoader and initial dataset creation
import torch
from torch.utils.data import DataLoader

# JAX/Flax/Optax imports
import jax
import jax.numpy as jnp
import flax.linen as lnn
from flax.training import train_state
import optax

# Other imports
import numpy as np
import matplotlib

matplotlib.use("Agg")  # <-- Use Agg backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import logging

# Local imports
from thesis.experiments.synthetics import registry
from thesis.utils.logger import logger
from thesis.experiments.synthetics.spectron import Spectron, SpectronConfig, get_spectral_filters
from thesis.experiments.synthetics.jax_scan_attn_mqar import TransformerConfig, Transformer
from typing import Any, Dict, Tuple, Iterator

# Constants
IGNORE_IDX = -100
SEED = 1746

# ===----------------------------------------------------------------------=== #
# JAX Setup and Utilities
# ===----------------------------------------------------------------------=== #

# Configure JAX logging to suppress DEBUG messages (e.g., compilation cache details)
# Use WARNING to see only warnings and errors, INFO for slightly more detail
jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.WARNING)

# JAX device setup
try:
    # Check for GPU/TPU
    device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("tpu")[0]
    logger.info(f"Using JAX device: {device.platform.upper()}")
except IndexError:
    device = jax.devices("cpu")[0]
    logger.warning("No GPU/TPU found, using CPU.")


class TrainState(train_state.TrainState):
    # Optionally add more state fields, e.g., batch stats for BatchNorm
    pass


def numpy_collate(batch):
    """Collate function to convert PyTorch tensors to NumPy arrays."""
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], (torch.Tensor, np.ndarray)):
        return np.stack(batch)
    return batch


def torch_loader_to_jax_iterator(loader: DataLoader, device: jax.Device) -> Iterator[Tuple[jnp.ndarray, ...]]:
    """Creates an iterator yielding JAX arrays placed on the specified device."""
    for batch in loader:
        # Assume batch is a tuple of (inputs, targets) or similar
        jax_batch = jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x), device), batch)
        yield jax_batch


@partial(jax.jit, static_argnames=("ignore_index",))
def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, ignore_index: int = IGNORE_IDX) -> jnp.ndarray:
    """Computes cross-entropy loss over the full sequence, ignoring specified index."""
    # Logits shape: [B, L, V], Targets shape: [B, L]
    B, L, V = logits.shape
    targets = targets.reshape(B * L)  # Flatten targets
    logits = logits.reshape(B * L, V)  # Flatten logits

    # Ensure targets are valid before one-hot encoding
    valid_targets = jnp.maximum(targets, 0)  # Replace ignore_index with 0 for one_hot
    one_hot_targets = jax.nn.one_hot(valid_targets, num_classes=V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Element-wise loss for each token position
    token_losses = -jnp.sum(one_hot_targets * log_probs, axis=-1)  # Shape [B*L,]

    # Mask out ignored indices
    valid_mask = (targets != ignore_index).astype(jnp.float32)  # Shape [B*L,]
    masked_losses = token_losses * valid_mask

    # Compute mean loss over valid tokens across the entire batch * sequence
    mean_loss = jnp.sum(masked_losses) / jnp.maximum(jnp.sum(valid_mask), 1e-9)
    return mean_loss


@partial(jax.jit, static_argnames=("ignore_index",))
def compute_accuracy(logits: jnp.ndarray, targets: jnp.ndarray, ignore_index: int = IGNORE_IDX) -> jnp.ndarray:
    """Computes accuracy over the full sequence, ignoring specified index."""
    # Logits shape: [B, L, V], Targets shape: [B, L]
    B, L, V = logits.shape
    targets = targets.reshape(B * L)  # Flatten targets
    logits = logits.reshape(B * L, V)  # Flatten logits

    predictions = jnp.argmax(logits, axis=-1)
    valid_mask = targets != ignore_index
    correct = (predictions == targets) & valid_mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(valid_mask), 1e-9)
    return accuracy


@partial(jax.jit, static_argnames=("ignore_index",))
def compute_metrics(
    logits: jnp.ndarray, targets: jnp.ndarray, ignore_index: int = IGNORE_IDX
) -> Dict[str, jnp.ndarray]:
    """Computes loss and accuracy."""
    loss = cross_entropy_loss(logits, targets, ignore_index)
    accuracy = compute_accuracy(logits, targets, ignore_index)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


# ===----------------------------------------------------------------------=== #
# Training Step and Evaluation Step
# ===----------------------------------------------------------------------=== #


@partial(jax.jit, static_argnames=("model_apply_fn", "ignore_index", "schedule"))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, ...],
    model_apply_fn: Any,
    schedule: Any,
    ignore_index: int = IGNORE_IDX,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Performs a single training step."""
    inputs, targets = batch

    def loss_fn(params):
        logits = model_apply_fn({"params": params}, inputs, training=True)
        loss = cross_entropy_loss(logits, targets, ignore_index)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)

    # Apply gradients
    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, targets, ignore_index)
    # Add learning rate to metrics by calling the schedule with the current step
    current_lr = schedule(state.step)
    metrics["learning_rate"] = current_lr

    return state, metrics


@partial(jax.jit, static_argnames=("model_apply_fn", "ignore_index"))
def eval_step(
    state: TrainState, batch: Tuple[jnp.ndarray, ...], model_apply_fn: Any, ignore_index: int = IGNORE_IDX
) -> Dict[str, jnp.ndarray]:
    """Performs a single evaluation step."""
    inputs, targets = batch
    logits = model_apply_fn({"params": state.params}, inputs, training=False)
    metrics = compute_metrics(logits, targets, ignore_index)
    return metrics


# ===----------------------------------------------------------------------=== #
# Main Training Function
# ===----------------------------------------------------------------------=== #


def run_training(
    task_config: Dict[str, Any],
    model_config: Dict[str, Any],
    lr: float,
    wd: float,
    num_epochs: int,
    difficulty_dim: str,
    difficulty_val: Any,
    **kwargs,
) -> Dict[str, Any] | None:
    """Runs the training loop for a given configuration and returns results dict."""
    task_name = task_config["task_name"]
    batch_size = task_config["batch_size"]
    num_train_examples = task_config["num_train"]

    # --- Logging the run configuration ---
    run_desc = f"Task: {task_name}, Sweep: {difficulty_dim}={difficulty_val}, LR: {lr:.1e}, WD: {wd:.1e}"
    logger.info(f"--- Starting Run: {run_desc} ---")
    # logger.info(f"Task Config: {task_config}") # Keep concise for now
    # logger.info(f"Model Config: {model_config}") # Keep concise for now

    # Create JAX data iterators
    # logger.info("Creating JAX data iterators...") # Less verbose
    try:
        train_iterator, val_iterator = registry.create_data_loaders(**task_config)
        # logger.info("Data iterators created.") # Less verbose
    except Exception as e:
        logger.error(f"Error creating data loaders for {run_desc}: {e}")
        # Optionally log failure to results and return
        return None  # Return None on error

    # Calculate steps per epoch and total steps
    if num_train_examples < batch_size:
        logger.warning(
            f"num_train ({num_train_examples}) < batch_size ({batch_size}). "
            f"Setting steps_per_epoch=1. Expect partial batches."
        )
        steps_per_epoch = 1
    else:
        steps_per_epoch = num_train_examples // batch_size
    max_steps = steps_per_epoch * num_epochs
    warmup_steps = max_steps // 10  # 10% of total steps for warmup
    eval_period = max(1, steps_per_epoch // 2)  # Evaluate twice per epoch
    logger.info(f"Training for {num_epochs} epochs ({steps_per_epoch} steps/epoch, {max_steps} total steps).")
    # logger.info(f"Warmup steps: {warmup_steps}, Eval period: {eval_period} steps.") # Less verbose

    # Initialize Spectron model
    seq_len = task_config["seq_len"]  # Get seq_len from task config
    vocab_size = task_config["vocab_size"]  # Get vocab_size from task config
    spectral_filters = get_spectral_filters(seq_len, 24)  # Hardcoded 24 filters
    effective_model_config = SpectronConfig(
        dim=model_config.get("dim", 128),
        inter_dim=model_config.get("inter_dim", 512),
        num_heads=model_config.get("num_heads", 1),  # Fixed at 1 head
        num_layers=model_config.get("num_layers", 2),  # Will be changed later to 4
        seq_len=seq_len,  # Use seq_len from task config
        vocab_size=vocab_size,  # Use vocab_size from task config
        bsz=batch_size,
        dtype=jnp.float32,
        use_tensordot=False,  # Set to True as in backup
        spectral_filters=spectral_filters,
    )
    # logger.info(f"Effective Spectron Configuration: {effective_model_config}") # Less verbose
    model = Spectron(effective_model_config)

    # Initialize JAX PRNG keys
    key = jax.random.PRNGKey(SEED)
    # key, init_key, dropout_key = jax.random.split(key, 3) # No dropout key needed
    key, init_key = jax.random.split(key, 2)

    # Initialize model parameters
    dummy_input = jnp.zeros((batch_size, effective_model_config.seq_len), dtype=jnp.int32)
    try:
    variables = model.init(init_key, dummy_input, training=False)
    params = variables["params"]
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params)) / 1e6
        logger.info(f"Model initialized with {param_count:.2f}M parameters.")
    except Exception as e:
        logger.error(f"Error initializing model for {run_desc}: {e}")
        # Log failure
        return None  # Return None on error

    # Create learning rate schedule (Cosine decay with warmup)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,  # Start from 0
        peak_value=lr,  # Peak at the specified learning rate
        warmup_steps=warmup_steps,
        decay_steps=max_steps,  # Decay over the total number of steps
        end_value=lr / 10.0,  # End at 1/10th of peak LR
    )

    # Create optimizer
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=wd, b1=0.9, b2=0.98)

    # Create TrainState
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    state = jax.device_put(state, device)

    # Compute baseline loss
    total_vocab = effective_model_config.vocab_size
    baseline_loss = math.log(total_vocab)

    # Training loop setup
    loss_history = []
    acc_history = []
    lr_history = []
    eval_steps_list = []
    curr_step = 0
    running_val_acc = 0.0
    examples_seen = 0
    epochs_completed = 0
    best_val_acc = 0.0
    # Initialize the iterator for the first epoch
    train_epoch_iterator = iter(train_iterator)

    # --- Update tqdm description ---
    pbar = tqdm(
        total=max_steps,
        desc=run_desc,  # Use the detailed run description
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        # smoothing=0.1 # Optional: Smoothes the rate estimate
    )

    # Main training loop
    start_time = time.time()
    try:
    while curr_step < max_steps:
        try:
            batch = next(train_epoch_iterator)
                # Ensure batch has expected structure (input, target)
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    logger.error(f"Unexpected batch format: {type(batch)}. Skipping step {curr_step}.")
                    curr_step += 1  # Increment step even if batch is skipped
                    pbar.update(1)
                    continue
                # Check if batch is empty
                if batch[0].shape[0] == 0:
                    logger.warning(f"Empty batch encountered at step {curr_step}. Recreating iterator.")
                    # Don't increment epoch here, just get new iterator
                    train_epoch_iterator = iter(train_iterator)
                    continue  # Skip to next iteration to fetch new batch

        except StopIteration:
            # Epoch finished, reset iterator for the next epoch
            epochs_completed += 1
                if epochs_completed >= num_epochs:
                    logger.info(f"Target epochs ({num_epochs}) reached.")
                    break  # Exit loop if target epochs completed
                # logger.info(f"Epoch {epochs_completed} finished. Resetting training iterator.") # Less verbose
            train_epoch_iterator = iter(train_iterator)  # Re-create the iterator
                try:
            batch = next(train_epoch_iterator)
                except StopIteration:
                    logger.error(
                        f"Training iterator exhausted immediately after reset for epoch {epochs_completed + 1}. Stopping run."
                    )
                    break  # Stop if iterator is empty right away

            except Exception as e:
                logger.exception(f"Error fetching batch at step {curr_step}, epoch {epochs_completed}: {e}")
                # Decide whether to break or continue
                break  # Example: Stop the run on data loading errors

            # Check batch size, handle partial last batch if necessary
            current_batch_size = batch[0].shape[0]
            examples_seen += current_batch_size

            # --- Perform Training Step ---
            try:
                state, metrics = train_step(
                    state,
                    batch,
                    model_apply_fn=model.apply,
                    schedule=schedule,
                    ignore_index=IGNORE_IDX,
                )
            except Exception as e:
                logger.exception(f"Error during train_step at step {curr_step}: {e}")
                # Potentially log the error state (e.g., shapes) and break
                # logger.error(f"Input shape: {batch[0].shape}, Target shape: {batch[1].shape}")
                break  # Stop the run on training errors

        curr_loss = metrics["loss"].item()
        curr_lr = metrics["learning_rate"].item()
        loss_history.append(curr_loss)
        lr_history.append(curr_lr)
        curr_step += 1

        # Periodic evaluation
        if curr_step % eval_period == 0 or curr_step == max_steps:
            val_metrics_list = []
            num_val_batches_processed = 0
                max_val_batches = 50  # Evaluate on a fixed number of batches
            try:
                    # Use a fresh iterator for validation if needed, or reset
                    val_epoch_iterator = iter(val_iterator)
                    for val_batch in val_epoch_iterator:
                    if num_val_batches_processed >= max_val_batches:
                        break
                        # Check validation batch format
                        if not isinstance(val_batch, (tuple, list)) or len(val_batch) != 2:
                            logger.error(
                                f"Unexpected validation batch format: {type(val_batch)}. Skipping eval batch."
                            )
                            continue
                        if val_batch[0].shape[0] == 0:
                            logger.warning(f"Empty validation batch encountered.")
                            continue

                        try:
                            val_metrics = eval_step(
                                state, val_batch, model_apply_fn=model.apply, ignore_index=IGNORE_IDX
                            )
                    val_metrics_list.append(val_metrics)
                    num_val_batches_processed += 1
            except Exception as e:
                            logger.exception(
                                f"Error during eval_step at step {curr_step}, batch {num_val_batches_processed}: {e}"
                            )
                            # Decide if we should skip this batch or stop validation
                            break  # Stop validation phase on error

                except Exception as e:
                    logger.exception(f"Error iterating validation data at step {curr_step}: {e}")

            # Aggregate validation metrics
                if val_metrics_list:
                val_loss = jnp.mean(jnp.array([m["loss"] for m in val_metrics_list])).item()
                val_acc = jnp.mean(jnp.array([m["accuracy"] for m in val_metrics_list])).item()
                    running_val_acc = val_acc * 100.0
                acc_history.append(running_val_acc)
                eval_steps_list.append(curr_step)
                    best_val_acc = max(best_val_acc, running_val_acc)
            else:
                    logger.warning(f"Validation metrics list empty at step {curr_step}. No validation performed?")
                val_loss = float("nan")
                running_val_acc = float("nan")
                    # Don't append nan acc, record step
                eval_steps_list.append(curr_step)

            # Update progress bar postfix
        pbar.set_postfix(
            loss=f"{curr_loss:.3f}",
                # base=f"{baseline_loss:.3f}",
                v_acc=f"{running_val_acc:.1f}%",
            lr=f"{curr_lr:.1e}",
                # ex=f"{examples_seen // 1000}k",
            ep=f"{epochs_completed}",
                # best_v_acc=f"{best_val_acc:.1f}%" # Optionally show best accuracy
        )
        pbar.update(1)

        # --- End of training loop ---
    except Exception as e:
        logger.exception(f"Unhandled exception during training loop for {run_desc}: {e}")
        return None  # Return None on failure
    finally:
    pbar.close()
        end_time = time.time()
        training_duration = end_time - start_time

    logger.info(f"--- Run Finished: {run_desc} --- ")
    final_loss = loss_history[-1] if loss_history else float("nan")
    final_acc = acc_history[-1] if acc_history else float("nan")
    logger.info(f"Training duration: {training_duration:.2f} seconds")
    logger.info(f"Final loss: {final_loss:.4f} (Baseline: {baseline_loss:.4f})")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (Final: {final_acc:.2f}%) ")
    # logger.info(f"Total examples seen: {examples_seen}") # Less verbose
    # logger.info(f"Epochs completed: {epochs_completed}") # Less verbose

    # Log results
    result_data = {
        "task_name": task_name,
        "difficulty_dim": difficulty_dim,
        "difficulty_val": difficulty_val,
        "lr": lr,
        "wd": wd,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
        "final_train_loss": final_loss,
        "final_val_acc": final_acc,
        "best_val_acc": best_val_acc,
        "baseline_loss": baseline_loss,
        "param_count_M": param_count,
        "duration_s": training_duration,
        "completed_epochs": epochs_completed,
        "total_examples_seen": examples_seen,
        # Add model config details
        "model_dim": effective_model_config.dim,
        "model_inter_dim": effective_model_config.inter_dim,
        "model_heads": effective_model_config.num_heads,
        "model_layers": effective_model_config.num_layers,
        "model_filters": 24,  # Hardcoded
        # Add task config details
        "task_seq_len": task_config["seq_len"],
        "task_vocab_size": task_config["vocab_size"],
        "task_num_train": num_train_examples,
        "task_num_test": task_config["num_test"],
    }
    # Add specific task parameters if they exist
    if task_name == "fuzzy_in_context_recall":
        result_data["task_k_motif"] = task_config.get("k_motif_size")
        result_data["task_v_motif"] = task_config.get("v_motif_size")
    elif task_name == "noisy_in_context_recall":
        result_data["task_error_rate"] = task_config.get("error_rate")
    elif task_name == "selective_copying":
        result_data["task_tokens_to_copy"] = task_config.get("num_tokens_to_copy")

    # --- Return results dictionary ---
    # new_row = pd.DataFrame([result_data]) # Don't create df row here
    # results_df = pd.concat([results_df, new_row], ignore_index=True) # Don't modify df here
    return result_data  # Return the dict


def create_task_config(
    task_name: str,
    batch_size: int,
    seq_len: int | None = None,  # Allow override
    vocab_size: int | None = None,  # Allow override
    num_train: int | None = None,  # Allow override
    **task_kwargs,
) -> Dict[str, Any]:
    """Creates the configuration dictionary for a specific task."""
    # --- Base configurations, potentially overridden by sweeps ---
    base_config = {
        "task_name": task_name,
        "batch_size": batch_size,
        "backend": "jax",
        "device": device,
        "in_memory": True,
        "num_workers": 0,
        # --- Default difficulty parameters (will be overridden by sweeps) ---
        "seq_len": 128,
        "vocab_size": 16,
        "num_train": 12800,  # <-- Changed default from 64000
        "num_test": 1000,  # Use a fixed, smaller test set
    }

    # --- Task-specific defaults (will be overridden by sweeps) ---
    if task_name == "fuzzy_in_context_recall":
        base_config.update(
            {
                "k_motif_size": 3,
                "v_motif_size": 3,
            }
        )
    elif task_name == "noisy_in_context_recall":
        base_config.update(
            {
                "error_rate": 0.05,  # Default error rate
            }
        )
    elif task_name == "selective_copying":
        base_config.update(
            {
                "num_tokens_to_copy": 16,
            }
        )

    # --- Override with sweep-specific values if provided ---
    if seq_len is not None:
        base_config["seq_len"] = seq_len
    if vocab_size is not None:
        base_config["vocab_size"] = vocab_size
    if num_train is not None:
        base_config["num_train"] = num_train

    # --- Override with any specific task_kwargs passed ---
    base_config.update(task_kwargs)

    # --- Ensure derived values are consistent ---
    # Make num_test a fraction of num_train if not explicitly set otherwise
    if "num_test" not in task_kwargs:
        base_config["num_test"] = max(100, base_config["num_train"] // 10)

    # --- Final check for required keys ---
    required_keys = ["task_name", "batch_size", "seq_len", "vocab_size", "num_train", "num_test"]
    missing_keys = [k for k in required_keys if k not in base_config or base_config[k] is None]
    if missing_keys:
        raise ValueError(
            f"Missing required config keys for task '{task_name}': {missing_keys}. Base config: {base_config}"
        )

    return base_config


# ===----------------------------------------------------------------------=== #
# Main Experiment Orchestration
# ===----------------------------------------------------------------------=== #


def main():
    # === Fixed Hyperparameters ===
    num_epochs = 200
    base_model_config = {
        "dim": 128,
        "inter_dim": 512,
        "num_heads": 1,
        "num_layers": 2,
    }
    base_batch_size = 128

    # === Sweeps ===
    # Appendix B values
    learning_rates = [1e-4, 5e-4, 1e-3]
    weight_decays = [0.0, 0.1]
    difficulty_sweeps = {
        "fuzzy_in_context_recall": {
            "seq_len": [128, 256, 512],
            "vocab_size": [16, 32, 64],
            "num_train": [12800, 6400, 3200, 1600, 800],
            # k/v_motif_size fixed by default in create_task_config
        },
        "noisy_in_context_recall": {
            "seq_len": [128, 256, 512],
            "vocab_size": [16, 32, 64],
            "num_train": [12800, 6400, 3200, 1600, 800],
            "error_rate": [0.01, 0.05, 0.1, 0.2],
        },
        "selective_copying": {
            "seq_len": [128, 256, 512, 1024],
            "vocab_size": [16, 32, 64, 128],
            "num_train": [12800, 6400, 3200, 1600, 800],
            # num_tokens_to_copy fixed by default
        },
    }

    # --- Results DataFrame ---
    results_df = pd.DataFrame()
    results_filename = "spectron_synthetics_results.csv"

    # --- Experiment Loop ---
    total_runs = sum(
        len(dim_values) * len(learning_rates) * len(weight_decays)
        for task, sweeps in difficulty_sweeps.items()
        for dim_name, dim_values in sweeps.items()
    )
    run_counter = 0
    logger.info(f"Starting experiment with {total_runs} total runs planned.")

    for task_name, sweeps in difficulty_sweeps.items():
        logger.info(f"===== Starting Task Group: {task_name} =====")
        for dim_name, dim_values in sweeps.items():
            logger.info(f"--- Sweeping Difficulty Dimension: {dim_name} ({len(dim_values)} values) ---")
            for dim_value in dim_values:
                current_sweep_results = []  # Store results for the LR/WD runs

                for lr in learning_rates:
                    for wd in weight_decays:
                        run_counter += 1
                        logger.info(f"--- Preparing Run {run_counter}/{total_runs} ---")

                        # Create task config, overriding the swept dimension
                        task_kwargs = {dim_name: dim_value}
                        try:
                            task_config = create_task_config(
                                task_name=task_name,
                                batch_size=base_batch_size,
                                **task_kwargs,
                            )
                        except ValueError as e:
                            logger.error(f"Error creating task config for run {run_counter}: {e}. Skipping run.")
                            continue  # Skip to next run

                        # --- Run Training (returns dict or None) ---
                        run_result_data = run_training(
                            task_config=task_config,
                            model_config=base_model_config,
                            lr=lr,
                            wd=wd,
                            num_epochs=num_epochs,
                            difficulty_dim=dim_name,  # Pass dimension name
                            difficulty_val=dim_value,  # Pass dimension value
                            # No results_df passed here
                        )

                        if run_result_data is not None:
                            current_sweep_results.append(run_result_data)
                        else:
                            logger.warning(f"Run {run_counter} failed or returned no results.")

                # --- Select Best Run from Sweep ---
                if not current_sweep_results:
                    logger.error(
                        f"No successful runs for sweep: Task={task_name}, {dim_name}={dim_value}. Skipping results saving for this setting."
                    )
                    continue  # Skip to next dim_value

                best_run = max(current_sweep_results, key=lambda x: x.get("best_val_acc", -1.0))
                logger.info(
                    f"Best run for Task={task_name}, {dim_name}={dim_value}: Acc={best_run.get('best_val_acc', -1.0):.2f}%, LR={best_run.get('lr', 0):.1e}, WD={best_run.get('wd', 0):.1e}"
                )

                # --- Append Best Run to DataFrame ---
                new_row = pd.DataFrame([best_run])
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                # --- Save Intermediate Results (after each difficulty level sweep) ---
                try:
                    results_df.to_csv(results_filename, index=False)
                    logger.info(f"Intermediate results saved ({len(results_df)} rows total).")
                except Exception as e:
                    logger.error(f"Error saving intermediate results: {e}")

    logger.info(f"===== Experiment Finished =====")
    logger.info(f"Completed {run_counter} runs.")
    logger.info(f"Final results saved to {results_filename}")

    # Optional: Final analysis or plotting based on results_df
    # ...


if __name__ == "__main__":
    # Configure base logger
    log_level = logging.INFO  # Or DEBUG for more details
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    # Make pandas display wider columns
    pd.set_option("display.max_colwidth", 120)
    # pd.set_option('display.max_rows', None) # Optional: show all rows

    main()
