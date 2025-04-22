# -*- coding: utf-8 -*-
"""Trains a Spectron model on synthetic sequence tasks."""

import time
import math

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
from thesis.experiments.synthetics.spectron_zoo import SimpleSpectron, Spectron, SpectronConfig, get_spectral_filters
from thesis.experiments.synthetics.spectron_zoo import TransformerConfig, Transformer
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
    dropout_rng: jax.random.PRNGKey,
    model_apply_fn: Any,
    schedule: Any,
    ignore_index: int = IGNORE_IDX,
) -> Tuple[TrainState, Dict[str, jnp.ndarray], jax.random.PRNGKey]:
    """Performs a single training step."""
    inputs, targets = batch
    new_dropout_rng, dropout_key = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits = model_apply_fn({"params": params}, inputs, training=True, rngs={"dropout": dropout_key})
        loss = cross_entropy_loss(logits, targets, ignore_index)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)

    # Compute gradient norm before clipping
    global_grad_norm = optax.global_norm(grads)

    # Apply gradients (optimizer update includes clipping)
    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, targets, ignore_index)
    # Add learning rate and gradient norm to metrics
    current_lr = schedule(state.step)
    metrics["learning_rate"] = current_lr
    metrics["grad_norm"] = global_grad_norm  # Report grad norm

    return state, metrics, new_dropout_rng


@partial(jax.jit, static_argnames=("model_apply_fn", "ignore_index"))
def eval_step(
    state: TrainState, batch: Tuple[jnp.ndarray, ...], model_apply_fn: Any, ignore_index: int = IGNORE_IDX
) -> Dict[str, jnp.ndarray]:
    """Performs a single evaluation step."""
    inputs, targets = batch
    # Get logits from model in eval mode
    logits = model_apply_fn({"params": state.params}, inputs, training=False)

    # Compute metrics (loss and accuracy) across the whole batch
    metrics = compute_metrics(logits, targets, ignore_index)
    return metrics


# ===----------------------------------------------------------------------=== #
# Main Training Function
# ===----------------------------------------------------------------------=== #


def main():
    E = 12800  # num_train_examples
    L = 256  # Sequence length
    V = 128  # Vocab size (base)
    batch_size = 128
    max_steps = 60000
    warmup_steps = max_steps // 100  # Changed from 10% to 1%
    eval_period = 250
    max_lr = 3e-4
    min_lr = max_lr * 0.1  # Set min_lr to 10% of max_lr for cosine decay
    DELIMS = 3  # Number of delimiter tokens

    # Create JAX data iterators directly using the registry
    logger.info("Creating JAX data iterators...")

    # Define common task parameters
    common_params = {
        "batch_size": batch_size,
        "num_train": E,
        "num_test": E // 10,
        "backend": "jax",
        "device": device,
        "in_memory": True,
        "vocab_size": V,
        "seq_len": L,
        "num_workers": 0,  # Set to 0 for JAX preloading
    }

    # Choose the task to run (example: in_context_recall)
    # You can switch this easily
    task_name_to_run = "selective_copying"
    task_specific_params = {}
    if task_name_to_run == "in_context_recall":
        task_specific_params = {"multi_query": False}
    elif task_name_to_run == "mqar":
        task_specific_params = {"num_pairs": 32, "alpha": 0.1}
    elif task_name_to_run == "copying":
        task_specific_params = {"num_tokens_to_copy": 16}
    elif task_name_to_run == "selective_copying":
        task_specific_params = {"num_tokens_to_copy": 16}
    elif task_name_to_run == "memorization":
        task_specific_params = {}
    elif task_name_to_run == "compression":
        task_specific_params = {}

    logger.info(f"Creating data loaders for task: {task_name_to_run}")
    train_iterator, val_iterator = registry.create_data_loaders(
        task_name=task_name_to_run, **common_params, **task_specific_params
    )
    logger.info("Data iterators created.")

    # Initialize Spectron model
    spectral_filters = get_spectral_filters(L, 24)
    config = SpectronConfig(
        dim=128,
        inter_dim=512,
        num_heads=1,
        num_layers=2,
        seq_len=L,
        vocab_size=V,
        bsz=batch_size,
        dtype=jnp.float32,
        use_tensordot=True,
        spectral_filters=spectral_filters,
    )
    logger.info(f"Spectron Configuration: {config}")
    model = Spectron(config)
    # config = TransformerConfig(
    #     dim=128,
    #     num_heads=1,
    #     num_layers=2,
    #     inter_dim=512,
    #     vocab_size=V,
    #     seq_len=L,
    #     dtype=jnp.float32,
    # )
    # model = Transformer(config)

    # Initialize JAX PRNG keys
    key = jax.random.PRNGKey(SEED)
    key, init_key, dropout_key = jax.random.split(key, 3)

    # Initialize model parameters
    # Ensure dummy input matches expected batch size and seq len
    # Use correct dtype if specified in config
    dummy_input = jnp.zeros((batch_size, config.seq_len), dtype=jnp.int32)
    variables = model.init(init_key, dummy_input, training=False)
    params = variables["params"]
    logger.info(
        f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M parameters."
    )

    # Create learning rate schedule (linear warmup, cosine decay)
    warmup_phase = optax.linear_schedule(
        init_value=0.0, end_value=max_lr, transition_steps=warmup_steps
    )  # Start warmup from 0
    # Calculate decay steps and alpha for cosine decay
    decay_steps = max_steps - warmup_steps
    alpha = min_lr / max_lr  # Ratio of final LR to peak LR (should be 0.1)
    cosine_decay_phase = optax.cosine_decay_schedule(init_value=max_lr, decay_steps=decay_steps, alpha=alpha)
    schedule = optax.join_schedules(
        schedules=[warmup_phase, cosine_decay_phase],
        boundaries=[warmup_steps],  # Switch to decay phase after warmup_steps
    )

    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.adamw(learning_rate=schedule, weight_decay=1e-2),
    )

    # Create TrainState
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    # Put state on device
    state = jax.device_put(state, device)

    # Compute baseline loss
    total_vocab = config.vocab_size
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
    reached_90 = False

    # Initialize the iterator for the first epoch
    train_epoch_iterator = iter(train_iterator)

    pbar = tqdm(
        total=max_steps,
        desc="Training",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    # Main training loop
    while curr_step < max_steps:
        try:
            batch = next(train_epoch_iterator)
        except StopIteration:
            # Epoch finished, reset iterator for the next epoch
            epochs_completed += 1
            logger.info(f"Epoch {epochs_completed} finished. Resetting training iterator.")
            train_epoch_iterator = iter(train_iterator)  # Re-create the iterator
            batch = next(train_epoch_iterator)

        examples_seen += batch[0].shape[0]  # Batch size

        state, metrics, dropout_key = train_step(
            state, batch, dropout_key, model_apply_fn=model.apply, schedule=schedule, ignore_index=IGNORE_IDX
        )

        curr_loss = metrics["loss"].item()
        curr_lr = metrics["learning_rate"].item()
        curr_gnorm = metrics["grad_norm"].item()  # Get grad norm
        loss_history.append(curr_loss)
        lr_history.append(curr_lr)
        curr_step += 1

        # Periodic evaluation
        if curr_step % eval_period == 0 or curr_step == max_steps:
            val_metrics_list = []
            num_val_batches_processed = 0
            max_val_batches = 10  # Limit evaluation batches like in mqar script
            try:
                # JAX iterator is reusable, iterate directly but limit batches
                for val_batch in val_iterator:
                    if num_val_batches_processed >= max_val_batches:
                        break
                    val_metrics = eval_step(state, val_batch, model_apply_fn=model.apply, ignore_index=IGNORE_IDX)
                    val_metrics_list.append(val_metrics)
                    num_val_batches_processed += 1
            except Exception as e:
                # Handle potential errors during validation iteration (e.g., if iterator is exhausted unexpectedly)
                logger.error(f"Error during validation loop: {e}")

            # Aggregate validation metrics
            if val_metrics_list:  # Ensure list is not empty
                val_loss = jnp.mean(jnp.array([m["loss"] for m in val_metrics_list])).item()
                val_acc = jnp.mean(jnp.array([m["accuracy"] for m in val_metrics_list])).item()
                running_val_acc = val_acc * 100.0  # Convert to percentage
                acc_history.append(running_val_acc)
                eval_steps_list.append(curr_step)
            else:
                logger.warning("Validation iterator yielded no batches.")
                val_loss = float("nan")
                running_val_acc = float("nan")
                # Don't append nan accuracy, but record step
                eval_steps_list.append(curr_step)

            if not reached_90 and running_val_acc >= 90.0:
                logger.info(
                    f"Reached 90% accuracy at step {curr_step}, examples seen: {examples_seen}, epochs: {epochs_completed}"
                )
                reached_90 = True

        pbar.set_postfix(
            loss=f"{curr_loss:.3f}",
            base=f"{baseline_loss:.3f}",
            acc=f"{running_val_acc:.1f}%",
            lr=f"{curr_lr:.1e}",
            ex=f"{examples_seen // 1000}k",
            ep=f"{epochs_completed}",
            gnorm=f"{curr_gnorm:.2f}",  # Report grad norm in postfix
        )
        pbar.update(1)

    pbar.close()
    logger.info("\nTraining complete!")
    logger.info(f"Final loss: {loss_history[-1]:.4f} (Baseline: {baseline_loss:.4f})")
    logger.info(f"Final accuracy: {acc_history[-1]:.2f}%")
    logger.info(f"Total examples seen: {examples_seen}")
    logger.info(f"Epochs completed: {epochs_completed}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="Training Loss")
    plt.plot(lr_history, label="Learning Rate (scaled)", color="purple", alpha=0.5)  # Plot LR too
    plt.axhline(y=baseline_loss, color="r", linestyle="--", label="Baseline Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss / LR")
    plt.title("Training Loss & LR Over Steps")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(eval_steps_list, acc_history, marker="o", color="orange", label="Validation Accuracy")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Over Steps")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("spectron_training_curves.png")
    logger.info("Saved training plots to spectron_training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
