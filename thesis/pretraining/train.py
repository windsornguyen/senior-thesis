import time
from datetime import timedelta

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim import (
    AdamW,
)  # TODO: See what best optimizer is for training in build_optimizers (anything good from torchao?)

from thesis.pretraining.metrics import build_device_memory_monitor, build_metric_logger
from thesis.pretraining.nd_dataloader import build_nd_data_loader
from thesis.pretraining.utils import clip_grad_norm_

# TODO: Move some more systems-related code here
from thesis.utils.systems import GarbageCollection, get_peak_flops

from thesis.utils.pretraining_config import JobConfig
from thesis.utils.logger import init_logger, color
from thesis.utils.optimizer import build_optimizers
from thesis.utils.scheduler import build_lr_schedulers
from thesis.utils.profiling import profile, enable_memory_snapshot
from thesis.utils.pytorch import get_num_params
from thesis.distributed.checkpoint import CheckpointManager, TrainState
from thesis.distributed import (
    setup_distributed,
    setup_env,
    cleanup_distributed,
    create_context_parallel_ctx,
    device_type,
    device_module,
    dist_max,
    dist_mean,
    set_determinism,
    set_pg_timeouts,
    validate_params_on_meta_device,
    validate_no_params_on_meta_device,
    EnvironmentArgs,
)

from thesis.distributed import get_train_context
from thesis.distributed.parallelisms.parallel_dims import ParallelDims
from thesis.distributed.parallelisms import models_pipelining_fns, models_parallelize_fns
from thesis.distributed.float8 import Float8Handler
from thesis.models import models_config, model_name_to_cls

# TODO: put this somewhere else
def linear_decay_with_warmup( # https://arxiv.org/pdf/2310.07831
    current_step: int, 
    warmup_steps: int, 
    num_steps: int, 
    max_lr: float = 3e-4, 
    min_lr: float = 3e-5,
) -> float:
    if current_step < warmup_steps:
        return min_lr + (max_lr - min_lr) * float(current_step) / float(max(warmup_steps, 1))
    else:
        return max_lr - (max_lr - min_lr) * float(current_step - warmup_steps) / float(max(num_steps - warmup_steps, 1))

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss as CrossEntropyLoss
    # from thesis.nn.functional.chunked_cross_entropy import ChunkedCrossEntropyLoss as CrossEntropyLoss # TODO: add this eventually?
except ImportError as e:
    print(f"Unable to import Triton-based cross entropy loss: {e}. Falling back to PyTorch implementation.")
    from torch.nn import CrossEntropyLoss


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    setup_env(EnvironmentArgs())

    use_color = not job_config.metrics.disable_color_printing
    logger = init_logger(color_enabled=use_color)

    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    gc_handler = GarbageCollection(gc_freq=job_config.training.gc_freq)

    device, global_rank, world_size, master_process = setup_distributed(job_config=job_config)
    device_module.set_device(device)

    parallel_dims = ParallelDims(
        dp_shard=getattr(job_config.distributed, "parallelism.data_parallel_shard_degree"),
        dp_replicate=getattr(job_config.distributed, "parallelism.data_parallel_replicate_degree"),
        cp=getattr(job_config.distributed, "parallelism.context_parallel_degree"),
        tp=getattr(job_config.distributed, "parallelism.tensor_parallel_degree"),
        pp=getattr(job_config.distributed, "parallelism.pipeline_parallel_degree"),
        world_size=world_size,
        enable_loss_parallel=not getattr(job_config.distributed, "parallelism.disable_loss_parallel"),
    )

    # Initialize device memory monitor and get peak FLOPs for MFU calculations
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = get_peak_flops(device_memory_monitor.device_name) # TODO: Incorporate GPU memory monitor
    logger.info(f"Peak FLOPs for MFU calculations: {gpu_peak_flops:.3e}")

    # Build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0  # TODO: Is this dp_rank a global rank??

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed and optionally enable deterministic mode (for debugging)
    set_determinism(world_mesh, device, job_config.training.seed, job_config.training.deterministic)

    # Build tokenizer
    # TODO

    # Initialize the dataset iterator
    logger.info("Initializing the dataset iterator...")
    # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size ~0.5M tokens
    # This is dataset and batch size dependent.
    total_tokens = 10_000_000_000
    dim_sizes = list(world_mesh.shape)
    train_loader = build_nd_data_loader(
        dim_sizes=dim_sizes,
        global_rank=global_rank,
        dataset_path=job_config.training.dataset_path,
        batch_size=job_config.training.microbatch_size,
        seq_len=job_config.training.seq_len,
        infinite=True,
        split="train",
    )
    val_loader = build_nd_data_loader(  # TODO: Add validation step
        dim_sizes=dim_sizes,
        global_rank=global_rank,  # TODO: Do we need global ranks for the nd-dataloader?
        dataset_path=job_config.training.dataset_path,
        batch_size=job_config.training.microbatch_size,
        seq_len=job_config.training.seq_len,
        infinite=True,
        split="val",
    )
    data_iterator = iter(train_loader)
    logger.info("Successfully initialized the dataset iterator!")

    model_name = job_config.model.name
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.variant]

    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = 200064  # o200k_base tokenizer
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(f"Building {model_name} {job_config.model.variant} with {model_config}...")
    # TODO: use MetricLoggerCallback to get model params (?) Do we need that class?
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
        validate_params_on_meta_device(model)

    # A no-op handler if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)

    # Swap to Float8Handler based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # Log the model size
    model_param_count = get_num_params(model, exclude_embeddings=True)
    num_flops_per_token = model.get_num_flops_per_token(
        num_non_embed_params=model_param_count,
        num_layers=model_config.num_layers,
        dim=model_config.dim,
        seq_len=job_config.training.seq_len,
    )
    logger.info(
        f"{color.BLUE}Model {model_name} {job_config.model.variant} "
        f"{color.RED}size: {model_param_count:,} total parameters{color.RESET}"
    )

    # Define a loss function to be shared by Pipeline Parallel and SPMD training
    # TODO: This can be optimized later via online softmax, CUDA/Triton kernels, etc.
    def loss_fn(inputs, targets):
        return torch.nn.functional.cross_entropy(inputs.flatten(0, 1).float(), targets.flatten(0, 1))

    # TODO: Compiling loss function can cause CUDA errors, disabling for now...
    logger.warning("Compiling the loss function can cause CUDA errors, disabling for now...")
    # if job_config.kernel.compile:
    #     loss_fn = torch.compile(loss_fn)

    # Move sharded model to the CPU/GPU to initialize weights via DTensor
    if getattr(job_config.distributed, "checkpoint.create_seed_checkpoint"):
        # (1) If we're creating a new "seed" checkpoint for the model, we typically
        #     put all parameters on CPU for simplicity. We also don't need a separate
        #     buffer device here, so buffer_device=None.
        init_device = "cpu"
        buffer_device = None
    elif getattr(job_config.distributed, "parallelism.enable_cpu_offload"):
        # (2) If CPU offload is enabled, we still want to initialize parameters on CPU,
        #     but we might create certain large buffers (e.g., RoPE frequencies) on GPU
        #     directly for runtime efficiency. So we set buffer_device = device_type (GPU).
        init_device = "cpu"
        buffer_device = device_type
    else:
        # (3) Otherwise, default both parameters and buffers to the same device_type
        #     (e.g., GPU).
        init_device = device_type
        buffer_device = None

    # Apply parallelisms and initializations
    if parallel_dims.pp_enabled:
        # Apply PT-D Pipeline Parallelism on meta device
        pp_schedule, model_parts = models_pipelining_fns[model_name](
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )

        """
        For PP with looped schedules, each item in model parts is one stage-model-chunk.

        We need to iterate through the model parts to apply SPMD parallelisms, compilation,
        optimizer, and checkpointing.
        """
        for model_part in model_parts:
            # Apply SPMD-style PT-D techniques
            models_parallelize_fns[model_name](model_part, world_mesh, parallel_dims, job_config)
            model_part.to_empty(device=init_device)
            with torch.no_grad():
                model_part.init_weights(buffer_device=buffer_device)
            model_part.train()
    else:
        # Apply PT-D Tensor Parallelism, activation checkpointing, torch.compile, and Data Parallelism
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
            # TODO: add dtype conversion here
        model.train()
        model_parts = [model]
    validate_no_params_on_meta_device(model)

    # Device memory statistics
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # We now build the optimizer after applying parallelisms and dtype conversions to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers, job_config)

    # Initialize the training state
    logger.info("Initializing training state...")
    train_state = TrainState()

    # Load initial checkpoints, if any
    logger.info("Initializing the checkpoint manager...")
    checkpoint = CheckpointManager(
        dataloader=train_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if getattr(job_config.distributed, "checkpoint.create_seed_checkpoint"):  # TODO: What is a seed checkpoint?
        assert world_size == 1, "Must create seed checkpoint using one GPU, to disable sharding..."
        checkpoint.save(curr_step=0, force=True)
        logger.info("Successfully created the seed checkpoint!")
        return

    checkpoint.load(step=getattr(job_config.distributed, "checkpoint.load_step"))

    # Initialize the metric logger
    logger.info("Initializing the metric logger...")
    metric_logger = build_metric_logger(job_config, parallel_dims)
    logger.info("Successfully initialized the metric logger!")

    """
    Plot losses loaded from checkpoint (if any) to TensorBoard.

    NOTE: Loss info after the last log step before checkpoint save will NOT be plotted.
    This can be avoided by setting checkpoint.interval to be a multiple of log_freq.
    """
    if train_state.step > 0:
        for i, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[i],
                "loss_metrics/global_max_loss": train_state.global_max_losses[i],
            }
            metric_logger.log(metrics, step=step)

    # Initialize the training loop
    logger.info("Initializing the training loop...")
    train_context = get_train_context(job_config=job_config)

    # Initialize variables used to track info for metric logging
    grad_norm = -1.0
    losses_since_last_log = []
    num_toks_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()
    checkpoint.reset()

    microbatch_size = job_config.training.microbatch_size
    seq_len = job_config.training.seq_len
    desired_tokens_per_step = job_config.training.desired_tokens_per_step
    training_steps = job_config.training.steps
    warmup = job_config.training.warmup_steps
    num_epochs = job_config.training.num_epochs

    total_tokens = 10_000_000_000

    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with microbatch_size={microbatch_size}, seq_len={seq_len}, "
        f"desired_tokens_per_step={desired_tokens_per_step}, total steps={training_steps} "
        f"(warmup={warmup})"
    )

    # Basic epoch math (if you have total_tokens and num_epochs)
    steps_per_epoch = total_tokens // desired_tokens_per_step
    max_steps = steps_per_epoch * num_epochs

    # Tokens processed by a single forward pass (one GPU microbatch) * all GPUs
    tokens_per_step = microbatch_size * seq_len * world_size

    # Basic checks to avoid bizarre configs
    if desired_tokens_per_step < tokens_per_step:
        raise ValueError(
            f"Invalid config: desired_tokens_per_step ({desired_tokens_per_step}) is smaller than "
            f"the tokens in one forward pass ({tokens_per_step}). "
            "This would produce zero or negative gradient accumulation steps."
        )

    if (desired_tokens_per_step % tokens_per_step) != 0:
        logger.warning(
            f"desired_tokens_per_step ({desired_tokens_per_step}) is not perfectly divisible by "
            f"tokens_per_step ({tokens_per_step}). "
            f"You may get partial tokens or need fractional accum steps."
        )

    # Number of gradient accumulation steps to reach `global_bsz` tokens each update
    grad_accum_steps = desired_tokens_per_step // tokens_per_step
    
    # Current gradient accumulating step (restore from checkpoint else 0)
    train_state.accum_step = train_state.accum_step or 0

    if master_process:
        logger.info("──────────────────────────────────────────────────────────────────────────")
        logger.info("Token Flow Per Gradient Update:")
        logger.info(
            f"- Microbatch size per GPU: {microbatch_size}\n"
            f"- Sequence length:         {seq_len}\n"
            f"- Total GPUs (world_size): {world_size}\n"
            f"→ Single forward pass = {tokens_per_step:,} tokens across all GPUs"
        )
        logger.info(
            f"→ Gradient accumulation steps = {grad_accum_steps} "
            f"→ Total tokens per gradient update = {desired_tokens_per_step:,}"
        )
        logger.info(f"Training for {training_steps} steps in total.")

        if training_steps != max_steps:
            logger.warning(
                f"Mismatch in configured steps vs. dataset-based steps:\n"
                f" - Configured steps = {training_steps}\n"
                f" - Computed steps   = {max_steps}  "
                f"(= {steps_per_epoch} steps/epoch × {num_epochs} epoch(s))\n"
                "Ensure your `steps` aligns with total tokens and desired tokens per update."
            )
        logger.info("──────────────────────────────────────────────────────────────────────────")

    logger.info("Training loop initialized! The training run will now begin.")

    # loop here?
    # we have num_epochs, TODO: Add support for multiple epochs, or maybe just read from num_steps (if num_steps > 1 epoch)
    with (
        profile(job_config, global_step=train_state.step) as torch_profiler,
        enable_memory_snapshot(job_config, global_step=train_state.step) as memory_profiler,
    ):
        # or loop here?
        while train_state.step < job_config.training.steps:
            is_last_step = train_state.step == job_config.training.steps - 1
            gc_handler.run(train_state.step)

            # Clear gradients at start of each accumulation cycle
            if train_state.accum_step == 0:
                optimizers.zero_grad()

            train_state.accum_step += 1
            
            # Some technical state variables
            accum_steps_left_in_cycle = grad_accum_steps - train_state.accum_step
            steps_left_in_training   = training_steps - train_state.step
            current_accum_target = min(accum_steps_left_in_cycle, steps_left_in_training)
            should_sync = (train_state.accum_step == current_accum_target) or is_last_step

            # Time the dataloading latency
            data_load_start = time.perf_counter()
            batch = next(data_iterator) # TODO: Ensure that for each new epoch, we shuffle the data or something
            data_loading_times.append(time.perf_counter() - data_load_start)
            inputs, targets = batch
            num_toks_since_last_log += targets.numel()

            # Move inputs, targets to device
            inputs, targets = inputs.to(device_type), targets.to(device_type)

            # Reduce allreduce comms overhead if not gradient syncing
            if grad_accum_steps > 1 and world_size > 1:
                model.set_requires_gradient_sync(should_sync)

            # Create context/sequence parallelism context, if enabled
            maybe_context_parallel = ( # TODO: Log and disable if model doesn't have any attention in it for now.
                create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[inputs, targets] + [m.freqs_cis for m in model_parts],  # TODO: We do not have RoPE in Flash STU...?
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={inputs, targets},
                    cp_rotate_method=getattr(job_config.distributed, "parallelism.context_parallel_rotate_method"),
                )
                if parallel_dims.cp_enabled
                else None
            )

            # PP path (NOTE: bwd pass is computed within step() here)
            if parallel_dims.pp_enabled:  
                is_first_stage = pp_mesh.get_local_rank() == 0
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                # TODO: Gradient scaling kind of hard to do with PP
                # See: https://github.com/pytorch/torchtitan/issues/803#issuecomment-2613298730
                with train_context(maybe_context_parallel):
                    if is_first_stage:
                        pp_schedule.step(inputs)
                    elif is_last_stage:
                        losses = []
                        pp_schedule.step(target=targets, losses=losses)
                    else:
                        pp_schedule.step()

                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                # Accumulate losses across pipeline microbatches
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if is_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else: # Non-PP path
                with train_context(maybe_context_parallel):
                    preds = model(inputs)
                    loss = loss_fn(preds, targets) / current_accum_target

                    del preds  # Free before bwd to avoid peaking memory

                    loss.backward()

            # TODO: Need to add validation loader somewhere

            if parallel_dims.pp_enabled:
                loss_for_logging = loss.detach() # Bwd pass already done, so we don't rescale
            else:
                loss_for_logging = loss.detach() * grad_accum_steps

            if should_sync:
                # clip_grad_norm_ gathers all rank-wise norms and returns the final global norm
                grad_norm = clip_grad_norm_(
                    [param for model_part in model_parts for param in model_part.parameters()],
                    job_config.training.max_norm,
                    foreach=True,
                    pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
                )

                loss = loss_for_logging

                # Step the optimizers and schedulers
                checkpoint.sync_staging()
                optimizers.step()
                lr_schedulers.step()

                # Calculate float8 dynamic amax/scale for all parameters for FSDP2
                # This issues a single all-reduce for all parameters at once for better perf
                float8_handler.maybe_precompute_float8_dynamic_scale_for_fsdp(model_parts)

                # Update only after a real gradient update
                train_state.step += 1
                train_state.accum_step = 0
                checkpoint.save(train_state.step, force=is_last_step)

            # Log every real global step
            if train_state.step == 1 or (train_state.step % job_config.metrics.log_freq == 0 and is_last_step):
                losses_since_last_log.append(loss)

                # Convert each item to float for average
                loss_values = [l.item() for l in losses_since_last_log]
                avg_loss = sum(loss_values) / len(loss_values)
                max_loss = max(loss_values)

                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    avg_loss, max_loss = (
                        dist_mean(avg_loss, world_mesh["dp_cp"]),
                        dist_max(max_loss, world_mesh["dp_cp"]),
                    )

                # Update the training state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(avg_loss)
                train_state.global_max_losses.append(max_loss)

                time_delta = time.perf_counter() - time_last_log

                # Tokens per second, per device
                tps = num_toks_since_last_log / (time_delta * parallel_dims.non_data_parallel_size)

                # Compute MFU - for its defn and calculation, see the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flops_per_token * tps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times) if data_loading_times else 0
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta if time_delta > 0 else 0

                device_mem_stats = device_memory_monitor.get_peak_stats()

                # Log our majestic metrics
                metrics = {
                    "loss_metrics/global_avg_loss": avg_loss,
                    "loss_metrics/global_max_loss": max_loss,
                    "throughput(tps)": tps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                    "stability_metrics/norm": grad_norm,
                }
                metric_logger.log(metrics, step=train_state.step)
                logger.info(
                    f"{color.CYAN}step: {train_state.step:2}  "
                    f"{color.GREEN}loss: {avg_loss:7.4f}  "
                    f"{color.YELLOW}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.BLUE}tps: {round(tps):,}  "
                    f"{color.MAGENTA}mfu: {mfu:.2f}%{color.RESET}"
                    f"{color.WHITE}grad_norm: {grad_norm:.2f}{color.RESET}"
                )

                # Clear for the next logging window
                losses_since_last_log.clear()
                num_toks_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            else:
                # If we did not hit the log step, just accumulate losses for next time.
                losses_since_last_log.append(loss)

            # Step the profilers
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # Reduce timeout after first real training step for faster signal
            if train_state.step == 1:
                set_pg_timeouts(
                    timeout=timedelta(seconds=getattr(job_config.distributed, "comm.train_timeout_seconds")),
                    world_mesh=world_mesh,
                )

    # Training loop concluded!
    cleanup_distributed()

    metric_logger.close()
    logger.info("Training run concluded.")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
