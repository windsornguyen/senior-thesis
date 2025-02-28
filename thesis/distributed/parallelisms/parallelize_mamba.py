import time

import torch
import torch.nn as nn

from collections import defaultdict

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as dist_ckpt_wrapper,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from thesis.models.mamba import get_no_recompute_ops
from thesis.distributed.parallelisms.parallel_dims import ParallelDims
from thesis.utils.logger import logger
from thesis.utils.pretraining_config import TORCH_DTYPE_MAP, JobConfig

NO_RECOMPUTE_OPS = get_no_recompute_ops()


def parallelize_mamba(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> None:
    """
    Applies the following to the model:
        (1) Tensor parallelism
        (2) Activation checkpointing
        (3) Torch Compile
        (4) Data Parallelism

    NOTE: The passed-in model preferably should be on the meta device.
    Otherwise, the model must fit within the GPU or CPU memory.
    """
    if parallel_dims.tp_enabled:
        if (
            job_config.distributed.parallelism.enable_async_tensor_parallel
            and not job_config.kernel.compile
        ):
            raise RuntimeError("Async tensor parallelism requires --kernel.compile")
        apply_tp(
            model=model,
            tp_mesh=world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=job_config.distributed.float8.enable_float8_linear,
            enable_async_tp=job_config.distributed.parallelism.enable_async_tensor_parallel,
        )

    if job_config.distributed.activation_checkpoint.get("mode") != "none":
        apply_ac(model=model, ac_config=job_config.distributed.activation_checkpoint)

    # Enable per-layer compiling after AC wrapping but before FSDP
    if job_config.kernel.compile:
        apply_compile(model=model)

    # Apply FSDP or HSDP, potentially with Context Parallel
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model=model,
            dp_mesh=world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Successfully Applied HSDP to the model!")
        else:
            logger.info("Successfully Applied FSDP to the model!")

        if parallel_dims.cp_enabled:
            logger.warning(
                "Context Parallel (CP) is currently only supported for attention-based "
                "modules in native PyTorch. Mamba (SSM) does not use attention, so CP will "
                "NOT be applied effectively. We recommend disabling CP for this model."
            )

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model!")

    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP does not support >1 1D parallelism.")
        apply_ddp(
            model=model,
            dp_mesh=world_mesh,
            enable_compile=job_config.kernel.compile,
            enable_compiled_autograd=job_config.distributed.parallelism.enable_compiled_autograd,
        )


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8: bool,
    enable_async_tp: bool,
) -> None:
    """
    Apply Tensor Parallelism using the following strategy:
        (1) Parallelize the embeddings and shard its outputs.
        (2) Parallelize the root norm layer over the sequence dim.
        (3) Parallelize the final linear output layer.
    """
    parallelize_plan = {
        "tok_emb": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )
    }
    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=parallelize_plan,
    )

    # Parallel styles used for linear weights and their inputs may be different for float8 linear layers
    if enable_float8:
        """
        TODO: Once float8 configuration supports delayed scaling, add a check here
        to enforce supported float8 all-gather configurations.
        """
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            Preparefloat8ModuleInput,
        )
        rwp, cwp, prep = Float8RowwiseParallel, Float8ColwiseParallel, Preparefloat8ModuleInput
    else:
        rwp, cwp, prep = RowwiseParallel, ColwiseParallel, PrepareModuleInput

    # Apply tensor + sequence parallelism to every Mamba block
    for _, mamba_block in model.layers.items():
        layer_plan = {
            "ssm_norm": SequenceParallel(),
            "ssm": prep(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            # TODO: Mamba may or may not have a dedicated MLP layer
            "mlp_norm": SequenceParallel(),
            "mlp": prep(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.w1": cwp(),
            "mlp.w2": rwp(output_layouts=Shard(1)),
            "mlp.w3": cwp(),
        }

        parallelize_module(
            module=mamba_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model!"
    )


def _apply_ac_to_mamba_block(module: nn.Module, ac_config: dict) -> nn.Module:
    valid_ac_modes = ("full", "selective")
    if ac_config.get("mode") not in valid_ac_modes:
        raise ValueError(
            f"Invalid activation checkpointing mode: {ac_config.get('mode')}. "
            f"Valid modes: {valid_ac_modes}"
        )

    if ac_config.get("mode") == "full":
        return dist_ckpt_wrapper(module, preserve_rng_state=False)

    assert ac_config.get("mode") == "selective", f"Invalid activation checkpointing mode: {ac_config.get('mode')}"
    use_op_sac = ac_config.get("selective_ac_option") == "op"
    use_layer_sac = ac_config.get("selective_ac_option", "").isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective activation checkpointing option: {ac_config.get('selective_ac_option')}. "
            f"Valid options: 'op' or a positive int representing layer frequency."
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta) -> CheckpointPolicy:
            def _custom_policy(ctxt, func, *args, **kwargs):
                mode = "recompute" if ctxt.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1

                # Save output of all compute ops, except every second mm
                to_save = func in NO_RECOMPUTE_OPS and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )

                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def selective_checkpointing_ctxt_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return dist_ckpt_wrapper(
            module=module,
            selective_checkpointing_contexts_fn=selective_checkpointing_ctxt_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.get("selective_ac_option", "0"))
        dist_ckpt_wrapper.__dict__.setdefault("_count", 0)
        dist_ckpt_wrapper._count += 1
        if not ac_freq or dist_ckpt_wrapper._count % ac_freq == 0:
            return dist_ckpt_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, ac_config: dict) -> None:
    """
    Apply activation checkpointing to the model.
    """
    for layer_id, mamba_block in model.layers.named_children():
        mamba_block = _apply_ac_to_mamba_block(mamba_block, ac_config)
        model.layers.register_module(layer_id, mamba_block)
    logger.info(f"Applied {ac_config.get('mode')} activation checkpointing to the model!")


def apply_compile(model: nn.Module) -> None:
    """
    Apply torch.compile to each Mamba block. This makes compilation efficient
    due to repeated structure. Alternatively, one can compile the whole model
    (after applying DP).
    """
    logger.info("Compiling each Mamba block with torch.compile...")
    num_layers = len(list(model.layers.named_children()))
    start = time.perf_counter()
    for layer_id, mamba_block in model.layers.named_children():
        mamba_block = torch.compile(mamba_block, mode="max-autotune", fullgraph=True)
        model.layers.register_module(layer_id, mamba_block)
    end = time.perf_counter()
    logger.info(f"Finished compiling {num_layers} Mamba layers in {end - start:.4f} seconds.")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
) -> None:
    """
    Apply data parallelism to the model using Fully Sharded Data Parallel (FSDP).
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffloadPolicy()

    for layer_id, mamba_block in model.layers.items():
        if pp_enabled:
            # For pipeline parallelism, we do not reshard after the fwd pass
            # to avoid per-microbatch all-gathers, which can be expensive and non-overlapped!
            reshard_after_forward = False
        else:
            # As a small optimization, we do not reshard after fwd pass of last layer
            # since FSDP would just prefetch it immediately anyway
            reshard_after_forward = int(layer_id) < len(model.layers) - 1

        # Use new FSDP2 API to fully shard each layer
        fully_shard(
            module=mamba_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
) -> None:
    """
    Apply Distributed Data Parallelism (DDP) to the model.
    """
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"
    replicate(
        module=model,
        device_mesh=dp_mesh,
        bucket_cap_mb=100,
    )
    logger.info("Successfully applied DDP to the model!")
