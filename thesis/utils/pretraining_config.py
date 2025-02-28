"""
Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/config_manager.py
"""

import argparse
import sys

from collections import defaultdict
from typing import Union

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def string_list(raw_arg):
    return [s.strip() for s in raw_arg.split(",") if s.strip()]


class JobConfig:
    """
    JobConfig is a helper class that manages the various training configurations.

    A default configuration is loaded from a TOML file. If a TOML file is not
    provided, or if there are missing keys, then the default configurations are
    loaded from the argparse defaults.

    If additional explicit command line arguments are provided in addition to the
    TOML file, they will override the TOML configuration and the argparse defaults.

    TL;DR -- The precedence order is as follows:
    1. Command line arguments
    2. TOML file
    3. Argparse defaults

    The argparsing logic is as follows:
    1. Each argument starts with <prefix>_ which is the section name in the TOML file.
    2. The name of the option in the TOML file is the same as the name of the argument.

    Example: model.name would translate to
        ```toml
        [model]
        name
        ```
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training argument parser.")

        # ==========================
        # [job] Section
        # ==========================
        self.parser.add_argument(
            "--job.config_file",
            type=str,
            default=None,
            help="Job config file",
        )

        self.parser.add_argument(
            "--job.dump_folder",
            type=str,
            default="./outputs",
            help="Folder to dump job outputs",
        )
        self.parser.add_argument(
            "--job.description",
            type=str,
            default="Mamba Debugging",
            help="Description and purpose of the job",
        )
        self.parser.add_argument(
            "--job.validate_job",
            default=False,
            action="store_true",
            help="Validate the setup of the job to ensure correctness",
        )
        self.parser.add_argument(
            "--job.print_args",
            action="store_true",
            help="Print the args to terminal",
        )

        # ==========================
        # [profiling] Section
        # ==========================
        self.parser.add_argument(
            "--profiling.enable_profiling",
            action="store_true",
            default=True,
            help="Whether to enable the PyTorch profiler",
        )
        self.parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profile_trace",
            help="Trace files location",
        )
        self.parser.add_argument(
            "--profiling.profile_freq",
            type=int,
            default=100,
            help="How often to collect profiler traces, in iterations",
        )
        self.parser.add_argument(
            "--profiling.enable_memory_snapshot",
            action="store_true",
            default=False,
            help="Whether to dump memory snapshot",
        )
        self.parser.add_argument(
            "--profiling.save_memory_snapshot_folder",
            type=str,
            default="memory_snapshot",
            help="Memory snapshot files location",
        )

        # ==========================
        # [metrics] Section
        # ==========================
        self.parser.add_argument(
            "--metrics.disable_color_printing",
            action="store_true",
            default=False,
            help="Disable colored output in metric printing",
        )
        self.parser.add_argument(
            "--metrics.log_freq",
            type=int,
            default=10,
            help="How often to log metrics to TensorBoard, in iterations",
        )
        self.parser.add_argument(
            "--metrics.enable_tensorboard",
            action="store_true",
            default=True,
            help="Whether to log metrics to TensorBoard",
        )
        self.parser.add_argument(
            "--metrics.save_tb_folder",
            type=str,
            default="tb",
            help="Folder to dump TensorBoard states",
        )
        self.parser.add_argument(
            "--metrics.enable_wandb",
            action="store_true",
            default=False,
            help="Whether to log metrics to the Weights and Biases platform",
        )
        self.parser.add_argument(
            "--metrics.rank_0_only",
            default=True,
            action="store_true",
            help="""
                Whether to save TensorBoard metrics only for rank 0 or for all ranks.
                When PP degree > 1, this option uses the 0th rank of the last stage pipeline group,
                which is the only stage that computes loss metrics.
            """,
        )

        # ==========================
        # [model] Section
        # ==========================
        self.parser.add_argument(
            "--model.name",
            type=str,
            default="mamba",
            help="The model to train",
        )
        self.parser.add_argument(
            "--model.variant",
            type=str,
            default="debug",  # Can be "learnable_filters", "tensorized_filters", etc.
            help="Which model config to train",
        )
        self.parser.add_argument(
            "--model.norm_type",
            type=str,
            default="rmsnorm",
            choices=["layernorm", "np_layernorm", "rmsnorm"],
            help="Type of layer normalization to use [layernorm, np_layernorm, rmsnorm]",
        )

        # ==========================
        # [optimizer] Section
        # ==========================
        self.parser.add_argument(
            "--optimizer.name",
            type=str,
            default="AdamW",
            help="Optimizer to use",
        )
        self.parser.add_argument(
            "--optimizer.lr",
            type=float,
            default=3e-4,
            help="Learning rate to use",
        )
        self.parser.add_argument(
            "--optimizer.fused",
            default=False,
            action="store_true",
            help="Whether the fused implementation (CUDA only) is used.",
        )
        self.parser.add_argument(
            "--optimizer.early_step_in_backward",
            action="store_true",
            help="""
            Whether to apply optimizer in the backward. Caution, optimizer_in_backward
            is not compatible with gradients clipping, users should not call
            register_post_accumulate_grad_hook after the optimizer is built.""",
        )

        # ==========================
        # [training] Section
        # ==========================
        self.parser.add_argument(
            "--training.dataset_path",
            type=str,
            help=(
                "Path to the dataset in the file system. If provided, data will be "
                "loaded from this path instead of downloaded."
            ),
        )
        self.parser.add_argument(
            "--training.num_epochs",
            type=int,
            default=1,
            help=("Number of times to go through the training dataset."),
        )
        self.parser.add_argument(
            "--training.microbatch_size",
            type=int,
            default=2,
            help="Number of sequences processed per GPU in a single forward pass (microbatch).",
        )
        self.parser.add_argument(
            "--training.seq_len",
            type=int,
            default=8192,
            help="Number of tokens in each sequence (input length).",
        )
        self.parser.add_argument(
            "--training.desired_tokens_per_step",
            type=int,
            default=524288,
            help="Total tokens to accumulate (across all GPUs) before performing a single optimizer step.",
        )
        self.parser.add_argument(
            "--training.warmup_steps",
            type=int,
            default=1907,  # lr scheduler warm up, normally 20% of the train steps
            help="Steps for lr scheduler warmup, normally 20% of --training.steps",
        )
        self.parser.add_argument(
            "--training.max_norm",
            type=Union[float, int],
            default=1.0,
            help="Max norm for gradient clipping",
        )
        self.parser.add_argument(
            "--training.model_dtype",
            type=str,
            default="bfloat16",
            choices=["float8", "float16", "float32", "bfloat16"],
            help="Model dtype for training",
        )
        self.parser.add_argument(
            "--training.steps",
            type=int,
            default=19073,
            help="How many train steps to run",
        )
        self.parser.add_argument(
            "--training.mixed_precision_param",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="""
                torch dtype to use for parameters when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_shard_degree > 1.
            """,
        )
        self.parser.add_argument(
            "--training.mixed_precision_reduce",
            type=str,
            default="float32",
            choices=["float32"],
            help="""
                torch dtype to use for reductions when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_degree > 1.
            """,
        )
        self.parser.add_argument(
            "--kernel.compile",
            action="store_true",
            default=True,
            help="Whether to compile the model",
        )
        self.parser.add_argument(
            "--training.gc_freq",
            type=int,
            default=50,
            help="Python garbage control scheduling interval, in steps",
        )
        self.parser.add_argument(
            "--training.spawn_method",
            type=str,
            default="forkserver",
            choices=["forkserver", "spawn", "fork"],
            help="Specify the multiprocessing start method to use.",
        )
        self.parser.add_argument(
            "--kernel.compile_cache_size_limit",
            type=int,
            default=8,
            help="Set the compile cache size limit in GB (default: 8 GB).",
        )
        self.parser.add_argument(
            "--training.seed",
            type=int,
            default=1746,
            help="Implement reproducibility by setting a Python, PyTorch and CUDA seed",
        )
        self.parser.add_argument(
            "--training.deterministic",
            action="store_true",
            default=False,
            help="Use deterministic algorithms wherever possible, may be slower",
        )

        # ==========================
        # [kernel] Section
        # ==========================
        self.parser.add_argument(
            "--kernel.matmul_allow_tf32",
            action="store_true",
            default=False,
            help="Allow TF32 matrix multiplication if supported (default: False).",
        )

        self.parser.add_argument(
            "--kernel.allow_bf16_reduced_precision_reduction",
            action="store_true",
            default=False,
            help="Allow BF16 reduced precision reduction if supported (default: False).",
        )

        self.parser.add_argument(
            "--kernel.set_float32_matmul_precision",
            type=str,
            default="high",
            choices=["medium", "high", "highest", None],
            help="Specify the precision for float32 matrix multiplications (default: None).",
        )

        self.parser.add_argument(
            "--kernel.detect_anomaly",
            action="store_true",
            default=False,
            help="Enable PyTorch anomaly detection for debugging (default: False).",
        )

        # ==========================
        # [distributed] Section
        # ==========================
        # ─────────────────────────────────────────────────────────────────────────────
        #  [distributed.parallelism] Sub-section
        # ─────────────────────────────────────────────────────────────────────────────
        self.parser.add_argument(
            "--distributed.activation_offloading.enable",
            action="store_true",
            default=False,
            help="Whether to enable activation offloading to CPU memory",
        )

        self.parser.add_argument(
            "--distributed.activation_offloading.use_pin_memory",
            action="store_true",
            default=True,
            help="Whether to use pinned memory for CPU tensors during offloading",
        )

        self.parser.add_argument(
            "--distributed.activation_offloading.use_streams",
            type=str,
            default="auto",  # Will be converted to None in OffloadConfig for auto-detection
            choices=["auto", "true", "false"],
            help="Whether to use multiple CUDA streams for overlap. 'auto' detects based on PyTorch version",
        )

        self.parser.add_argument(
            "--distributed.activation_offloading.max_fwd_stash_size",
            type=int,
            default=5,
            help="Maximum number of tensors to keep in forward stash",
        )

        self.parser.add_argument(
            "--distributed.activation_offloading.min_offload_size",
            type=int,
            default=1024,
            help="Minimum tensor size in bytes to qualify for offloading",
        )

        self.parser.add_argument(
            "--distributed.activation_offloading.virtual_memory_safe_pct",
            type=float,
            default=60.0,
            help="Maximum safe percentage of virtual memory to use",
        )

        self.parser.add_argument(
            "--distributed.parallelism.data_parallel_replicate_degree",
            type=int,
            default=1,
            help="""
                The `data_parallel_replicate_degree` argument specifies the degree of
                data parallelism for weight replication.
                
                When this value is greater
                than 1, weights will be replicated across `data_parallel_replicate_degree`
                ranks.
                
                If `data_parallel_shard_degree` is also greater than 1, the parallelism
                method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
                parallelism method used is DDP (Distributed Data Parallelism).

                1 means disabled.
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.data_parallel_shard_degree",
            type=int,
            default=-1,
            help="""
                The `data_parallel_shard_degree` argument specifies the degree of data
                parallelism for weight sharding.
                
                When this value is greater than 1, weights
                will be sharded across `data_parallel_shard_degree` ranks.
                
                If
                `data_parallel_replicate_degree` is also greater than 1, the parallelism
                method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
                parallelism method used is FSDP (Fully Sharded Data Parallelism).

                -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
                only one of `data_parallel_replicate_degree` and `data_parallel_shard_degree`
                can be negative.
                1 means disabled.
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.enable_cpu_offload",
            action="store_true",
            default=False,
            help="""
                Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP.
            """,
        )

        # Tensor Parallelism
        self.parser.add_argument(
            "--distributed.parallelism.tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor Parallelism degree. 1 means disabled.",
        )
        self.parser.add_argument(
            "--distributed.parallelism.disable_loss_parallel",
            action="store_true",
            default=False,
            help="Whether to apply loss parallel when sequence parallel is enabled",
        )

        # Pipeline Parallelism
        self.parser.add_argument(
            "--distributed.parallelism.pipeline_parallel_degree",
            type=int,
            default=1,
            help="""
                Pipeline Parallelism degree, or number of ranks. 1 means disabled.
                If using looped schedules, this still specifies the number of physical ranks, not the number
                of stages. Stages per rank are inferred from split points degree, and schedule.
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.pipeline_parallel_split_points",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Specify comma-separated names of modules to use as the beginning of a split point.

                e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
                the first containing all the layers up to layers.0,
                the second containing layers.0 and up to layers.2,
                the third containing layers.2 and all the remaining layers.

                Note: fully-automated splitting may be enabled in the future,
                but currently the split points must be specified manually.
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.pipeline_parallel_schedule",
            type=str,
            default="1F1B",
            help="""
                Specify the Pipeline Parallel schedule to use. The supported schedules are:
                https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
                The schedule must be compatible with the split points and stages_per_rank.

                Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
                and split_points = number of stages - 1.
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.pipeline_parallel_schedule_csv",
            type=str,
            default="",
            help="""
                Specify the path to the pipeline parallel schedule csv file to use.
                The pipeline_parallel_schedule argument must be either
                PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.pipeline_parallel_microbatches",
            type=int,
            default=None,
            help="""
                How many microbatches to split the global training batch into when using pipeline parallelism.
                The global training batch size must be evenly divisible by the number of microbatches.
                The default value will be the number of pipeline stages, if unspecified.
            """,
        )

        # Asynchronous and Experimental Parallelism Features
        self.parser.add_argument(
            "--distributed.parallelism.enable_async_tensor_parallel",
            action="store_true",
            default=False,
            help="""
                Whether to apply async tensor parallel (currently only effective when compile is enabled).
            """,
        )
        self.parser.add_argument(
            "--distributed.parallelism.enable_compiled_autograd",
            action="store_true",
            default=False,
            help="Enable CompiledAutograd to compile the backward.",
        )
        self.parser.add_argument(
            "--distributed.parallelism.context_parallel_degree",
            type=int,
            default=1,
            help="Context parallelism degree. 1 means disabled.",
        )
        self.parser.add_argument(
            "--distributed.parallelism.context_parallel_rotate_method",
            type=str,
            default="allgather",
            choices=["allgather", "alltoall"],
            help="""
                The collective to use in context parallel SDPA for kv shards exchange.
                'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
                'alltoall' means to all-to-all shuffle the kv shards.
                The default value is 'allgather'.
            """,
        )

        # ─────────────────────────────────────────────────────────────────────────────
        #  [distributed.checkpoint] Sub-section
        # ─────────────────────────────────────────────────────────────────────────────
        self.parser.add_argument(
            "--distributed.checkpoint.enable_checkpoint",
            action="store_true",
            default=False,
            help="Whether to enable checkpoint",
        )
        self.parser.add_argument(
            "--distributed.checkpoint.folder",
            type=str,
            default="checkpoint",
            help="""
                The folder to store the checkpoints.
                When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--distributed.checkpoint.folder}.
            """,
        )
        self.parser.add_argument(
            "--distributed.checkpoint.interval_type",
            type=str,
            default="steps",
            choices=["steps", "seconds"],
            help="Checkpointing interval unit of measurement ['steps', 'seconds']",
        )
        self.parser.add_argument(
            "--distributed.checkpoint.interval",
            type=int,
            default=500,
            help="Checkpointing interval, in steps or seconds depending on --distributed.checkpoint.interval_type",
        )
        self.parser.add_argument(
            "--distributed.checkpoint.model_weights_only",
            action="store_true",
            default=False,
            help="""
                When model_weights_only=True, only model weights will be saved at the end of training.
                With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
                When model_weights_only=False, the full checkpoint will be saved.
                A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
                The default value is false.
            """,
        )
        self.parser.add_argument(
            "--distributed.checkpoint.export_dtype",
            type=str,
            default="float32",
            choices=["float16", "bfloat16", "float32"],
            help="""
                Converts to the specified precision when training completes and model_weights_only=true.
                Currently supports float32, float16, and bfloat16.
                The default value is float32.
            """,
        )
        self.parser.add_argument(
            "--distributed.checkpoint.create_seed_checkpoint",
            action="store_true",
            default=False,
            help="""
                Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
                Note: requires user to call train.py without specifying any parallelisms, e.g., NGPU=1.
                Could be implemented as a separate script, but this way shares more code.
            """,
        )
        self.parser.add_argument(
            "--distributed.checkpoint.async_mode",
            type=str,
            default="disabled",
            choices=["disabled", "async", "async_with_pinned_mem"],
            help="""
                Which async checkpoint mode to use. Currently there are 3 different modes.
                1. "disabled": synchronized checkpointing will be used.
                2. "async": torch.distributed.checkpoint.async_save will be used.
                3. "async_with_pinned_mem": this option utilizes a dedicated pinned memory
                   space and creates a separate process for faster GPU->CPU transfer
                   performance and eliminating GIL contention. The cost is increased CPU
                   memory usage. If insufficient CPU memory is available, performance may
                   degrade due to memory paging. For most users, "async" should suffice as
                   the performance overhead is typically small (on the order of tens of
                   seconds) compared to checkpointing frequency. This mode can be employed
                   to pursue near-zero checkpointing times (e.g., < 1 second) given
                   appropriate hardware support such as ample CPU memory and fast PCIe.

                "disabled" is the default mode.
            """,
        )
        self.parser.add_argument(
            "--distributed.checkpoint.keep_latest_k",
            type=int,
            default=0,
            help="""
                Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
                0 is the default value.
            """,
        )
        self.parser.add_argument(
            "--distributed.checkpoint.load_step",
            type=int,
            default=-1,
            help="Load the checkpoint at the specified step. If -1, load the latest checkpoint.",
        )

        # ─────────────────────────────────────────────────────────────────────────────
        #  [distributed.activation_checkpoint] Sub-section
        # ─────────────────────────────────────────────────────────────────────────────
        self.parser.add_argument(
            "--distributed.activation_checkpoint.mode",
            type=str,
            default="full",
            choices=["none", "selective", "full"],
            help="Type of activation checkpointing to use ['none', 'selective', 'full']",
        )
        self.parser.add_argument(
            "--distributed.activation_checkpoint.selective_ac_option",
            type=str,
            default="2",  # 2 = checkpoint every other layer
            help="""
                Selective activation checkpointing options ['int', 'op'].
                'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
            """,
        )

        # ─────────────────────────────────────────────────────────────────────────────
        #  [distributed.float8] Sub-section
        # ─────────────────────────────────────────────────────────────────────────────
        self.parser.add_argument(
            "--distributed.float8.enable_float8_linear",
            action="store_true",
            default=True,
            help="""
                If true, swaps `torch.nn.Linear` with `Float8Linear`.
                This feature requires you to install 'torchao' which can be found
                here: https://github.com/pytorch/ao
            """,
        )
        self.parser.add_argument(
            "--distributed.float8.enable_fsdp_float8_all_gather",
            action="store_true",
            default=True,
            help="Whether to enable float8 all-gather in FSDP",
        )
        self.parser.add_argument(
            "--distributed.float8.precompute_float8_dynamic_scale_for_fsdp",
            action="store_true",
            default=True,
            help="Whether to precompute float8 scales dynamically for FSDP",
        )

        # ─────────────────────────────────────────────────────────────────────────────
        #  [distributed.comm] Sub-section
        # ─────────────────────────────────────────────────────────────────────────────
        self.parser.add_argument(
            "--distributed.comm.init_timeout_seconds",
            type=int,
            default=300,
            help="Timeout for communication operations during initialization and first train step.",
        )
        self.parser.add_argument(
            "--distributed.comm.train_timeout_seconds",
            type=int,
            default=100,
            help=(
                "Timeout for communication operations after the first train step -- "
                "usually a tighter bound than during initialization."
            ),
        )
        self.parser.add_argument(
            "--distributed.comm.trace_buf_size",
            type=int,
            default=20000,
            help="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled",
        )

        # ==========================
        # [memory_estimation] Section
        # ==========================
        self.parser.add_argument(
            "--memory_estimation.enabled",
            action="store_true",
            default=False,
            help="Whether to estimate memory usage for FSDP",
        )
        self.parser.add_argument(
            "--memory_estimation.disable_fake_mode",
            action="store_true",
            default=False,
            help="Whether to estimate memory under FakeTensorMode",
        )

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_cli(args_list)
        config_file = getattr(args, "job.config_file", None)

        # Build up a two-level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # To prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                from thesis.utils.logger import logger

                logger.error(f"Error while loading the configuration file: {config_file}")
                logger.error(f"Error details: {str(e)}")
                raise e

        # If split-points came from 'args' (from cmd line) it would have already been parsed into a list by that parser
        if (
            "distributed" in args_dict
            and "parallelism" in args_dict["distributed"]
            and "pipeline_parallel_split_points" in args_dict["distributed"]["parallelism"]
            and isinstance(args_dict["distributed"]["parallelism"]["pipeline_parallel_split_points"], str)
        ):
            parallelism = args_dict["distributed"]["parallelism"]
            parallelism["pipeline_parallel_split_points"] = string_list(parallelism["pipeline_parallel_split_points"])

        # Override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        self.args_dict = args_dict

        for section, section_values in args_dict.items():
            class_type = type(section.title().replace("_", ""), (), {})
            for key, value in section_values.items():
                setattr(class_type, key, value)
            setattr(self, section, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(dict)
        for k, v in vars(args).items():
            if "." in k:
                first_level_key, second_level_key = k.split(".", 1)
                args_dict[first_level_key][second_level_key] = v
            else:
                # Handle global or incorrectly prefixed arguments if any
                args_dict["global"][k] = v
        return args_dict

    def _validate_config(self) -> None:
        # TODO: Add more mandatory validations
        assert hasattr(self, "model") and hasattr(self.model, "name"), "Model name is required."
        assert hasattr(self, "model") and hasattr(self.model, "variant"), "Model variant is required."

    def parse_args_from_cli(self, args_list) -> tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # Aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument("--" + arg, action="store_true" if val else "store_false")
            elif arg.endswith("pipeline_parallel_split_points"):
                # Special case for list arguments
                aux_parser.add_argument("--" + arg, type=string_list, nargs="+")
            else:
                # Handle 'None' defaults appropriately
                arg_type = type(val)
                if arg_type is type(None):
                    arg_type = str  # Default to string if type is NoneType
                aux_parser.add_argument("--" + arg, type=arg_type)

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
