import contextlib
import os
import pickle
import time

import torch

from thesis.utils.pretraining_config import JobConfig
from thesis.utils.logger import logger

# The number of warm-up steps before the active step in each profiling cycle
WARMUP = 3

# How much memory allocation (bytes) / free ops to record in memory snapshot
MEMORY_SNAPSHOT_MAX_ENTRIES = 100_000

@contextlib.contextmanager
def profile(job_config: JobConfig, *, global_step: int = 0):
    enable_profiling = job_config.profiling.enable_profiling
    
    if enable_profiling:
        dump_dir = job_config.job.dump_folder
        save_trace_dir = job_config.profiling.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = job_config.profiling.profile_freq
        
        rank = torch.distributed.get_rank()
        
        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)
                
            logger.info(f"Dumping profiler traces at step {prof.step_num}.")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            logger.info(f"Finished dumping profiler traces in {time.monotonic() - begin:.4f} seconds.")
        
        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
        
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)
        
        warmup, active = WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert wait >= 0, f"profile_freq ({profile_freq}) must be greater than or equal to warmup + active ({warmup} + {active})"

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None

@contextlib.contextmanager
def enable_memory_snapshot(job_config: JobConfig, *, global_step: int = 0):
    enable_snapshot = job_config.profiling.enable_memory_snapshot
    
    if enable_snapshot:
        snapshot_folder = job_config.profiling.save_memory_snapshot_folder
        snapshot_dir = os.path.join(job_config.job.dump_folder, snapshot_folder)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                # Start recording memory history
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                self.step_num = step_num
                self.freq = freq
            
            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                # Only dump a snapshot if:
                #   1. We're exiting the context due to OOM, OR
                #   2. We've hit a multiple of 'freq'
                if (not exit_ctx) and (self.step_num % self.freq != 0):
                    return

                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # If OOM on step_num, dump as iteration_(step_num-1)_exit
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)

                logger.info(f"Dumping memory snapshot at step {curr_step}.")
                begin = time.monotonic()

                with open(
                    f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle", "wb"
                ) as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)

                logger.info(
                    f"Finished dumping memory snapshot in "
                    f"{time.monotonic() - begin:.4f} seconds."
                )
        
        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}.")
        profiler = MemoryProfiler(global_step, job_config.profiling.profile_freq)

        try:
            # Enter context
            yield profiler

        except torch.OutOfMemoryError as oom:
            logger.error(
                f"OOM at step {profiler.step_num}: {oom}. "
                "Dumping memory snapshot before exiting."
            )
            profiler.step(exit_ctx=True)
            raise  # Re-raise so code can handle or terminate properly

    else:
        # If snapshot is disabled, yield a no-op
        yield None
