import contextlib
import logging
import math
import os
import sys
import time
from datetime import timedelta
from functools import lru_cache
from typing import Generator, Optional

import torch.distributed as dist
from dotenv import load_dotenv
from thesis.utils.color import Color

load_dotenv()

# TODO: remove these lol they should be in distributed
@lru_cache()
def is_distributed() -> bool:
    """
    Check if running in distributed mode.

    Verifies all required environment variables for torch.distributed
    are properly set and the package is available.

    Returns:
        bool: True if in a valid distributed environment
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    return (
        dist.is_available()
        and bool(port and addr)
        and size >= 1
        and rank >= 0
    )

@lru_cache()
def is_slurm_job() -> bool:
    """
    Check if running as a SLURM job (not torchrun).

    Returns:
        bool: True if running under SLURM but not in a torch.distributed environment
    """
    return ("SLURM_JOB_ID" in os.environ) and (not is_distributed())

@lru_cache()
def get_global_rank() -> int:
    """
    Get process rank, defaulting to 0 for non-distributed or local runs.

    Returns:
        int: Global rank of current process
    """
    if is_distributed():
        return int(os.environ["RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    return 0

@contextlib.contextmanager
def clean_env() -> Generator[None, None, None]:
    """
    A context manager to temporarily remove any environment variables related to
    distributed or cluster setups (SLURM, Submitit, WANDB, etc.), then restore them.
    """
    distributed_vars = {
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    }

    removed_vars = {}
    for var in list(os.environ.keys()):
        if (
            var in distributed_vars
            or var.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_"))
        ):
            removed_vars[var] = os.environ.pop(var)

    try:
        yield
    finally:
        os.environ.update(removed_vars)

color = Color(enabled=True)

class ColorLogFormatter(logging.Formatter):
    """
    Custom formatter for adding color, timing, and rank information to log messages.
    
    Features:
      - Colored output based on log level
      - Process rank information for distributed runs
      - Precise timestamps with microseconds
      - Time delta from program start (showing how long after start the message occurred)
      - Source file and line information
      - Multi-line message handling with proper indentation
      - Exception and stack trace formatting with colors
    """
    # Instead of referencing Color.*, use the color instance:
    LOG_COLORS = {
        logging.DEBUG:  color.combine(color.BRIGHT_MAGENTA, color.BOLD),
        logging.INFO:   color.CYAN,
        logging.WARNING: color.combine(color.YELLOW, color.BOLD),
        logging.ERROR:  color.combine(color.RED, color.BOLD),
        logging.CRITICAL: color.combine(color.RED, color.BOLD),
    }

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()
        self.rank = get_global_rank()
        self.show_rank = is_distributed() or is_slurm_job()

    def formatTime(self, record: logging.LogRecord) -> str:
        """
        Format the log record's creation time with microseconds and
        a delta showing time elapsed since program start.
        """
        subsecond, seconds = math.modf(record.created)
        formatted_time = time.strftime("%H:%M:%S", time.localtime(seconds))
        ms = f"{int(subsecond * 1000):03d}"
        timestamp = f"{formatted_time}.{ms}"

        time_delta = timedelta(seconds=round(record.created - self.start_time))
        elapsed = f"(+{time_delta})" if time_delta.total_seconds() > 0 else ""

        # Use the instance-based color:
        return color.wrap(f"{timestamp} {elapsed:>12}", color.GREEN)

    def formatPrefix(self, record: logging.LogRecord) -> str:
        """
        Build the prefix string, including rank (if any), log level, time, and file info.
        Ensures consistent alignment for all components.
        """
        level_color = self.LOG_COLORS.get(record.levelno, color.WHITE)
        level_str = color.wrap(f"{record.levelname:<8}", level_color)
        
        time_str = self.formatTime(record)

        # Format filename (truncate if needed)
        filename = record.filename
        if len(filename) > 12:
            filename = "..." + filename[-9:]
        file_info = color.wrap(f"{filename}:{record.lineno:<4}", color.BLUE)

        # Bold separator
        separator = color.wrap("┃", color.combine(color.WHITE, color.BOLD))

        # Handle rank display
        if self.show_rank and self.rank > 0:
            rank_str = color.wrap(f"[Rank {self.rank}]", color.BRIGHT_WHITE)
            return f"{rank_str:<10} {level_str} {time_str} {file_info} {separator} "

        return f"{level_str} {time_str} {file_info} {separator} "

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.formatPrefix(record)
        
        def strip_colors(s: str) -> str:
            """Remove all color codes for proper indent calculation."""
            s = s.replace(color.RESET, "")
            for code in self.LOG_COLORS.values():
                s = s.replace(code, "")
            s = s.replace(color.BLUE, "")
            s = s.replace(color.GREEN, "")
            s = s.replace(color.BRIGHT_WHITE, "")
            s = s.replace(color.combine(color.WHITE, color.BOLD), "")
            return s

        visible_length = len(strip_colors(prefix))
        indent = " " * visible_length
        
        content = record.getMessage()
        if "\n" in content:
            content = content.replace("\n", f"\n{indent}")

        # Exception info
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                formatted_exc = []
                for line in record.exc_text.split("\n"):
                    if "Traceback" in line:
                        formatted_exc.append(color.wrap(line, color.YELLOW))
                    elif any(x in line for x in ["Error:", "Exception:"]):
                        formatted_exc.append(color.wrap(line, self.LOG_COLORS[logging.ERROR]))
                    else:
                        formatted_exc.append(color.wrap(line, color.WHITE))
                content += f"\n{indent}" + f"\n{indent}".join(formatted_exc)

        # Stack info
        if record.stack_info:
            stack_text = self.formatStack(record.stack_info)
            content += f"\n{indent}{color.wrap(stack_text, color.WHITE)}"
            
        return f"{prefix}{content}"

def init_logger(
    log_file: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: str = "NOTSET",
    color_enabled: bool = True,   # <-- optional param if you want dynamic on/off
) -> logging.Logger:
    """
    Initialize a logger with color support, file output (optional), and
    correct streaming behavior (info below goes to stdout, warning above goes to stderr).
    """
    # You can override color.enabled here if you wish:
    color.enabled = color_enabled

    if os.environ.get("DEBUG", "").lower() in ("true", "1", "t"):
        level = "DEBUG"

    os.environ["KINETO_LOG_LEVEL"] = "5"

    root_logger = logging.getLogger()
    try:
        root_logger.setLevel(level.upper())
    except ValueError:
        root_logger.warning(f"Invalid logging level '{level}'. Falling back to NOTSET.")
        root_logger.setLevel(logging.NOTSET)

    logger = logging.getLogger(name)
    logger.handlers.clear()

    # Use the updated ColorLogFormatter that references the color instance
    formatter = ColorLogFormatter()

    # Handler: stdout for INFO and below
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Handler: stderr for WARNING and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Handler: optional file (rank 0 only)
    if log_file and get_global_rank() == 0:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Instantiate a default logger on import
logger = init_logger()

if __name__ == "__main__":
    print("\n=== Logger Test ===")

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    # Multi-line example
    logger.info(
        "Multi-line message showing indentation:\n"
        "First level indent\n"
        "  Second level indent\n"
        "    Third level indent"
    )

    # Exception example
    try:
        raise ValueError("Test exception for demonstration")
    except ValueError:
        logger.exception("Caught a test exception")

    if not is_distributed():
        print("\n=== Simulated Distributed Environment ===")
        with clean_env():
            # Simulate a distributed setup
            os.environ.update({
                "MASTER_PORT": "12345",
                "MASTER_ADDR": "localhost",
                "WORLD_SIZE": "2",
                "RANK": "1", # TODO: Does it make sense to have a logger be anything other than rank 0?
            })

            # Clear cached environment checks
            is_distributed.cache_clear()
            get_global_rank.cache_clear()

            dist_logger = init_logger()
            dist_logger.info("Message from rank 1")
            dist_logger.warning("Warning from rank 1")
