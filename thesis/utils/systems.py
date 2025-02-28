import gc
import subprocess

from thesis.utils.logger import logger


class GarbageCollection:
    def __init__(self, gc_freq=1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        gc.collect(1)

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            gc.collect(1)

def get_peak_flops(device_name: str) -> int:
    """
    NOTE: We use hard-coded bf16 peak FLOPs numbers for NVIDIA's A100, H100, and H200 GPUs.
    """
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)

        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [line for line in result.stdout.splitlines() if "NVIDIA" in line and "H100" in line]

        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci command to get GPU name: {e}, using the device_name instead.")

    if "A100" in device_name:
        return 312e12  # Per https://www.nvidia.com/en-us/data-center/a100/
    elif "H100" in device_name:
        # Per https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # For H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # Per https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    else:  # For other GPU types, assume A100 (smh, you gpu poor)
        logger.warning(f"Peak FLOPs undefined for: {device_name}, assuming A100.")
        return 312e12

