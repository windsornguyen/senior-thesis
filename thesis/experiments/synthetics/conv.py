import torch
import torch.nn as nn
import torch.autograd
import time
import numpy as np
from torch.profiler import profile, ProfilerActivity
import uuid
import os

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available")


# Define Conv1d and FFT-based convolution functions (depthwise style)
def conv1d_fn(x, weight, bias=None):
    return nn.functional.conv1d(x, weight, bias, groups=x.size(1))  # Depthwise


def fft_conv1d_fn(x, weight, bias=None):
    # Zero-pad weight to match input length
    pad_size = x.size(-1) - weight.size(-1)
    weight_padded = nn.functional.pad(weight, (0, pad_size))  # [channels, 1, seq_len]

    # FFT-based convolution
    x_fft = torch.fft.rfft(x, dim=-1)  # [batch, channels, seq_len//2+1]
    weight_fft = torch.fft.rfft(weight_padded, dim=-1)  # [channels, 1, seq_len//2+1]

    # Squeeze weight_fft for broadcasting with x_fft in depthwise fashion
    weight_fft = weight_fft.squeeze(1)  # [channels, seq_len//2+1]

    out_fft = x_fft * weight_fft  # [B, C, S_fft] * [C, S_fft] -> [B, C, S_fft]
    out = torch.fft.irfft(out_fft, n=x.size(-1), dim=-1)

    if bias is not None:
        out += bias.view(1, -1, 1)
    return out


# Compile functions for specific input shapes
def get_compiled_functions(batch, channels, seq_len, kernel_size):
    # Define sample inputs
    x = torch.randn(batch, channels, seq_len, device=device)
    weight = torch.randn(channels, 1, kernel_size, device=device)  # Depthwise weight
    bias = torch.randn(channels, device=device)

    # Compile with static shape
    compiled_conv1d = torch.compile(conv1d_fn, dynamic=False)
    compiled_fft_conv1d = torch.compile(fft_conv1d_fn, dynamic=False)

    # Warmup to ensure compilation
    compiled_conv1d_fn = None
    try:
        for _ in range(5):
            compiled_conv1d(x, weight, bias)
            torch.cuda.synchronize()
        compiled_conv1d_fn = compiled_conv1d
    except RuntimeError as e:
        if "canUse32BitIndexMath" in str(e):
            print(
                f"Warning: Conv1d failed compilation/warmup for shape (B={batch}, C={channels}, L={seq_len}) due to size limits. Skipping."
            )
        else:
            raise e  # Re-raise unexpected errors

    # FFT warmup (assuming it doesn't hit the same limit, or we want to see its error if it does)
    for _ in range(5):
        compiled_fft_conv1d(x, weight, bias)
        torch.cuda.synchronize()

    return compiled_conv1d_fn, compiled_fft_conv1d


# Profile function
def profile_function(fn, x, weight, bias, name, shape_info):
    # Warmup
    for _ in range(3):
        fn(x, weight, bias)
        torch.cuda.synchronize()

    # Sleep for power throttling
    time.sleep(0.1)

    # Profile speed and memory
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            fn(x, weight, bias)
            torch.cuda.synchronize()

    # Extract metrics
    # Filter for CUDA events first
    cuda_events = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    # Use self_device_time_total for kernel time and self_device_memory_usage for memory
    total_time = sum(event.self_device_time_total for event in cuda_events)
    total_memory = max(
        (event.self_device_memory_usage for event in cuda_events if event.self_device_memory_usage is not None),
        default=0,
    )

    return {
        "name": name,
        "batch": shape_info["batch"],
        "channels": shape_info["channels"],
        "seq_len": shape_info["seq_len"],
        "kernel_size": shape_info["kernel_size"],
        "time_us": total_time / 10,  # Average over 10 runs
        "memory_bytes": total_memory,
    }


# Test configurations
batch_sizes = [1, 2, 4]
channel_sizes = [128]
seq_lengths = [256, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288]
kernel_size = 16  # Fixed kernel size

# Collect results
results = []

# Run profiling
for batch in batch_sizes:
    for channels in channel_sizes:
        for seq_len in seq_lengths:
            shape_info = {"batch": batch, "channels": channels, "seq_len": seq_len, "kernel_size": kernel_size}

            # Generate inputs
            x = torch.randn(batch, channels, seq_len, device=device)
            weight = torch.randn(channels, 1, kernel_size, device=device)  # Depthwise
            bias = torch.randn(channels, device=device)

            # Get compiled functions for this specific shape
            compiled_conv1d, compiled_fft_conv1d = get_compiled_functions(batch, channels, seq_len, kernel_size)

            # Interleaved profiling
            if compiled_conv1d is not None:
                result_conv1d = profile_function(compiled_conv1d, x, weight, bias, "Conv1d", shape_info)
            else:
                # Create dummy result if conv1d failed compilation/warmup
                result_conv1d = {"name": "Conv1d", **shape_info, "time_us": float("inf"), "memory_bytes": float("inf")}

            # Profile FFT (assuming it works, or let it raise its own error if it fails)
            result_fft = profile_function(compiled_fft_conv1d, x, weight, bias, "FFT_Conv1d", shape_info)

            results.append(result_conv1d)
            results.append(result_fft)

            # Clear memory
            torch.cuda.empty_cache()

# Save results to file
output_file = f"conv1d_vs_fft_profile_{uuid.uuid4().hex}.txt"
with open(output_file, "w") as f:
    f.write("Conv1d vs FFT Convolution Profiling Results\n")
    f.write("=" * 50 + "\n")
    for res in results:
        f.write(f"Method: {res['name']}\n")
        f.write(
            f"Batch: {res['batch']}, Channels: {res['channels']}, Seq_Len: {res['seq_len']}, Kernel_Size: {res['kernel_size']}\n"
        )
        f.write(f"Time: {res['time_us']:.2f} us\n")
        f.write(f"Memory: {res['memory_bytes'] / 1024**2:.2f} MB\n")
        f.write("-" * 50 + "\n")

# Print summary
print(f"Profiling complete. Results saved to {output_file}")
print("Summary of when FFT is faster than Conv1d:")
for batch in batch_sizes:
    for channels in channel_sizes:
        for seq_len in seq_lengths:
            conv_res = next(
                r
                for r in results
                if r["name"] == "Conv1d"
                and r["batch"] == batch
                and r["channels"] == channels
                and r["seq_len"] == seq_len
            )
            fft_res = next(
                r
                for r in results
                if r["name"] == "FFT_Conv1d"
                and r["batch"] == batch
                and r["channels"] == channels
                and r["seq_len"] == seq_len
            )
            if fft_res["time_us"] < conv_res["time_us"]:
                print(f"FFT faster at Batch={batch}, Channels={channels}, Seq_Len={seq_len}:")
                print(f"  Conv1d: {conv_res['time_us']:.2f} us, FFT: {fft_res['time_us']:.2f} us")
