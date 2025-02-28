import torch
import einops
import time
from statistics import mean, stdev
from colorama import Fore, Style

def warmup(func, *args):
    for _ in range(10):
        _ = func(*args)
    torch.cuda.synchronize()

def time_function(func, *args, trials=30):
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        _ = func(*args)

        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times

def einops_method(y_shifted, M_y):
    return einops.einsum(y_shifted, M_y, "b s i d, d i o -> b s o")

def torch_einsum_method(y_shifted, M_y):
    return torch.einsum("bsid,dio->bso", y_shifted, M_y)

def main():
    # Test for a range of sequence lengths
    sequence_lengths = [1024, 2048, 4096, 8192, 16384]
    batch_size, k_y, d_hidden, d_out = 32, 3, 256, 128

    for seq_len in sequence_lengths:
        print(f"{Fore.CYAN}Running tests for sequence length: {seq_len}{Style.RESET_ALL}")

        y_shifted = torch.randn(batch_size, seq_len, k_y, d_hidden, device='cuda')
        M_y = torch.randn(d_hidden, k_y, d_out, device='cuda')

        # Warm up
        print(f"{Fore.YELLOW}Warming up...{Style.RESET_ALL}")
        warmup(einops_method, y_shifted, M_y)
        warmup(torch_einsum_method, y_shifted, M_y)

        # Benchmark einops method
        print(f"{Fore.YELLOW}Benchmarking einops method...{Style.RESET_ALL}")
        einops_times = time_function(einops_method, y_shifted, M_y)

        # Benchmark torch.einsum method
        print(f"{Fore.YELLOW}Benchmarking torch.einsum method...{Style.RESET_ALL}")
        torch_einsum_times = time_function(torch_einsum_method, y_shifted, M_y)

        # Calculate mean and variance
        einops_mean, einops_std = mean(einops_times), stdev(einops_times)
        torch_mean, torch_std = mean(torch_einsum_times), stdev(torch_einsum_times)
        einops_var, torch_var = einops_std ** 2, torch_std ** 2

        # Determine superior and inferior methods
        if einops_mean < torch_mean:
            superior = "einops"
            inferior = "torch.einsum"
            percentage_diff = ((torch_mean - einops_mean) / torch_mean) * 100
        else:
            superior = "torch.einsum"
            inferior = "einops"
            percentage_diff = ((einops_mean - torch_mean) / einops_mean) * 100

        # Results
        print(f"{Fore.GREEN}einops method: {einops_mean:.4f} ms ± {einops_std:.4f} ms (variance: {einops_var:.6f} ms²){Style.RESET_ALL}")
        print(f"{Fore.GREEN}torch.einsum method: {torch_mean:.4f} ms ± {torch_std:.4f} ms (variance: {torch_var:.6f} ms²){Style.RESET_ALL}")
        print(f"{Fore.BLUE}Superior method: {superior} ({percentage_diff:.2f}% faster than {inferior}){Style.RESET_ALL}")
        print("-" * 80)

if __name__ == "__main__":
    main()
