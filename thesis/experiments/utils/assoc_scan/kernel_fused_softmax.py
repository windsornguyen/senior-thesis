"""NOT CAUSAL WITHOUT MASK (!!!)"""

import torch
import triton
import triton.language as tl

from triton.testing import do_bench


def get_softmax_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=s, num_warps=w)
        for bs in [64, 128, 256, 512, 1024]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    if triton.runtime.driver.active.get_current_target().backend == "hip":
        configs.extend(
            [triton.Config({"BLOCK_SIZE": bs}, num_stages=1, num_warps=4, waves_per_eu=3) for bs in [64, 128, 256]]
        )
    return configs


def keep_config(conf, n_cols=None):
    BLOCK_SIZE = conf.kwargs["BLOCK_SIZE"]
    num_warps = conf.num_warps
    if BLOCK_SIZE >= 512 and num_warps < 8:
        return False
    if BLOCK_SIZE < 128 and num_warps > 4:
        return False
    if n_cols is not None and BLOCK_SIZE > n_cols:
        return False
    return True


@triton.autotune(
    configs=[conf for conf in get_softmax_configs() if keep_config(conf)],
    key=["n_cols"],
)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, x_stride_0, out_stride_0, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_idx = pid
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = x_ptr + row_idx * x_stride_0 + col_offsets
    output_ptrs = out_ptr + row_idx * out_stride_0 + col_offsets
    x = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    m = tl.max(x, axis=0)
    y = x - m
    exp_y = tl.exp(y)
    sum_exp_y = tl.sum(exp_y, axis=0)
    out = exp_y / sum_exp_y
    tl.store(output_ptrs, out, mask=col_offsets < n_cols)


def softmax(x):
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (n_rows,)
    softmax_kernel[grid](x, out, n_cols, x.stride(0), out.stride(0))
    return out


if __name__ == "__main__":
    x = torch.randn(4096, 16384, dtype=torch.float32, device="cuda")
    out = softmax(x)
    out_ref = torch.softmax(x, dim=1)
    print((out - out_ref).abs().max().item())
    time_triton = do_bench(lambda: softmax(x))
    time_pt = do_bench(lambda: torch.softmax(x, dim=1))
    print(f"Time Triton: {time_triton:.4f} ms")
    print(f"Time PyTorch: {time_pt:.4f} ms")
    num_bytes = x.numel() * x.element_size() + out.numel() * out.element_size()
    ideal_time_ms = num_bytes / 3.35e12 * 1e3  # On H100
    print(f"Ideal time on H100: {ideal_time_ms:.4f} ms")
    print(f"Bandwidth util of Triton: {ideal_time_ms / time_triton * 100:.2f} %")
