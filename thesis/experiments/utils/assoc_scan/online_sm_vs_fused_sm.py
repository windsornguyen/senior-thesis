import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def softmax_kernel(x_ptr,
                   out_ptr,
                   n_cols,
                   x_stride_0,  # assume x_stride_1 = 1
                   out_stride_0,  # assume out_stride_1 = 1
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    pid = tl.program_id(0)
    row_idx = pid
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = x_ptr + row_idx * x_stride_0 + col_offsets  # Range of addresses
    output_ptrs = out_ptr + row_idx * out_stride_0 + col_offsets
    # Load the input with pipelining
    x = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    m = tl.max(x, axis=0)
    y = x - m
    exp_y = tl.exp(y)
    sum_exp_y = tl.sum(exp_y, axis=0)
    out = exp_y / sum_exp_y
    # Store the output
    tl.store(output_ptrs, out, mask=col_offsets < n_cols)


def softmax(x):
    x = x.contiguous()
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    grid = lambda meta: (n_rows, )
    # Set num_stages for software pipelining
    num_stages = 2
    softmax_kernel[grid](x, out, n_cols, x.stride(0), out.stride(0),
                         BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return out


if __name__ == "__main__":
    x = torch.randn(20000, 3000, dtype=torch.float32, device="cuda")
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
