import torch
import torch.nn.functional as F

from flashfftconv import FlashFFTConv
from thesis.utils import nearest_power_of_two


def stu_convolve(u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference function."""
    bsz, seq_len, d_in = u.shape

    # sign pattern
    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    # reshape filter
    if use_tensordot:
        d_out = v.size(1)
        v = v.view(1, -1, d_out, 1).to(torch.float32)
    else:
        K = v.size(1)
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32)  # (bsz, seq_len, K, d_in, stack)
        u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    # fft
    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)

    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)

    # apply sign
    U_minus = U_minus * sgn

    # cast back to the original dtype
    U_plus = U_plus.to(u.dtype)
    U_minus = U_minus.to(u.dtype)
    return U_plus, U_minus


def flash_stu_convolve(
    u: torch.Tensor,
    v: torch.Tensor,
    flash_fft: FlashFFTConv,
    use_tensordot: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashFFTConv-based function."""
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    sgn = torch.full((1, 1, padded_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1

    if use_tensordot:
        # shape = (B, d_in, L) after transpose
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        # stack for positive/negative
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
        u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

    U_conv = flash_fft(u_conv, v_padded)  # forward pass
    U_conv = U_conv[..., :seq_len]        # crop back to original seq length

    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

    if use_tensordot:
        # apply sign to negative branch
        u_minus = u_minus * sgn[:, :, :seq_len]
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

    return U_plus, U_minus

def main():
    # just pick some sample shapes
    B, L, d_in = 1, 8192, 64
    K = 32  # number of filters

    # create random test data
    u = torch.randn(B, L, d_in, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(K, d_in, device="cuda", dtype=torch.bfloat16)

    # build a FlashFFTConv object
    n = nearest_power_of_two(L, round_up=True)
    flash_fft = FlashFFTConv(seqlen=n, dtype=torch.bfloat16).cuda()

    # helper to print difference stats
    def print_diff_stats(t_ref, t_test, label):
        diff = (t_ref - t_test).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        std_diff = diff.std().item()
        print(f"{label}: max diff={max_diff:.6f}, mean diff={mean_diff:.6f}, std diff={std_diff:.6f}")

    for use_tensordot in [True, False]:
        U_plus_orig, U_minus_orig = stu_convolve(u, v, n=n, use_tensordot=use_tensordot)
        U_plus_flash, U_minus_flash = flash_stu_convolve(u, v, flash_fft, use_tensordot=use_tensordot)

        # check shapes & dtypes
        print(f"\n==== use_tensordot={use_tensordot} ====")
        print(f"[orig ] U_plus shape={U_plus_orig.shape}, dtype={U_plus_orig.dtype}")
        print(f"[flash] U_plus shape={U_plus_flash.shape}, dtype={U_plus_flash.dtype}")
        print(f"[orig ] U_minus shape={U_minus_orig.shape}, dtype={U_minus_orig.dtype}")
        print(f"[flash] U_minus shape={U_minus_flash.shape}, dtype={U_minus_flash.dtype}")

        # allclose checks
        allclose_plus = torch.allclose(U_plus_orig, U_plus_flash, atol=1e-4, rtol=1e-3)
        allclose_minus = torch.allclose(U_minus_orig, U_minus_flash, atol=1e-4, rtol=1e-3)

        print(f"U_plus allclose?  {allclose_plus}")
        print(f"U_minus allclose? {allclose_minus}")

        # difference stats
        print_diff_stats(U_plus_orig, U_plus_flash, "U_plus")
        print_diff_stats(U_minus_orig, U_minus_flash, "U_minus")

if __name__ == "__main__":
    main()
