import torch
import torchaudio
import math
from typing import Callable


# Helper function needed by the first convfft implementation
def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))


class ConvTester:
    def convfft(
        self,
        signal: torch.Tensor,
        filt: torch.Tensor,
    ) -> torch.Tensor:
        """Causal 1D convolution via FFT for [L,] filters.

        Given `s[t]` (length *L*) and kernel `f[t]` (same length),
        returns the *causal* convolution

            y[t] = Σ_{τ=0..t}  s[τ] · f[t-τ]   for  t = 0..L-1.

        Implementation: zero-pad both vectors to *2 L* FFT length, multiply
        in frequency domain, IFFT, then keep the first *L* samples.
        """
        L = signal.size(0)
        # Ensure n is large enough for linear convolution (L + L - 1)
        n = nearest_power_of_two(2 * L - 1, round_up=True)
        # Compute FFTs
        signal_fft = torch.fft.rfft(signal, n=n)
        filt_fft = torch.fft.rfft(filt, n=n)
        # Perform convolution in frequency domain
        conv_fft = signal_fft * filt_fft
        # Inverse FFT and truncate to original length L
        y = torch.fft.irfft(conv_fft, n=n)
        return y[:L]

    def bhld_convfft(self, v, u):          # v: [L,K] , u: [B,H,L,h]
        causal = lambda ker, sig: self.convfft(sig, ker)   # (L,)x(L,)→(L,)

        kmap = torch.vmap(causal, in_dims=(1, None), out_dims=0)  # over K of v
        cmap = torch.vmap(kmap,  in_dims=(None, 1), out_dims=1)   # over h of u
        hmap = torch.vmap(cmap,  in_dims=(None, 0), out_dims=0)   # over H
        bmap = torch.vmap(hmap,  in_dims=(None, 0), out_dims=0)   # over B

        return bmap(v, u).permute(0, 1, 4, 2, 3)   # [B,H,L,K,h]

    def bhld_convfft_torchaudio(
        self,
        v: torch.Tensor,  # [L, K]
        u: torch.Tensor,  # [B, H, L, h]
    ) -> torch.Tensor:
        """Compute FFT-based convolution for multi-headed inputs. Impl 2 (torchaudio).

        Args:
            v: Spectral filters of shape [L, K].
            u: Inputs of shape [B, H, L, h].

        Returns:
            Convolution output of shape [B, H, L, K, h].
        """
        L = u.shape[2]
        tr_conv = lambda filter_k, channel_h: torchaudio.functional.fftconvolve(channel_h, filter_k)[:L]
        conv_filter_with_channels = torch.vmap(tr_conv, in_dims=(None, 1), out_dims=1)
        conv_all_filters_channels = torch.vmap(conv_filter_with_channels, in_dims=(1, None), out_dims=1)
        conv_one_head = lambda u1, v_filters: conv_all_filters_channels(v_filters, u1)
        conv_heads = torch.vmap(conv_one_head, in_dims=(0, None), out_dims=0)
        conv_batch = torch.vmap(conv_heads, in_dims=(0, None), out_dims=0)
        return conv_batch(u, v)

    def test_comparison(self, B=2, H=1, L=16, h=4, K=5, dtype=torch.float32, device="cpu"):
        print(f"--- Testing Convolution Implementations ---")
        print(f"Parameters: B={B}, H={H}, L={L}, h={h}, K={K}, dtype={dtype}, device={device}")

        # Create test data
        u = torch.randn(B, H, L, h, dtype=dtype, device=device)
        v = torch.randn(L, K, dtype=dtype, device=device)  # Filters

        # Ensure requires_grad is False if not needed, or handle appropriately if grads are tested
        u.requires_grad_(False)
        v.requires_grad_(False)

        # Run both implementations
        print("Running bhld_convfft (custom)...")
        try:
            output1 = self.bhld_convfft(v, u)
            print("  Success.")
        except Exception as e:
            print(f"  Failed: {e}")
            output1 = None

        print("Running bhld_convfft_torchaudio...")
        try:
            output2 = self.bhld_convfft_torchaudio(v, u)
            print("  Success.")
        except Exception as e:
            print(f"  Failed: {e}")
            output2 = None

        # Compare results
        if output1 is not None and output2 is not None:
            print(f"Comparing outputs...")
            # Check shapes first
            if output1.shape != output2.shape:
                print(f"  Shape Mismatch: {output1.shape} vs {output2.shape}")
                are_close = False
            else:
                print(f"  Shapes Match: {output1.shape}")
                # Use torch.allclose for numerical comparison
                are_close = torch.allclose(output1, output2, atol=1e-6)  # Adjust tolerance if needed
                if are_close:
                    print(f"  Outputs are numerically close (atol=1e-6).")
                else:
                    diff = torch.abs(output1 - output2).max()
                    print(f"  Outputs differ numerically (max difference: {diff.item()}).")
            return are_close
        else:
            print("Comparison skipped due to failure in one or both implementations.")
            return False


if __name__ == "__main__":
    tester = ConvTester()
    # Run tests with different parameters if desired
    tester.test_comparison(B=2, H=1, L=64, h=4, K=5)
    print("-" * 40)
    tester.test_comparison(B=1, H=3, L=32, h=8, K=10)
    print("-" * 40)
    if torch.cuda.is_available():
        print("Testing on CUDA...")
        tester.test_comparison(B=4, H=2, L=128, h=16, K=8, device="cuda")
        print("-" * 40)
    else:
        print("CUDA not available, skipping CUDA test.")
