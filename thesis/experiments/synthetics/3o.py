import torch
import torch.nn.functional as F
import torchaudio.functional as AF

def stu_conv_vmap(filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r"""Applies K shared filters causally to all channels using a batched vmap.

        Equivalent to :func:`stu_conv`, but implemented by reshaping the
        ``B, H, h`` dimensions into a single batch dimension and using nested
        `torch.vmap` optimized for batching. Results in an expanded output
        dimension ``K_num``.

        See ``test_conv.py`` for vmap and batched versions.

        Args:
            filters (torch.Tensor): Bank of shared filters.
                Shape: ``[K_len, K_num]``.
            inputs (torch.Tensor): Input sequences (typically pre-projected).
                Shape: ``[B, H, L, h]``.

        Returns:
            torch.Tensor: Output tensor after convolution. Shape: ``[B, H, L, K_num, h]``.
        """
        if filters.dim() != 2:
            raise ValueError("filters must be 2D tensor of shape [K_len, K_num]")
        if inputs.dim() != 4:
            raise ValueError("inputs must be 4D tensor of shape [B, H, L, h]")

        B, H, L, h = inputs.shape
        K_len, K_num = filters.shape

        # Flatten B, H, h dims into one batch dim
        inputs_flat = inputs.permute(0, 1, 3, 2).reshape(B * H * h, L)  # [BHh, L]

        causal_base = lambda sig, ker: AF.fftconvolve(sig, ker, mode="full")[..., : sig.shape[-1]]

        # Inner function applies all K_num filters to a single signal
        apply_all_filters = torch.vmap(
            causal_base, in_dims=(None, 1), out_dims=0
        )  # sig[L], filters[K_len, K_num] -> y[K_num, L]

        # Outer vmap maps this inner function over the flattened input batch
        vmap_flat_batch = torch.vmap(apply_all_filters, in_dims=(0, None), out_dims=0)

        y_flat = vmap_flat_batch(inputs_flat, filters)  # [BHh, K_num, L]

        # Reshape back
        y_reshaped = y_flat.reshape(B, H, h, K_num, L)  # [B, H, h, K_num, L]
        return y_reshaped.permute(0, 1, 4, 3, 2)  # [B, H, L, K_num, h]

def stu_conv_torchaudio(filters: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    B, H, L, h = inputs.shape
    K_len, K_num = filters.shape
    # flatten inputs
    x = inputs.permute(0,1,3,2).reshape(B*H*h, 1, L)  # [BHh,1,L]
    w = filters.T.unsqueeze(0)                        # [1,K_num,K_len]
    y_full = AF.convolve(x, w, mode='full')           # [BHh,K_num,L+K_len-1]
    y = y_full[..., :L]                               # causal slice
    y = y.view(B, H, h, K_num, L).permute(0,1,4,3,2)
    return y
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, L, h = 2, 3, 16, 5
    K_len, K_num = 7, 4

    filters = torch.randn(K_len, K_num)
    inputs  = torch.randn(B, H, L, h)

    y1 = stu_conv_vmap(filters, inputs)
    y2 = stu_conv_torchaudio(filters, inputs)

    print("Match?        ", torch.allclose(y1, y2, rtol=1e-5, atol=1e-6))
    print("Max abs diff: ", (y1 - y2).abs().max().item())
