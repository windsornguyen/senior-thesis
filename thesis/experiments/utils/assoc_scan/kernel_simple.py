import torch
import triton
import triton.language as tl

# -*- Reference -*-
class AssociativeAttention(nn.Module):
    """Online Associative Attention module."""

    def __init__(self, config: SpectronConfig) -> None:
        super().__init__()
        
        # Initialized at top-level, i.e. Spectron
        self.spectral_filters     = None
        self.spectral_filters_fft = None

        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.wo.SCALE_INIT = 1

        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_local_heads = config.num_local_heads
        
        self.wg = nn.Linear(config.dim, 1)
        self.eps = config.eps

        # Per-head (dim_v x dim_k) scaling matrix, broadcasted across batch and sequence dims
        # We normalize both K and V which bounds their values to unit magnitude, so this init suffices
        self.kv_norm_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, self.head_dim, self.head_dim))

        # Standard choice of init for QK norm for each head, broadcasted across batch and sequence dims
        self.qk_norm_scale = nn.Parameter(torch.full((1, self.num_heads, 1), 1 / math.sqrt(self.head_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, h = self.num_heads, self.head_dim
        
        # [B, H, L, h]
        q = self.wq(x).view(B, L, H, h).transpose(1, 2)
        k = self.wk(x).view(B, L, H, h).transpose(1, 2)
        v = self.wv(x).view(B, L, H, h).transpose(1, 2)
        
        # Compute similarity between queries and keys
        sim = torch.einsum("bhlq,bhlk->bhl", q, k)
        sim = sim * self.qk_norm_scale
        
        # Take L2-norm and apply KV norm (analogous to QK norm)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)
        
        # Apply spectral basis
        if self.use_tensordot:
            filters = self.tensordot_proj(self.spectral_basis)  # [L, K] @ [K, D] -> [L, D]
            k = self.tensordot_conv(filters, k)
            v = self.tensordot_conv(filters, v)
        else:
            k = self.full_conv(self.spectral_basis, k)
            v = self.full_conv(self.spectral_basis, v)
        
        # Compute pairwise interactions via outer product
        if self.use_tensordot:
            Z = torch.einsum("bhld,bhle->bhlde", v, k)
        else:
            Z = torch.einsum("bhlkd,bhlke->bhlde", v, k)  # Contract over K dim
        Z = Z * self.kv_norm_scale
    
        # [B, H, L, h^2] -> [B, H, L, 1]
        gate_inputs = Z.view(B, H, L, h**2)
        gates_logits = self.wg(gate_inputs)

        # [B, H, L, 1]
        gates = sq_relu(gates_logits) + self.eps

        # [B, H, L, 1, 1]
        gates = gates.unsqueeze(-1)
        
        # Apply gating to Z
        gated_Z = gates * Z
        
        # Vmap over H dim
        hmap = torch.vmap(self.scan_fn, in_dims=(0, 0, 0, 0))
        
        # Vmap over B dim
        bmap = torch.vmap(hmap, in_dims=(0, 0, 0, 0))
        
        # Scan over all dims simultaneously
        m_scan, s_scan, n_scan, Z_scan, g_scan = bmap(
            qk_slice=sim, v_slice=v, Z_slice=gated_Z, g_slice=gates
        )

        # -*- Compute final attention outputs -*-
        
        # Compute online softmax in safe manner
        softmax_weights = torch.exp(sim - m_scan).unsqueeze(-1).unsqueeze(-1) \
                        / (s_scan.unsqueeze(-1).unsqueeze(-1) + self.eps)
        
        # Compute gated accumulation normalization
        gated_weights = Z_scan / (g_scan + self.eps)
        
        # Multiplicatively modulate gated weights w/ softmax weights
        attn_weights = gated_weights * (1.0 + F.silu(softmax_weights))
        
        # Query from the accumulated state history
        ctxt = torch.einsum("bhld,bhlde->bhle", q, attn_weights)
        
        # Concatenate the heads back together
        ctxt = ctxt.transpose(1, 2).reshape(B, L, -1)
        output = self.wo(ctxt)
        
        return output

    def combine_fn(self, x: Tuple[Any], y: Tuple[Any]) -> Tuple[Any]:
        """Combine two leaves of the associative scan tree for AA.
        
        Args:
            x: First leaf of the scan tree, containing the current state.
            y: Second leaf of the scan tree, containing the new input.
        
        Returns:
            A tuple representing the combined state of the two leaves.

        NOTE:

            Each leaf is a tuple (m, s, n, Z, g) where:
                - m: Running maximum (for stable online softmax)
                - s: Running sum of exp(score - m)
                - n: Running sum of exp(score - m) * value
                - Z: Running sum of gated outer products 
                - g: Running sum of gates

        """
        m_x, s_x, n_x, Z_x, g_x = x
        m_y, s_y, n_y, Z_y, g_y = y
        
        # Compute the new maximum
        m_new = torch.max(m_x, m_y)
        
        # Scaling factors
        exp_x = torch.exp(m_x - m_new)
        exp_y = torch.exp(m_y - m_new)
        
        # Update softmax aggregator components
        s_new = s_x * exp_x + s_y * exp_y
        n_new = n_x * exp_x.unsqueeze(-1) + n_y * exp_y.unsqueeze(-1)
        
        # Update gated Z and gate accumulation
        Z_new = Z_x + Z_y
        g_new = g_x + g_y
        
        return (m_new, s_new, n_new, Z_new, g_new)
    
    def scan_fn(self, qk_slice, v_slice, Z_slice, g_slice):
        """Process a single [B, H] slice.
        

        """
        leaves_m = qk_slice                    # [L,]
        leaves_s = torch.zeros_like(qk_slice)  # [L,]
        leaves_n = torch.zeros_like(qk_slice)  # [L, h]
        leaves_Z = Z_slice                     # [L, h, h]
        leaves_g = g_slice                     # [L, 1, 1]

        leaves = (leaves_m, leaves_s, leaves_n, leaves_Z, leaves_g)

        # TODO: Settle assoc scan API (enforce L to be dim=0?)
        scan = associative_scan(input=leaves, combine_fn=self.combine_fn)
        return scan

@triton.jit
def fwd_scan_kernel(gated_Z_ptr,
                    gates_ptr,
                    out_gated_Z_ptr,
                    out_gates_ptr,
                    batch_size: int,
                    feature_size: int,
                    seq_len: int,
                    stride_b: int,
                    stride_d: int,
                    stride_l: int,
                    BLOCK_SIZE: tl.constexpr):
    """
    Forward kernel for cumulative associative scan.
    
    Scans along the sequence (L) dimension for each batch (B) and feature (D).
    
    Args:
        gated_Z_ptr: Pointer to input gated tensor [B, D, L].
        gates_ptr: Pointer to input gates tensor [B, D, L].
        out_gated_Z_ptr: Pointer to output cumulative gated tensor.
        out_gates_ptr: Pointer to output cumulative gates tensor.
        batch_size: Batch size (B).
        feature_size: Feature dimension (D).
        seq_len: Sequence length (L).
        stride_b: Stride of the batch dimension.
        stride_d: Stride of the feature dimension.
        stride_l: Stride of the sequence dimension.
        BLOCK_SIZE: Number of elements per block in the sequence dimension.
    """
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d
    offsets = start_offset + tl.arange(0, BLOCK_SIZE) * stride_l
    mask = tl.arange(0, BLOCK_SIZE) < seq_len

    gated_Z = tl.load(gated_Z_ptr + offsets, mask=mask, other=0.0)
    gates = tl.load(gates_ptr + offsets, mask=mask, other=0.0)

    result_gated_Z, result_gates = tl.associative_scan(
        (gated_Z, gates), axis=0, combine_fn=combine_fn, reverse=False
    )

    tl.store(out_gated_Z_ptr + offsets, result_gated_Z, mask=mask)
    tl.store(out_gates_ptr + offsets, result_gates, mask=mask)


@triton.jit
def bwd_scan_kernel(grad_cumulative_gated_Z_ptr,
                    grad_cumulative_gates_ptr,
                    grad_gated_Z_ptr,
                    grad_gates_ptr,
                    batch_size: int,
                    feature_size: int,
                    seq_len: int,
                    stride_b: int,
                    stride_d: int,
                    stride_l: int,
                    BLOCK_SIZE: tl.constexpr):
    """
    Backward kernel for associative scan.
    
    Computes gradients for the associative scan by performing a forward scan
    on the reversed gradient sequence.
    
    Args:
        grad_cumulative_gated_Z_ptr: Pointer to gradient of cumulative gated tensor [B, D, L].
        grad_cumulative_gates_ptr: Pointer to gradient of cumulative gates tensor [B, D, L].
        grad_gated_Z_ptr: Pointer to output gradient for gated tensor.
        grad_gates_ptr: Pointer to output gradient for gates tensor.
        batch_size: Batch size (B).
        feature_size: Feature dimension (D).
        seq_len: Sequence length (L).
        stride_b: Stride of the batch dimension.
        stride_d: Stride of the feature dimension.
        stride_l: Stride of the sequence dimension.
        BLOCK_SIZE: Number of elements per block in the sequence dimension.
    """
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    start_offset = pid_batch * stride_b + pid_feature * stride_d

    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < seq_len

    # Compute reversed offsets to capture the triangular Jacobian pattern.
    rev_offsets = start_offset + (seq_len - 1 - indices) * stride_l

    grad_in_Z_rev = tl.load(grad_cumulative_gated_Z_ptr + rev_offsets, mask=mask, other=0.0)
    grad_in_gates_rev = tl.load(grad_cumulative_gates_ptr + rev_offsets, mask=mask, other=0.0)

    result_grad_Z_rev, result_grad_gates_rev = tl.associative_scan(
        (grad_in_Z_rev, grad_in_gates_rev), axis=0, combine_fn=combine_fn, reverse=False
    )

    tl.store(grad_gated_Z_ptr + rev_offsets, result_grad_Z_rev, mask=mask)
    tl.store(grad_gates_ptr + rev_offsets, result_grad_gates_rev, mask=mask)


class AssociativeScan(torch.autograd.Function):
    """
    PyTorch autograd wrapper for an associative scan using Triton.
    
    Implements a cumulative associative scan along the sequence dimension for
    input tensors of shape [B, D, L].
    """
    @staticmethod
    def forward(ctx, gated_Z: torch.Tensor, gates: torch.Tensor):
        """
        Forward pass.
        
        Args:
            gated_Z: Input gated tensor of shape [B, D, L].
            gates: Input gates tensor of shape [B, D, L].
        
        Returns:
            A tuple (cumulative_gated_Z, cumulative_gates), each of shape [B, D, L],
            containing the cumulative scan results.
        """
        batch_size, feature_size, seq_len = gated_Z.shape
        gated_Z = gated_Z.contiguous().cuda()
        gates = gates.contiguous().cuda()
        cumulative_gated_Z = torch.empty_like(gated_Z)
        cumulative_gates = torch.empty_like(gates)

        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        grid = (batch_size, feature_size, 1)

        fwd_scan_kernel[grid](
            gated_Z,
            gates,
            cumulative_gated_Z,
            cumulative_gates,
            batch_size,
            feature_size,
            seq_len,
            gated_Z.stride(0),
            gated_Z.stride(1),
            gated_Z.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(gated_Z, gates)
        return cumulative_gated_Z, cumulative_gates

    @staticmethod
    def backward(ctx, grad_cumulative_gated_Z: torch.Tensor, grad_cumulative_gates: torch.Tensor):
        """
        Backward pass.
        
        Args:
            grad_cumulative_gated_Z: Gradient of cumulative gated tensor [B, D, L].
            grad_cumulative_gates: Gradient of cumulative gates tensor [B, D, L].
        
        Returns:
            Gradients with respect to the input tensors gated_Z and gates.
        """
        gated_Z, gates = ctx.saved_tensors
        batch_size, feature_size, seq_len = gated_Z.shape
        grad_cumulative_gated_Z = grad_cumulative_gated_Z.contiguous().cuda()
        grad_cumulative_gates = grad_cumulative_gates.contiguous().cuda()
        grad_gated_Z = torch.empty_like(gated_Z)
        grad_gates = torch.empty_like(gates)

        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        grid = (batch_size, feature_size, 1)

        bwd_scan_kernel[grid](
            grad_cumulative_gated_Z,
            grad_cumulative_gates,
            grad_gated_Z,
            grad_gates,
            batch_size,
            feature_size,
            seq_len,
            grad_cumulative_gated_Z.stride(0),
            grad_cumulative_gated_Z.stride(1),
            grad_cumulative_gated_Z.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_gated_Z, grad_gates


def associative_scan(gated_Z: torch.Tensor, gates: torch.Tensor):
    """
    Executes an associative scan on the provided tensors.
    
    Args:
        gated_Z: Tensor of shape [B, D, L] representing the gated values.
        gates: Tensor of shape [B, D, L] representing the gating coefficients.

    Returns:
        A tuple (cumulative_gated_Z, cumulative_gates), each of shape [B, D, L],
        corresponding to the cumulative associative scan results along the sequence dimension.
    """
    return AssociativeScan.apply(gated_Z, gates)
