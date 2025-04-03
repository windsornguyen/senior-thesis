import torch
from torch._higher_order_ops import associative_scan
import torch.nn.functional as F
from typing import Tuple

class ScanAttention(torch.nn.Module):
    """
    Implementation of the scan-based attention variant using PyTorch's experimental
    associative_scan operator.
    """
    def __init__(self, feature_dim, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
    
    # Modified forward method to optionally return components
    def forward(self, Q, K, V, return_components=False):
        """
        Compute scan-based attention with log-domain normalization.
        
        Args:
            Q: Query tensor (B, H, T, P)
            K: Key tensor (B, H, T, P)
            V: Value tensor (B, H, T, P)
            return_components: If True, return intermediates for analysis
            
        Returns:
            Output tensor (B, T, H*P) or tuple with components if return_components=True
        """
        # Feature map
        Q_feat = torch.exp(Q)  # shape: (B, H, T, P)
        K_feat = torch.exp(K)  # shape: (B, H, T, P)
        
        # Initialize the scan components
        m_initial = torch.log(K_feat)  # (B, H, T, P)
        num_initial = torch.einsum("bhtp,bhtn->bhtpn", V, K_feat)  # (B, H, T, P, P)
        denom_initial = K_feat  # (B, H, T, P)
        
        # Define the combine function for associative scan
        def combine_fn(x, y):
            m_x, N_x, D_x = x
            m_y, N_y, D_y = y
            
            # Compute new reference point (elementwise max)
            m_new = torch.maximum(m_x, m_y)
            
            # Compute scaling factors
            scale_x = torch.exp(m_x - m_new)
            scale_y = torch.exp(m_y - m_new)
            
            # Combine numerators and denominators
            N_new = N_x * scale_x.unsqueeze(-1) + N_y * scale_y.unsqueeze(-1)
            D_new = D_x * scale_x + D_y * scale_y
            
            return m_new, N_new, D_new
        
        # Package the inputs for scan
        tuple_initial = (m_initial, num_initial, denom_initial)
        
        # Use associative_scan along time dimension (dim=2)
        try:
            m_cum, num_cum, denom_cum = associative_scan(
                combine_fn, tuple_initial, dim=2, combine_mode="generic"
            )
        except RuntimeError as e:
            # Fallback to sequential implementation if scan fails
            print(f"Warning: associative_scan failed with error: {e}")
            print("Falling back to sequential implementation")
            
            # Sequential implementation as fallback
            B, H, T, P = Q.shape
            m_cum = m_initial.clone()
            num_cum = num_initial.clone()
            denom_cum = denom_initial.clone()
            
            for t in range(1, T):
                m_prev, N_prev, D_prev = m_cum[:,:,t-1], num_cum[:,:,t-1], denom_cum[:,:,t-1]
                m_curr, N_curr, D_curr = m_initial[:,:,t], num_initial[:,:,t], denom_initial[:,:,t]
                
                m_new = torch.maximum(m_prev, m_curr)
                scale_prev = torch.exp(m_prev - m_new)
                scale_curr = torch.exp(m_curr - m_new)
                
                N_new = N_prev * scale_prev.unsqueeze(-1) + N_curr * scale_curr.unsqueeze(-1)
                D_new = D_prev * scale_prev + D_curr * scale_curr
                
                m_cum[:,:,t] = m_new
                num_cum[:,:,t] = N_new
                denom_cum[:,:,t] = D_new
        
        # Now, for each time step, combine with Q_feat to compute the output
        Y_num = torch.einsum("bhtp,bhtpq->bhtq", Q_feat, num_cum)
        Y_den = torch.einsum("bhtp,bhtp->bht", Q_feat, denom_cum)
        
        # Apply normalization with small epsilon for stability
        Y = Y_num / (Y_den.unsqueeze(-1) + self.eps)
        
        # Merge heads for final output: shape (B, T, H*P)
        Y_out = Y.permute(0, 2, 1, 3).reshape(Q.shape[0], Q.shape[2], -1)
        
        if return_components:
            return Y_out, m_cum, num_cum, denom_cum, Y_den
        
        return Y_out


def test_scan_attention():
    """Test the scan attention implementation."""
    B, H, T, P = 2, 2, 6, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, T, P, device=device, requires_grad=True)
    K = torch.randn(B, H, T, P, device=device, requires_grad=True)
    V = torch.randn(B, H, T, P, device=device, requires_grad=True)
    
    # Create model
    model = ScanAttention(feature_dim=P).to(device)
    
    # Forward pass
    Y = model(Q, K, V)
    print(f"Output shape: {Y.shape}")
    
    # Test backward pass
    if Q.requires_grad:
        loss = Y.sum()
        loss.backward()
        print(f"Gradients computed: Q.grad norm = {Q.grad.norm().item()}")
    
    # Compare with vanilla softmax attention
    def vanilla_attention(Q, K, V):
        Q_feat = torch.exp(Q)
        K_feat = torch.exp(K)
        
        scores = torch.einsum('bhtp,bhsp->bhts', Q_feat, K_feat)
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        scores = torch.where(mask, scores, torch.tensor(-1e9, device=device))
        
        weights = F.softmax(scores, dim=-1)
        Y = torch.einsum('bhts,bhsp->bhtp', weights, V)
        
        # Reshape to match scan output
        return Y.permute(0, 2, 1, 3).reshape(B, T, -1)
    
    Y_vanilla = vanilla_attention(Q, K, V)
    diff = (Y - Y_vanilla).abs().max().item()
    print(f"Max difference from vanilla softmax: {diff}")
    
    # Let's also visualize the row sums to confirm our theory
    with torch.no_grad():
        Q_feat = torch.exp(Q)
        K_feat = torch.exp(K)
        
        scores = torch.einsum('bhtp,bhsp->bhts', Q_feat, K_feat)
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        masked_scores = torch.where(mask, scores, torch.tensor(-1e9, device=device))
        
        # Vanilla weights (softmax)
        weights_vanilla = F.softmax(masked_scores, dim=-1)
        row_sums_vanilla = weights_vanilla.sum(dim=-1)
        
        # Reconstructing scan-based weights
        weights_scan = torch.zeros_like(weights_vanilla)
        
        # Forward pass of scan to get denominators
        Y_out, m_cum, num_cum, denom_cum, Y_den = model.forward(Q, K, V, return_components=True)
                
        # Manually calculate effective weights
        for t in range(T):
            for s in range(t+1):
                numerator = (Q_feat[:,:,t] * K_feat[:,:,s]).sum(dim=-1)
                weights_scan[:,:,t,s] = numerator
            
            # Normalize by denominator from scan
            if t < T:  # Avoid index error
                weights_scan[:,:,t] /= (Y_den[:,:,t].unsqueeze(-1) + model.eps)
        
        row_sums_scan = weights_scan.sum(dim=-1)
        
        print("\nVanilla row sums (should all be close to 1.0):")
        print(row_sums_vanilla[0,0])
        
        print("\nScan-based row sums (typically increase with position):")
        print(row_sums_scan[0,0])
    
    return "Test completed successfully!"

def test_gradcheck():
    """Test that gradients are computed correctly using torch.autograd.gradcheck."""
    B, H, T, P = 2, 2, 3, 4  # Using smaller values for speed
    device = "cpu"  # gradcheck needs double precision on CPU
    
    # Random inputs (requires double precision for gradcheck)
    torch.manual_seed(42)
    Q = torch.randn(B, H, T, P, device=device, dtype=torch.double, requires_grad=True)
    K = torch.randn(B, H, T, P, device=device, dtype=torch.double, requires_grad=True)
    V = torch.randn(B, H, T, P, device=device, dtype=torch.double, requires_grad=True)
    
    # Create model with double precision
    model = ScanAttention(feature_dim=P).to(device).double()
    
    # Define function for gradcheck that returns a scalar
    def func(q, k, v):
        return model(q, k, v).sum()
    
    # Run gradcheck
    from torch.autograd import gradcheck
    
    # Gradcheck tests if the gradient is computed correctly via finite differences
    print("Running gradcheck (this may take a while)...")
    result = gradcheck(func, (Q, K, V), eps=1e-6, atol=1e-4)
    print(f"Gradcheck passed: {result}")
    
    return result


if __name__ == "__main__":
    # Enable float64 for more precision in testing
    torch.set_default_dtype(torch.float64)
    test_scan_attention()
    test_gradcheck()
