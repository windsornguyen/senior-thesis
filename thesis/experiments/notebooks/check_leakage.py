import torch
import torch.nn as nn
from typing import Tuple, List, Optional

def get_hankel(seq_len: int, use_hankel_L: bool = False, device=None) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z, UPLO="U")
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k = phi_k * sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)

class LearnableSpectralFilters(nn.Module):
    def __init__(self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        # Pre-project x to shape (B, T, seq_len) if needed.
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        
        # Q is now a spectral filter.
        self.Q = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        # K remains a spectral filter.
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        # V becomes a linear layer.
        self.v_proj = nn.Linear(seq_len, k).to(device)
        
        # Final projection from k back to d_model.
        self.o_proj = nn.Linear(k, d_model).to(device)
        # Decay parameter: one per time step.
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))
        # Hankel matrix L (shape: [T, T]); we'll mask it to be lower-triangular.
        self.L = nn.Parameter(get_hankel(seq_len, use_hankel_L, device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)  # (B, T, seq_len)
        
        # Q computed as a spectral filter: (B, T, k)
        Q = torch.einsum("bti,ik->btk", x_proj, self.Q())
        
        # K remains a spectral filter: (B, T, k)
        K = torch.einsum("bti,ik->btk", x_proj, self.K())
        
        # V is now a linear projection: (B, T, k)
        V = self.v_proj(x_proj)
        
        # Compute Z as the outer product between V and K per time step,
        # scaled by the decay factor.
        # V: (B, T, k) [indices "btp"]
        # K: (B, T, k) [indices "btn"]
        # decay: (T,) [index "t"]
        # Z: (B, T, k, k)
        Z = torch.einsum("btp,btn,t->btpn", V, K, self.decay)
        
        # Prepare the Hankel mask: force L to be lower-triangular and add batch dim.
        L_masked = torch.tril(self.L).unsqueeze(0)  # (1, T, T)
        
        # Aggregate Z over time using L_masked.
        # For each b, t, p, n:
        #    H[b, t, p, n] = sum_s L_masked[0, t, s] * Z[b, s, p, n]
        H = torch.einsum("bts,bspn->btpn", L_masked, Z)
        
        # Query the aggregated result with Q.
        # Q: (B, T, k) and H: (B, T, k, k) → Y: (B, T, k)
        Y = torch.einsum("btk,btkn->btn", Q, H)
        
        return self.o_proj(Y)

def test_causality_comprehensive(
    model: nn.Module,
    seq_len: int,
    d_model: int,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> Tuple[bool, List[str]]:
    """
    Comprehensive test suite for checking causality in attention mechanisms.
    
    Args:
        model: The attention model to test
        seq_len: Length of input sequences
        d_model: Dimension of model input
        batch_size: Batch size for testing
        device: Device to run tests on
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons
    
    Returns:
        Tuple of (passed_all: bool, failures: List[str])
    """
    if device is None:
        device = next(model.parameters()).device
    
    failures = []
    
    # Test 1: Systematic position-by-position perturbation
    def position_perturbation_test():
        x_base = torch.randn(batch_size, seq_len, d_model, device=device)
        y_base = model(x_base)
        
        for t in range(seq_len - 1):
            # Create perturbation for all future positions
            x_perturbed = x_base.clone()
            x_perturbed[:, t+1:] += torch.randn_like(x_perturbed[:, t+1:]) * 100.0
            
            y_perturbed = model(x_perturbed)
            
            # Check outputs up to position t are unchanged
            diff = torch.abs(y_base[:, :t+1] - y_perturbed[:, :t+1])
            max_diff = diff.max().item()
            
            if not torch.allclose(y_base[:, :t+1], y_perturbed[:, :t+1], rtol=rtol, atol=atol):
                failures.append(f"Position {t} leaks to past with max diff {max_diff}")
    
    # Test 2: Extreme value test
    def extreme_value_test():
        x_base = torch.randn(batch_size, seq_len, d_model, device=device)
        y_base = model(x_base)
        
        # Test with very large values
        x_extreme = x_base.clone()
        x_extreme[:, seq_len//2:] = 1e6
        y_extreme = model(x_extreme)
        
        if not torch.allclose(y_base[:, :seq_len//2], y_extreme[:, :seq_len//2], rtol=rtol, atol=atol):
            failures.append("Large value perturbation causes leakage")
        
        # Test with very small values
        x_tiny = x_base.clone()
        x_tiny[:, seq_len//2:] = 1e-6
        y_tiny = model(x_tiny)
        
        if not torch.allclose(y_base[:, :seq_len//2], y_tiny[:, :seq_len//2], rtol=rtol, atol=atol):
            failures.append("Small value perturbation causes leakage")
    
    # Test 3: Gradient-based test
    def gradient_test():
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        y = model(x)
        
        for t in range(seq_len - 1):
            # Compute gradients with respect to future tokens
            loss = y[:, t].abs().sum()
            loss.backward(retain_graph=True)
            
            # Check if gradients exist for future tokens
            future_grads = x.grad[:, t+1:]
            if torch.any(torch.abs(future_grads) > atol):
                failures.append(f"Position {t} has gradient flow from future tokens")
            
            x.grad = None
    
    # Test 4: Random seed invariance
    def random_seed_test():
        x_base = torch.randn(batch_size, seq_len, d_model, device=device)
        seeds = [42, 123, 999]
        
        base_outputs = []
        for seed in seeds:
            torch.manual_seed(seed)
            x_perturbed = x_base.clone()
            x_perturbed[:, seq_len//2:] += torch.randn_like(x_perturbed[:, seq_len//2:])
            y = model(x_perturbed)
            base_outputs.append(y[:, :seq_len//2].clone())
        
        # Check all outputs are identical up to position seq_len//2
        for i in range(1, len(base_outputs)):
            if not torch.allclose(base_outputs[0], base_outputs[i], rtol=rtol, atol=atol):
                failures.append(f"Random seed {seeds[i]} produces different past outputs")
    
    # Test 5: Batch consistency
    def batch_consistency_test():
        if batch_size > 1:
            x_base = torch.randn(batch_size, seq_len, d_model, device=device)
            y_base = model(x_base)
            
            # Perturb future tokens in only one batch element
            x_perturbed = x_base.clone()
            x_perturbed[0, seq_len//2:] += torch.randn_like(x_perturbed[0, seq_len//2:]) * 100.0
            y_perturbed = model(x_perturbed)
            
            # Check other batch elements are unchanged
            if not torch.allclose(y_base[1:, :seq_len//2], y_perturbed[1:, :seq_len//2], rtol=rtol, atol=atol):
                failures.append("Batch independence violated")
    
    # Run all tests
    test_functions = [
        position_perturbation_test,
        extreme_value_test,
        gradient_test,
        random_seed_test,
        batch_consistency_test
    ]
    
    for test_fn in test_functions:
        try:
            test_fn()
        except Exception as e:
            failures.append(f"Test {test_fn.__name__} failed with error: {str(e)}")
    
    return len(failures) == 0, failures

# Example usage:
if __name__ == "__main__":
    # Setup model and test parameters
    T = 10  # sequence length
    d = 10  # input dimension
    model = SpectralAttention(T, d, 10, use_hankel_L=False, device=torch.device('cuda'))
    
    # Run comprehensive test suite
    passed, failures = test_causality_comprehensive(
        model=model,
        seq_len=T,
        d_model=d,
        batch_size=2,
        rtol=1e-5,
        atol=1e-5
    )
    
    if passed:
        print("✅ All causality tests passed!")
    else:
        print("❌ Causality tests failed:")
        for failure in failures:
            print(f"  - {failure}")