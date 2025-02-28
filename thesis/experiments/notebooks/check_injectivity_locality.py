import torch
import torch.nn as nn
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        self.Q_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.K_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.v_proj = nn.Linear(d_model, k).to(device)
        self.o_proj = nn.Linear(k, d_model).to(device)
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)
        Q = torch.einsum("bti,ik->btk", x_proj, self.Q_filt())
        K = torch.einsum("bti,ik->btk", x_proj, self.K_filt())
        V = self.v_proj(x)
        Z = torch.einsum("btp,btn->btpn", V, K)
        decay = self.decay.view(1, T, 1, 1)
        Z = Z * decay
        H = torch.cumsum(Z, dim=1)
        Y = torch.einsum("btk,btkn->btn", Q, H)
        return self.o_proj(Y)
    
def test_injectivity(
    model: SpectralAttention,
    num_samples: int = 1000,
    epsilon: float = 1e-5,
    device: torch.device = None
) -> Tuple[bool, List[str]]:
    """
    Test if the attention mechanism is injective by checking if different queries
    map to different attention distributions.
    
    From the paper: An attention mechanism is injective if distinct queries q₁ ≠ q₂
    produce distinct attention distributions.
    
    Args:
        model: The SpectralAttention model to test
        num_samples: Number of random query pairs to test
        epsilon: Threshold for considering attention distributions as different
        device: Device to run tests on
    
    Returns:
        Tuple of (is_injective: bool, failure_messages: List[str])
    """
    if device is None:
        device = next(model.parameters()).device
    
    failures = []
    seq_len = model.seq_len
    d_model = model.o_proj.out_features
    
    # Create a simple batch with two different sequences
    for i in range(num_samples):
        # Generate two different random sequences
        x1 = torch.randn(1, seq_len, d_model, device=device)
        x2 = torch.randn(1, seq_len, d_model, device=device)
        
        # Get their attention distributions
        out1 = model(x1)
        out2 = model(x2)
        
        # If inputs are different but outputs are same (within epsilon),
        # the function is not injective
        inputs_diff = torch.norm(x1 - x2).item()
        outputs_diff = torch.norm(out1 - out2).item()
        
        if inputs_diff > epsilon and outputs_diff < epsilon:
            failures.append(
                f"Found non-injective case: inputs diff={inputs_diff:.6f}, "
                f"outputs diff={outputs_diff:.6f}"
            )
            break
    
    return len(failures) == 0, failures

def test_locality(
    model: SpectralAttention,
    device: torch.device = None,
    window_size: int = 3,
    visualize: bool = False
) -> Tuple[bool, float]:
    """
    Test if the attention mechanism exhibits local bias by measuring the
    attention weights assigned to local neighborhoods.
    
    From the paper: Effective local modeling is crucial for attention mechanisms,
    showing strong bias towards local neighborhoods.
    
    Args:
        model: The SpectralAttention model to test
        device: Device to run tests on
        window_size: Size of local window to measure attention weight concentration
        visualize: Whether to create a heatmap visualization of attention weights
    
    Returns:
        Tuple of (has_local_bias: bool, local_attention_score: float)
    """
    if device is None:
        device = next(model.parameters()).device
    
    seq_len = model.seq_len
    d_model = model.o_proj.out_features
    
    # Create a test sequence with a clear local structure
    x = torch.zeros(1, seq_len, d_model, device=device)
    # Add some local patterns
    for i in range(seq_len):
        if i < seq_len - 1:
            x[0, i, :] = torch.sin(torch.tensor([i/seq_len * 2 * 3.14159]))
    
    # Get model output
    output = model(x)
    
    # Analyze attention weights through the cumsum pattern
    # This is an approximation since we don't have direct access to attention weights
    attention_pattern = torch.zeros(seq_len, seq_len)
    
    # Calculate local attention score
    total_attention = 0
    local_attention = 0
    
    for i in range(seq_len):
        # Consider a window of size window_size around position i
        start_idx = max(0, i - window_size//2)
        end_idx = min(seq_len, i + window_size//2 + 1)
        
        # Measure attention through output dependencies
        center_impact = output[0, i].abs().sum()
        local_impact = output[0, start_idx:end_idx].abs().sum()
        
        total_attention += center_impact
        local_attention += local_impact
        
        attention_pattern[i, start_idx:end_idx] = local_impact / center_impact
    
    local_attention_score = local_attention / total_attention
    
    if visualize:
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_pattern.detach().cpu().numpy(), 
                   cmap='viridis',
                   xticklabels=5, 
                   yticklabels=5)
        plt.title('Attention Weight Distribution')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.show()
    
    # Consider the model to have local bias if local_attention_score > 0.5
    # This threshold can be adjusted based on requirements
    has_local_bias = local_attention_score > 0.5
    
    return has_local_bias, local_attention_score

import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, n_heads: int = 8, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Lower triangular mask for causal attention
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer('mask', mask.view(1, 1, seq_len, seq_len))
        
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        
        # Project and reshape to multi-head format
        Q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # B,H,T,d
        K = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # B,H,T,d
        V = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # B,H,T,d
        
        # Linear attention with cumsum for O(T) complexity
        Z = torch.einsum('bhsp,bhsn->bhspn', V, K)  # combine along sequence dim
        Z_cumsum = torch.cumsum(Z, dim=2)           # cumulative sum to apply mask implicitly
        Y = torch.einsum('bhtn,bhtpn->bhtp', Q, Z_cumsum)
        
        # Reshape and project back
        Y = Y.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(Y)

    
def run_full_test_suite(
    model: SpectralAttention,
    device: torch.device = None,
    visualize: bool = False
) -> None:
    """
    Run both injectivity and locality tests and print comprehensive results.
    """
    print("Running full test suite for Spectral Attention...\n")
    
    # Test injectivity
    print("Testing injectivity...")
    is_injective, failure_messages = test_injectivity(model, device=device)
    print(f"Injectivity test {'PASSED' if is_injective else 'FAILED'}")
    if not is_injective:
        print("Failure cases:")
        for msg in failure_messages:
            print(f"  - {msg}")
    print()
    
    # Test locality
    print("Testing locality...")
    has_local_bias, local_score = test_locality(model, device=device, visualize=visualize)
    print(f"Locality test {'PASSED' if has_local_bias else 'FAILED'}")
    print(f"Local attention score: {local_score:.3f}")
    
    print("\nOverall assessment:")
    print(f"✓ Injectivity: {'Yes' if is_injective else 'No'}")
    print(f"✓ Local bias: {'Yes' if has_local_bias else 'No'} (score: {local_score:.3f})")


if __name__ == "__main__":
    # Test parameters
    seq_len = 10
    d_model = 64
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and test input
    model = LinearAttention(seq_len, d_model, device=device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Run tests
    run_full_test_suite(model, device=device, visualize=True)
    
# # Example usage
# if __name__ == "__main__":
#     # Setup model and parameters
#     seq_len = 10
#     d_model = 10
#     k = 10
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model = SpectralAttention(
#         seq_len=seq_len,
#         d_model=d_model,
#         k=k,
#         use_hankel_L=False,
#         device=device
#     )
    
#     # Run tests
#     run_full_test_suite(model, device=device, visualize=True)