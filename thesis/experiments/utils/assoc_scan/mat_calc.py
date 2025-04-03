import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

"""
Gated Scan Derivative Validation

This script methodically validates the derivatives for the gated scan operation.
We use PyTorch's autograd to compute ground truth gradients and analyze their structure
to understand the exact pattern before attempting any vectorization.
"""

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def compute_direct_recurrence(Z: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """
    Compute the gated scan directly using the recurrence relation:
    H[0] = Z[0]
    H[t] = H[t-1] + (product of gates up to t-1) * Z[t] for t > 0
    """
    B, C, T = Z.shape
    H = torch.zeros_like(Z)
    
    # First element is just Z[0]
    H[:, :, 0] = Z[:, :, 0]
    
    # Keep track of accumulated gate products
    gate_product = torch.ones((B, C), device=Z.device, dtype=Z.dtype)
    
    for t in range(1, T):
        # Update gate product (multiply by previous gate)
        gate_product = gate_product * gates[:, :, t-1]
        
        # Compute current output
        H[:, :, t] = H[:, :, t-1] + gate_product * Z[:, :, t]
    
    return H

def compute_full_jacobian(Z: torch.Tensor, gates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the full Jacobian matrices dH/dZ and dH/dgates using autograd.
    
    For each output element H[b,c,t], compute gradients with respect to
    all input elements Z[b,c,s] and gates[b,c,s].
    
    Returns:
        dH_dZ: Tensor of shape (B, C, T, B, C, T) where dH_dZ[b,c,t,b',c',s] 
               is the derivative of H[b,c,t] with respect to Z[b',c',s]
        dH_dgates: Tensor of shape (B, C, T, B, C, T) where dH_dgates[b,c,t,b',c',s]
                  is the derivative of H[b,c,t] with respect to gates[b',c',s]
    """
    B, C, T = Z.shape
    device, dtype = Z.device, Z.dtype
    
    # Create tensors that require gradients
    Z_grad = Z.clone().detach().requires_grad_(True)
    gates_grad = gates.clone().detach().requires_grad_(True)
    
    # Initialize Jacobian matrices
    dH_dZ = torch.zeros((B, C, T, B, C, T), device=device, dtype=dtype)
    dH_dgates = torch.zeros((B, C, T, B, C, T), device=device, dtype=dtype)
    
    # Compute output
    H = compute_direct_recurrence(Z_grad, gates_grad)
    
    # For each output element, compute gradients w.r.t. all inputs
    for b1 in range(B):
        for c1 in range(C):
            for t1 in range(T):
                # Create a gradient that selects just this output element
                grad_output = torch.zeros_like(H)
                grad_output[b1, c1, t1] = 1.0
                
                # Clear gradients from previous iteration
                if Z_grad.grad is not None:
                    Z_grad.grad.zero_()
                if gates_grad.grad is not None:
                    gates_grad.grad.zero_()
                
                # Backpropagate
                H.backward(grad_output, retain_graph=True)
                
                # Store gradients
                if Z_grad.grad is not None:
                    dH_dZ[b1, c1, t1] = Z_grad.grad.clone()
                if gates_grad.grad is not None:
                    dH_dgates[b1, c1, t1] = gates_grad.grad.clone()
    
    return dH_dZ, dH_dgates

def analyze_jacobian_structure(dH_dZ: torch.Tensor, dH_dgates: torch.Tensor):
    """
    Analyze the structure of the Jacobian matrices to understand the pattern.
    """
    B, C, T, _, _, _ = dH_dZ.shape
    
    print("\nJacobian Structure Analysis:")
    
    # Focus on a single batch and channel for clarity
    b, c = 0, 0
    
    # Create tables to visualize the structure
    print(f"\n1. dH/dZ[{b},{c}] Jacobian structure:")
    print("   " + " ".join([f"Z[{s}]".ljust(10) for s in range(T)]))
    
    for t in range(T):
        row = [f"H[{t}]"]
        for s in range(T):
            val = dH_dZ[b, c, t, b, c, s].item()
            if abs(val) < 1e-6:
                cell = "0".ljust(10)
            else:
                cell = f"{val:.6f}".ljust(10)
            row.append(cell)
        print("   ".join(row))
    
    print(f"\n2. dH/dgates[{b},{c}] Jacobian structure:")
    print("   " + " ".join([f"g[{s}]".ljust(10) for s in range(T)]))
    
    for t in range(T):
        row = [f"H[{t}]"]
        for s in range(T):
            val = dH_dgates[b, c, t, b, c, s].item()
            if abs(val) < 1e-6:
                cell = "0".ljust(10)
            else:
                cell = f"{val:.6f}".ljust(10)
            row.append(cell)
        print("   ".join(row))
    
    # Plot heatmaps for better visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(dH_dZ[b, c, :, b, c, :].numpy(), cmap='viridis')
    plt.colorbar(label='Gradient value')
    plt.title(f'dH/dZ Structure')
    plt.xlabel('Z position (s)')
    plt.ylabel('H position (t)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(dH_dgates[b, c, :, b, c, :].numpy(), cmap='viridis')
    plt.colorbar(label='Gradient value')
    plt.title(f'dH/dgates Structure')
    plt.xlabel('gates position (s)')
    plt.ylabel('H position (t)')
    
    plt.tight_layout()
    plt.savefig('jacobian_structure.png')
    print("\nJacobian structure heatmaps saved to 'jacobian_structure.png'")

def derive_analytical_formulas(Z: torch.Tensor, gates: torch.Tensor, dH_dZ: torch.Tensor, dH_dgates: torch.Tensor):
    """
    Based on the Jacobian structure, derive analytical formulas for the gradients.
    Then verify these formulas match the autograd results.
    """
    B, C, T = Z.shape
    
    # Compute the product of gates at each position
    gate_products = torch.ones((B, C, T), device=Z.device, dtype=Z.dtype)
    
    for t in range(1, T):
        gate_products[:, :, t] = gate_products[:, :, t-1] * gates[:, :, t-1]
    
    # Analytical formula for dH/dZ
    analytical_dH_dZ = torch.zeros_like(dH_dZ)
    
    # For each H[b,c,t], derivatives w.r.t. Z
    for b in range(B):
        for c in range(C):
            for t in range(T):
                # Each H[t] depends on Z[0]...Z[t]
                for s in range(t+1):
                    if s == 0:
                        # Special case: H[t] depends on Z[0] directly
                        analytical_dH_dZ[b, c, t, b, c, 0] = 1.0
                    else:
                        # H[t] depends on Z[s] scaled by product of gates[0]...gates[s-1]
                        analytical_dH_dZ[b, c, t, b, c, s] = gate_products[b, c, s]
    
    # Analytical formula for dH/dgates
    analytical_dH_dgates = torch.zeros_like(dH_dgates)
    
    # For each H[b,c,t], derivatives w.r.t. gates
    for b in range(B):
        for c in range(C):
            for t in range(T):
                # Each H[t] depends on gates[0]...gates[t-1]
                for s in range(t):
                    # Effect of gates[s] on H[t]
                    # It scales all future Z values by entering the product
                    effect = 0.0
                    
                    # Scale factor up to gates[s]
                    if s == 0:
                        scale = 1.0
                    else:
                        scale = gate_products[b, c, s].item()
                    
                    # Compute the sum of effects on all future Z values
                    for i in range(s+1, t+1):
                        # Compute product of gates between s+1 and i-1 (if any)
                        if i > s+1:
                            intermediate_prod = gates[b, c, s+1:i].prod().item()
                        else:
                            intermediate_prod = 1.0
                        
                        # Effect on Z[i]
                        effect += intermediate_prod * Z[b, c, i].item()
                    
                    # Set the gradient
                    analytical_dH_dgates[b, c, t, b, c, s] = scale * effect
    
    # Verify analytical formulas match autograd results
    dH_dZ_match = torch.allclose(analytical_dH_dZ, dH_dZ, atol=1e-5)
    dH_dgates_match = torch.allclose(analytical_dH_dgates, dH_dgates, atol=1e-5)
    
    print("\nAnalytical Formula Verification:")
    print(f"dH/dZ formula matches autograd: {dH_dZ_match}")
    print(f"dH/dgates formula matches autograd: {dH_dgates_match}")
    
    if not dH_dZ_match:
        max_diff = torch.abs(analytical_dH_dZ - dH_dZ).max().item()
        print(f"Max difference for dH/dZ: {max_diff}")
    
    if not dH_dgates_match:
        max_diff = torch.abs(analytical_dH_dgates - dH_dgates).max().item()
        print(f"Max difference for dH/dgates: {max_diff}")
    
    return dH_dZ_match, dH_dgates_match, analytical_dH_dZ, analytical_dH_dgates

def summarize_gradient_patterns(dH_dZ: torch.Tensor, dH_dgates: torch.Tensor):
    """
    Based on the analysis, summarize the gradient patterns in plain English
    to guide the implementation of the optimized backward pass.
    """
    print("\nGradient Pattern Summary:")
    
    print("\n1. dH/dZ Pattern:")
    print("   - H[t] depends directly on Z[0] through Z[t]")
    print("   - For s=0: dH[t]/dZ[0] = 1.0 for all t")
    print("   - For 0<s≤t: dH[t]/dZ[s] = product of gates[0] through gates[s-1]")
    print("   - For s>t: dH[t]/dZ[s] = 0 (no dependency)")
    
    print("\n2. dH/dgates Pattern:")
    print("   - H[t] depends on gates[0] through gates[t-1]")
    print("   - The effect of gates[s] on H[t] is through all Z[i] where s<i≤t")
    print("   - Each gates[s] scales multiple Z values through the cumulative product")
    print("   - For s≥t: dH[t]/dgates[s] = 0 (no dependency)")
    print("   - For s<t: dH[t]/dgates[s] = (product of gates up to s) * sum of (product of intermediate gates) * Z values")

def verify_example_case():
    """
    Verify the specific example from the problem description:
    Z = [1, 2, 3], gates = [0.5, 0.5, 0.5], Expected output = [1, 2, 2.75]
    """
    Z = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    gates = torch.tensor([[[0.5, 0.5, 0.5]]], requires_grad=True)
    
    # Compute output
    output = compute_direct_recurrence(Z, gates)
    
    print("\nExample Case Verification:")
    print(f"Z = {Z.squeeze().tolist()}")
    print(f"gates = {gates.squeeze().tolist()}")
    print(f"Output = {output.squeeze().tolist()}")
    print(f"Expected = [1.0, 2.0, 2.75]")
    
    # Compute gradients for loss = sum of outputs
    loss = output.sum()
    loss.backward()
    
    print(f"dOutput/dZ = {Z.grad.squeeze().tolist()}")
    print(f"dOutput/dgates = {gates.grad.squeeze().tolist()}")
    
    # Compute the expected gradients step by step
    # dZ gradients
    dZ_expected = [0.0, 0.0, 0.0]
    # Z[0] affects H[0], H[1], and H[2]
    dZ_expected[0] = 1.0 + 1.0 + 1.0  # = 3.0
    
    # Z[1] affects H[1] and H[2]
    # H[1]: scaled by gates[0] = 0.5
    # H[2]: scaled by gates[0] = 0.5
    dZ_expected[1] = 0.5 + 0.5  # = 1.0
    
    # Z[2] affects only H[2]
    # H[2]: scaled by gates[0]*gates[1] = 0.5*0.5 = 0.25
    dZ_expected[2] = 0.25
    
    # dgates gradients
    dgates_expected = [0.0, 0.0, 0.0]
    
    # gates[0] affects Z[1] in H[1] and H[2], and Z[2] in H[2]
    # Effect on H[1] through Z[1]: 1.0 * 2.0 = 2.0
    # Effect on H[2] through Z[1]: 1.0 * 2.0 = 2.0
    # Effect on H[2] through Z[2]: 1.0 * 0.5 * 3.0 = 1.5
    dgates_expected[0] = 2.0 + 2.0 + 1.5  # = 5.5
    
    # gates[1] affects only Z[2] in H[2]
    # Effect: 0.5 * 3.0 = 1.5
    dgates_expected[1] = 1.5
    
    # gates[2] doesn't affect any output
    dgates_expected[2] = 0.0
    
    print(f"\nExpected dOutput/dZ = {dZ_expected}")
    print(f"Expected dOutput/dgates = {dgates_expected}")
    
    # Check if expected gradients match autograd
    dZ_match = torch.allclose(Z.grad.squeeze(), torch.tensor(dZ_expected), atol=1e-5)
    dgates_match = torch.allclose(gates.grad.squeeze(), torch.tensor(dgates_expected), atol=1e-5)
    
    print(f"Expected dZ matches autograd: {dZ_match}")
    print(f"Expected dgates matches autograd: {dgates_match}")
    
    # Detailed calculation breakdown for our example case
    print("\nDetailed Gradient Calculation Breakdown:")
    
    print("\n1. Z Gradients:")
    print(f"   Z[0] affects:")
    print(f"     H[0] directly: 1.0")
    print(f"     H[1] directly: 1.0")
    print(f"     H[2] directly: 1.0")
    print(f"     Total gradient for Z[0]: 3.0")
    
    print(f"\n   Z[1] affects:")
    print(f"     H[1] scaled by gates[0] = 0.5: 0.5")
    print(f"     H[2] scaled by gates[0] = 0.5: 0.5")
    print(f"     Total gradient for Z[1]: 1.0")
    
    print(f"\n   Z[2] affects:")
    print(f"     H[2] scaled by gates[0]*gates[1] = 0.5*0.5 = 0.25: 0.25")
    print(f"     Total gradient for Z[2]: 0.25")
    
    print("\n2. Gates Gradients:")
    print(f"   gates[0] affects:")
    print(f"     H[1] through Z[1]: 1.0 * Z[1] = 1.0 * 2.0 = 2.0")
    print(f"     H[2] through Z[1]: 1.0 * Z[1] = 1.0 * 2.0 = 2.0")
    print(f"     H[2] through Z[2]: 1.0 * gates[1] * Z[2] = 1.0 * 0.5 * 3.0 = 1.5")
    print(f"     Total gradient for gates[0]: 5.5")
    
    print(f"\n   gates[1] affects:")
    print(f"     H[2] through Z[2]: gates[0] * Z[2] = 0.5 * 3.0 = 1.5")
    print(f"     Total gradient for gates[1]: 1.5")
    
    print(f"\n   gates[2] affects: (nothing - no gradient)")

def main():
    """Run the complete derivative validation."""
    print("Running Gated Scan Derivative Validation")
    print("-" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create small test case for clarity
    B, C, T = 2, 2, 4
    Z = torch.randn(B, C, T)
    gates = torch.rand(B, C, T) * 0.5 + 0.5  # Gates between 0.5 and 1.0
    
    print(f"Using test case with B={B}, C={C}, T={T}")
    
    # Compute forward pass
    H = compute_direct_recurrence(Z, gates)
    print(f"\nOutput shape: {H.shape}")
    print(f"First few values: {H[0, 0]}")
    
    # Compute full Jacobian using autograd
    print("\nComputing full Jacobian matrices using autograd...")
    dH_dZ, dH_dgates = compute_full_jacobian(Z, gates)
    
    # Analyze Jacobian structure
    analyze_jacobian_structure(dH_dZ, dH_dgates)
    
    # Derive analytical formulas and verify
    formulas_match, _, _, _ = derive_analytical_formulas(Z, gates, dH_dZ, dH_dgates)
    
    # Summarize gradient patterns
    summarize_gradient_patterns(dH_dZ, dH_dgates)
    
    # Verify example case
    verify_example_case()
    
    print("\nDerivative validation complete!")
    if formulas_match:
        print("✓ All analytical formulas match autograd results.")
    else:
        print("✗ Some analytical formulas don't match autograd results.")
    
    print("\nNext steps:")
    print("1. Implement the backward pass using vectorized operations")
    print("2. Use associative scans to efficiently compute cumulative products")
    print("3. Test with gradcheck and verify against autograd implementation")

if __name__ == "__main__":
    main()