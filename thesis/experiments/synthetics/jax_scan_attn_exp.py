import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
from tqdm import tqdm

def attention_mechanism_analysis():
    """Comprehensive analysis and experimentation with attention mechanisms."""
    print("ANALYZING ATTENTION MECHANISMS FOR ASSOCIATIVE RECALL")
    print("="*80)
    
    # ----------------------
    # 1. SETUP & DEFINITIONS
    # ----------------------
    # Define small test case for visualizing attention patterns
    B, H, T, P = 1, 1, 8, 2  # Small dimensions for analysis
    
    # Create reproducible test inputs
    key = jax.random.PRNGKey(42)
    Q = jax.random.normal(key, (B, H, T, P))
    key, subkey = jax.random.split(key)
    K = jax.random.normal(subkey, (B, H, T, P))
    key, subkey = jax.random.split(key)
    V = jax.random.normal(subkey, (B, H, T, P))
    
    print("Query input:")
    print(Q[0, 0])
    print("\nKey input:")
    print(K[0, 0])
    print("\nValue input:")
    print(V[0, 0])
    
    # Helper functions for both attention mechanisms
    def softplus(x, beta=1.0):
        """Softplus activation with configurable beta parameter."""
        return (1.0 / beta) * jnp.log(1.0 + jnp.exp(beta * x))
    
    def combine_fn(x, y):
        """Bounded, numerically stable combine function for log-domain scan."""
        m_x, log_N_x, log_D_x = x
        m_y, log_N_y, log_D_y = y

        # Compute new max for stability
        m_new = jnp.maximum(m_x, m_y)

        # Shift values to new log-domain reference
        log_N_x_shifted = log_N_x + (m_x - m_new)[..., None]
        log_N_y_shifted = log_N_y + (m_y - m_new)[..., None]

        log_D_x_shifted = log_D_x + (m_x - m_new)
        log_D_y_shifted = log_D_y + (m_y - m_new)

        # Use logsumexp for stable addition
        log_N_new = jsp.special.logsumexp(jnp.stack([log_N_x_shifted, log_N_y_shifted]), axis=0)
        log_D_new = jsp.special.logsumexp(jnp.stack([log_D_x_shifted, log_D_y_shifted]), axis=0)

        return m_new, log_N_new, log_D_new
    
    # -------------------------------
    # 2. ANALYZE VANILLA VS SCAN
    # -------------------------------
    print("\n2. BASELINE COMPARISON: VANILLA SOFTMAX VS SCAN ATTENTION")
    
    def compute_vanilla_attention(Q, K, V, beta=1.0, temperature=1.0, sharpening=1.0):
        """Compute vanilla softmax attention."""
        # Apply temperature scaling
        if temperature != 1.0:
            Q = Q / temperature
            K = K / temperature
        
        # Apply feature map
        if beta > 0:
            Q_feat = softplus(Q, beta)
            K_feat = softplus(K, beta)
        else:
            # Use exp if beta <= 0
            Q_feat = jnp.exp(Q)
            K_feat = jnp.exp(K)
        
        # Compute attention scores
        scores = jnp.einsum('bhtp,bhsp->bhts', Q_feat, K_feat)
        
        # Create causal mask
        batch_size, num_heads, seq_len, _ = Q_feat.shape
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        
        # Apply mask
        masked_scores = jnp.where(mask, scores, -1e9)
        
        # Apply softmax
        weights = jax.nn.softmax(masked_scores, axis=-1)
        
        # Apply sharpening if requested
        if sharpening != 1.0:
            weights = weights ** sharpening
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-6)
        
        # Compute output
        Y = jnp.einsum('bhts,bhsp->bhtp', weights, V)
        
        return Y, weights, scores
            
    def compute_scan_attention(Q, K, V, beta=1.0, temperature=1.0, sharpening=1.0, pre_normalization=False):
        """Compute scan-based attention."""
        # Apply temperature scaling
        if temperature != 1.0:
            Q = Q / temperature
            K = K / temperature
        
        # Apply feature map
        if beta > 0:
            Q_feat = softplus(Q, beta)
            K_feat = softplus(K, beta)
        else:
            # Use exp if beta <= 0
            Q_feat = jnp.exp(Q)
            K_feat = jnp.exp(K)
        
        # Scan-based attention
        m_initial = jnp.log(K_feat + 1e-6)
        num_initial = jnp.einsum("bhtp,bhtn->bhtpn", V, K_feat)
        denom_initial = K_feat
        
        if pre_normalization:
            # Normalize QK products to be more similar to softmax
            qk_sums = jnp.sum(K_feat, axis=-1, keepdims=True)
            K_feat_normalized = K_feat / (qk_sums + 1e-6)
            num_initial = jnp.einsum("bhtp,bhtn->bhtpn", V, K_feat_normalized)
            denom_initial = K_feat_normalized
        
        tuple_initial = (m_initial, num_initial, denom_initial)
        m_cum, num_cum, denom_cum = jax.lax.associative_scan(combine_fn, tuple_initial, axis=2)
        
        Y_num = jnp.einsum("bhtp,bhtpq->bhtq", Q_feat, num_cum)
        Y_den = jnp.einsum("bhtp,bhtp->bht", Q_feat, denom_cum)
        epsilon = 1e-6
        
        # Compute effective weights for analysis
        weights = np.zeros((B, H, T, T))
        scores = jnp.einsum('bhtp,bhsp->bhts', Q_feat, K_feat)
        
        batch_size, num_heads, seq_len, _ = Q_feat.shape
        for t in range(seq_len):
            for s in range(t+1):
                weights[0, 0, t, s] = jnp.sum(Q_feat[0, 0, t] * K_feat[0, 0, s])
            
            if Y_den[0, 0, t] > 0:
                weights[0, 0, t, :] /= Y_den[0, 0, t]
                
                # Apply optional sharpening
                if sharpening != 1.0:
                    mask = weights[0, 0, t, :] > 0
                    if np.any(mask):
                        weights[0, 0, t, mask] = weights[0, 0, t, mask] ** sharpening
                        weights[0, 0, t, :] /= (np.sum(weights[0, 0, t, :]) + 1e-6)
        
        Y = Y_num / (Y_den[..., None] + epsilon)
        
        # Apply optional post-processing sharpening to outputs
        if sharpening != 1.0:
            sign = jnp.sign(Y)
            Y = sign * jnp.abs(Y) ** sharpening
            
        return Y, weights, scores
    
    # Compute vanilla softmax attention
    Y_vanilla, weights_vanilla, scores_vanilla = compute_vanilla_attention(
        Q, K, V, beta=-1.0)
    
    # Compute scan-based attention with various configurations
    variants = [
        {"name": "Scan (Basic)", "params": {"beta": -1.0}},
        {"name": "Scan (Beta=2.0)", "params": {"beta": 2.0}},
        {"name": "Scan (Beta=5.0)", "params": {"beta": 5.0}},
        {"name": "Scan (Temp=0.5)", "params": {"beta": 2.0, "temperature": 0.5}},
        {"name": "Scan (Sharp=2.0)", "params": {"beta": 2.0, "sharpening": 2.0}},
        {"name": "Scan (PreNorm)", "params": {"beta": 2.0, "pre_normalization": True}}
    ]
    
    results = {}
    results["Vanilla"] = {"Y": Y_vanilla, "weights": weights_vanilla, "scores": scores_vanilla}
    
    for variant in variants:
        Y, weights, scores = compute_scan_attention(Q, K, V, **variant["params"])
        results[variant["name"]] = {"Y": Y, "weights": weights, "scores": scores}
    
    # --------------------------
    # 3. VISUALIZE ATTENTION PATTERNS
    # --------------------------
    print("\n3. VISUALIZING ATTENTION PATTERNS")
    
    def plot_attention_comparison(results, query_positions=None):
        """Compare attention patterns between mechanisms, focusing on query positions."""
        if query_positions is None:
            # Use last two positions by default
            query_positions = [T-2, T-1]
            
        variant_names = list(results.keys())
        num_variants = len(variant_names)
        
        # Create figure with rows for each query position, columns for each variant
        fig, axes = plt.subplots(len(query_positions), num_variants, 
                                figsize=(4*num_variants, 4*len(query_positions)))
        
        # For a single query position, ensure axes is 2D
        if len(query_positions) == 1:
            axes = axes.reshape(1, -1)
        
        # Plot attention patterns for each query position and variant
        for i, query_pos in enumerate(query_positions):
            for j, variant in enumerate(variant_names):
                weights = results[variant]["weights"]
                
                # Plot attention weights for this query position
                im = axes[i, j].imshow(weights[0, 0, query_pos, :query_pos+1].reshape(1, -1), 
                                      cmap='Blues', aspect='auto')
                axes[i, j].set_title(f"{variant}\nQuery {query_pos}")
                
                # Show attention weight values
                for k in range(query_pos+1):
                    weight = weights[0, 0, query_pos, k]
                    axes[i, j].text(k, 0, f"{weight:.2f}", ha="center", va="center", 
                                   color="white" if weight > 0.3 else "black")
                
                # Clean up ticks
                axes[i, j].set_xticks(range(query_pos+1))
                axes[i, j].set_yticks([])
        
        plt.tight_layout()
        return fig
    
    # Plot attention patterns for the query positions
    query_positions = [T-2, T-1]  # Last two positions
    fig_attention = plot_attention_comparison(results, query_positions=query_positions)
    
    # --------------------------
    # 4. ANALYZE KL DIVERGENCE
    # --------------------------
    print("\n4. MEASURING DISTRIBUTION DIVERGENCE")
    
    def compute_kl_divergence(p, q, axis=-1, eps=1e-10):
        """Compute KL divergence between two distributions."""
        # Ensure proper normalization
        p = p / (jnp.sum(p, axis=axis, keepdims=True) + eps)
        q = q / (jnp.sum(q, axis=axis, keepdims=True) + eps)
        
        # Apply smoothing to avoid zeros
        p = p + eps
        q = q + eps
        
        # Renormalize after smoothing
        p = p / jnp.sum(p, axis=axis, keepdims=True)
        q = q / jnp.sum(q, axis=axis, keepdims=True)
        
        # Compute KL divergence
        kl = jnp.sum(p * jnp.log(p / q), axis=axis)
        return kl
    
    print("\nKL Divergence from Vanilla Softmax:")
    variant_names = list(results.keys())
    for variant in variant_names[1:]:  # Skip vanilla
        # Get weights for query positions
        w_vanilla = weights_vanilla[0, 0, query_positions, :]
        w_variant = results[variant]["weights"][0, 0, query_positions, :]
        
        # Compute KL divergence for each query position
        kl_divs = []
        for i, query_pos in enumerate(query_positions):
            # Only consider valid positions (causal attention)
            kl_div = compute_kl_divergence(
                w_vanilla[i, :query_pos+1], w_variant[i, :query_pos+1])
            kl_divs.append(kl_div)
        
        # Print results
        print(f"{variant}: {np.mean(kl_divs):.4f}")
    
    # --------------------------
    # 5. IMPROVED SCAN ATTENTION
    # --------------------------
    print("\n5. DESIGNING IMPROVED SCAN ATTENTION")
    
    def improved_scan_attention(Q, K, V, beta=2.0, temperature=0.5, sharpening=1.5, 
                               attention_scaling=True, post_normalization=False):
        """
        Improved scan attention with multiple enhancements:
        1. Softplus with tuned beta
        2. Temperature scaling
        3. Output sharpening
        4. Attention scaling
        5. Optional post-normalization to sum to 1
        """
        # Apply temperature scaling
        scaled_Q = Q / temperature
        scaled_K = K / temperature
        
        # Apply feature map
        Q_feat = softplus(scaled_Q, beta)
        K_feat = softplus(scaled_K, beta)
        
        # Initialize scan components
        m_initial = jnp.log(K_feat + 1e-6)
        num_initial = jnp.einsum("bhtp,bhtn->bhtpn", V, K_feat)
        denom_initial = K_feat
        
        # Use associative scan
        tuple_initial = (m_initial, num_initial, denom_initial)
        m_cum, num_cum, denom_cum = jax.lax.associative_scan(combine_fn, tuple_initial, axis=2)
        
        # Compute output
        Y_num = jnp.einsum("bhtp,bhtpq->bhtq", Q_feat, num_cum)
        Y_den = jnp.einsum("bhtp,bhtp->bht", Q_feat, denom_cum)
        epsilon = 1e-6
        
        # Apply normalization
        Y = Y_num / (Y_den[..., None] + epsilon)
        
        # Apply attention scaling - reweight outputs based on position
        if attention_scaling:
            # Scale outputs based on position (later positions get higher weight)
            batch_size, num_heads, seq_len, d_head = Y.shape
            position_scaling = jnp.arange(1, seq_len + 1) / seq_len
            position_scaling = position_scaling.reshape(1, 1, seq_len, 1)
            Y = Y * position_scaling
        
        # Apply sharpening (if requested)
        if sharpening != 1.0:
            sign = jnp.sign(Y)
            Y = sign * jnp.abs(Y) ** sharpening
        
        # Compute effective weights for visualization
        weights = np.zeros((B, H, T, T))
        for t in range(T):
            for s in range(t+1):
                weights[0, 0, t, s] = jnp.sum(Q_feat[0, 0, t] * K_feat[0, 0, s])
            
            if Y_den[0, 0, t] > 0:
                weights[0, 0, t, :] /= Y_den[0, 0, t]
                
                # Apply sharpening
                if sharpening != 1.0:
                    mask = weights[0, 0, t, :] > 0
                    if np.any(mask):
                        weights[0, 0, t, mask] = weights[0, 0, t, mask] ** sharpening
                
                # Force sum to 1 per row if requested
                if post_normalization:
                    row_sum = np.sum(weights[0, 0, t, :])
                    if row_sum > 0:
                        weights[0, 0, t, :] /= row_sum
        
        return Y, weights
    
    # Test our improved scan attention
    Y_improved, weights_improved = improved_scan_attention(
        Q, K, V, beta=2.0, temperature=0.5, sharpening=1.5, 
        attention_scaling=True, post_normalization=True)
    
    results["Improved Scan"] = {"Y": Y_improved, "weights": weights_improved}
    
    # Plot comparison with improved version included
    fig_improved = plot_attention_comparison(
        {"Vanilla": results["Vanilla"], "Basic Scan": results["Scan (Basic)"], 
         "Improved Scan": results["Improved Scan"]}, 
        query_positions=query_positions)
    
    # --------------------------
    # 6. SYSTEMATIC PARAMETER SWEEP
    # --------------------------
    print("\n6. SYSTEMATIC PARAMETER SWEEP")
    
    # Test parameters to find optimal configuration
    betas = [1.0, 2.0, 3.0]
    temperatures = [0.5, 0.75, 1.0]
    sharpenings = [1.0, 1.5, 2.0]
    
    # Create shorter parameter list for demo
    sweep_results = []
    
    # Perform parameter sweep (simplified for demonstration)
    for beta in betas:
        for temp in temperatures:
            for sharp in sharpenings:
                # Run improved scan with these parameters
                Y, weights = improved_scan_attention(
                    Q, K, V, beta=beta, temperature=temp, sharpening=sharp, 
                    attention_scaling=True, post_normalization=True)
                
                # Compute KL divergence from vanilla softmax
                kl_divs = []
                for i, query_pos in enumerate(query_positions):
                    # Handle possible empty slices
                    if query_pos > 0:
                        kl_div = compute_kl_divergence(
                            weights_vanilla[0, 0, query_pos, :query_pos+1], 
                            weights[0, 0, query_pos, :query_pos+1])
                        kl_divs.append(kl_div)
                
                avg_kl = float(np.mean(kl_divs))
                sweep_results.append({
                    "beta": beta,
                    "temperature": temp,
                    "sharpening": sharp,
                    "kl_divergence": avg_kl
                })
    
    # Find best parameters
    sweep_results.sort(key=lambda x: x["kl_divergence"])
    best_params = sweep_results[0]
    
    print(f"\nBest parameters to match softmax:")
    print(f"Beta: {best_params['beta']}")
    print(f"Temperature: {best_params['temperature']}")
    print(f"Sharpening: {best_params['sharpening']}")
    print(f"KL Divergence: {best_params['kl_divergence']:.4f}")
    
    # Create visualization of parameter sweep
    def plot_parameter_sweep(sweep_results):
        """Visualize parameter sweep results."""
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Extract data for plotting
        params = ["beta", "temperature", "sharpening"]
        
        for i, param in enumerate(params):
            # Group by parameter
            param_values = sorted(set(r[param] for r in sweep_results))
            kl_by_param = {}
            
            for val in param_values:
                kl_by_param[val] = [r["kl_divergence"] for r in sweep_results if r[param] == val]
            
            # Plot boxplot
            boxes = axes[i].boxplot([kl_by_param[val] for val in param_values], 
                                   labels=[str(val) for val in param_values])
            axes[i].set_title(f"Effect of {param} on KL Divergence")
            axes[i].set_ylabel("KL Divergence from Softmax")
            axes[i].set_xlabel(param)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    fig_sweep = plot_parameter_sweep(sweep_results)
    
    # --------------------------
    # 7. FINAL RECOMMENDED MODEL
    # --------------------------
    print("\n7. FINAL RECOMMENDATION")
    
    # Implement the final recommended model class
    class ImprovedScanAttention(nn.Module):
        """Scan attention optimized for associative recall tasks."""
        
        dim: int
        num_heads: int
        beta_init: float = 2.0
        temperature_init: float = 0.5
        sharpening_init: float = 1.5
        eps: float = 1e-6
        learnable_params: bool = True
        
        def setup(self):
            assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
            self.head_dim = self.dim // self.num_heads
            
            self.wq = nn.Dense(self.dim)
            self.wk = nn.Dense(self.dim)
            self.wv = nn.Dense(self.dim)
            self.out_proj = nn.Dense(self.dim)
            
            if self.learnable_params:
                # Learnable parameters (per head)
                self.log_beta = self.param('log_beta', 
                                        lambda key, shape: jnp.ones(shape) * jnp.log(self.beta_init),
                                        (self.num_heads,))
                
                self.log_temp = self.param('log_temp',
                                        lambda key, shape: jnp.ones(shape) * jnp.log(self.temperature_init),
                                        (self.num_heads,))
                
                self.log_sharp = self.param('log_sharp',
                                          lambda key, shape: jnp.ones(shape) * jnp.log(self.sharpening_init),
                                          (self.num_heads,))
            else:
                # Fixed parameters
                self.beta = self.beta_init
                self.temperature = self.temperature_init
                self.sharpening = self.sharpening_init
        
        def softplus(self, x, beta):
            """Softplus with configurable beta."""
            return (1.0 / beta) * jnp.log(1.0 + jnp.exp(beta * x))
        
        def combine_fn(self, x, y):
            m_x, log_N_x, log_D_x = x
            m_y, log_N_y, log_D_y = y
            
            m_new = jnp.maximum(m_x, m_y)
            log_N_x_shifted = log_N_x + (m_x - m_new)[..., None]
            log_N_y_shifted = log_N_y + (m_y - m_new)[..., None]
            log_D_x_shifted = log_D_x + (m_x - m_new)
            log_D_y_shifted = log_D_y + (m_y - m_new)
            
            log_N_new = jsp.special.logsumexp(jnp.stack([log_N_x_shifted, log_N_y_shifted]), axis=0)
            log_D_new = jsp.special.logsumexp(jnp.stack([log_D_x_shifted, log_D_y_shifted]), axis=0)
            
            return m_new, log_N_new, log_D_new
        
        def __call__(self, x, training=False):
            batch_size, seq_len, _ = x.shape
            
            # Linear projections
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            
            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Transpose for batched operations
            q = q.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
            k = k.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
            v = v.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
            
            if self.learnable_params:
                # Get learnable parameters
                beta = jnp.exp(self.log_beta)
                temperature = jnp.exp(self.log_temp)
                sharpening = jnp.exp(self.log_sharp)
                
                # Reshape for broadcasting
                beta = beta[:, None, None]
                temperature = temperature[:, None, None]
                sharpening = sharpening[:, None, None]
            else:
                # Use fixed parameters
                beta = self.beta
                temperature = self.temperature
                sharpening = self.sharpening
            
            # Apply temperature scaling
            q = q / temperature
            k = k / temperature
            
            # Apply feature map with learnable beta
            Q_feat = self.softplus(q, beta)
            K_feat = self.softplus(k, beta)
            
            # Initialize scan components
            m_initial = jnp.log(K_feat + self.eps)
            num_initial = jnp.einsum("bhtp,bhtn->bhtpn", v, K_feat)
            denom_initial = K_feat
            
            # Perform associative scan
            tuple_initial = (m_initial, num_initial, denom_initial)
            m_cum, num_cum, denom_cum = jax.lax.associative_scan(
                self.combine_fn, tuple_initial, axis=2)
            
            # Compute output
            Y_num = jnp.einsum("bhtp,bhtpq->bhtq", Q_feat, num_cum)
            Y_den = jnp.einsum("bhtp,bhtp->bht", Q_feat, denom_cum)
            
            # Apply normalization
            Y = Y_num / (Y_den[..., None] + self.eps)
            
            # Apply sharpening
            if self.learnable_params:
                # Per-head sharpening
                sign = jnp.sign(Y)
                Y = sign * jnp.abs(Y) ** sharpening
            else:
                # Fixed sharpening
                sign = jnp.sign(Y)
                Y = sign * jnp.abs(Y) ** sharpening
            
            # Position-dependent scaling - boost later positions
            position_scaling = jnp.arange(1, seq_len + 1) / seq_len
            position_scaling = position_scaling.reshape(1, 1, seq_len, 1)
            Y = Y * position_scaling
            
            # Reshape output
            Y = Y.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
            Y = Y.reshape(batch_size, seq_len, self.dim)
            
            # Output projection
            out = self.out_proj(Y)
            return out
    
    print("\nFinal Recommended Improvements:")
    print("""
    1. Replace exp with softplus (beta ≈ 2.0) for controlled sharpness
    2. Use temperature scaling (temp ≈ 0.5) for more concentrated attention
    3. Apply output sharpening (power ≈ 1.5) to enhance focus
    4. Add position-dependent scaling to boost later sequence positions
    5. Make all parameters (beta, temperature, sharpening) learnable per head
    
    With these improvements, scan-based attention should perform competitively
    with vanilla softmax attention on associative recall tasks while maintaining
    its computational efficiency advantages.
    """)
    
    return {
        "attention_comparison": fig_attention,
        "improved_comparison": fig_improved,
        "parameter_sweep": fig_sweep,
        "best_params": best_params,
        "results": results
    }

# Run the analysis
results = attention_mechanism_analysis()