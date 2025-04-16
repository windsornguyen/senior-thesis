import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve

@jax.jit
def causal_conv(filters: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Performs causal 1D convolution via FFT with K filters across heads and channels.
    
    Args:
        filters: jnp.ndarray of shape [L, K]
        inputs: jnp.ndarray of shape [B, H, L, D]
    
    Returns:
        jnp.ndarray of shape [B, H, L, K * D]
    """
    # [L,] with [L,] -> [L,]
    conv_1d = lambda x, f: convolve(x, f, method='fft')[:x.shape[0]]
    
    # [L,] with [L, K] -> [L, K]
    conv_filters = lambda x: jax.vmap(conv_1d, in_axes=(None, 1), out_axes=1)(x, filters)
    
    # [L, D] -> [L, K, D]
    conv_channels = lambda x: jax.vmap(conv_filters, in_axes=1, out_axes=2)(x)
    
    # [H, L, D] -> [H, L, K, D]
    conv_heads = lambda x: jax.vmap(conv_channels, in_axes=0, out_axes=0)(x)
    
    # [B, H, L, D] -> [B, H, L, K, D]
    conv_batch = lambda x: jax.vmap(conv_heads, in_axes=0, out_axes=0)(x)
    
    # Apply and reshape: [B, H, L, K, D] -> [B, H, L, K * D]
    output = conv_batch(inputs)
    B, H, L, K, D = output.shape
    return output.reshape(B, H, L, K * D)

# Simple verification
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    B, H, L, D, K = 2, 4, 16, 3, 2
    x = jnp.linspace(-3, 3, L)
    filters = jnp.stack([jnp.exp(-x**2 / (2 * 1.0**2)), jnp.exp(-x**2 / (2 * 1.5**2))], axis=1)
    inputs = jax.random.normal(key, (B, H, L, D))
    output = causal_conv(filters, inputs)
    assert output.shape == (B, H, L, K * D)
    print("Shape check passed:", output.shape, "matches", (B, H, L, K * D))