import torch
from thesis.nn.attention_masks import generate_top_k_mask

def main(device: str = "cpu"):
    """
    Visualizes the top-k attention mask mod.

    For demonstration, random query and key tensors (with decently large dimensions)
    are generated, their dot-product is computed as importance scores, and then the
    top-k mask mod is built. Finally, attn_gym's visualization utility displays the resulting
    attention grid.
    
    Args:
        device (str): Device for computation.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 64, 32

    def make_tensor():
        # Generate random tensors so that dot-product scores vary.
        return torch.randn(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    # Compute dot-product attention scores: shape [B, H, SEQ_LEN, SEQ_LEN]
    importance_scores = torch.matmul(query, key.transpose(-1, -2))

    k = 3  # Select top 3 keys per query.
    topk_mask_fn = generate_top_k_mask(k, importance_scores, causal=True)

    # attn_gym will supply its own coordinate grids when invoking the mask mod.
    visualize_attention_scores(
        query, key, mask_mod=topk_mask_fn, device=device, name="top_k_mask"
    )

if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    from jsonargparse import CLI
    CLI(main)
