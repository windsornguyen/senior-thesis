import torch
from thesis.nn.attention_masks import (
    causal_mask,
    generate_sliding_window_mask,
    generate_dilated_sliding_window_mask,
)

from omegaconf import DictConfig


def get_attention_mask(config: DictConfig):
    """Get the appropriate attention mask based on config."""
    attn_config = config.model.get("attention", {"type": "causal"})
    mask_type = attn_config.get("type", "causal")

    if mask_type == "causal":
        return causal_mask
    elif mask_type == "sliding_window":
        return generate_sliding_window_mask(
            window_size=config.model.window_size, causal=attn_config.get("causal", True)
        )
    elif mask_type == "dilated_window":
        return generate_dilated_sliding_window_mask(
            window_size=config.model.window_size,
            dilation=attn_config.get("dilation", 1),
            causal=attn_config.get("causal", True),
        )
    else:
        raise ValueError(f"Unknown attention mask type: {mask_type}")


def create_model(config: DictConfig, expected_shapes: dict, device: torch.device):
    """Create and validate model based on config and expected dataset shapes."""
    model_type = config.get("model_type", None)
    if model_type is None:
        raise ValueError("No model_type specified in config.")

    if model_type.lower() == "transformer":
        from thesis.experiments.models.transformer import Transformer, TransformerConfig

        # Validate dimensions against dataset expectations
        if config.model.d_in != expected_shapes["d_in"]:
            raise ValueError(
                f"Model input dimension (d_in={config.model.d_in}) does not match "
                f"dataset input dimension ({expected_shapes['d_in']})"
            )
        if config.model.d_out != expected_shapes["d_out"]:
            raise ValueError(
                f"Model output dimension (d_out={config.model.d_out}) does not match "
                f"dataset output dimension ({expected_shapes['d_out']})"
            )
        if config.model.seq_len < expected_shapes["seq_len"]:
            raise ValueError(
                f"Model sequence length (seq_len={config.model.seq_len}) is smaller than "
                f"dataset sequence length ({expected_shapes['seq_len']})"
            )

        # Get the appropriate attention mask
        mask_mod = get_attention_mask(config)

        # TODO: Write a score_mod

        torch_dtype = getattr(torch, config.model.torch_dtype)
        config = TransformerConfig(
            bsz=config.model.bsz,
            dim=config.model.dim,
            d_in=config.model.d_in,
            d_out=config.model.d_out,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            seq_len=config.model.seq_len,
            window_size=config.model.window_size,
            vocab_size=config.model.vocab_size,
            mlp_scale=config.model.mlp_scale,
            bias=config.model.bias,
            dropout=config.model.dropout,
            softcap=config.model.softcap,
            theta=config.model.theta,
            use_alibi=config.model.use_alibi,
            torch_dtype=torch_dtype,
            device=device,
        )
        model = Transformer(config, mask_mod=mask_mod).to(device=device, dtype=torch_dtype)
    else:
        raise ValueError(f"Model '{model_type}' has not yet been implemented!")

    return model
