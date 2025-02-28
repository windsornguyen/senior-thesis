from .model import MambaConfig, BaseMamba, Mamba, get_no_recompute_ops

__all__ = ["Mamba"]

mamba_configs = {
    "debug": MambaConfig(
        dim=64,
        num_layers=2,
        num_heads=8,
        state_dim=128,
        num_groups=1,
        vocab_size=200064,
    ),
    "124M": MambaConfig(
        
    ),
    "2B": MambaConfig(
       
    ),
}
