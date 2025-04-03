import argparse

parser = argparse.ArgumentParser(description="Train models on Associative Recall task")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["transformer", "spectron", "flashstu", "mamba", "sparse_flashstu", "all"],
    help="Model to train (transformer, spectron, flashstu, mamba, sparse_flashstu, or all)",
)

# Dataset configs
parser.add_argument("--num_examples", type=int, default=640000, help="Number of training examples")
parser.add_argument("--steps", type=int, default=100000, help="Number of training steps")
parser.add_argument("--eval", type=int, default=250, help="Steps between evaluations")
parser.add_argument("--bsz", type=int, default=64, help="Batch size for training")
parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
parser.add_argument("--vocab_size", type=int, default=8192, help="Vocabulary size")
parser.add_argument("--num_pairs", type=int, default=32, help="Number of pairs")
parser.add_argument("--num_queries", type=int, default=3, help="Number of queries")
parser.add_argument("--random_non_queries", type=bool, default=True, help="Replace all the 0's with random values in the input.")
parser.add_argument("--seed", type=int, default=1746, help="Random seed")

# Model configs
parser.add_argument("--dim", type=int, default=64, help="Model dimension")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
parser.add_argument("--theta", type=float, default=10000.0, help="Theta for rotary embeddings")
parser.add_argument("--bias", type=bool, default=False, help="Use bias in linear layers")
parser.add_argument("--weight_tying", type=bool, default=False, help="Weight tying embeddings with output layer")
parser.add_argument("--mlp_scale", type=int, default=4, help="MLP scale")
parser.add_argument("--use_alibi", type=bool, default=False, help="Use ALiBi")
parser.add_argument("--use_hankel_L", type=bool, default=False, help="Use Hankel L")
parser.add_argument("--use_tensordot", type=bool, default=False, help="Use tensordot")
parser.add_argument("--use_flash_fft", type=bool, default=False, help="Use flash FFT")
parser.add_argument("--use_mem_eff_path", type=bool, default=True, help="Use memory efficient path")
parser.add_argument("--window_size", type=int, default=512, help="Local attention window size")
parser.add_argument("--softcap", type=float, default=50.0, help="Logit softcap for attention")
parser.add_argument("--dtype", type=str, default="float32", help="Training dtype")
parser.add_argument("--num_eigh", type=int, default=16, help="Number of eigenvectors")

# Training configs
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")

# Optimizations configs
parser.add_argument("--torch_compile", type=bool, default=False, help="Use torch.compile")

# RoPE configs
parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
parser.add_argument("--rope_factor", type=float, default=40.0, help="RoPE factor")
parser.add_argument("--beta_fast", type=int, default=32, help="Beta fast")
parser.add_argument("--beta_slow", type=int, default=1, help="Beta slow")

args = parser.parse_args()
