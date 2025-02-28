import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_dim = 4 * dim
        self.gate_proj = nn.Linear(dim, self.hidden_dim)
        self.up_proj = nn.Linear(dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        gate = self.gate_proj(x)
        modulated_gate = F.silu(gate)
        up = self.up_proj(x)
        fuse = modulated_gate * up
        outputs = self.down_proj(fuse)
        outputs = self.dropout(outputs)
        return outputs


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k**0.25
    phi_k = phi_k.to(device=device, dtype=dtype)
    return phi_k


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z


def compute_dimensions(n: int) -> tuple[int, int, int]:
    if n <= 2:
        raise ValueError("n must be greater than 2")

    T_prime = (math.ceil(math.sqrt(n - 2))) ** 2 + 2
    sqrt_T_prime = math.ceil(math.sqrt(T_prime - 2))
    k_max = sqrt_T_prime
    return T_prime, sqrt_T_prime, k_max


def get_tensorized_spectral_filters(
    n: int = 8192,
    k: int = 24,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute tensorized spectral filters for given sequence length and filter count.

    Args:
        n: Sequence length
        k: Number of filters
        use_hankel_L: Hankel_main ⊗ Hankel_L? Default is Hankel_main ⊗ Hankel_main.
        device: Computation device
        dtype: Computation dtype
    """
    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    Z = get_hankel(sqrt_T_prime)
    sigma, phi = torch.linalg.eigh(Z)
    phi_i = phi[:, -k:] * sigma[-k:] ** 0.25

    if use_hankel_L:  # TODO: We may want to use Hankel_L above too if use_hankel_L is true, make another variable for this (mix != use_hankel_L)
        Z_L = get_hankel(sqrt_T_prime, True)
        sigma_L, phi_L = torch.linalg.eigh(Z_L)
        phi_j = phi_L[:, -k:] * sigma_L[-k:] ** 0.25
    else:
        phi_j = phi_i

    filters = torch.kron(phi_i, phi_j)
    return filters.to(device=device, dtype=dtype)


class LearnableSpectralFilters(nn.Module):
    def __init__(self, seq_len, k, use_hankel_L=False, device=None, dtype=torch.float32):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        self.use_hankel_L = use_hankel_L
        self.device = device

        filters = get_tensorized_spectral_filters(
            n=seq_len,
            k=k,
            use_hankel_L=use_hankel_L,
            device=device,
            dtype=dtype,
        )
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


class SpectralAttention(nn.Module):
    """
    Implements the linear form of structured masked attention using spectral filters.
    According to the linear form:

    Z = contract(V,K) -> (B,S,i,j) with i,j=k
    H = contract(L,Z) -> (B,T,i,j)
    Y = contract(Q,H) -> (B,T,j)

    Then project Y back to (B,T,dim).

    We print shapes at each step for verification.
    """

    def __init__(self, seq_len, n_embd, k, use_hankel_L=False, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.k = k

        self.Q_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters.transpose(0, 1)
        self.K_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters.transpose(0, 1)
        self.V_filt = LearnableSpectralFilters(seq_len, k, use_hankel_L, device).filters.transpose(0, 1)

        self.i_proj = nn.Linear(n_embd, self.Q_filt.shape[0])
        self.o_proj = nn.Linear(self.Q_filt.shape[0], n_embd)

    def forward(self, x, L):
        bsz, T, dim = x.shape
        x_proj = self.i_proj(x)

        Q = torch.einsum("bth,hk->btk", x_proj, self.Q_filt)
        K = torch.einsum("bth,hk->btk", x_proj, self.K_filt)
        V = torch.einsum("bth,hk->btk", x_proj, self.V_filt)

        Z = torch.einsum("bsi, bsj -> bsij", V, K)
        H = torch.einsum("bts, bsij-> btij", L, Z)
        Y = torch.einsum("bti, btij -> btj", Q, H)

        Y = self.o_proj(Y)
        return Y


class SpectralAttentionLayer(nn.Module):
    def __init__(self, seq_len, n_embd, k, dropout=0.1, use_hankel_L=False, device=None):
        super().__init__()
        self.attn_norm = nn.RMSNorm(n_embd)
        self.mlp_norm = nn.RMSNorm(n_embd)
        self.attn = SpectralAttention(seq_len, n_embd, k, use_hankel_L, device)
        self.mlp = MLP(n_embd)

    def forward(self, x, L):
        x = x + self.attn(self.attn_norm(x), L)
        # x = x + self.mlp(self.mlp_norm(x))
        return x


class SpectralTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                SpectralAttentionLayer(config.seq_len, config.n_embd, config.k, config.dropout, device=config.device)
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        bsz, seq_len = x.size()

        # Construct lower-triangular mask L: (B,T,S)
        L = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        L = L.unsqueeze(0).expand(bsz, -1, -1)  # (B,T,S)

        x = self.dropout(self.tok_emb(x))  # (B,T,embed)
        for layer in self.layers:
            x = layer(x, L)
        x = self.norm(x)
        return self.output(x)


def test_causality_leakage(
    model,
    vocab_size=64,
    seq_len=20,
    num_tests=5,
    device="cpu",
    early_timesteps_to_check=(0, 1, 2),
    modify_future_tokens_count=3,
):
    model.eval()
    total_diff = 0.0
    comparisons = 0

    for test_idx in range(num_tests):
        # new seed each time
        seed = 1746 + test_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # create a batch of sequences
        B = 4  # multiple sequences per test
        x = torch.randint(0, vocab_size, (B, seq_len), device=device)

        with torch.no_grad():
            out_original = model(x)  # (B, T, vocab_size)

        # modify several future tokens
        x_modified = x.clone()
        # choose some future positions to drastically alter
        # we'll pick positions in the second half of the sequence
        future_positions = range(seq_len // 2, seq_len)
        chosen_future_tokens = random.sample(
            list(future_positions), min(modify_future_tokens_count, len(future_positions))
        )
        for pos in chosen_future_tokens:
            # add a big random value
            x_modified[:, pos] = (
                x_modified[:, pos] + torch.randint(low=vocab_size // 2, high=vocab_size, size=(B,), device=device)
            ) % vocab_size

        with torch.no_grad():
            out_modified = model(x_modified)

        # measure difference at multiple early timesteps
        for t in early_timesteps_to_check:
            diff = torch.norm(out_original[:, t, :] - out_modified[:, t, :], p=2, dim=-1)  # (B)
            total_diff += diff.sum().item()
            comparisons += B

    avg_diff = total_diff / comparisons if comparisons > 0 else 0.0
    print(f"Average difference across {num_tests} tests and multiple sequences: {avg_diff:.6f}")
    if avg_diff > 1e-9:
        print("Causality leakage suspected! Future token changes influenced earlier outputs.")
    else:
        print("No significant leakage detected.")


# Example usage:
if __name__ == "__main__":

    class SimpleConfig:
        def __init__(self):
            self.num_layers = 2
            self.n_heads = 4
            self.n_embd = 16
            self.k = 4
            self.use_hankel_L = False
            self.dropout = 0.0
            self.batch_size = 2
            self.lr = 3e-4
            self.num_epochs = 1
            self.device = torch.device("cpu")
            self.seed = 1746
            self.torch_compile = False
            self.vocab_size = 64
            self.seq_len = 10

    config = SimpleConfig()
    model = SpectralTransformer(config).to(config.device)

    test_causality_leakage(
        model,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_tests=10,
        device=config.device,
        early_timesteps_to_check=(0, 1, 2),
        modify_future_tokens_count=3,
    )
