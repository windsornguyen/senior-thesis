import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

###############################################
# Dataset Generation (LDS and Adding Problem)
###############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def generate_lds(
    num_examples: int = 10,
    sequence_len: int = 512,
    num_regimes: int = 1,
    state_size: int = 3,
    input_size: int = 2,
    output_size: int = 2,
    noise_level: float = 0.1,
    obs_noise: float = 0.0,
    stability_factor: float = 0.95,
    min_duration: int = 100,
    randomness_factor: float = 0.25,
    symmetric: bool = False,
    seed: int = 1_337,
) -> TensorDataset:
    torch.manual_seed(seed)

    def generate_regime_matrices():
        if symmetric:
            Q, _ = torch.linalg.qr(torch.randn(state_size, state_size))
            eigenvalues = (torch.rand(state_size) * 2 - 1) * stability_factor
            D = torch.diag(eigenvalues)
            A = Q @ D @ Q.T
        else:
            A = torch.randn(state_size, state_size)
            spec_norm = torch.linalg.norm(A, ord=2)
            A = (A / spec_norm) * stability_factor
        B = torch.randn(state_size, input_size) / math.sqrt(state_size)
        C = torch.randn(output_size, state_size) / math.sqrt(state_size)
        D = torch.randn(output_size, input_size) / math.sqrt(input_size)
        return A, B, C, D

    if num_regimes == 1:
        A_fixed, B_fixed, C_fixed, D_fixed = generate_regime_matrices()
        A_mats = A_fixed.unsqueeze(0)
        B_mats = B_fixed.unsqueeze(0)
        C_mats = C_fixed.unsqueeze(0)
        D_mats = D_fixed.unsqueeze(0)
        schedules = torch.zeros(num_examples, sequence_len, dtype=torch.long)
    else:
        A_list, B_list, C_list, D_list = [], [], [], []
        for _ in range(num_regimes):
            A_, B_, C_, D_ = generate_regime_matrices()
            A_list.append(A_)
            B_list.append(B_)
            C_list.append(C_)
            D_list.append(D_)
        A_mats = torch.stack(A_list, dim=0)
        B_mats = torch.stack(B_list, dim=0)
        C_mats = torch.stack(C_list, dim=0)
        D_mats = torch.stack(D_list, dim=0)

        def generate_regime_schedule():
            ideal_segment = sequence_len / num_regimes
            regime_changes = []
            for r in range(1, num_regimes):
                min_valid = r * min_duration
                max_valid = sequence_len - (num_regimes - r) * min_duration
                offset_range = int(randomness_factor * ideal_segment)
                random_offset = torch.randint(-offset_range, offset_range + 1, (1,)).item()
                switch_point = int(r * ideal_segment + random_offset)
                switch_point = max(min_valid, min(switch_point, max_valid))
                regime_changes.append(switch_point)
            schedule = torch.zeros(sequence_len, dtype=torch.long)
            current_regime = 0
            for t in range(sequence_len):
                if regime_changes and t >= regime_changes[0]:
                    current_regime += 1
                    regime_changes.pop(0)
                schedule[t] = current_regime
            return schedule

        schedules = torch.stack([generate_regime_schedule() for _ in range(num_examples)])

    u = torch.randn(num_examples, sequence_len, input_size)
    x = torch.zeros(num_examples, state_size)
    controls = torch.zeros(num_examples, sequence_len, input_size)
    observations = torch.zeros(num_examples, sequence_len, output_size)
    for t in range(sequence_len):
        controls[:, t] = u[:, t]
        regime_idx = schedules[:, t]
        A_batch = A_mats[regime_idx]
        B_batch = B_mats[regime_idx]
        C_batch = C_mats[regime_idx]
        D_batch = D_mats[regime_idx]
        y_t = torch.bmm(C_batch, x.unsqueeze(2)).squeeze(2) + torch.bmm(D_batch, u[:, t].unsqueeze(2)).squeeze(2)
        if obs_noise > 0:
            y_t += torch.randn_like(y_t) * obs_noise
        observations[:, t] = y_t
        x_next = torch.bmm(A_batch, x.unsqueeze(2)).squeeze(2) + torch.bmm(B_batch, u[:, t].unsqueeze(2)).squeeze(2)
        if noise_level > 0:
            x_next += torch.randn_like(x_next) * noise_level
        x = x_next
    return TensorDataset(controls, observations)


def generate_adding_problem(num_examples: int = 1000, sequence_len: int = 50) -> TensorDataset:
    numbers = torch.rand(num_examples, sequence_len, 1)
    indicators = torch.zeros(num_examples, sequence_len, 1)
    for i in range(num_examples):
        idx = torch.randperm(sequence_len)[:2]
        indicators[i, idx, 0] = 1.0
    inputs = torch.cat([numbers, indicators], dim=-1)
    target_sum = (numbers * indicators).sum(dim=1, keepdim=True)
    targets = target_sum.expand(-1, sequence_len, -1)
    return TensorDataset(inputs, targets)


###############################################
# Spectral (Spectron) Attention Components
###############################################


# We use a simple Hankel-based spectral filter to build a tensorized (Kronecker) filter.
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
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    epsilon = 1e-9
    sigma_k = sigma_k.clamp_min(epsilon)
    phi_k *= sigma_k**0.25
    return phi_k.to(device=device, dtype=dtype)


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
    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)
    Z = get_hankel(sqrt_T_prime)
    sigma, phi = torch.linalg.eigh(Z)
    phi_i = phi[:, -k:] * sigma[-k:] ** 0.25
    if use_hankel_L:
        Z_L = get_hankel(sqrt_T_prime, True)
        sigma_L, phi_L = torch.linalg.eigh(Z_L)
        phi_j = phi_L[:, -k:] * sigma_L[-k:] ** 0.25
    else:
        phi_j = phi_i
    filters = torch.kron(phi_i, phi_j)
    return filters.to(device=device, dtype=dtype)


def poly_mul_x(poly):
    # Multiply polynomial by x: shift coefficients right by one index.
    return [0] + poly


def poly_scale(poly, factor):
    # Scale polynomial coefficients by factor.
    return [coef * factor for coef in poly]


def poly_sub(poly1, poly2):
    # Subtract poly2 from poly1; extend with zeros if necessary.
    length = max(len(poly1), len(poly2))
    result = []
    for i in range(length):
        coef1 = poly1[i] if i < len(poly1) else 0
        coef2 = poly2[i] if i < len(poly2) else 0
        result.append(coef1 - coef2)
    return result


def chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x)
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    T_nm2 = [1]  # T_0(x)
    T_nm1 = [0, 1]  # T_1(x)
    for _ in range(2, n + 1):
        # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        term = poly_mul_x(T_nm1)
        term = poly_scale(term, 2)
        T_n = poly_sub(term, T_nm2)
        T_nm2, T_nm1 = T_nm1, T_n
    return T_n


def normalized_chebyshev_coeff(n):
    # Returns the coefficients of the nth Chebyshev polynomial T_n(x) normalized by 2**(n-1).
    # Coefficients are in ascending order: [a0, a1, ..., an] represents a0 + a1*x + ... + an*x^n.
    coeff = chebyshev_coeff(n)
    leading_term = coeff[-1]
    return [c / leading_term for c in coeff]


def integrate_polar_monomial(a, b, beta):
    """
    Compute the integral of z^a * z̄^b over the polar wedge:
      r ∈ [0, 1], θ ∈ [-beta, beta],
    in closed form:
      if a==b: 2*beta/(a+b+2)
      else:   2*sin((a-b)*beta)/((a-b)*(a+b+2))
    Here a and b are tensors (floats).
    """
    diff = a - b
    denom = a + b + 2
    return torch.where(
        condition=diff == 0,
        input=2 * beta / denom,
        other=2 * torch.sin(diff * beta) / (diff * denom),
    )


def get_polynomial_hankel(
    n, beta, t, chunk_size=2048, device="cuda", dtype=torch.bfloat16
):
    """ """
    matrix_size = t - n

    # Compute Chebyshev coefficients
    poly_coeff = normalized_chebyshev_coeff(n)
    poly = torch.tensor(poly_coeff, device=device)  # (n+1,)

    # Outer product of polynomial coefficients
    P = torch.outer(poly, poly).unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

    # Precompute the index arrays for the summation indices (ii, jj)
    ii = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    jj = torch.arange(0, n + 1, device=device, dtype=torch.float32)
    ii, jj = torch.meshgrid(ii, jj, indexing="ij")  # (n+1, n+1)
    ii = ii.unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)
    jj = jj.unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

    # Allocate the result matrix
    Z = torch.empty((matrix_size, matrix_size), dtype=torch.complex64, device=device)

    # Process in chunks to save memory.
    for i_start in range(0, matrix_size, chunk_size):
        # Create i indices
        i_end = min(i_start + chunk_size, matrix_size)
        i_vals = torch.arange(
            i_start, i_end, device=device, dtype=torch.float32
        ).view(-1, 1, 1, 1)

        for j_start in range(0, matrix_size, chunk_size):
            # Create j indices
            j_end = min(j_start + chunk_size, matrix_size)
            j_vals = torch.arange(
                j_start, j_end, device=device, dtype=torch.float32
            ).view(1, -1, 1, 1)

            # Compute A and B for the chunk: shape (chunk_i, chunk_j, n+1, n+1)
            A = i_vals + ii  # Compute i + ii for each chunk element
            B = j_vals + jj

            # Compute the closed-form integral for each (i+ii, j+jj)
            int_vals = integrate_polar_monomial(A, B, beta)

            # Multiply by P and sum over the polynomial indices to yield the (i,j) entry
            chunk_Z = torch.sum(int_vals * P, dim=(2, 3))

            # Write the computed chunk to the result matrix.
            Z[i_start:i_end, j_start:j_end] = chunk_Z.to(torch.complex64)

    return Z


def get_opt_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7 / 6) * math.log2(seq_len)))

def get_polynomial_spectral_filters(
    seq_len: int,
    k: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    n = get_opt_degree(seq_len)
    beta = 1.0 / (64.0 * n**2)
    Z = get_polynomial_hankel(n, beta, seq_len + n, device=device)
    _, phi = torch.linalg.eigh(Z, UPLO="U")
    phi_k = phi[:, -k:] / math.sqrt(seq_len)

    # Validate that the eigenvectors are real since Z is Hermitian
    if torch.abs(phi_k.imag).max() > 1e-7:
        raise ValueError("Unexpectedly large imaginary components in eigenvectors")

    # Take real part only (imaginary part is due to floating point imprecision)
    return phi_k.real.to(dtype)


class LearnableSpectralFilters(nn.Module):
    def __init__(self, seq_len: int, k: int, use_hankel_L: bool = False, device=None, dtype=torch.float32):
        super().__init__()
        filters = get_spectral_filters(seq_len, k, use_hankel_L, device, dtype)
        if filters.shape[0] < seq_len:
            pad_size = seq_len - filters.shape[0]
            filters = F.pad(filters, (0, pad_size, 0, pad_size), mode="constant", value=0)
        elif filters.shape[0] > seq_len:
            filters = filters[:seq_len, :seq_len]
        # Convert filters to a proper nn.Parameter so it moves with the model
        self.filters = nn.Parameter(filters)

    def forward(self):
        return self.filters


class SpectralAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, k: int, use_hankel_L: bool = False, device=None):
        super().__init__()
        self.seq_len = seq_len
        # Pre-project x to shape (B, T, seq_len) if needed.
        self.pre_proj = nn.Linear(d_model, seq_len) if d_model != seq_len else nn.Identity()
        # Q projection: maps pre-projected input to k dimensions.
        self.q_proj = nn.Linear(seq_len, k).to(device)
        # Learnable spectral filters for K and V.
        self.K = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        self.V = LearnableSpectralFilters(seq_len, k, use_hankel_L, device)
        # Final projection from k back to d_model.
        self.o_proj = nn.Linear(k, d_model).to(device)
        # Decay parameter: one per time step.
        self.decay = nn.Parameter(torch.ones(seq_len, device=device))
        # Hankel matrix L (shape: [T, T]); we'll mask it to be lower-triangular.
        self.L = nn.Parameter(get_hankel(seq_len, use_hankel_L, device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_proj = self.pre_proj(x)  # (B, T, seq_len)

        # Compute Q: (B, T, k)
        Q = self.q_proj(x_proj)
        
        # Compute K and V: (B, T, k)
        K = torch.einsum("bti,ik->btk", x_proj, self.K())
        V = torch.einsum("bti,ik->btk", x_proj, self.V())
        
        # Compute Z as the outer product between V and K per time step,
        # scaled by the decay factor. In our notation:
        #   V: (B, T, k) with indices "btp"
        #   K: (B, T, k) with indices "btn"
        #   decay: (T,) with index "t"
        # Result Z: (B, T, k, k)
        Z = torch.einsum("btp,btn,t->btpn", V, K, self.decay)
        
        # Prepare the Hankel mask: force L to be lower-triangular and reshape to (1, T, T)
        # so it broadcasts over batch.
        L_masked = torch.tril(self.L).unsqueeze(0)  # (1, T, T)
        
        # Aggregate Z over time using L_masked.
        # Here we interpret L_masked as weighting contributions from all time steps s to output time t.
        # That is, for each b, t, p, n:
        #    H[b,t,p,n] = sum_s L_masked[0,t,s] * Z[b,s,p,n]
        H = torch.einsum("bts,bspn->btpn", L_masked, Z)
        
        # Query the aggregated result with Q.
        # Q: (B, T, k), H: (B, T, k, k) → Y: (B, T, k)
        Y = torch.einsum("btk,btkn->btn", Q, H)
        
        return self.o_proj(Y)


class SpectralAttnModel(nn.Module):
    """
    Spectral attention model.
    For LDS: input and output dims are 1.
    For the adding problem, input dim is 2 and output is 1.

    The main model projects the input to a fixed d_model space,
    applies spectral attention, adds a residual connection,
    and finally projects to the output dimension.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        k: int,
        d_in: int,
        d_out: int,
        use_hankel_L: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.spec_attn = SpectralAttention(seq_len, d_model, k, use_hankel_L, device)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        x_proj = self.in_proj(x)  # (B, T, d_model)
        out = x_proj + self.spec_attn(x_proj)
        return self.out_proj(out)  # (B, T, d_out)


###############################################
# Training Loops for LDS and Adding Problem using SpectralAttnModel
###############################################


def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask.bool(), float("-inf"))


def train_model_spec(model, loader, epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for controls, targets in loader:
            controls = controls.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(controls)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
    return losses


# --- Train on LDS Dataset ---
print("Training SpectralAttnModel on LDS dataset...")
SEQ_LEN_LDS = 512
D_MODEL_LDS = 8
K_LDS = 4
spectral_model_lds = SpectralAttnModel(seq_len=SEQ_LEN_LDS, d_model=D_MODEL_LDS, k=K_LDS, d_in=1, d_out=1).to(device)
losses_spec_lds = train_model_spec(
    spectral_model_lds,
    DataLoader(
        generate_lds(
            num_examples=64,
            sequence_len=SEQ_LEN_LDS,
            num_regimes=5,
            state_size=16,
            input_size=1,
            output_size=1,
            noise_level=0.1,
            obs_noise=0.1,
            stability_factor=0.95,
            seed=1746,
            symmetric=False,
            min_duration=2,
        ),
        batch_size=8,
        shuffle=True,
    ),
    epochs=100,
)

# --- Train on Adding Problem ---
print("\nTraining SpectralAttnModel on Adding Problem...")
SEQ_LEN_ADD = 50
D_MODEL_ADD = 16
K_ADD = 7
spectral_model_add = SpectralAttnModel(seq_len=SEQ_LEN_ADD, d_model=D_MODEL_ADD, k=K_ADD, d_in=2, d_out=1).to(device)
losses_spec_add = train_model_spec(
    spectral_model_add,
    DataLoader(generate_adding_problem(num_examples=1000, sequence_len=SEQ_LEN_ADD), batch_size=64, shuffle=True),
    epochs=100,
)

# Plot loss curves for the Adding Problem experiment:
plt.figure(figsize=(8, 5))
plt.plot(losses_spec_add, label="SpectralAttnModel (Adding Problem)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Spectral Attention Model Training Loss (Adding Problem)")
plt.legend()
plt.show()
