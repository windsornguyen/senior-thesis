import torch
import torch.nn as nn

from thesis.pretraining.utils import nearest_power_of_two
from thesis.utils import get_spectral_filters, get_tensorized_spectral_filters
from thesis.utils.conv import convolve, flash_convolve
from flashfftconv import FlashFFTConv

class STU(nn.Module):
    def __init__(self, config) -> None:
        super(STU, self).__init__()
        self.config = config
        phi = get_tensorized_spectral_filters() if config.use_tensorized_filters else get_spectral_filters()
        self.filters = nn.Parameter(phi)

        # Task dependent
        self.d_in = config.dim
        self.d_out = config.dim

        self.seq_len = nearest_power_of_two(config.seq_len, round_up=True)
        self.flashfftconv = FlashFFTConv(self.seq_len, dtype=torch.bfloat16)
        self.use_tensordot = config.use_tensordot

        if self.use_tensordot:
            self.M_inputs = nn.Parameter(torch.empty(self.d_in, self.d_out, dtype=config.torch_dtype))
            self.M_filters = nn.Parameter(torch.empty(config.num_eigh, self.d_in, dtype=config.torch_dtype))
        else:
            self.M_phi_plus = nn.Parameter(torch.empty(config.num_eigh, self.d_in, self.d_out, dtype=config.torch_dtype))
            self.M_phi_minus = nn.Parameter(torch.empty(config.num_eigh, self.d_in, self.d_out, dtype=config.torch_dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.use_tensordot:
            # Contract inputs and filters over the K and d_in dimensions.
            u_proj = u @ self.M_inputs
            phi_proj = self.phi @ self.M_filters

            # Then, convolve.
            spectral_plus, spectral_minus = flash_convolve(u_proj, phi_proj, self.flashfftconv, self.use_tensordot)
        else:
            # Convolve first.
            U_plus, U_minus = flash_convolve(u, self.filters, self.flashfftconv, self.use_tensordot)

            # Then, contract over the K and d_in dimensions.
            spectral_plus = torch.einsum("blko,oki->blo", U_plus, self.M_phi_plus)
            spectral_minus = torch.einsum("blko,oki->blo", U_minus, self.M_phi_minus)

        y = spectral_plus + spectral_minus
        return y
