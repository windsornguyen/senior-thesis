import torch
import math
import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
from torchaudio.functional import fftconvolve

# Helper functions
def get_opt_degree(seq_len: int) -> int:
    return int(math.ceil((7.0 / 6.0) * math.log2(seq_len)))

def get_cheby_coeffs(seq_len: int, *, device=None, dtype=torch.float32):
    n = get_opt_degree(seq_len)
    Tn = chebyshev.Chebyshev.basis(n).convert(kind=chebyshev.Polynomial)
    coef = torch.tensor(Tn.coef, device=device, dtype=dtype)
    coef = coef.flip(dims=[0]).contiguous()
    return coef / (2 ** (n - 1))

def cheby_conv(coeffs: torch.Tensor, inputs: np.ndarray) -> np.ndarray:
    # Simple numpy convolution for analysis
    return np.convolve(inputs, coeffs.cpu().numpy(), mode='full')[:len(inputs)]

# Settings
L = 64

# Compute coefficients
coeffs = get_cheby_coeffs(L, dtype=torch.float32, device='cpu').cpu().numpy()
n = len(coeffs)
t = np.arange(L)

# 1) Plot Chebyshev coefficients
plt.figure()
plt.plot(np.arange(n), coeffs)
plt.title('Chebyshev Filter Coefficients (Degree {})'.format(n-1))
plt.xlabel('Coefficient Index')
plt.ylabel('Value')
plt.show()

# 2) Impulse response
impulse = np.zeros(L)
impulse[0] = 1.0
imp_response = cheby_conv(torch.tensor(coeffs), impulse)
plt.figure()
plt.stem(np.arange(len(imp_response)), imp_response, use_line_collection=True)
plt.title('Impulse Response (should match coefficients)')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.show()

# 3) Step response
step = np.ones(L)
step_response = cheby_conv(torch.tensor(coeffs), step)
plt.figure()
plt.plot(t, step_response)
plt.title('Step Response')
plt.xlabel('n')
plt.ylabel('Step Output')
plt.show()

# 4) Frequency response (magnitude)
freq_resp = np.fft.rfft(coeffs, n=512)
freq = np.fft.rfftfreq(512, d=1)
plt.figure()
plt.plot(freq, np.abs(freq_resp))
plt.title('Frequency Response Magnitude')
plt.xlabel('Normalized Frequency')
plt.ylabel('|H(f)|')
plt.show()

# 5) Sinusoidal input responses at different frequencies
for f in [0.05, 0.15, 0.3]:
    sine = np.sin(2 * np.pi * f * t)
    out = cheby_conv(torch.tensor(coeffs), sine)
    plt.figure()
    plt.plot(t, out)
    plt.title(f'Sine Response (f = {f:.2f})')
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.show()
