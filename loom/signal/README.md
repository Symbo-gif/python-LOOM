# loom Signal Processing Module

**FFT, convolution, digital filtering, and windowing**

## Status: ✅ Production-ready (v1.1.0)

_Validated by FFT, convolution, windowing, and filter tests in the v1.1.0 release suite._

## Features

### Fourier Transform
- `fft`: 1D Fast Fourier Transform (Cooley-Tukey algorithm for 2^n lengths).
- `ifft`: Inverse FFT.
- Automatic fallback to naive DFT for non-power-of-2 lengths.

### Convolution
- `convolve`: 1D convolution with 'full', 'same', and 'valid' modes.
- `convolve2d`: 2D convolution for image/grid data.

### Digital Filters (Phase 8)
- `lfilter(b, a, x)`: Apply a linear filter to data (direct form difference equation).
- `filtfilt(b, a, x)`: Zero-phase filtering (forward-backward application).

### Window Functions (Phase 8)
- `boxcar(n)`: Rectangular window.
- `hamming(n)`: Hamming window.
- `hanning(n)`: Hanning (von Hann) window.
- `blackman(n)`: Blackman window.
- `bartlett(n)`: Bartlett (triangular) window.

## Usage Example

```python
from loom.signal import fft, convolve, lfilter, hamming

# FFT
data = [1, 0, 1, 0]
freq = fft(data)

# 1D Convolution
a = [1, 2, 3]
v = [1, 1]
c = convolve(a, v, mode='same')

# Digital filter (exponential smoothing)
b = [0.5]
a = [1.0, -0.5]
y = lfilter(b, a, [1.0, 0.0, 0.0])  # [0.5, 0.25, 0.125]

# Windowing
w = hamming(64)  # 64-point Hamming window
```
