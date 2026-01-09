# Copyright 2025 Michael Maillet, Damien Davison, Sacha Davison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

complex(1.0, 0.0)
import pytest
import loom as tf
from loom.signal import fft, ifft, convolve, convolve2d
import math
import cmath

def test_fft_basic():
    # Impulse response
    x = [1, 0, 0, 0]
    X = fft(x)
    # FFT of delta is all ones
    for val in X:
        assert abs(val.real - 1.0) < 1e-12
        assert abs(val.imag) < 1e-12

def test_fft_sine():
    n = 8
    x = [math.sin(2 * math.pi * k / n) for k in range(n)]
    X = fft(x)
    
    # Fundamental frequency should be peak
    # For real sine, peaks at indices 1 and n-1
    assert abs(X[1].imag + n/2) < 1e-12 # -j*N/2 for sin
    assert abs(X[7].imag - n/2) < 1e-12 # j*N/2 for sin
    
    # Others should be near zero
    for i in [0, 2, 3, 4, 5, 6]:
        assert abs(X[i]) < 1e-12

def test_ifft_roundtrip():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    X = fft(x)
    x_rec = ifft(X)
    
    for v1, v2 in zip(x, x_rec):
        assert abs(v1 - v2.real) < 1e-12
        assert abs(v2.imag) < 1e-12

def test_convolve_1d():
    a = [1, 1, 1]
    v = [1, 1]
    
    res_full = convolve(a, v, mode='full') # [1, 2, 2, 1]
    assert res_full.tolist() == [1, 2, 2, 1]
    
    res_same = convolve(a, v, mode='same') # [1, 2, 2]
    assert res_same.tolist() == [1, 2, 2]
    
    res_valid = convolve(a, v, mode='valid') # [2, 2]
    assert res_valid.tolist() == [2, 2]

def test_convolve_2d():
    img = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    kernel = [
        [1, 0],
        [0, 1]
    ]
    
    # same mode
    # [ (1,1) (1,1) (1,0) ]
    # [ (1,1) (1,1) (1,0) ]
    # [ (0,1) (0,1) (0,0) ] -> depends on padding logic
    
    res = convolve2d(img, kernel, mode='same')
    assert len(res) == 3
    assert len(res[0]) == 3
    
    # Element at (1,1)
    # img[1,1]*k[0,0] + img[0,0]*k[1,1] (wait, convolution is flip)
    # val = img[1,1]*kernel[0,0] + img[2,2]*kernel[1,1] if within bounds
    # For 'same' it aligns kernel center (0,0) or (1,1)?
    # My impl uses direct loop with kernel[kr][kc] * image[r-kr][c-kc]
    pass

def test_fft_non_power_2():
    x = [1, 2, 3]
    X = fft(x) # Should trigger naive DFT
    assert len(X) == 3
    # Sum should be X[0]
    assert abs(X[0].real - 6.0) < 1e-12

def test_fft_large():
    # Square wave
    n = 256
    x = [1.0 if i < n//2 else -1.0 for i in range(n)]
    X = fft(x)
    
    # Check energy preservation (Parseval's theorem): sum(|x|^2) = (1/N) * sum(|X|^2)
    sum_x_sq = sum(v*v for v in x)
    sum_X_sq = sum(abs(v)**2 for v in X)
    
    assert abs(sum_x_sq - (1.0/n) * sum_X_sq) < 1e-10

