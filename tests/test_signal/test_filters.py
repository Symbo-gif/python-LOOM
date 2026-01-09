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

"""
Tests for digital filters and windows.
"""

import pytest
import math
import loom as tf
import loom.signal as signal

def test_lfilter_moving_average():
    # Simple 3-point moving average
    # y[n] = 1/3 * (x[n] + x[n-1] + x[n-2])
    b = [1/3, 1/3, 1/3]
    a = [1.0]
    x = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    y = signal.lfilter(b, a, x)
    y_list = y.tolist()
    
    assert math.isclose(y_list[0], 1/3, rel_tol=1e-4)
    assert math.isclose(y_list[1], 2/3, rel_tol=1e-4)
    assert math.isclose(y_list[2], 1.0, rel_tol=1e-4)
    assert math.isclose(y_list[3], 1.0)
    assert math.isclose(y_list[4], 1.0)

def test_lfilter_recursive():
    # Exponential smoothing: y[n] = 0.5*x[n] + 0.5*y[n-1]
    # a[0]*y[n] = b[0]*x[n] - a[1]*y[n-1]
    # 1.0*y[n] = 0.5*x[n] + 0.5*y[n-1]  => a=[1, -0.5], b=[0.5]
    b = [0.5]
    a = [1.0, -0.5]
    x = [1.0, 0.0, 0.0]
    
    y = signal.lfilter(b, a, x)
    y_list = y.tolist()
    
    # n=0: y[0] = 0.5*1.0 + 0.5*0 = 0.5
    # n=1: y[1] = 0.5*0 + 0.5*0.5 = 0.25
    # n=2: y[2] = 0.5*0 + 0.5*0.25 = 0.125
    assert y_list == [0.5, 0.25, 0.125]

def test_hamming_window():
    w = signal.hamming(5)
    w_list = w.tolist()
    # Symmetry check
    assert math.isclose(w_list[0], w_list[4])
    assert math.isclose(w_list[1], w_list[3])
    # Peak at middle for odd N
    assert w_list[2] == 1.0

def test_hanning_window():
    w = signal.hanning(5)
    w_list = w.tolist()
    # Ends should be 0 (usually)
    assert math.isclose(w_list[0], 0.0, abs_tol=1e-15)
    assert math.isclose(w_list[4], 0.0, abs_tol=1e-15)
    assert w_list[2] == 1.0

