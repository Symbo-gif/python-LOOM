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

import pytest
import loom as tf
from loom.interpolate import interp1d, CubicSpline
import math

def test_linear_interpolation():
    x = [0, 1, 2]
    y = [0, 1, 4] # y = x^2 at nodes
    
    f = interp1d(x, y, kind='linear')
    
    # Midpoint test
    assert f(0.5) == 0.5
    assert f(1.5) == 2.5
    
    # Boundary test
    assert f(-1) == 0
    assert f(3) == 4
    
    # Tensor input
    x_new = tf.array([0.5, 1.5])
    y_new = f(x_new)
    assert abs(y_new.tolist()[0] - 0.5) < 1e-7
    assert abs(y_new.tolist()[1] - 2.5) < 1e-7

def test_cubic_spline():
    x = [0, 1, 2, 3]
    y = [0, 1, 0, 1] # oscillation
    
    f = CubicSpline(x, y)
    
    # Check nodes
    assert abs(f(0) - 0) < 1e-12
    assert abs(f(1) - 1) < 1e-12
    assert abs(f(2) - 0) < 1e-12
    
    # Check midpoint smoothness
    # x=0.5 should be between 0 and 1
    val = f(0.5)
    assert 0 < val < 1
    
    # Natural spline condition: second derivative at ends is zero
    # This is internal, but we check if it produces reasonable output
    assert abs(f(1.5) - 0.5) < 0.2 # Roughly in middle

def test_cubic_spline_stress():
    # Large number of points
    x = [float(i) for i in range(100)]
    y = [math.sin(xi) for xi in x]
    
    f = CubicSpline(x, y)
    
    # Check many points
    for i in range(len(x)-1):
        mid = x[i] + 0.5
        val = f(mid)
        # Should be close to sin(mid)
        assert abs(val - math.sin(mid)) < 0.1

