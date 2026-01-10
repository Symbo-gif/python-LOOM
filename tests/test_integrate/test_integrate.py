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
from loom.integrate import trapezoid, simpson, solve_ivp
import math

def test_quadrature():
    # Integral of x^2 from 0 to 1 is 1/3
    x = [i * 0.1 for i in range(11)]
    y = [xi**2 for xi in x]
    
    res_trap = trapezoid(y, x)
    # Trapezoid is slightly over-estimating for convex x^2
    # Use .item() to convert Tensor to float for comparison
    assert abs(res_trap.item() - 1/3) < 0.01
    
    res_simp = simpson(y, dx=0.1)
    # Simpson is exact for polynomials up to degree 3
    assert abs(res_simp.item() - 1/3) < 1e-12

def test_ode_linear():
    # dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
    def f(t, y):
        return -y
        
    sol = solve_ivp(f, (0, 5), [1.0], method='RK45')
    assert sol.success
    # Check final point
    assert abs(sol.y[-1][0] - math.exp(-5)) < 1e-4

def test_ode_oscillator():
    # Harmonic oscillator: y'' + y = 0
    # y1' = y2
    # y2' = -y1
    def f(t, y):
        y1, y2 = y
        return [y2, -y1]
        
    y0 = [1.0, 0.0] # starts at (1,0)
    sol = solve_ivp(f, (0, 2*math.pi), y0, method='RK45')
    
    # After one period, should be back at around [1, 0]
    last_y = sol.y[-1]
    assert abs(last_y[0] - 1.0) < 5e-3
    assert abs(last_y[1] - 0.0) < 5e-3

def test_ode_stress():
    # Test high frequency or long term stability
    def f(t, y):
        return [math.cos(t)]
        
    sol = solve_ivp(f, (0, 100), [0.0], method='RK45')
    assert sol.success
    # Integral of cos(t) is sin(t)
    # sin(100)
    assert abs(sol.y[-1][0] - math.sin(100)) < 1e-2

