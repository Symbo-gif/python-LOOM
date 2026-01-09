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
from loom.optimize import bisect, newton, brentq, minimize
import math

def test_root_finding_scalar():
    # f(x) = x^2 - 2 (root at sqrt(2))
    def f(x): return x**2 - 2
    
    res_bisect = bisect(f, 0, 2)
    assert abs(res_bisect - math.sqrt(2)) < 1e-6
    
    res_brent = brentq(f, 0, 2)
    assert abs(res_brent - math.sqrt(2)) < 1e-7
    
    res_newton = newton(f, 1.5)
    assert abs(res_newton - math.sqrt(2)) < 1e-7

def test_minimize_scalar():
    # f(x) = (x-3)^2 + 4 (min at x=3)
    def f(x): return (x.item() - 3)**2 + 4
    
    res = minimize(f, [0.0], method='BFGS')
    assert abs(res.x.item() - 3.0) < 1e-3
    
    res_nm = minimize(f, [0.0], method='Nelder-Mead')
    assert abs(res_nm.x.item() - 3.0) < 1e-3

def test_minimize_rosenbrock():
    def rosen(x):
        x = x.tolist()
        return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
        
    res = minimize(rosen, [0.5, 0.5], method='BFGS', tol=1e-6)
    assert abs(res.x.tolist()[0] - 1.0) < 5e-3
    assert abs(res.x.tolist()[1] - 1.0) < 5e-3
    
    res_nm = minimize(rosen, [0.5, 0.5], method='Nelder-Mead', tol=1e-6)
    assert abs(res_nm.x.tolist()[0] - 1.0) < 5e-3
    assert abs(res_nm.x.tolist()[1] - 1.0) < 5e-3

def test_minimize_stress():
    # Sphere function in 5D
    def sphere(x):
        return (tf.array(x)**2).sum().item()
        
    res = minimize(sphere, [1.0, 2.0, 3.0, 4.0, 5.0], method='BFGS')
    assert res.success
    assert (res.x**2).sum().item() < 1e-5

