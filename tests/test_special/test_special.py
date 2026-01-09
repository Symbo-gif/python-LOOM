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
from loom.special import gamma, loggamma, beta, erf, erfc
import math

def test_gamma():
    # Known values
    assert abs(gamma(1) - 1.0) < 1e-12
    assert abs(gamma(2) - 1.0) < 1e-12
    assert abs(gamma(3) - 2.0) < 1e-12
    assert abs(gamma(4) - 6.0) < 1e-12
    assert abs(gamma(0.5) - math.sqrt(math.pi)) < 1e-12

def test_loggamma():
    assert abs(loggamma(5) - math.log(24)) < 1e-12

def test_beta():
    # B(2, 2) = G(2)*G(2)/G(4) = 1*1/6 = 1/6
    assert abs(beta(2, 2) - 1/6) < 1e-12

def test_erf():
    assert abs(erf(0) - 0.0) < 1e-12
    # erf(1) ~ 0.8427
    assert abs(erf(1) - 0.84270079) < 1e-6
    # erf(-inf) = -1
    assert abs(erf(10) - 1.0) < 1e-12
    assert abs(erf(-10) + 1.0) < 1e-12

def test_erfc():
    assert abs(erfc(0) - 1.0) < 1e-12
    assert abs(erfc(1) - (1 - erf(1))) < 1e-12

def test_gamma_stress():
    # Large values
    val = gamma(10)
    assert abs(val - math.factorial(9)) < 1e-5
    
    # Negative non-integers
    val_neg = gamma(-0.5)
    # G(0.5) = G(-0.5+1) = (-0.5)*G(-0.5) => G(-0.5) = G(0.5)/(-0.5) = -2*sqrt(pi)
    assert abs(val_neg + 2*math.sqrt(math.pi)) < 1e-12

