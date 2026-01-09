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
Tests for the stats module.
"""

import pytest
import math
import loom as tf
import loom.stats as stats

def test_normal_pdf():
    x = 0
    mu = 0
    sigma = 1.0
    # PDF(0) = 1/sqrt(2*pi) approx 0.3989
    res = stats.normal_pdf(x, mu, sigma)
    assert math.isclose(res.item(), 1.0 / math.sqrt(2 * math.pi), abs_tol=1e-5)

def test_normal_cdf():
    # CDF(0) should be 0.5 for standard normal
    res = stats.normal_cdf(0.0)
    assert math.isclose(res.item(), 0.5, abs_tol=1e-7)
    
    # CDF(large) approx 1
    res_large = stats.normal_cdf(10.0)
    assert math.isclose(res_large.item(), 1.0, abs_tol=1e-5)

def test_poisson_pmf():
    # lam=2, k=1 -> (2^1 * exp(-2)) / 1! = 2 * 0.1353... = 0.27067
    res = stats.poisson_pmf(1, lam=2.0)
    expected = 2.0 * math.exp(-2.0)
    assert math.isclose(res.item(), expected, abs_tol=1e-7)

def test_skew_metrics():
    # Symmetric data should have 0 skew
    data = [1, 2, 2, 3] # mean=2, symmetric
    s = stats.skew(data)
    assert math.isclose(s.item(), 0.0, abs_tol=1e-7)
    
    # Positive skew
    data_pos = [1, 1, 1, 10]
    s_pos = stats.skew(data_pos)
    assert s_pos.item() > 0

def test_percentile():
    data = list(range(1, 101)) # 1 to 100
    p = stats.percentile(data, 50)
    # Median of 1..100 is 50.5
    assert math.isclose(p.item(), 50.5, abs_tol=1e-7)
    
    p0 = stats.percentile(data, 0)
    assert p0.item() == 1.0
    
    p100 = stats.percentile(data, 100)
    assert p100.item() == 100.0

def test_ttest_1samp():
    # Data with mean 10, testing against null mean 10
    data = [9, 10, 11]
    t_stat, p_val = stats.ttest_1samp(data, 10.0)
    assert math.isclose(t_stat.item(), 0.0, abs_tol=1e-7)
    assert math.isclose(p_val, 1.0, abs_tol=1e-7)
    
    # Testing against mean 0
    t_stat_2, p_val_2 = stats.ttest_1samp(data, 0.0)
    assert t_stat_2.item() > 0
    assert p_val_2 < 0.05 # significantly different from 0

def test_ttest_ind():
    a = [1, 2, 3]
    b = [1, 2, 3]
    t, p = stats.ttest_ind(a, b)
    assert math.isclose(t.item(), 0.0, abs_tol=1e-7)
    assert p == 1.0
    
    c = [10, 11, 12]
    t2, p2 = stats.ttest_ind(a, c)
    assert t2.item() < 0
    assert p2 < 0.05

