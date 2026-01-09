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
Statistical Hypothesis Tests.
"""

import math
from typing import Union, List, Tuple, Optional
import loom as tf
from loom.core.tensor import Tensor, array
from loom.stats.metrics import std, variance


def ttest_1samp(a: Union[Tensor, List], popmean: float) -> Tuple[Tensor, float]:
    """
    Calculate the T-test for the mean of ONE group of scores.
    
    Returns (t_statistic, p_value).
    Uses normal approximation for large samples.
    """
    a = array(a)
    n = a.size
    if n < 2:
        raise ValueError("ttest_1samp requires at least 2 observations")
    
    mu = a.mean()
    s = std(a, ddof=1)
    
    if s.item() < 1e-15:
        # No variance, t-stat is undefined or infinite
        if abs(mu.item() - popmean) < 1e-15:
            return array(0.0), 1.0
        else:
            return array(float('inf') if mu.item() > popmean else float('-inf')), 0.0
    
    t_stat = (mu - popmean) / (s / math.sqrt(n))
    
    # Use normal approximation (valid for large n, reasonable for small n)
    from loom.stats.distributions import normal_cdf
    p_val = 2 * (1 - normal_cdf(abs(t_stat.item())).item())
    
    return t_stat, p_val


def ttest_ind(a: Union[Tensor, List], b: Union[Tensor, List], equal_var: bool = True) -> Tuple[Tensor, float]:
    """
    Calculate the T-test for the means of TWO INDEPENDENT samples of scores.
    
    Returns (t_statistic, p_value).
    """
    a = array(a)
    b = array(b)
    n1 = a.size
    n2 = b.size
    
    if n1 < 2 or n2 < 2:
        raise ValueError("ttest_ind requires at least 2 observations in each sample")
    
    m1 = a.mean()
    m2 = b.mean()
    v1 = variance(a, ddof=1)
    v2 = variance(b, ddof=1)
    
    if equal_var:
        # Pooled variance
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        denom = (sp2 * (1.0/n1 + 1.0/n2)).sqrt()
    else:
        # Welch's t-test
        denom = (v1/n1 + v2/n2).sqrt()
    
    if denom.item() < 1e-15:
        if abs(m1.item() - m2.item()) < 1e-15:
            return array(0.0), 1.0
        else:
            return array(float('inf') if m1.item() > m2.item() else float('-inf')), 0.0
        
    t_stat = (m1 - m2) / denom
    
    from loom.stats.distributions import normal_cdf
    p_val = 2 * (1 - normal_cdf(abs(t_stat.item())).item())
    
    return t_stat, p_val


def chisquare(f_obs: Union[Tensor, List], f_exp: Optional[Union[Tensor, List]] = None) -> Tuple[Tensor, float]:
    """
    Calculate a one-way chi-square test.
    
    Returns (chi_statistic, p_value).
    
    The p-value is calculated using the chi-square CDF via the regularized
    incomplete gamma function.
    """
    f_obs = array(f_obs)
    k = f_obs.size  # Number of categories (degrees of freedom = k - 1)
    
    if k < 2:
        raise ValueError("chisquare requires at least 2 categories")
    
    if f_exp is None:
        # Uniform expected frequencies
        total = f_obs.sum().item()
        f_exp = array([total / k] * k)
    else:
        f_exp = array(f_exp)
        
    if f_exp.size != k:
        raise ValueError("f_obs and f_exp must have the same length")
    
    # Check for zeros in expected
    exp_list = f_exp.tolist()
    if any(e <= 0 for e in exp_list):
        raise ValueError("Expected frequencies must be positive")
    
    chi_stat = ((f_obs - f_exp)**2 / f_exp).sum()
    
    # Degrees of freedom
    df = k - 1
    
    # P-value: P(X > chi_stat) = 1 - CDF(chi_stat)
    # Chi-square CDF: P(x; k) = gammainc(k/2, x/2)
    from loom.special import gammainc
    p_val = 1.0 - gammainc(df / 2.0, chi_stat.item() / 2.0)
    
    return chi_stat, p_val

