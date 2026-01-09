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
Statistical Distributions.
"""

import math
from typing import Union, List, Optional
import loom as tf
from loom.core.tensor import Tensor, array

def normal_pdf(x: Union[Tensor, List, float], mu: float = 0.0, sigma: float = 1.0) -> Tensor:
    """
    Normal (Gaussian) probability density function.
    """
    x = array(x)
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * (-(x - mu)**2 / (2 * sigma**2)).exp()

def normal_cdf(x: Union[Tensor, List, float], mu: float = 0.0, sigma: float = 1.0) -> Tensor:
    """
    Normal (Gaussian) cumulative distribution function.
    Uses erf approximation.
    """
    from loom.special import erf
    x = array(x)
    return 0.5 * (1 + erf((x - mu) / (sigma * math.sqrt(2))))

def poisson_pmf(k: Union[Tensor, List, int], lam: float) -> Tensor:
    """
    Poisson probability mass function.
    P(k) = (lam^k * exp(-lam)) / k!
    """
    from loom.special import loggamma
    k = array(k)
    # Use log-space calculation for stability: k*log(lam) - lam - loggamma(k+1)
    lam_tensor = array(lam)
    log_p = k * lam_tensor.log() - lam_tensor - loggamma(k + 1)
    return log_p.exp()

def binomial_pmf(k: Union[Tensor, List, int], n: int, p: float) -> Tensor:
    """
    Binomial probability mass function.
    P(k) = (n choose k) * p^k * (1-p)^(n-k)
    """
    from loom.special import loggamma
    k = array(k)
    
    # Use log-space: log(n!/(k!(n-k)!)) + k*log(p) + (n-k)*log(1-p)
    # log(comb) = loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)
    
    log_comb = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
    
    # Handle p=0 or p=1 cases safely
    if p == 0:
        return (k == 0).cast("float64")
    if p == 1:
        return (k == n).cast("float64")
        
    log_prob = log_comb + k * math.log(p) + (n - k) * math.log(1 - p)
    return log_prob.exp()

def gamma_pdf(x: Union[Tensor, List, float], alpha: float, beta: float) -> Tensor:
    """
    Gamma probability density function.
    f(x; alpha, beta) = (beta^alpha * x^(alpha-1) * exp(-beta*x)) / gamma(alpha)
    """
    from loom.special import gamma
    x = array(x)
    return (beta**alpha * x**(alpha - 1) * (-beta * x).exp()) / gamma(alpha)

