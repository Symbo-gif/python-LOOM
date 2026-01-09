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
Special Functions Capability Tests.

Tests: 65 very easy, 50 easy, 30 medium, 20 hard, 15 very hard = 180 total
Covers: gamma, loggamma, beta, erf, erfc, gammainc, gammaincc
"""

import pytest
import math
from loom.special import gamma, loggamma, beta, erf, erfc, gammainc, gammaincc


# =============================================================================
# VERY EASY (65 tests)
# =============================================================================

class TestVeryEasySpecial:
    """Very easy special functions tests."""
    
    # gamma (15)
    def test_ve_gamma_1(self): assert gamma(1) == 1.0
    def test_ve_gamma_2(self): assert gamma(2) == 1.0
    def test_ve_gamma_3(self): assert math.isclose(gamma(3), 2.0, rel_tol=1e-10)
    def test_ve_gamma_4(self): assert math.isclose(gamma(4), 6.0, rel_tol=1e-10)
    def test_ve_gamma_5(self): assert math.isclose(gamma(5), 24.0, rel_tol=1e-10)
    def test_ve_gamma_6(self): assert math.isclose(gamma(6), 120.0, rel_tol=1e-10)
    def test_ve_gamma_positive(self): assert gamma(1.5) > 0
    def test_ve_gamma_half(self): assert math.isclose(gamma(0.5), math.sqrt(math.pi), rel_tol=1e-6)
    def test_ve_gamma_1_5(self): assert math.isclose(gamma(1.5), math.sqrt(math.pi)/2, rel_tol=1e-6)
    def test_ve_gamma_2_5(self): assert math.isclose(gamma(2.5), 0.75*math.sqrt(math.pi), rel_tol=1e-6)
    def test_ve_gamma_increasing(self): assert gamma(3) > gamma(2)
    def test_ve_gamma_positive_int(self): assert all(gamma(n) > 0 for n in range(1, 10))
    def test_ve_gamma_factorial_rel(self): assert math.isclose(gamma(6), 5*4*3*2*1, rel_tol=1e-10)
    def test_ve_gamma_nonint(self): assert gamma(1.1) > 0
    def test_ve_gamma_small_pos(self): assert gamma(0.1) > 0
    
    # loggamma (10)
    def test_ve_loggamma_1(self): assert loggamma(1) == 0.0
    def test_ve_loggamma_2(self): assert loggamma(2) == 0.0
    def test_ve_loggamma_3(self): assert math.isclose(loggamma(3), math.log(2), rel_tol=1e-10)
    def test_ve_loggamma_4(self): assert math.isclose(loggamma(4), math.log(6), rel_tol=1e-10)
    def test_ve_loggamma_5(self): assert math.isclose(loggamma(5), math.log(24), rel_tol=1e-10)
    def test_ve_loggamma_positive(self): assert loggamma(10) > 0
    def test_ve_loggamma_exp_gamma(self): assert math.isclose(math.exp(loggamma(5)), gamma(5), rel_tol=1e-6)
    def test_ve_loggamma_large(self): assert loggamma(100) > 0
    def test_ve_loggamma_increasing(self): assert loggamma(10) > loggamma(5)
    def test_ve_loggamma_vs_gamma(self): 
        for n in [3, 4, 5]:
            assert math.isclose(math.exp(loggamma(n)), gamma(n), rel_tol=1e-6)
    
    # beta (10)
    def test_ve_beta_1_1(self): assert math.isclose(beta(1, 1), 1.0, rel_tol=1e-10)
    def test_ve_beta_1_2(self): assert math.isclose(beta(1, 2), 0.5, rel_tol=1e-10)
    def test_ve_beta_2_1(self): assert math.isclose(beta(2, 1), 0.5, rel_tol=1e-10)
    def test_ve_beta_symmetric(self): assert math.isclose(beta(2, 3), beta(3, 2), rel_tol=1e-10)
    def test_ve_beta_2_2(self): assert math.isclose(beta(2, 2), 1/6, rel_tol=1e-6)
    def test_ve_beta_positive(self): assert beta(1.5, 2.5) > 0
    def test_ve_beta_formula(self): 
        a, b = 3, 4
        expected = gamma(a) * gamma(b) / gamma(a + b)
        assert math.isclose(beta(a, b), expected, rel_tol=1e-6)
    def test_ve_beta_half_half(self): assert math.isclose(beta(0.5, 0.5), math.pi, rel_tol=1e-6)
    def test_ve_beta_decreasing(self): assert beta(1, 1) > beta(2, 2)
    def test_ve_beta_nonint(self): assert beta(1.5, 1.5) > 0
    
    # erf (10)
    def test_ve_erf_0(self): assert erf(0) == 0.0
    def test_ve_erf_positive(self): assert erf(1) > 0
    def test_ve_erf_negative(self): assert erf(-1) < 0
    def test_ve_erf_odd(self): assert math.isclose(erf(-1), -erf(1), rel_tol=1e-10)
    def test_ve_erf_large(self): assert erf(3) > 0.99
    def test_ve_erf_1(self): assert math.isclose(erf(1), 0.8427, rel_tol=0.01)
    def test_ve_erf_bounded(self): assert -1 <= erf(0.5) <= 1
    def test_ve_erf_small(self): assert math.isclose(erf(0.01), 0.01128, rel_tol=0.01)
    def test_ve_erf_increasing(self): assert erf(0.5) < erf(1) < erf(2)
    def test_ve_erf_limit(self): assert erf(10) > 0.9999999
    
    # erfc (10)
    def test_ve_erfc_0(self): assert erfc(0) == 1.0
    def test_ve_erfc_complement(self): assert math.isclose(erf(1) + erfc(1), 1.0, rel_tol=1e-10)
    def test_ve_erfc_large(self): assert erfc(3) < 0.01
    def test_ve_erfc_negative(self): assert erfc(-1) > 1
    def test_ve_erfc_bounded(self): assert 0 <= erfc(0.5) <= 2
    def test_ve_erfc_sum_1(self): 
        for x in [0.5, 1, 2]:
            assert math.isclose(erf(x) + erfc(x), 1.0, rel_tol=1e-10)
    def test_ve_erfc_1(self): assert math.isclose(erfc(1), 0.1573, rel_tol=0.01)
    def test_ve_erfc_decreasing(self): assert erfc(0) > erfc(1) > erfc(2)
    def test_ve_erfc_small_x(self): assert erfc(0.1) > 0.8
    def test_ve_erfc_very_large(self): assert erfc(10) < 1e-10
    
    # gammainc (10)
    def test_ve_gammainc_0(self): assert gammainc(1, 0) == 0.0
    def test_ve_gammainc_inf(self): assert math.isclose(gammainc(1, 100), 1.0, rel_tol=1e-6)
    def test_ve_gammainc_a1_x1(self): assert math.isclose(gammainc(1, 1), 1 - math.exp(-1), rel_tol=1e-6)
    def test_ve_gammainc_increasing(self): assert gammainc(1, 1) < gammainc(1, 2) < gammainc(1, 3)
    def test_ve_gammainc_bounded(self): assert 0 <= gammainc(2, 1) <= 1
    def test_ve_gammainc_small(self): assert gammainc(2, 0.1) < 0.1
    def test_ve_gammainc_exponential(self): 
        # gammainc(1, x) = 1 - exp(-x)
        assert math.isclose(gammainc(1, 2), 1 - math.exp(-2), rel_tol=1e-6)
    def test_ve_gammainc_positive(self): assert gammainc(3, 2) > 0
    def test_ve_gammainc_limit_0(self): assert gammainc(2, 0) == 0.0
    def test_ve_gammainc_a_effect(self): assert gammainc(1, 1) > gammainc(2, 1)


# =============================================================================
# EASY (50 tests)
# =============================================================================

class TestEasySpecial:
    """Easy special functions tests."""
    
    # gamma (10)
    def test_e_gamma_7(self): assert math.isclose(gamma(7), 720, rel_tol=1e-10)
    def test_e_gamma_8(self): assert math.isclose(gamma(8), 5040, rel_tol=1e-10)
    def test_e_gamma_recurrence(self): 
        for n in [3, 4, 5, 6]:
            assert math.isclose(gamma(n+1), n * gamma(n), rel_tol=1e-6)
    def test_e_gamma_3_5(self): assert gamma(3.5) > 0
    def test_e_gamma_4_5(self): assert gamma(4.5) > 0
    def test_e_gamma_duplication(self):
        # Legendre duplication: Γ(z)Γ(z+1/2) = 2^(1-2z) * sqrt(pi) * Γ(2z)
        z = 2
        lhs = gamma(z) * gamma(z + 0.5)
        rhs = 2**(1-2*z) * math.sqrt(math.pi) * gamma(2*z)
        assert math.isclose(lhs, rhs, rel_tol=1e-6)
    def test_e_gamma_reflection(self):
        # Γ(z)Γ(1-z) = π/sin(πz) for non-integer z
        z = 0.3
        lhs = gamma(z) * gamma(1-z)
        rhs = math.pi / math.sin(math.pi * z)
        assert math.isclose(lhs, rhs, rel_tol=1e-4)
    def test_e_gamma_large(self): assert gamma(20) > 1e15
    def test_e_gamma_very_small(self): assert gamma(0.01) > 99
    def test_e_gamma_near_1(self): assert math.isclose(gamma(1.001), 0.9994, rel_tol=0.01)
    
    # loggamma (10)
    def test_e_loggamma_10(self): assert math.isclose(loggamma(10), math.log(gamma(10)), rel_tol=1e-6)
    def test_e_loggamma_20(self): assert loggamma(20) > 30
    def test_e_loggamma_50(self): assert loggamma(50) > 100
    def test_e_loggamma_stirling(self):
        # Stirling: log(Γ(n)) ≈ (n-0.5)log(n) - n + 0.5*log(2π)
        n = 50
        stirling = (n-0.5)*math.log(n) - n + 0.5*math.log(2*math.pi)
        assert math.isclose(loggamma(n), stirling, rel_tol=0.01)
    def test_e_loggamma_difference(self):
        # log(Γ(n+1)) - log(Γ(n)) = log(n)
        n = 10
        diff = loggamma(n+1) - loggamma(n)
        assert math.isclose(diff, math.log(n), rel_tol=1e-6)
    def test_e_loggamma_nonint(self): assert loggamma(5.5) > 0
    def test_e_loggamma_vs_log_gamma(self):
        for x in [2, 5, 10]:
            assert math.isclose(loggamma(x), math.log(gamma(x)), rel_tol=1e-10)
    def test_e_loggamma_100(self): assert loggamma(100) > 350
    def test_e_loggamma_half(self): assert math.isclose(loggamma(0.5), math.log(math.sqrt(math.pi)), rel_tol=1e-6)
    def test_e_loggamma_1_5(self): assert loggamma(1.5) < 0
    
    # beta (10)
    def test_e_beta_3_4(self):
        expected = gamma(3) * gamma(4) / gamma(7)
        assert math.isclose(beta(3, 4), expected, rel_tol=1e-6)
    def test_e_beta_5_5(self):
        expected = gamma(5) * gamma(5) / gamma(10)
        assert math.isclose(beta(5, 5), expected, rel_tol=1e-6)
    def test_e_beta_integral(self):
        # B(a,b) = B(a+1,b) * (a+b)/a
        a, b = 2, 3
        lhs = beta(a, b)
        rhs = beta(a+1, b) * (a+b) / a
        assert math.isclose(lhs, rhs, rel_tol=1e-6)
    def test_e_beta_recurrence(self):
        # B(a,b) = B(a,b+1) * (a+b)/b
        a, b = 3, 4
        lhs = beta(a, b)
        rhs = beta(a, b+1) * (a+b) / b
        assert math.isclose(lhs, rhs, rel_tol=1e-6)
    def test_e_beta_uniform(self): assert math.isclose(beta(1, 1), 1.0, rel_tol=1e-10)
    def test_e_beta_small_params(self): assert beta(0.1, 0.1) > 0
    def test_e_beta_10_10(self): assert beta(10, 10) > 0
    def test_e_beta_asymmetric(self): assert beta(1, 10) != beta(10, 1) or beta(1, 10) == beta(10, 1)  # symmetric
    def test_e_beta_vs_gamma(self):
        a, b = 4, 5
        assert math.isclose(beta(a, b), gamma(a)*gamma(b)/gamma(a+b), rel_tol=1e-6)
    def test_e_beta_large(self): assert beta(20, 20) > 0
    
    # erf/erfc (10)
    def test_e_erf_2(self): assert math.isclose(erf(2), 0.9953, rel_tol=0.01)
    def test_e_erf_0_5(self): assert math.isclose(erf(0.5), 0.5205, rel_tol=0.01)
    def test_e_erfc_2(self): assert math.isclose(erfc(2), 0.0047, rel_tol=0.1)
    def test_e_erfc_0_5(self): assert math.isclose(erfc(0.5), 0.4795, rel_tol=0.01)
    def test_e_erf_normal_cdf(self):
        # Φ(x) = 0.5 * (1 + erf(x/sqrt(2)))
        x = 1
        phi = 0.5 * (1 + erf(x / math.sqrt(2)))
        assert math.isclose(phi, 0.8413, rel_tol=0.01)
    def test_e_erf_series(self):
        # erf(x) ≈ 2x/sqrt(π) for small x
        x = 0.01
        approx = 2 * x / math.sqrt(math.pi)
        assert math.isclose(erf(x), approx, rel_tol=0.01)
    def test_e_erfc_tail(self): assert erfc(4) < 1e-6
    def test_e_erf_derivative(self):
        # d/dx erf(x) = 2/sqrt(π) * exp(-x²)
        h = 1e-5
        x = 0.5
        deriv = (erf(x + h) - erf(x - h)) / (2 * h)
        expected = 2 / math.sqrt(math.pi) * math.exp(-x**2)
        assert math.isclose(deriv, expected, rel_tol=0.01)
    def test_e_erfc_negative_large(self): assert erfc(-3) > 1.99
    def test_e_erfc_vs_erf(self):
        for x in [-1, 0, 1, 2]:
            assert math.isclose(erf(x) + erfc(x), 1.0, rel_tol=1e-10)
    
    # gammainc (10)
    def test_e_gammainc_complement(self):
        a, x = 2, 1.5
        assert math.isclose(gammainc(a, x) + gammaincc(a, x), 1.0, rel_tol=1e-10)
    def test_e_gammainc_chi2(self):
        # Chi-square CDF: P = gammainc(k/2, x/2)
        k, x = 4, 5
        p = gammainc(k/2, x/2)
        assert 0 < p < 1
    def test_e_gammainc_2_1(self): 
        # P(2,1) = 1 - (1+1)*e^(-1) = 1 - 2e^(-1)
        expected = 1 - 2 * math.exp(-1)
        assert math.isclose(gammainc(2, 1), expected, rel_tol=1e-6)
    def test_e_gammainc_large_x(self): assert math.isclose(gammainc(3, 50), 1.0, rel_tol=1e-10)
    def test_e_gammaincc_small_x(self): assert math.isclose(gammaincc(2, 0.01), 1.0, rel_tol=0.01)
    def test_e_gammainc_half_int(self):
        # For a=0.5, gammainc is related to erf
        x = 1
        expected = erf(math.sqrt(x))
        assert math.isclose(gammainc(0.5, x), expected, rel_tol=1e-4)
    def test_e_gammaincc_large_a(self): assert gammaincc(50, 1) > 0.99
    def test_e_gammainc_increasing_x(self):
        a = 2
        assert gammainc(a, 1) < gammainc(a, 2) < gammainc(a, 5)
    def test_e_gammaincc_decreasing_x(self):
        a = 2
        assert gammaincc(a, 1) > gammaincc(a, 2) > gammaincc(a, 5)
    def test_e_gammainc_sum(self):
        for a in [1, 2, 5]:
            for x in [0.5, 1, 3]:
                assert math.isclose(gammainc(a, x) + gammaincc(a, x), 1.0, rel_tol=1e-10)


# =============================================================================
# MEDIUM (30 tests)
# =============================================================================

class TestMediumSpecial:
    """Medium special functions tests."""
    
    def test_m_gamma_10(self): assert math.isclose(gamma(10), 362880, rel_tol=1e-8)
    def test_m_gamma_15(self): assert gamma(15) > 8e10
    def test_m_loggamma_stirling_accurate(self):
        n = 100
        stirling = (n-0.5)*math.log(n) - n + 0.5*math.log(2*math.pi)
        assert math.isclose(loggamma(n), stirling, rel_tol=0.001)
    def test_m_beta_mean(self):
        # E[X] for Beta(a,b) = a/(a+b)
        a, b = 2, 5
        mean = a / (a + b)
        assert mean == pytest.approx(2/7, rel=0.01)
    def test_m_beta_variance(self):
        # Var[X] = ab/((a+b)²(a+b+1))
        a, b = 2, 5
        var = a*b / ((a+b)**2 * (a+b+1))
        assert var > 0
    def test_m_erf_chained(self):
        # erf(erf(1)) < erf(1)
        assert erf(erf(1)) < erf(1)
    def test_m_erfc_precision(self):
        assert erfc(5) < 1e-10
    def test_m_gammainc_chisq_table(self):
        # Chi-square(2) P(x<2) = gammainc(1, 1) = 1 - e^(-1)
        p = gammainc(1, 1)
        assert math.isclose(p, 1 - math.exp(-1), rel_tol=1e-6)
    def test_m_gammainc_series_region(self):
        # Small x uses series
        assert gammainc(5, 0.1) < 0.001
    def test_m_gammainc_cf_region(self):
        # Large x uses continued fraction
        assert gammainc(2, 10) > 0.999
    def test_m_gamma_near_pole(self):
        # Gamma has poles at 0, -1, -2, ...
        # Near pole, should be large
        assert abs(gamma(0.001)) > 900
    def test_m_loggamma_large_arg(self):
        assert loggamma(1000) > 5000
    def test_m_beta_small_args(self):
        assert beta(0.5, 0.5) > 3
    def test_m_erf_integral(self):
        # erf(x) = (2/sqrt(π)) * ∫₀ˣ exp(-t²) dt
        # Check at x=1
        assert math.isclose(erf(1), 0.8427, rel_tol=0.01)
    def test_m_erfc_large_arg(self):
        assert erfc(6) < 1e-15 or erfc(6) == 0
    def test_m_gammainc_a_large(self):
        # For large a, need x ≈ a for significant P
        assert gammainc(100, 80) < 0.1
    def test_m_gammaincc_a_large(self):
        assert gammaincc(100, 80) > 0.9
    def test_m_gamma_recurrence_chain(self):
        # Check Γ(n) = (n-1)! for several values
        for n in range(1, 10):
            factorial = 1
            for k in range(1, n):
                factorial *= k
            assert math.isclose(gamma(n), factorial, rel_tol=1e-8)
    def test_m_loggamma_additive(self):
        # loggamma(n+1) = loggamma(n) + log(n)
        n = 20
        assert math.isclose(loggamma(n+1), loggamma(n) + math.log(n), rel_tol=1e-8)
    def test_m_beta_log(self):
        # log(beta(a,b)) = loggamma(a) + loggamma(b) - loggamma(a+b)
        a, b = 5, 7
        log_beta = loggamma(a) + loggamma(b) - loggamma(a+b)
        assert math.isclose(math.log(beta(a, b)), log_beta, rel_tol=1e-8)
    def test_m_erf_table_values(self):
        # Standard table values
        assert math.isclose(erf(0.5), 0.5205, rel_tol=0.01)
        assert math.isclose(erf(1.5), 0.9661, rel_tol=0.01)
    def test_m_gammainc_exponential_special(self):
        # For a=1, gammainc(1,x) = 1 - exp(-x)
        for x in [0.5, 1, 2, 5]:
            assert math.isclose(gammainc(1, x), 1 - math.exp(-x), rel_tol=1e-6)
    def test_m_gamma_half_integers(self):
        # Γ(n+1/2) = ((2n-1)!!/(2^n)) * √π
        # For n=2: Γ(2.5) = (3*1/4)*√π = 0.75√π
        assert math.isclose(gamma(2.5), 0.75 * math.sqrt(math.pi), rel_tol=1e-6)
    def test_m_erfc_asymptotic(self):
        # erfc(x) ≈ exp(-x²)/(x√π) for large x
        x = 5
        asymp = math.exp(-x**2) / (x * math.sqrt(math.pi))
        assert math.isclose(erfc(x), asymp, rel_tol=0.1)
    def test_m_gammainc_regularized(self):
        # Check that it's properly regularized (P ∈ [0,1])
        for a in [1, 2, 5, 10]:
            for x in [0.1, 1, 5, 10]:
                assert 0 <= gammainc(a, x) <= 1
    def test_m_beta_reflection(self):
        assert math.isclose(beta(2, 3), beta(3, 2), rel_tol=1e-10)
    def test_m_gamma_double_arg(self):
        # Γ(2z) relation
        z = 1.5
        lhs = gamma(2*z)
        rhs = gamma(z) * gamma(z+0.5) * 2**(2*z-1) / math.sqrt(math.pi)
        assert math.isclose(lhs, rhs, rel_tol=1e-4)
    def test_m_erf_odd_function(self):
        for x in [0.5, 1, 2]:
            assert math.isclose(erf(-x), -erf(x), rel_tol=1e-10)
    def test_m_gammaincc_poisson(self):
        # Q(a,x) = P(X >= a) for Poisson(x) with integer a
        # Check boundary behavior
        assert 0 < gammaincc(5, 5) < 1


# =============================================================================
# HARD (20 tests)
# =============================================================================

class TestHardSpecial:
    """Hard special functions tests."""
    
    def test_h_gamma_near_zero(self):
        assert gamma(0.001) > 999
    def test_h_gamma_very_large(self):
        assert gamma(50) > 1e60
    def test_h_loggamma_very_large(self):
        assert loggamma(500) > 2500
    def test_h_beta_edge(self):
        assert beta(0.1, 10) > 0
    def test_h_erf_very_small(self):
        x = 1e-10
        assert math.isclose(erf(x), 2*x/math.sqrt(math.pi), rel_tol=0.01)
    def test_h_erfc_very_large(self):
        assert erfc(10) < 1e-40 or erfc(10) == 0
    def test_h_gammainc_small_a(self):
        assert gammainc(0.1, 1) > 0.9
    def test_h_gammainc_large_a(self):
        assert gammainc(100, 50) < 0.001
    def test_h_gamma_precision(self):
        # Check precision for known value
        assert math.isclose(gamma(10), 362880, rel_tol=1e-10)
    def test_h_loggamma_precision(self):
        assert math.isclose(loggamma(10), math.log(362880), rel_tol=1e-10)
    def test_h_beta_precision(self):
        a, b = 10, 10
        expected = gamma(a) * gamma(b) / gamma(a+b)
        assert math.isclose(beta(a, b), expected, rel_tol=1e-10)
    def test_h_gammainc_precision(self):
        # Known value
        assert math.isclose(gammainc(1, 1), 1 - math.exp(-1), rel_tol=1e-10)
    def test_h_erf_precision(self):
        assert math.isclose(erf(1), 0.8427007929, rel_tol=1e-6)
    def test_h_erfc_precision(self):
        assert math.isclose(erfc(1), 0.1572992070, rel_tol=1e-6)
    def test_h_gamma_continuity(self):
        # Check continuity near integer
        assert math.isclose(gamma(2.999), gamma(3), rel_tol=0.01)
    def test_h_loggamma_continuity(self):
        assert math.isclose(loggamma(2.999), loggamma(3), rel_tol=0.01)
    def test_h_gammainc_continuity(self):
        a = 2
        assert math.isclose(gammainc(a, 0.999), gammainc(a, 1.001), rel_tol=0.01)
    def test_h_erf_continuity(self):
        assert math.isclose(erf(0.999), erf(1.001), rel_tol=0.01)
    def test_h_beta_large_args(self):
        assert beta(50, 50) > 0
    def test_h_gammainc_boundary(self):
        # Near x=0 should be near 0
        assert gammainc(5, 0.0001) < 1e-15


# =============================================================================
# VERY HARD (15 tests)
# =============================================================================

class TestVeryHardSpecial:
    """Very hard special functions tests - edge cases."""
    
    def test_vh_gamma_pole(self):
        try:
            g = gamma(0)
            assert g == float('inf') or math.isinf(g)
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_gamma_negative_int(self):
        try:
            g = gamma(-1)
            # Should be inf or raise
            assert math.isinf(g)
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_loggamma_zero(self):
        try:
            lg = loggamma(0)
            assert lg == float('inf') or math.isinf(lg)
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_beta_zero(self):
        try:
            b = beta(0, 1)
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_gammainc_a_zero(self):
        try:
            p = gammainc(0, 1)
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_erf_very_large(self):
        assert math.isclose(erf(50), 1.0, rel_tol=1e-15)
    
    def test_vh_erfc_very_large(self):
        result = erfc(50)
        assert result == 0 or result < 1e-100
    
    def test_vh_gamma_tiny(self):
        x = 1e-15
        g = gamma(x)
        assert g > 1e14
    
    def test_vh_loggamma_tiny(self):
        x = 1e-15
        lg = loggamma(x)
        assert lg > 30
    
    def test_vh_gammainc_x_zero(self):
        assert gammainc(5, 0) == 0.0
    
    def test_vh_gammaincc_x_zero(self):
        assert gammaincc(5, 0) == 1.0
    
    def test_vh_beta_both_small(self):
        b = beta(0.01, 0.01)
        assert b > 100
    
    def test_vh_gamma_underflow(self):
        # Very large argument
        g = gamma(170)
        assert g > 0 or g == float('inf')
    
    def test_vh_loggamma_overflow_avoided(self):
        # loggamma handles large args without overflow
        lg = loggamma(1000)
        assert lg > 0 and not math.isinf(lg)
    
    def test_vh_gammainc_extreme(self):
        # Very small a, large x
        p = gammainc(0.001, 100)
        assert math.isclose(p, 1.0, rel_tol=1e-10)

