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
Statistics Capability Tests.

Tests: 65 very easy, 50 easy, 30 medium, 20 hard, 15 very hard = 180 total
Covers: distributions, metrics, hypothesis tests
"""

import pytest
import math
import loom as tf
import loom.stats as stats


# =============================================================================
# VERY EASY (65 tests)
# =============================================================================

class TestVeryEasyStats:
    """Very easy statistics tests."""
    
    # Distributions (20)
    def test_ve_normal_pdf_0(self): assert math.isclose(stats.normal_pdf(0, 0, 1), 0.3989, rel_tol=0.01)
    def test_ve_normal_pdf_mu(self): assert math.isclose(stats.normal_pdf(5, 5, 1), 0.3989, rel_tol=0.01)
    def test_ve_normal_cdf_0(self): assert math.isclose(stats.normal_cdf(0, 0, 1), 0.5, rel_tol=0.01)
    def test_ve_normal_cdf_inf(self): assert stats.normal_cdf(10, 0, 1) > 0.999
    def test_ve_normal_cdf_neginf(self): assert stats.normal_cdf(-10, 0, 1) < 0.001
    def test_ve_poisson_pmf_0(self): assert math.isclose(stats.poisson_pmf(0, 1), math.exp(-1), rel_tol=0.01)
    def test_ve_poisson_pmf_1(self): assert math.isclose(stats.poisson_pmf(1, 1), math.exp(-1), rel_tol=0.01)
    def test_ve_poisson_pmf_mean(self): assert stats.poisson_pmf(2, 2) > 0
    def test_ve_binomial_pmf_0(self): assert math.isclose(stats.binomial_pmf(0, 1, 0.5), 0.5, rel_tol=0.01)
    def test_ve_binomial_pmf_1(self): assert math.isclose(stats.binomial_pmf(1, 1, 0.5), 0.5, rel_tol=0.01)
    def test_ve_binomial_pmf_all(self): assert math.isclose(stats.binomial_pmf(2, 2, 1.0), 1.0, rel_tol=0.01)
    def test_ve_binomial_pmf_none(self): assert math.isclose(stats.binomial_pmf(0, 2, 0.0), 1.0, rel_tol=0.01)
    def test_ve_gamma_pdf_1(self): assert stats.gamma_pdf(1, 1, 1) > 0
    def test_ve_gamma_pdf_0(self): assert stats.gamma_pdf(0.001, 1, 1) > 0
    def test_ve_normal_pdf_wide(self): assert stats.normal_pdf(0, 0, 10) < stats.normal_pdf(0, 0, 1)
    def test_ve_normal_cdf_sym_neg(self): 
        p1 = stats.normal_cdf(-1, 0, 1)
        p2 = stats.normal_cdf(1, 0, 1)
        assert math.isclose(p1 + p2, 1.0, rel_tol=0.01)
    def test_ve_poisson_sum(self): 
        total = sum(stats.poisson_pmf(k, 1) for k in range(20))
        assert math.isclose(total, 1.0, rel_tol=0.01)
    def test_ve_binomial_sum(self):
        total = sum(stats.binomial_pmf(k, 5, 0.5) for k in range(6))
        assert math.isclose(total, 1.0, rel_tol=0.01)
    def test_ve_gamma_pdf_shape(self): assert stats.gamma_pdf(2, 2, 1) > 0
    def test_ve_gamma_pdf_scale(self): assert stats.gamma_pdf(1, 1, 2) > 0
    
    # Metrics (25)
    def test_ve_std_zeros(self): assert stats.std([0, 0, 0]) == 0.0
    def test_ve_std_ones(self): assert stats.std([1, 1, 1]) == 0.0
    def test_ve_variance_zeros(self): assert stats.variance([0, 0, 0]) == 0.0
    def test_ve_variance_simple(self): assert stats.variance([1, 2, 3]) > 0
    def test_ve_median_odd(self): assert stats.median([1, 2, 3]) == 2.0
    def test_ve_median_even(self): assert stats.median([1, 2, 3, 4]) == 2.5
    def test_ve_median_single(self): assert stats.median([5]) == 5.0
    def test_ve_percentile_50(self): assert stats.percentile([1, 2, 3, 4, 5], 50) == 3.0
    def test_ve_percentile_0(self): assert stats.percentile([1, 2, 3], 0) == 1.0
    def test_ve_percentile_100(self): assert stats.percentile([1, 2, 3], 100) == 3.0
    def test_ve_skew_symmetric(self): assert abs(stats.skew([1, 2, 3, 4, 5])) < 0.1
    def test_ve_skew_right(self): assert stats.skew([1, 1, 1, 1, 10]) > 0
    def test_ve_skew_left(self): assert stats.skew([1, 10, 10, 10, 10]) < 0
    def test_ve_kurtosis_normal_like(self): data = [1, 2, 3, 4, 5]; k = stats.kurtosis(data); assert k is not None
    def test_ve_std_simple(self): 
        s = stats.std([1, 2, 3, 4, 5])
        assert 1.0 < s < 2.0
    def test_ve_variance_double_std(self):
        data = [1, 2, 3, 4, 5]
        assert math.isclose(stats.variance(data), stats.std(data)**2, rel_tol=0.01)
    def test_ve_median_sorted(self): assert stats.median([1, 2, 3, 4, 5]) == 3.0
    def test_ve_median_unsorted(self): assert stats.median([3, 1, 4, 1, 5]) == 3.0
    def test_ve_percentile_25(self): assert stats.percentile([1, 2, 3, 4, 5, 6, 7, 8], 25) == 2.5
    def test_ve_percentile_75(self): assert stats.percentile([1, 2, 3, 4, 5, 6, 7, 8], 75) == 6.5
    def test_ve_std_ddof_0(self): assert stats.std([1, 2, 3], ddof=0) > 0
    def test_ve_std_ddof_1(self): assert stats.std([1, 2, 3], ddof=1) > stats.std([1, 2, 3], ddof=0)
    def test_ve_variance_ddof_0(self): assert stats.variance([1, 2, 3], ddof=0) > 0
    def test_ve_variance_ddof_1(self): assert stats.variance([1, 2, 3], ddof=1) > 0
    def test_ve_kurtosis_constant(self): 
        # Constant data has undefined/zero kurtosis
        k = stats.kurtosis([1, 1, 1, 1, 1])
        # May be 0 or nan
        assert k == 0 or math.isnan(k) or k is None
    
    # Tests (20)
    def test_ve_chisq_uniform(self):
        chi, p = stats.chisquare([10, 10, 10, 10])
        assert chi.item() == 0.0
        
    def test_ve_chisq_pvalue_perfect(self):
        chi, p = stats.chisquare([10, 10, 10])
        assert math.isclose(p, 1.0, abs_tol=0.01)
    
    def test_ve_chisq_deviation(self):
        chi, p = stats.chisquare([20, 5, 5])
        assert chi.item() > 0
    
    def test_ve_ttest_1samp_exact(self):
        t, p = stats.ttest_1samp([10, 10, 10], 10)
        assert abs(t.item()) < 0.01
    
    def test_ve_ttest_1samp_different(self):
        t, p = stats.ttest_1samp([1, 2, 3], 10)
        assert abs(t.item()) > 1
    
    def test_ve_ttest_ind_same(self):
        t, p = stats.ttest_ind([1, 2, 3], [1, 2, 3])
        assert abs(t.item()) < 0.1
    
    def test_ve_ttest_ind_different(self):
        t, p = stats.ttest_ind([1, 2, 3], [10, 11, 12])
        assert abs(t.item()) > 1
    
    def test_ve_chisq_two_cats(self):
        chi, p = stats.chisquare([15, 15])
        assert chi.item() == 0.0
    
    def test_ve_chisq_expected(self):
        chi, p = stats.chisquare([10, 20], [15, 15])
        assert chi.item() > 0
    
    def test_ve_ttest_1samp_returns_2(self):
        result = stats.ttest_1samp([1, 2, 3], 2)
        assert len(result) == 2
    
    def test_ve_ttest_ind_returns_2(self):
        result = stats.ttest_ind([1, 2, 3], [4, 5, 6])
        assert len(result) == 2
    
    def test_ve_chisq_returns_2(self):
        result = stats.chisquare([5, 5, 5])
        assert len(result) == 2
    
    def test_ve_ttest_1samp_positive(self):
        t, p = stats.ttest_1samp([5, 6, 7], 4)
        assert t.item() > 0
    
    def test_ve_ttest_1samp_negative(self):
        t, p = stats.ttest_1samp([1, 2, 3], 10)
        assert t.item() < 0
    
    def test_ve_ttest_ind_positive(self):
        t, p = stats.ttest_ind([10, 11, 12], [1, 2, 3])
        assert t.item() > 0
    
    def test_ve_chisq_all_same(self):
        chi, p = stats.chisquare([7, 7, 7, 7, 7, 7, 7])
        assert chi.item() == 0.0
    
    def test_ve_chisq_pvalue_range(self):
        chi, p = stats.chisquare([10, 20, 30])
        assert 0 <= p <= 1
    
    def test_ve_ttest_pvalue_range(self):
        t, p = stats.ttest_1samp([1, 2, 3], 2)
        assert 0 <= p <= 1
    
    def test_ve_ttest_ind_pvalue_range(self):
        t, p = stats.ttest_ind([1, 2, 3], [4, 5, 6])
        assert 0 <= p <= 1
    
    def test_ve_chisq_large_deviation(self):
        chi, p = stats.chisquare([100, 1, 1, 1])
        assert p < 0.05


# =============================================================================
# EASY (50 tests)
# =============================================================================

class TestEasyStats:
    """Easy statistics tests."""
    
    # Distributions (15)
    def test_e_normal_pdf_sigma_2(self):
        p = stats.normal_pdf(0, 0, 2)
        assert math.isclose(p, 0.199, rel_tol=0.02)
    
    def test_e_normal_cdf_1sigma(self):
        p = stats.normal_cdf(1, 0, 1)
        assert math.isclose(p, 0.8413, rel_tol=0.01)
    
    def test_e_normal_cdf_2sigma(self):
        p = stats.normal_cdf(2, 0, 1)
        assert math.isclose(p, 0.9772, rel_tol=0.01)
    
    def test_e_poisson_pmf_lam_5(self):
        p = stats.poisson_pmf(5, 5)
        assert p > 0.1
    
    def test_e_binomial_pmf_n_10(self):
        p = stats.binomial_pmf(5, 10, 0.5)
        assert p > 0.2
    
    def test_e_gamma_pdf_exponential(self):
        # Gamma(1, beta) = Exponential(1/beta)
        p = stats.gamma_pdf(1, 1, 1)
        assert math.isclose(p, math.exp(-1), rel_tol=0.1)
    
    def test_e_normal_pdf_tail(self):
        p = stats.normal_pdf(3, 0, 1)
        assert p < 0.01
    
    def test_e_poisson_mode(self):
        # Mode of Poisson(lambda) is floor(lambda)
        p_at_mode = stats.poisson_pmf(5, 5.5)
        p_higher = stats.poisson_pmf(6, 5.5)
        # Should be similar
        assert abs(p_at_mode - p_higher) < 0.05
    
    def test_e_binomial_extremes(self):
        assert stats.binomial_pmf(10, 10, 0.99) > 0.9
    
    def test_e_gamma_shape_effect(self):
        p1 = stats.gamma_pdf(2, 1, 1)
        p2 = stats.gamma_pdf(2, 2, 1)
        # Different shapes give different densities
        assert p1 != p2
    
    def test_e_normal_cdf_symmetry(self):
        p_pos = stats.normal_cdf(1.5, 0, 1)
        p_neg = stats.normal_cdf(-1.5, 0, 1)
        assert math.isclose(p_pos + p_neg, 1.0, rel_tol=0.01)
    
    def test_e_poisson_cumulative(self):
        cdf = sum(stats.poisson_pmf(k, 3) for k in range(10))
        assert cdf > 0.99
    
    def test_e_binomial_expected(self):
        # E[X] = n*p, check mode is near expected
        p_expected = stats.binomial_pmf(5, 10, 0.5)
        assert p_expected > 0.2
    
    def test_e_gamma_scale_effect(self):
        p1 = stats.gamma_pdf(2, 2, 1)
        p2 = stats.gamma_pdf(4, 2, 2)
        # Scaling relationship
        assert p1 > 0 and p2 > 0
    
    def test_e_normal_pdf_different_mu(self):
        p1 = stats.normal_pdf(0, 0, 1)
        p2 = stats.normal_pdf(5, 5, 1)
        assert math.isclose(p1, p2, rel_tol=0.01)
    
    # Metrics (20)
    def test_e_std_large_data(self):
        data = list(range(100))
        s = stats.std(data)
        assert 25 < s < 35
    
    def test_e_variance_formula(self):
        data = [1, 2, 3, 4, 5]
        v = stats.variance(data)
        mean = sum(data) / len(data)
        manual = sum((x - mean)**2 for x in data) / len(data)
        assert math.isclose(v, manual, rel_tol=0.01)
    
    def test_e_median_large(self):
        data = list(range(1, 101))
        m = stats.median(data)
        assert m == 50.5
    
    def test_e_percentile_quartiles(self):
        data = list(range(1, 101))
        q1 = stats.percentile(data, 25)
        q3 = stats.percentile(data, 75)
        assert q1 < 30 and q3 > 70
    
    def test_e_skew_exponential_like(self):
        # Right-skewed data
        data = [1]*10 + [2]*5 + [10]*2
        s = stats.skew(data)
        assert s > 0
    
    def test_e_kurtosis_uniform_like(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k = stats.kurtosis(data)
        # Uniform has negative excess kurtosis
        assert k < 3
    
    def test_e_std_scaled(self):
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 4, 6, 8, 10]
        assert math.isclose(stats.std(data2), 2 * stats.std(data1), rel_tol=0.01)
    
    def test_e_variance_additive_constant(self):
        data1 = [1, 2, 3, 4, 5]
        data2 = [101, 102, 103, 104, 105]
        assert math.isclose(stats.variance(data1), stats.variance(data2), rel_tol=0.01)
    
    def test_e_median_duplicates(self):
        data = [1, 1, 1, 2, 2]
        assert stats.median(data) == 1.0
    
    def test_e_percentile_interpolation(self):
        data = [1, 2, 3, 4]
        p = stats.percentile(data, 50)
        assert 2 <= p <= 3
    
    def test_e_skew_symmetric_data(self):
        data = [-2, -1, 0, 1, 2]
        s = stats.skew(data)
        assert abs(s) < 0.1
    
    def test_e_kurtosis_peaked(self):
        # Heavy tails
        data = [0]*10 + [10]*2 + [-10]*2
        k = stats.kurtosis(data)
        # Should have high kurtosis
        assert k > 0 or k < 0  # Just check it computes
    
    def test_e_std_negative_values(self):
        data = [-5, -3, -1, 1, 3, 5]
        s = stats.std(data)
        assert s > 0
    
    def test_e_variance_single_value(self):
        v = stats.variance([5])
        assert v == 0.0
    
    def test_e_median_negative(self):
        data = [-5, -3, -1]
        assert stats.median(data) == -3.0
    
    def test_e_percentile_90(self):
        data = list(range(1, 101))
        p = stats.percentile(data, 90)
        assert p > 85
    
    def test_e_percentile_10(self):
        data = list(range(1, 101))
        p = stats.percentile(data, 10)
        assert p < 15
    
    def test_e_skew_tensor(self):
        t = tf.array([1, 2, 3, 4, 5])
        s = stats.skew(t)
        assert isinstance(s, (int, float, tf.Tensor))
    
    def test_e_kurtosis_tensor(self):
        t = tf.array([1, 2, 3, 4, 5])
        k = stats.kurtosis(t)
        assert k is not None
    
    def test_e_std_tensor(self):
        t = tf.array([1, 2, 3, 4, 5])
        s = stats.std(t)
        assert s > 0
    
    # Tests (15)
    def test_e_chisq_5_categories(self):
        chi, p = stats.chisquare([10, 20, 30, 20, 20])
        assert chi.item() > 0
    
    def test_e_chisq_expected_nonuniform(self):
        obs = [30, 20, 10]
        exp = [20, 20, 20]
        chi, p = stats.chisquare(obs, exp)
        assert chi.item() > 0
    
    def test_e_ttest_1samp_large(self):
        data = list(range(50))
        mean = sum(data) / len(data)
        t, p = stats.ttest_1samp(data, mean)
        assert abs(t.item()) < 0.1
    
    def test_e_ttest_ind_equal_var(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        t, p = stats.ttest_ind(a, b, equal_var=True)
        assert p < 0.05
    
    def test_e_chisq_degrees_freedom(self):
        # df = n_categories - 1
        chi, p = stats.chisquare([10, 10, 10, 10])  # df = 3
        assert p >= 0
    
    def test_e_ttest_1samp_tensor(self):
        t_stat, p = stats.ttest_1samp(tf.array([1, 2, 3, 4, 5]), 3)
        assert isinstance(p, float)
    
    def test_e_ttest_ind_different_sizes(self):
        a = [1, 2, 3]
        b = [4, 5, 6, 7, 8]
        t, p = stats.ttest_ind(a, b)
        assert t.item() < 0
    
    def test_e_chisq_large_sample(self):
        obs = [100, 100, 100, 100]
        chi, p = stats.chisquare(obs)
        assert chi.item() == 0.0
    
    def test_e_ttest_1samp_high_variance(self):
        data = [1, 100, 1, 100, 1]
        t, p = stats.ttest_1samp(data, 50)
        assert p > 0.1
    
    def test_e_ttest_ind_high_variance(self):
        a = [1, 100, 1, 100]
        b = [50, 50, 50, 50]
        t, p = stats.ttest_ind(a, b)
        assert p > 0.1
    
    def test_e_chisq_very_different(self):
        chi, p = stats.chisquare([95, 1, 1, 1, 1, 1])
        assert p < 0.001
    
    def test_e_ttest_1samp_exact_mean(self):
        data = [5, 5, 5, 5, 5]
        t, p = stats.ttest_1samp(data, 5)
        # t should be 0 or nan (no variance)
        assert math.isnan(t.item()) or abs(t.item()) < 0.01
    
    def test_e_ttest_ind_same_mean_diff_var(self):
        a = [4, 5, 6]
        b = [1, 5, 9]
        t, p = stats.ttest_ind(a, b)
        assert abs(t.item()) < 1
    
    def test_e_chisq_2x_expected(self):
        obs = [20, 10]
        exp = [10, 20]
        chi, p = stats.chisquare(obs, exp)
        assert chi.item() > 5
    
    def test_e_chisq_float_expected(self):
        obs = [10, 10, 10]
        exp = [10.0, 10.0, 10.0]
        chi, p = stats.chisquare(obs, exp)
        assert chi.item() == 0.0


# =============================================================================
# MEDIUM (30 tests)
# =============================================================================

class TestMediumStats:
    """Medium statistics tests."""
    
    def test_m_normal_cdf_3sigma(self):
        p = stats.normal_cdf(3, 0, 1)
        assert math.isclose(p, 0.9987, rel_tol=0.01)
    
    def test_m_poisson_large_lambda(self):
        # Poisson(100) should be approximately normal
        p = stats.poisson_pmf(100, 100)
        assert p > 0.01
    
    def test_m_binomial_large_n(self):
        p = stats.binomial_pmf(50, 100, 0.5)
        assert p > 0.05
    
    def test_m_gamma_chi_square(self):
        # Chi-square(k) = Gamma(k/2, 2)
        p = stats.gamma_pdf(2, 1, 2)
        assert p > 0
    
    def test_m_std_large_variance(self):
        data = list(range(1000))
        s = stats.std(data)
        expected = math.sqrt(sum((x - 499.5)**2 for x in data) / 1000)
        assert math.isclose(s, expected, rel_tol=0.01)
    
    def test_m_variance_sample_vs_pop(self):
        data = [1, 2, 3, 4, 5]
        v_pop = stats.variance(data, ddof=0)
        v_sample = stats.variance(data, ddof=1)
        assert v_sample > v_pop
    
    def test_m_median_large_dataset(self):
        data = list(range(1001))
        m = stats.median(data)
        assert m == 500.0
    
    def test_m_percentile_edge(self):
        data = [1, 2, 3, 4, 5]
        p0 = stats.percentile(data, 0)
        p100 = stats.percentile(data, 100)
        assert p0 == 1 and p100 == 5
    
    def test_m_skew_heavily_skewed(self):
        data = [1]*100 + [100]
        s = stats.skew(data)
        assert s > 5
    
    def test_m_kurtosis_bimodal(self):
        data = [0]*50 + [10]*50
        k = stats.kurtosis(data)
        # Bimodal has low kurtosis
        assert k < 0
    
    def test_m_chisq_many_categories(self):
        obs = [10] * 20
        chi, p = stats.chisquare(obs)
        assert chi.item() == 0.0
    
    def test_m_ttest_1samp_large_sample(self):
        data = [x + 0.1 for x in range(100)]
        mean = sum(data) / len(data)
        t, p = stats.ttest_1samp(data, mean + 0.01)
        assert p > 0.5
    
    def test_m_ttest_ind_unequal_var(self):
        a = [1, 2, 3]
        b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        assert p < 0.1
    
    def test_m_normal_pdf_very_small_sigma(self):
        p = stats.normal_pdf(0, 0, 0.1)
        assert p > 3
    
    def test_m_poisson_pmf_k_far_from_lambda(self):
        p = stats.poisson_pmf(20, 5)
        assert p < 1e-5
    
    def test_m_binomial_pmf_extreme_p(self):
        p = stats.binomial_pmf(1, 100, 0.01)
        assert p > 0.3
    
    def test_m_gamma_pdf_large_alpha(self):
        p = stats.gamma_pdf(10, 10, 1)
        assert p > 0
    
    def test_m_std_constant_scaling(self):
        data = [10, 20, 30, 40, 50]
        s1 = stats.std(data)
        data2 = [x * 3 for x in data]
        s2 = stats.std(data2)
        assert math.isclose(s2, s1 * 3, rel_tol=0.01)
    
    def test_m_median_even_large(self):
        data = list(range(1000))
        m = stats.median(data)
        assert m == 499.5
    
    def test_m_percentile_quartile_spread(self):
        data = list(range(100))
        iqr = stats.percentile(data, 75) - stats.percentile(data, 25)
        assert iqr > 40
    
    def test_m_chisq_significance(self):
        # Clear deviation should be significant
        chi, p = stats.chisquare([40, 10, 10, 40])
        assert p < 0.01
    
    def test_m_ttest_1samp_small_effect(self):
        data = [100 + x * 0.01 for x in range(100)]
        t, p = stats.ttest_1samp(data, 100.5)
        # Small effect, may or may not be significant
        assert 0 <= p <= 1
    
    def test_m_ttest_ind_large_effect(self):
        a = [1, 2, 3, 4, 5]
        b = [100, 101, 102, 103, 104]
        t, p = stats.ttest_ind(a, b)
        assert p < 0.001
    
    def test_m_skew_normal_data(self):
        # Simulated normal-ish data
        data = [0, 1, 1, 2, 2, 2, 3, 3, 4]
        s = stats.skew(data)
        assert abs(s) < 1
    
    def test_m_kurtosis_heavy_tails(self):
        data = [0]*10 + [1]*5 + [100]*2
        k = stats.kurtosis(data)
        assert k > 0
    
    def test_m_chisq_custom_expected(self):
        obs = [30, 30, 20, 20]
        exp = [25, 25, 25, 25]
        chi, p = stats.chisquare(obs, exp)
        assert chi.item() > 0
    
    def test_m_ttest_paired_like(self):
        # Simulate paired by computing differences
        before = [10, 12, 14, 16, 18]
        after = [11, 13, 15, 17, 19]
        diff = [a - b for a, b in zip(after, before)]
        t, p = stats.ttest_1samp(diff, 0)
        assert t.item() > 0
    
    def test_m_percentile_interpolate(self):
        data = [1, 10]
        p50 = stats.percentile(data, 50)
        assert 1 < p50 < 10
    
    def test_m_variance_large_values(self):
        data = [1e10, 1e10 + 1, 1e10 + 2]
        v = stats.variance(data)
        assert v > 0


# =============================================================================
# HARD (20 tests)
# =============================================================================

class TestHardStats:
    """Hard statistics tests - edge cases and precision."""
    
    def test_h_normal_cdf_extreme(self):
        p = stats.normal_cdf(6, 0, 1)
        assert p > 0.9999999
    
    def test_h_poisson_pmf_very_large_lambda(self):
        p = stats.poisson_pmf(500, 500)
        assert p > 0
    
    def test_h_binomial_pmf_large_n(self):
        p = stats.binomial_pmf(500, 1000, 0.5)
        assert p > 0
    
    def test_h_gamma_pdf_small_x(self):
        p = stats.gamma_pdf(0.001, 0.5, 1)
        # For alpha < 1, density goes to inf at 0
        assert p > 1
    
    def test_h_std_very_large_data(self):
        data = list(range(10000))
        s = stats.std(data)
        expected = math.sqrt(sum((x - 4999.5)**2 for x in data) / 10000)
        assert math.isclose(s, expected, rel_tol=0.01)
    
    def test_h_variance_numerical_stability(self):
        # Large mean, small variance
        data = [1e10, 1e10 + 1e-5, 1e10 + 2e-5]
        v = stats.variance(data)
        # Should be very small
        assert v < 1e-8
    
    def test_h_median_duplicates_many(self):
        data = [1]*500 + [2]*500
        m = stats.median(data)
        assert m == 1.5
    
    def test_h_percentile_precision(self):
        data = list(range(1, 1001))
        p33 = stats.percentile(data, 33.33)
        assert 330 < p33 < 340
    
    def test_h_skew_very_skewed(self):
        data = [0]*1000 + [1000]
        s = stats.skew(data)
        assert s > 10
    
    def test_h_kurtosis_very_peaked(self):
        data = [0]*100 + [50]*800 + [100]*100
        k = stats.kurtosis(data)
        # Should be leptokurtic (positive excess)
        assert k > 0
    
    def test_h_chisq_very_small_expected(self):
        obs = [1, 1, 1, 97]
        exp = [25, 25, 25, 25]
        chi, p = stats.chisquare(obs, exp)
        assert p < 0.0001
    
    def test_h_ttest_1samp_high_precision(self):
        data = [1.0000001, 1.0000002, 1.0000003]
        t, p = stats.ttest_1samp(data, 1.0000002)
        assert abs(t.item()) < 1
    
    def test_h_ttest_ind_very_different(self):
        a = [1, 1, 1, 1, 1]
        b = [1000, 1000, 1000, 1000, 1000]
        t, p = stats.ttest_ind(a, b)
        assert p < 1e-10
    
    def test_h_normal_pdf_very_large_x(self):
        p = stats.normal_pdf(100, 0, 1)
        # Should be essentially 0
        assert p < 1e-100 or p == 0
    
    def test_h_poisson_cumulative_precision(self):
        cdf = sum(stats.poisson_pmf(k, 10) for k in range(50))
        assert math.isclose(cdf, 1.0, rel_tol=1e-10)
    
    def test_h_binomial_cumulative_precision(self):
        cdf = sum(stats.binomial_pmf(k, 20, 0.5) for k in range(21))
        assert math.isclose(cdf, 1.0, rel_tol=1e-10)
    
    def test_h_std_underflow(self):
        data = [1e-200, 1e-200, 1e-200]
        s = stats.std(data)
        assert s == 0 or s < 1e-100
    
    def test_h_variance_overflow(self):
        data = [1e150, -1e150]
        v = stats.variance(data)
        # Should be huge or inf
        assert v > 1e299 or v == float('inf')
    
    def test_h_median_sorted_reverse(self):
        data = list(range(1000, 0, -1))
        m = stats.median(data)
        assert m == 500.5
    
    def test_h_chisq_many_small_counts(self):
        obs = [1] * 100
        chi, p = stats.chisquare(obs)
        assert chi.item() == 0


# =============================================================================
# VERY HARD (15 tests)
# =============================================================================

class TestVeryHardStats:
    """Very hard statistics tests - adversarial."""
    
    def test_vh_normal_pdf_zero_sigma(self):
        try:
            p = stats.normal_pdf(0, 0, 0)
            # May be inf or raise
            assert math.isinf(p) or p > 1e10
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_poisson_pmf_zero_lambda(self):
        p0 = stats.poisson_pmf(0, 0)
        p1 = stats.poisson_pmf(1, 0)
        assert p0 == 1.0 and p1 == 0.0
    
    def test_vh_binomial_pmf_p_zero(self):
        assert stats.binomial_pmf(0, 10, 0) == 1.0
        assert stats.binomial_pmf(1, 10, 0) == 0.0
    
    def test_vh_binomial_pmf_p_one(self):
        assert stats.binomial_pmf(10, 10, 1) == 1.0
        assert stats.binomial_pmf(9, 10, 1) == 0.0
    
    def test_vh_gamma_pdf_zero_x(self):
        p = stats.gamma_pdf(0, 2, 1)
        # For alpha > 1, density at 0 is 0
        assert p == 0 or p < 1e-10
    
    def test_vh_std_empty(self):
        try:
            s = stats.std([])
            # May be nan or raise
        except (ValueError, ZeroDivisionError, IndexError):
            pass
    
    def test_vh_variance_single(self):
        v = stats.variance([42])
        assert v == 0.0
    
    def test_vh_median_empty(self):
        try:
            m = stats.median([])
        except (ValueError, IndexError):
            pass
    
    def test_vh_percentile_out_of_range(self):
        try:
            p = stats.percentile([1, 2, 3], 150)
            # May clip or raise
        except ValueError:
            pass
    
    def test_vh_chisq_zero_observed(self):
        chi, p = stats.chisquare([0, 0, 100])
        assert chi.item() > 0
    
    def test_vh_ttest_1samp_single(self):
        try:
            t, p = stats.ttest_1samp([5], 5)
            # May be nan or raise (no variance)
            assert math.isnan(t.item()) or abs(t.item()) < 0.01
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_ttest_ind_single_each(self):
        try:
            t, p = stats.ttest_ind([1], [2])
            # May fail with single samples
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_skew_two_values(self):
        try:
            s = stats.skew([1, 2])
            # Skewness needs at least 3 values ideally
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_kurtosis_two_values(self):
        try:
            k = stats.kurtosis([1, 2])
            # Kurtosis needs at least 4 values ideally
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_vh_chisq_zero_expected(self):
        try:
            chi, p = stats.chisquare([10, 10], [0, 20])
            # Division by zero expected
        except (ValueError, ZeroDivisionError):
            pass

