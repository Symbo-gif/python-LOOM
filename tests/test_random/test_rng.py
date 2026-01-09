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
Comprehensive tests for random number generation.

PHASE STATUS: Phase 1 Weeks 8-9 - IMPLEMENTED

Test Coverage:
- Basic generation (rand, randn, randint)
- Distributions (uniform, normal, exponential, poisson)
- Sampling (choice, permutation)
- Reproducibility (seed)
- Statistical validation
- Stress tests
"""

import pytest
from loom import random
from loom.random import seed, rand, randn, randint, uniform, normal, exponential, poisson, choice, permutation


# =============================================================================
# BASIC GENERATION TESTS
# =============================================================================

class TestBasicGeneration:
    """Test basic random generation."""
    
    def test_rand_shape(self):
        """rand() returns correct shape."""
        t = rand(5)
        assert t.shape.dims == (5,)
        
    def test_rand_2d_shape(self):
        """rand() with 2D shape."""
        t = rand(3, 4)
        assert t.shape.dims == (3, 4)
    
    def test_rand_range(self):
        """rand() values in [0, 1)."""
        seed(42)
        t = rand(100)
        data = t.compute()
        assert all(0 <= x < 1 for x in data)
    
    def test_randn_shape(self):
        """randn() returns correct shape."""
        t = randn(10)
        assert t.shape.dims == (10,)
    
    def test_randint_basic(self):
        """randint() basic usage."""
        seed(42)
        t = randint(10, size=(20,))
        data = t.compute()
        assert all(0 <= x < 10 for x in data)
    
    def test_randint_range(self):
        """randint() with low and high."""
        seed(42)
        t = randint(5, 10, size=(20,))
        data = t.compute()
        assert all(5 <= x < 10 for x in data)


# =============================================================================
# DISTRIBUTION TESTS
# =============================================================================

class TestDistributions:
    """Test distribution functions."""
    
    def test_uniform_range(self):
        """uniform() values in [low, high)."""
        seed(42)
        t = uniform(10, 20, (100,))
        data = t.compute()
        assert all(10 <= x < 20 for x in data)
    
    def test_normal_mean(self):
        """normal() approximate mean."""
        seed(42)
        t = normal(100, 1, (10000,))
        data = t.compute()
        mean = sum(data) / len(data)
        assert abs(mean - 100) < 1  # Within 1 of expected mean
    
    def test_exponential_positive(self):
        """exponential() all positive."""
        seed(42)
        t = exponential(1.0, (100,))
        data = t.compute()
        assert all(x > 0 for x in data)
    
    def test_poisson_non_negative(self):
        """poisson() all non-negative integers."""
        seed(42)
        t = poisson(5.0, (100,))
        data = t.compute()
        assert all(x >= 0 and isinstance(x, int) for x in data)


# =============================================================================
# SAMPLING TESTS
# =============================================================================

class TestSampling:
    """Test sampling functions."""
    
    def test_choice_from_list(self):
        """choice() from list."""
        seed(42)
        t = choice([10, 20, 30, 40], size=5)
        data = t.compute()
        assert all(x in [10, 20, 30, 40] for x in data)
    
    def test_choice_from_int(self):
        """choice() from integer (range)."""
        seed(42)
        t = choice(5, size=10)
        data = t.compute()
        assert all(0 <= x < 5 for x in data)
    
    def test_permutation_complete(self):
        """permutation() contains all indices."""
        seed(42)
        t = permutation(5)
        data = sorted(t.compute())
        assert data == [0, 1, 2, 3, 4]


# =============================================================================
# REPRODUCIBILITY TESTS
# =============================================================================

class TestReproducibility:
    """Test seed-based reproducibility."""
    
    def test_same_seed_same_values(self):
        """Same seed produces same values."""
        seed(12345)
        a = rand(10).compute()
        
        seed(12345)
        b = rand(10).compute()
        
        assert a == b
    
    def test_different_seed_different_values(self):
        """Different seeds produce different values."""
        seed(111)
        a = rand(10).compute()
        
        seed(222)
        b = rand(10).compute()
        
        assert a != b


# =============================================================================
# STATISTICAL VALIDATION TESTS
# =============================================================================

class TestStatisticalValidation:
    """Statistical validation of distributions."""
    
    def test_uniform_distribution(self):
        """Uniform distribution is roughly uniform."""
        seed(42)
        t = rand(10000)
        data = t.compute()
        
        # Check quantiles
        sorted_data = sorted(data)
        q25 = sorted_data[2500]
        q50 = sorted_data[5000]
        q75 = sorted_data[7500]
        
        assert 0.20 < q25 < 0.30
        assert 0.45 < q50 < 0.55
        assert 0.70 < q75 < 0.80
    
    def test_normal_std(self):
        """Normal distribution has approximately correct std."""
        seed(42)
        t = normal(0, 5, (10000,))
        data = t.compute()
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = variance ** 0.5
        
        # Should be close to 5
        assert 4.5 < std < 5.5


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for random generation."""
    
    def test_large_generation(self):
        """Generate large number of values."""
        seed(42)
        t = rand(100000)
        data = t.compute()
        assert len(data) == 100000
    
    def test_many_calls(self):
        """Many successive calls."""
        seed(42)
        for _ in range(1000):
            t = rand(10)
            data = t.compute()
            assert len(data) == 10

