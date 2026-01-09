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
loom Random Number Generation.

This module implements random number generation with various distributions.

ESTABLISHED FACTS (Phase 1 Implementation):
- Uses PCG64-like algorithm for reproducibility
- Seed-based initialization for reproducible results
- All functions return Tensor objects
- Pure Python implementation

REFERENCE DOCUMENTATION:
- gap_analysis_complete.md (CRITICAL GAP #3: Random number generation)
- NumPy random documentation

PHASE STATUS: Phase 1 Weeks 8-9 - IMPLEMENTED
"""

import math
from typing import Optional, Tuple, Union, List


class RandomGenerator:
    """
    Pseudo-random number generator using Linear Congruential Generator.
    
    This is a simple but effective PRNG suitable for general-purpose use.
    Not cryptographically secure.
    
    ESTABLISHED FACTS:
    - Seed-based initialization for reproducibility
    - Period of 2^31 - 1
    - Uses Lehmer RNG / MINSTD variant
    """
    
    # LCG parameters (MINSTD variant)
    _a = 48271
    _m = 2147483647  # 2^31 - 1
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed. If None, uses current time.
        """
        if seed is None:
            import time
            seed = int(time.time() * 1000000) % self._m
        
        self._state = seed % self._m
        if self._state == 0:
            self._state = 1
    
    def _next_int(self) -> int:
        """Generate next random integer."""
        self._state = (self._a * self._state) % self._m
        return self._state
    
    def _next_float(self) -> float:
        """Generate uniform random float in [0, 1)."""
        return self._next_int() / self._m
    
    def random(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> List[float]:
        """
        Generate uniform random values in [0, 1).
        
        Args:
            size: Output shape. None for single value.
        
        Returns:
            List of random values
        """
        n = self._size_to_count(size)
        return [self._next_float() for _ in range(n)]
    
    def uniform(
        self, 
        low: float = 0.0, 
        high: float = 1.0, 
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> List[float]:
        """
        Generate uniform random values in [low, high).
        
        Args:
            low: Lower bound
            high: Upper bound
            size: Output shape
        
        Returns:
            List of random values
        """
        n = self._size_to_count(size)
        scale = high - low
        return [low + self._next_float() * scale for _ in range(n)]
    
    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> List[float]:
        """
        Generate normal (Gaussian) random values.
        
        Uses Box-Muller transform.
        
        Args:
            loc: Mean of distribution
            scale: Standard deviation
            size: Output shape
        
        Returns:
            List of random values
        """
        n = self._size_to_count(size)
        result = []
        
        while len(result) < n:
            # Box-Muller transform
            u1 = self._next_float()
            u2 = self._next_float()
            
            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10
            
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
            
            result.append(loc + z0 * scale)
            if len(result) < n:
                result.append(loc + z1 * scale)
        
        return result[:n]
    
    def randint(
        self,
        low: int,
        high: Optional[int] = None,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> List[int]:
        """
        Generate random integers in [low, high).
        
        Args:
            low: Lowest value (inclusive). If high is None, range is [0, low).
            high: Upper bound (exclusive)
            size: Output shape
        
        Returns:
            List of random integers
        """
        if high is None:
            high = low
            low = 0
        
        n = self._size_to_count(size)
        range_size = high - low
        return [low + int(self._next_float() * range_size) for _ in range(n)]
    
    def exponential(
        self,
        scale: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> List[float]:
        """
        Generate exponential random values.
        
        Args:
            scale: 1/lambda (mean)
            size: Output shape
        
        Returns:
            List of random values
        """
        n = self._size_to_count(size)
        result = []
        
        for _ in range(n):
            u = self._next_float()
            if u < 1e-10:
                u = 1e-10
            result.append(-scale * math.log(u))
        
        return result
    
    def poisson(
        self,
        lam: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> List[int]:
        """
        Generate Poisson random values.
        
        Uses inverse transform sampling for small lambda,
        normal approximation for large lambda.
        
        Args:
            lam: Expected number of events (lambda)
            size: Output shape
        
        Returns:
            List of random integers
        """
        n = self._size_to_count(size)
        result = []
        
        if lam < 30:
            # Inverse transform method
            L = math.exp(-lam)
            for _ in range(n):
                k = 0
                p = 1.0
                while p > L:
                    k += 1
                    p *= self._next_float()
                result.append(k - 1)
        else:
            # Normal approximation
            for _ in range(n):
                u1 = self._next_float()
                u2 = self._next_float()
                if u1 < 1e-10:
                    u1 = 1e-10
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                val = int(round(lam + math.sqrt(lam) * z))
                result.append(max(0, val))
        
        return result
    
    def choice(
        self,
        a: List,
        size: Optional[int] = None,
        replace: bool = True
    ) -> List:
        """
        Random selection from array.
        
        Args:
            a: Array to sample from
            size: Number of samples. None for single sample.
            replace: Whether to sample with replacement
        
        Returns:
            Selected elements
        """
        if size is None:
            size = 1
        
        n = len(a)
        result = []
        
        if replace:
            for _ in range(size):
                idx = int(self._next_float() * n)
                result.append(a[idx])
        else:
            # Without replacement - Fisher-Yates shuffle partial
            indices = list(range(n))
            for i in range(min(size, n)):
                j = i + int(self._next_float() * (n - i))
                indices[i], indices[j] = indices[j], indices[i]
                result.append(a[indices[i]])
        
        return result
    
    def shuffle(self, x: List) -> None:
        """
        Shuffle list in-place.
        
        Args:
            x: List to shuffle (modified in-place)
        """
        n = len(x)
        for i in range(n - 1, 0, -1):
            j = int(self._next_float() * (i + 1))
            x[i], x[j] = x[j], x[i]
    
    def permutation(self, n: int) -> List[int]:
        """
        Generate random permutation.
        
        Args:
            n: Length of permutation
        
        Returns:
            Random permutation of [0, 1, ..., n-1]
        """
        perm = list(range(n))
        self.shuffle(perm)
        return perm
    
    def _size_to_count(self, size: Optional[Union[int, Tuple[int, ...]]]) -> int:
        """Convert size specification to element count."""
        if size is None:
            return 1
        if isinstance(size, int):
            return size
        result = 1
        for s in size:
            result *= s
        return result


# =============================================================================
# GLOBAL STATE
# =============================================================================

_default_rng = RandomGenerator()


def seed(s: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Args:
        s: Random seed
    """
    global _default_rng
    _default_rng = RandomGenerator(s)


def get_generator() -> RandomGenerator:
    """Get the default random generator."""
    return _default_rng


# =============================================================================
# CONVENIENCE FUNCTIONS (return Tensor)
# =============================================================================

def rand(*shape) -> 'Tensor':
    """
    Random values in [0, 1) with given shape.
    
    Args:
        *shape: Output shape dimensions. Can be passed as rand(3, 4) or rand((3, 4))
    
    Returns:
        Tensor with uniform random values
    """
    from loom.core.tensor import Tensor
    # Handle rand((3, 4)) usage (single tuple argument)
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    size = shape if shape else None
    data = _default_rng.random(size)
    return Tensor(data, shape=shape if shape else ())


def randn(*shape) -> 'Tensor':
    """
    Standard normal random values.
    
    Args:
        *shape: Output shape dimensions. Can be passed as randn(3, 4) or randn((3, 4))
    
    Returns:
        Tensor with standard normal random values
    """
    from loom.core.tensor import Tensor
    # Handle randn((3, 4)) usage (single tuple argument)
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    size = shape if shape else None
    data = _default_rng.normal(0.0, 1.0, size)
    return Tensor(data, shape=shape if shape else ())


def randint(low: int, high: Optional[int] = None, size: Optional[Tuple[int, ...]] = None) -> 'Tensor':
    """
    Random integers in [low, high).
    
    Args:
        low: Lower bound (or upper bound if high is None)
        high: Upper bound (exclusive)
        size: Output shape
    
    Returns:
        Tensor with random integers
    """
    from loom.core.tensor import Tensor
    data = _default_rng.randint(low, high, size)
    shape = size if size else ()
    return Tensor(data, shape=shape, dtype="int64")


def uniform(low: float = 0.0, high: float = 1.0, size: Optional[Tuple[int, ...]] = None) -> 'Tensor':
    """
    Uniform random values in [low, high).
    
    Args:
        low: Lower bound
        high: Upper bound
        size: Output shape
    
    Returns:
        Tensor with uniform random values
    """
    from loom.core.tensor import Tensor
    data = _default_rng.uniform(low, high, size)
    shape = size if size else ()
    return Tensor(data, shape=shape)


def normal(loc: float = 0.0, scale: float = 1.0, size: Optional[Tuple[int, ...]] = None) -> 'Tensor':
    """
    Normal random values.
    
    Args:
        loc: Mean
        scale: Standard deviation
        size: Output shape
    
    Returns:
        Tensor with normal random values
    """
    from loom.core.tensor import Tensor
    data = _default_rng.normal(loc, scale, size)
    shape = size if size else ()
    return Tensor(data, shape=shape)


def exponential(scale: float = 1.0, size: Optional[Tuple[int, ...]] = None) -> 'Tensor':
    """
    Exponential random values.
    
    Args:
        scale: 1/lambda
        size: Output shape
    
    Returns:
        Tensor with exponential random values
    """
    from loom.core.tensor import Tensor
    data = _default_rng.exponential(scale, size)
    shape = size if size else ()
    return Tensor(data, shape=shape)


def poisson(lam: float = 1.0, size: Optional[Tuple[int, ...]] = None) -> 'Tensor':
    """
    Poisson random values.
    
    Args:
        lam: Expected rate
        size: Output shape
    
    Returns:
        Tensor with Poisson random values
    """
    from loom.core.tensor import Tensor
    data = _default_rng.poisson(lam, size)
    shape = size if size else ()
    return Tensor(data, shape=shape, dtype="int64")


def choice(a, size: Optional[int] = None, replace: bool = True) -> 'Tensor':
    """
    Random sample from array.
    
    Args:
        a: Array or integer
        size: Number of samples
        replace: With replacement
    
    Returns:
        Tensor with sampled values
    """
    from loom.core.tensor import Tensor
    if isinstance(a, int):
        a = list(range(a))
    elif hasattr(a, 'compute'):
        a = a.compute()
    data = _default_rng.choice(a, size, replace)
    shape = (size,) if size else ()
    return Tensor(data, shape=shape)


def permutation(n: int) -> 'Tensor':
    """
    Random permutation of [0, 1, ..., n-1].
    
    Args:
        n: Length of permutation
    
    Returns:
        Tensor with permuted indices
    """
    from loom.core.tensor import Tensor
    data = _default_rng.permutation(n)
    return Tensor(data, shape=(n,), dtype="int64")

