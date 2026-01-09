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
loom Random Number Generation Module.

PHASE STATUS: Phase 1 Weeks 8-9 - IMPLEMENTED

Provides:
- rand(*shape): Uniform [0, 1)
- randn(*shape): Standard normal
- randint(low, high, size): Random integers
- uniform(low, high, size): Uniform [low, high)
- normal(loc, scale, size): Normal distribution
- exponential(scale, size): Exponential distribution
- poisson(lam, size): Poisson distribution
- choice(a, size, replace): Random sampling
- permutation(n): Random permutation
- seed(s): Set random seed
"""

from loom.random.rng import (
    RandomGenerator,
    seed,
    get_generator,
    rand,
    randn,
    randint,
    uniform,
    normal,
    exponential,
    poisson,
    choice,
    permutation,
)

__all__ = [
    "RandomGenerator",
    "seed",
    "get_generator",
    "rand",
    "randn",
    "randint",
    "uniform",
    "normal",
    "exponential",
    "poisson",
    "choice",
    "permutation",
]

