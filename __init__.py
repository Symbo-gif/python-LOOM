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
loom: Native Python Mathematical Computing Framework

Zero external dependencies. Pure Python with optional Cython/CUDA acceleration.

This package provides a unified framework for:
- Numerical computing (NumPy replacement)
- Symbolic algebra (SymPy replacement)
- Scientific computing (SciPy replacement)
- Agent-based distributed computation (ASTRA architecture)

PHASE STATUS: Phase 9 COMPLETE (Production Quality, Hardened)
"""

__version__ = "0.9.0"
__author__ = "loom Contributors"
__license__ = "Apache-2.0"

# =============================================================================
# CORE EXPORTS
# =============================================================================

from loom.core.tensor import (
    Tensor, 
    Symbol, 
    array, 
    zeros, 
    ones, 
    full, 
    eye
)
from loom.core.shape import Shape
from loom.core.dtype import DType

# =============================================================================
# OPERATION EXPORTS
# =============================================================================

from loom.ops.complex_ops import (
    conj, 
    real, 
    imag, 
    angle, 
    polar,
    rect
)

# Random is typically accessed via tf.random
from loom import random
# Convenience exports for common random functions
from loom.random import randn, rand, randint, seed

# Linear Algebra is accessed via tf.linalg
from loom import linalg

# Symbolic is accessed via tf.symbolic
from loom import symbolic

# Optimization and Integration (SciPy replacements)
from loom import optimize
from loom import integrate

# Signal, Interpolation, and Special Functions (Phase 5)
from loom import signal
from loom import interpolate
from loom import special
from loom import io

# Sparse and Spatial Algorithms (Phase 6)
from loom import sparse
from loom import spatial

# Field Tensors and Agent Orchestration (Phase 7)
from loom import field
from loom import agent

# Stats and Numeric Optimization (Phase 8)
from loom import stats

# =============================================================================
# NAMESPACE CONFIGURATION
# =============================================================================

from loom import config

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Tensor",
    "Symbol",
    "Shape",
    "DType",
    "array",
    "zeros",
    "ones",
    "full",
    "eye",
    "conj",
    "real",
    "imag",
    "angle",
    "polar",
    "random",
    "randn",
    "rand",
    "randint",
    "seed",
    "linalg",
    "symbolic",
    "optimize",
    "integrate",
    "signal",
    "interpolate",
    "special",
    "io",
    "sparse",
    "spatial",
    "field",
    "agent",
    "stats",
    "config",
]

