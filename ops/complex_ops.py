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
loom Complex Number Operations.

This module implements complex number operations for tensors.

ESTABLISHED FACTS (Phase 1 Week 10):
- Complex64 uses float32 components
- Complex128 uses float64 components
- Operations: conj, real, imag, abs, angle, polar
- Pure Python implementation using Python's native complex type

REFERENCE DOCUMENTATION:
- gap_analysis_complete.md (CRITICAL GAP #4: Complex number support)
- NumPy complex documentation

PHASE STATUS: Phase 1 Week 10 - IMPLEMENTED
"""

import cmath
import math
from typing import List, Tuple, Any, Union
from loom.ops.arithmetic import UnaryOp
from loom.core.shape import Shape


class ConjOp(UnaryOp):
    """Complex conjugate operation."""
    
    @property
    def name(self) -> str:
        return "conj"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        return [x.conjugate() if isinstance(x, complex) else x for x in data]


class RealOp(UnaryOp):
    """Extract real part of complex numbers."""
    
    @property
    def name(self) -> str:
        return "real"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        return [x.real if isinstance(x, complex) else x for x in data]


class ImagOp(UnaryOp):
    """Extract imaginary part of complex numbers."""
    
    @property
    def name(self) -> str:
        return "imag"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        return [x.imag if isinstance(x, complex) else 0.0 for x in data]


class ComplexAbsOp(UnaryOp):
    """Absolute value (magnitude) of complex numbers."""
    
    @property
    def name(self) -> str:
        return "complex_abs"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        return [abs(x) for x in data]


class AngleOp(UnaryOp):
    """Phase angle of complex numbers in radians."""
    
    @property
    def name(self) -> str:
        return "angle"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        result = []
        for x in data:
            if isinstance(x, complex):
                result.append(cmath.phase(x))
            else:
                result.append(0.0 if x >= 0 else math.pi)
        return result


# =============================================================================
# SINGLETON OPERATIONS
# =============================================================================

CONJ = ConjOp()
REAL = RealOp()
IMAG = ImagOp()
COMPLEX_ABS = ComplexAbsOp()
ANGLE = AngleOp()


# =============================================================================
# CONVENIENCE FUNCTIONS (return Tensor)
# =============================================================================

def conj(t) -> 'Tensor':
    """
    Complex conjugate.
    """
    from loom.core.tensor import Tensor
    return Tensor(
        shape=t.shape.dims,
        dtype=t.dtype,
        _op=CONJ,
        _args=(t,),
    )


def real(t) -> 'Tensor':
    """
    Real part of complex values.
    """
    from loom.core.tensor import Tensor
    from loom.core.dtype import DType
    return Tensor(
        shape=t.shape.dims,
        dtype=DType.FLOAT64,
        _op=REAL,
        _args=(t,),
    )


def imag(t) -> 'Tensor':
    """
    Imaginary part of complex values.
    """
    from loom.core.tensor import Tensor
    from loom.core.dtype import DType
    return Tensor(
        shape=t.shape.dims,
        dtype=DType.FLOAT64,
        _op=IMAG,
        _args=(t,),
    )


def angle(t) -> 'Tensor':
    """
    Phase angle in radians.
    """
    from loom.core.tensor import Tensor
    from loom.core.dtype import DType
    return Tensor(
        shape=t.shape.dims,
        dtype=DType.FLOAT64,
        _op=ANGLE,
        _args=(t,),
    )


def polar(magnitude, angle_rad=None) -> Any:
    """
    Polar coordinates support.
    
    If one argument provided: decomposes complex tensor to (r, theta).
    If two arguments provided: creates complex tensor from polar coordinates (r, theta).
    
    Args:
        magnitude: Complex tensor (if 1 arg) or r values (if 2 args)
        angle_rad: theta values in radians (if 2 args)
        
    Returns:
        Tuple (r, theta) or Complex tensor
    """
    if angle_rad is None:
        return to_polar(magnitude)
        
    from loom.core.tensor import Tensor, array
    from loom.core.dtype import DType
    
    # Compute mag * (cos(angle) + i*sin(angle))
    m_data = magnitude.compute() if hasattr(magnitude, 'compute') else ([magnitude] if not isinstance(magnitude, list) else magnitude)
    a_data = angle_rad.compute() if hasattr(angle_rad, 'compute') else ([angle_rad] if not isinstance(angle_rad, list) else angle_rad)
    
    result = []
    # Basic zip loop
    for m, a in zip(m_data, a_data):
        result.append(complex(m * math.cos(a), m * math.sin(a)))
    
    shape = magnitude.shape if hasattr(magnitude, 'shape') else Shape(())
    return Tensor(result, shape=shape.dims, dtype=DType.COMPLEX128)


def rect(magnitude, angle_rad) -> 'Tensor':
    """
    Create complex from polar coordinates (r, theta).
    Alias for polar(r, theta).
    """
    return polar(magnitude, angle_rad)


def to_polar(z) -> Tuple['Tensor', 'Tensor']:
    """
    Convert complex tensor to polar coordinates (r, theta).
    """
    return z.abs(), angle(z)

