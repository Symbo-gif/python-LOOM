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
loom Data Type System.

This module defines the type system for loom tensors.

ESTABLISHED FACTS:
- DType enum ensures type safety at construction
- Supports: float32, float64, int32, int64, complex64, complex128, bool, symbolic
- Pure Python implementation (no NumPy dependency)

DESIGN DECISIONS:
- Use enum for compile-time type checking
- "symbolic" dtype indicates unevaluated symbolic expression
- "hybrid" dtype indicates mixed symbolic-numeric content

REFERENCE DOCUMENTATION:
- loom-native-complete.md Section 2.1

PHASE STATUS: Phase 0 (Skeleton)
"""

from enum import Enum
from typing import Dict, Type


class DType(Enum):
    """
    Unified data type system for loom.
    
    Supports both numerical types (like NumPy) and symbolic types (like SymPy).
    
    Numerical Types:
        FLOAT32, FLOAT64: Single and double precision floating point
        INT32, INT64: 32-bit and 64-bit signed integers
        COMPLEX64, COMPLEX128: Complex numbers with float32/float64 components
        BOOL: Boolean values
    
    Symbolic Types:
        SYMBOLIC: Unevaluated symbolic expression
        HYBRID: Mixed symbolic and numeric content
    
    Example:
        >>> from loom.core import DType
        >>> DType.FLOAT32
        <DType.FLOAT32: 'float32'>
        >>> DType.FLOAT32.value
        'float32'
    """
    
    # Floating point types
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    
    # Integer types
    INT32 = "int32"
    INT64 = "int64"
    
    # Complex types
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    
    # Boolean
    BOOL = "bool"
    
    # Symbolic types (unique to loom)
    SYMBOLIC = "symbolic"
    HYBRID = "hybrid"


# =============================================================================
# TYPE PROPERTIES
# =============================================================================

# Byte sizes for each dtype (used for memory calculations)
DTYPE_SIZES: Dict[DType, int] = {
    DType.FLOAT32: 4,
    DType.FLOAT64: 8,
    DType.INT32: 4,
    DType.INT64: 8,
    DType.COMPLEX64: 8,
    DType.COMPLEX128: 16,
    DType.BOOL: 1,
    DType.SYMBOLIC: 0,  # Variable size
    DType.HYBRID: 0,    # Variable size
}

# Python types corresponding to each dtype
DTYPE_PYTHON_TYPES: Dict[DType, Type] = {
    DType.FLOAT32: float,
    DType.FLOAT64: float,
    DType.INT32: int,
    DType.INT64: int,
    DType.COMPLEX64: complex,
    DType.COMPLEX128: complex,
    DType.BOOL: bool,
    DType.SYMBOLIC: object,  # Expression objects
    DType.HYBRID: object,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_dtype(dtype_input) -> DType:
    """
    Parse various dtype inputs into DType enum.
    
    Args:
        dtype_input: Can be:
            - DType enum member
            - String like "float32", "float64", etc.
            - None (returns default FLOAT32)
    
    Returns:
        DType enum member
    
    Raises:
        ValueError: If dtype string is not recognized
        TypeError: If input type is not supported
    
    Example:
        >>> parse_dtype("float32")
        <DType.FLOAT32: 'float32'>
        >>> parse_dtype(DType.INT64)
        <DType.INT64: 'int64'>
    """
    if dtype_input is None:
        return DType.FLOAT64
    
    if isinstance(dtype_input, DType):
        return dtype_input
    
    if isinstance(dtype_input, str):
        dtype_str = dtype_input.lower()
        try:
            return DType(dtype_str)
        except ValueError:
            # Try uppercase enum name
            try:
                return DType[dtype_str.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown dtype: '{dtype_input}'. "
                    f"Valid dtypes: {[d.value for d in DType]}"
                )
    
    raise TypeError(
        f"dtype must be DType enum or string, got {type(dtype_input).__name__}"
    )


def get_dtype_size(dtype: DType) -> int:
    """Return byte size for dtype (0 for variable-size types)."""
    return DTYPE_SIZES.get(dtype, 0)


def is_numeric_dtype(dtype: DType) -> bool:
    """Return True if dtype is numerical (not symbolic)."""
    return dtype not in (DType.SYMBOLIC, DType.HYBRID)


def is_complex_dtype(dtype: DType) -> bool:
    """Return True if dtype is complex-valued."""
    return dtype in (DType.COMPLEX64, DType.COMPLEX128)


def is_floating_dtype(dtype: DType) -> bool:
    """Return True if dtype is floating-point."""
    return dtype in (DType.FLOAT32, DType.FLOAT64)


def is_integer_dtype(dtype: DType) -> bool:
    """Return True if dtype is integer."""
    return dtype in (DType.INT32, DType.INT64)


# =============================================================================
# CONVENIENCE TYPE ALIASES (for backward compatibility)
# =============================================================================

# These aliases allow code like `from loom.core.dtype import float32, int64`
# to work similar to numpy

float32 = DType.FLOAT32
float64 = DType.FLOAT64
int32 = DType.INT32
int64 = DType.INT64
complex64 = DType.COMPLEX64
complex128 = DType.COMPLEX128

__all__ = [
    'DType',
    'parse_dtype',
    'get_dtype_size',
    'is_numeric_dtype',
    'is_complex_dtype',
    'is_floating_dtype',
    'is_integer_dtype',
    # Convenience aliases
    'float32',
    'float64',
    'int32',
    'int64',
    'complex64',
    'complex128',
]

