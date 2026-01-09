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
loom custom exceptions.

This module defines the exception hierarchy for loom.
All exceptions inherit from loomError for easy catching.

ESTABLISHED FACTS:
- All loom exceptions inherit from loomError
- Exception hierarchy follows domain-specific grouping
- Each exception includes descriptive docstring

DESIGN DECISION:
- Exceptions are defined early (Phase 0) to establish error handling patterns
- New exceptions should be added as modules are implemented
"""


class loomError(Exception):
    """
    Base exception for all loom errors.
    
    Catch this to handle any loom-specific error:
        try:
            result = tensor.compute()
        except loomError as e:
            print(f"loom error: {e}")
    """
    pass


# =============================================================================
# SHAPE AND TYPE ERRORS
# =============================================================================

class ShapeError(loomError):
    """
    Raised when tensor shapes are incompatible.
    
    Examples:
        - Broadcasting failure: shapes (3, 4) and (5,) cannot be broadcast
        - Matrix multiplication: shapes (3, 4) and (5, 6) are incompatible
        - Shape mismatch in element-wise operations
    """
    pass


class DTypeError(loomError):
    """
    Raised when data types are incompatible or invalid.
    
    Examples:
        - Unknown dtype string: "float33"
        - Type mismatch: cannot add symbolic and numeric without conversion
        - Invalid type for operation: complex numbers for integer-only op
    """
    pass


# =============================================================================
# COMPUTATION ERRORS
# =============================================================================

class ComputationError(loomError):
    """
    Raised during tensor computation.
    
    Examples:
        - Division by zero
        - Numerical overflow/underflow
        - Backend execution failure
    """
    pass


class CacheError(loomError):
    """
    Raised when caching operations fail.
    
    Examples:
        - Cache size exceeded
        - Cache lookup failure
        - Invalid cache state
    """
    pass


# =============================================================================
# SYMBOLIC ERRORS
# =============================================================================

class SymbolicError(loomError):
    """
    Raised in symbolic operations.
    
    Examples:
        - Cannot evaluate expression with free symbols
        - Simplification rule failure
        - Invalid symbolic operation
    """
    pass


class DifferentiationError(SymbolicError):
    """
    Raised during differentiation.
    
    Examples:
        - Cannot differentiate non-differentiable function
        - Variable not found in expression
        - Higher-order derivative failure
    """
    pass


# =============================================================================
# OPTIMIZATION AND INTEGRATION ERRORS
# =============================================================================

class OptimizationError(loomError):
    """
    Raised in optimization routines.
    
    Examples:
        - Convergence failure
        - Invalid bounds or constraints
        - Singular Hessian
    """
    pass


class IntegrationError(loomError):
    """
    Raised in integration routines.
    
    Examples:
        - ODE solver failure
        - Step size too small
        - Stiff problem requires different solver
    """
    pass


# =============================================================================
# BACKEND ERRORS
# =============================================================================

class BackendError(loomError):
    """
    Raised when backend operations fail.
    
    Examples:
        - Backend not available
        - GPU memory allocation failure
        - Kernel compilation error
    """
    pass


class CUDAError(BackendError):
    """
    Raised for CUDA-specific errors.
    
    Examples:
        - CUDA not available
        - GPU out of memory
        - CUDA kernel launch failure
    """
    pass


# =============================================================================
# I/O ERRORS
# =============================================================================

class IOError(loomError):
    """
    Raised for file I/O operations.
    
    Examples:
        - File not found
        - Invalid file format
        - Corrupted data
    """
    pass


class FormatError(IOError):
    """
    Raised for data format errors.
    
    Examples:
        - Unknown file format
        - Version mismatch
        - Invalid header
    """
    pass

