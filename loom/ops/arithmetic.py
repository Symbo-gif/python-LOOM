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
loom Arithmetic Operations.

This module implements element-wise arithmetic operations for tensors.

ESTABLISHED FACTS (Phase 1 Implementation):
- Operations use broadcasting for shape compatibility
- Operations create DAG nodes for lazy evaluation
- Scalar operands are automatically promoted to tensors
- All operations support both numeric and future symbolic modes

REFERENCE DOCUMENTATION:
- loom-native-complete.md (ops/ module)
- loom-three-must-haves.md (Broadcasting §1)

PHASE STATUS: Phase 1 - IMPLEMENTED
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, List
from loom.core.shape import Shape, broadcast_shapes
from loom.core.dtype import DType
import math
import cmath


class Operation(ABC):
    """
    Abstract base class for all tensor operations.
    
    Operations are the nodes in the computation DAG. Each operation
    defines how to:
    1. Infer output shape from inputs
    2. Execute the computation
    3. Compute gradients (for autodiff)
    
    ESTABLISHED FACTS:
    - Operations are stateless (no mutable internal state)
    - Operations work on raw data (lists), not Tensor objects
    - Shape inference happens at DAG construction time
    """



    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return operation name for debugging."""
        pass
    
    @abstractmethod
    def infer_shape(self, *args) -> Shape:
        """
        Infer output shape from input shapes.
        
        Args:
            *args: Input tensors or values
            
        Returns:
            Shape of output tensor
        """
        pass
    
    @abstractmethod
    def execute(self, *args) -> List:
        """
        Execute the operation on concrete data.
        
        Args:
            *args: Input data (flat lists)
            
        Returns:
            Output data (flat list)
        """
        pass
    
    def __repr__(self) -> str:
        return f"Operation({self.name})"


class BinaryOp(Operation):
    """
    Base class for binary element-wise operations.
    
    Handles broadcasting logic common to all binary ops.
    """
    
    def _get_data(self, arg):
        """Get computed data from a tensor or scalar.
        
        Checks _cached_result first (for DAG nodes), then _dense_data.
        """
        if not hasattr(arg, '_dense_data'):
            # Scalar value
            return [arg]
        
        # Check cached result first (for computed DAG nodes)
        if hasattr(arg, '_cached_result') and arg._cached_result is not None:
            return arg._cached_result
        
        # Fall back to dense data
        if arg._dense_data is not None:
            return arg._dense_data
        
        # Should not reach here if compute() was called first
        raise ValueError(f"Tensor has no computed data: {arg}")
    
    def infer_shape(self, a, b) -> Shape:
        """Infer output shape using broadcasting."""
        shape_a = a.shape if hasattr(a, 'shape') else Shape(())
        shape_b = b.shape if hasattr(b, 'shape') else Shape(())
        return broadcast_shapes(shape_a, shape_b)
    
    def _broadcast_and_apply(
        self, 
        a_data: List, 
        a_shape: Shape,
        b_data: List,
        b_shape: Shape,
        op_func
    ) -> Tuple[List, Shape]:
        """
        Apply binary operation with broadcasting.
        
        Args:
            a_data: Flat list of a's elements
            a_shape: Shape of a
            b_data: Flat list of b's elements  
            b_shape: Shape of b
            op_func: Function to apply elementwise
            
        Returns:
            (result_data, result_shape)
        """
        result_shape = broadcast_shapes(a_shape, b_shape)
        result_size = result_shape.size
        
        # Compute strides for indexing
        a_strides = self._compute_strides(a_shape.dims)
        b_strides = self._compute_strides(b_shape.dims)
        result_strides = self._compute_strides(result_shape.dims)
        
        result = []
        for flat_idx in range(result_size):
            # Convert flat index to multi-dimensional
            multi_idx = self._flat_to_multi(flat_idx, result_shape.dims)
            
            # Map to source array indices with broadcasting
            a_idx = self._broadcast_index(multi_idx, a_shape.dims, result_shape.dims)
            b_idx = self._broadcast_index(multi_idx, b_shape.dims, result_shape.dims)
            
            # Get flat indices in source arrays
            a_flat = self._multi_to_flat(a_idx, a_strides)
            b_flat = self._multi_to_flat(b_idx, b_strides)
            
            # Apply operation
            a_val = a_data[a_flat] if a_flat < len(a_data) else a_data[0]
            b_val = b_data[b_flat] if b_flat < len(b_data) else b_data[0]
            result.append(op_func(a_val, b_val))
        
        return result, result_shape
    
    def _compute_strides(self, dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute strides for row-major (C-style) indexing."""
        if not dims:
            return ()
        strides = [1]
        for d in reversed(dims[1:]):
            strides.append(strides[-1] * d)
        return tuple(reversed(strides))
    
    def _flat_to_multi(self, flat_idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional index (row-major/C-style)."""
        if not dims:
            return ()
        result = []
        # Iterate from rightmost dimension to leftmost
        for d in reversed(dims):
            result.append(flat_idx % d)
            flat_idx //= d
        return tuple(reversed(result))
    
    def _multi_to_flat(self, multi_idx: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
        """Convert multi-dimensional index to flat index."""
        if not multi_idx:
            return 0
        return sum(i * s for i, s in zip(multi_idx, strides))
    
    def _broadcast_index(
        self, 
        result_idx: Tuple[int, ...],
        source_dims: Tuple[int, ...],
        result_dims: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Map result index to source index with broadcasting."""
        # Pad source dims to match result dims
        pad_len = len(result_dims) - len(source_dims)
        padded_source = (1,) * pad_len + source_dims
        
        result = []
        for r_idx, s_dim in zip(result_idx, padded_source):
            # If source dim is 1, broadcast (use index 0)
            result.append(0 if s_dim == 1 else r_idx)
        
        # Remove padding from result
        return tuple(result[pad_len:])


class AddOp(BinaryOp):
    """Element-wise addition: a + b"""
    
    @property
    def name(self) -> str:
        return "add"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x + y
        )
        return result


class SubOp(BinaryOp):
    """Element-wise subtraction: a - b"""
    
    @property
    def name(self) -> str:
        return "sub"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x - y
        )
        return result


class MulOp(BinaryOp):
    """Element-wise multiplication: a * b"""
    
    @property
    def name(self) -> str:
        return "mul"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x * y
        )
        return result


class DivOp(BinaryOp):
    """Element-wise division: a / b"""
    
    @property
    def name(self) -> str:
        return "div"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x / y if y != 0 else float('inf') if x > 0 else float('-inf') if x < 0 else float('nan')
        )
        return result


class FloorDivOp(BinaryOp):
    """Element-wise floor division: a // b"""
    
    @property
    def name(self) -> str:
        return "floordiv"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x // y if y != 0 else 0
        )
        return result


class PowOp(BinaryOp):
    """Element-wise power: a ** b"""
    
    @property
    def name(self) -> str:
        return "pow"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x ** y
        )
        return result


class ModOp(BinaryOp):
    """Element-wise modulo: a % b"""
    
    @property
    def name(self) -> str:
        return "mod"
    
    def execute(self, a, b) -> List:
        a_data = self._get_data(a)
        b_data = self._get_data(b)
        a_shape = a.shape if hasattr(a, 'shape') else Shape(())
        b_shape = b.shape if hasattr(b, 'shape') else Shape(())
        
        result, _ = self._broadcast_and_apply(
            a_data, a_shape, b_data, b_shape,
            lambda x, y: x % y if y != 0 else 0
        )
        return result


class UnaryOp(Operation):
    """Base class for unary operations."""
    
    def _get_data(self, arg):
        """Get computed data from a tensor or scalar."""
        if not hasattr(arg, '_dense_data'):
            return [arg]
        if hasattr(arg, '_cached_result') and arg._cached_result is not None:
            return arg._cached_result
        if arg._dense_data is not None:
            return arg._dense_data
        raise ValueError(f"Tensor has no computed data: {arg}")
    
    def infer_shape(self, a) -> Shape:
        """Unary ops preserve shape."""
        return a.shape if hasattr(a, 'shape') else Shape(())


class NegOp(UnaryOp):
    """Element-wise negation: -a"""
    
    @property
    def name(self) -> str:
        return "neg"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        return [-x for x in a_data]


class AbsOp(UnaryOp):
    """Element-wise absolute value: abs(a)"""
    
    @property
    def name(self) -> str:
        return "abs"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        return [abs(x) for x in a_data]


class SqrtOp(UnaryOp):
    """Element-wise square root: sqrt(a)"""
    
    @property
    def name(self) -> str:
        return "sqrt"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        result = []
        for x in a_data:
            if isinstance(x, complex):
                result.append(cmath.sqrt(x))
            elif x < 0:
                result.append(float('nan'))
            else:
                result.append(math.sqrt(x))
        return result


class ExpOp(UnaryOp):
    """Element-wise exponential: exp(a)"""
    
    @property
    def name(self) -> str:
        return "exp"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        result = []
        for x in a_data:
            try:
                if isinstance(x, complex):
                     result.append(cmath.exp(x))
                else:
                     result.append(math.exp(x))
            except OverflowError:
                if isinstance(x, complex):
                     result.append(complex(float('inf'), float('inf'))) 
                else:
                     result.append(float('inf') if x > 0 else 0.0)
        return result


class LogOp(UnaryOp):
    """Element-wise natural logarithm: log(a)"""
    
    @property
    def name(self) -> str:
        return "log"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        result = []
        for x in a_data:
            if isinstance(x, complex):
                result.append(cmath.log(x))
            else:
                if x == 0:
                    result.append(float('-inf'))
                elif x < 0:
                    result.append(float('nan'))
                else:
                    try:
                        result.append(math.log(x))
                    except ValueError:
                         result.append(float('nan'))
        return result


class SinOp(UnaryOp):
    """Element-wise sine: sin(a)"""
    
    @property
    def name(self) -> str:
        return "sin"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        return [math.sin(x) for x in a_data]


class CosOp(UnaryOp):
    """Element-wise cosine: cos(a)"""
    
    @property
    def name(self) -> str:
        return "cos"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        return [math.cos(x) for x in a_data]


class TanOp(UnaryOp):
    """Element-wise tangent: tan(a)"""
    
    @property
    def name(self) -> str:
        return "tan"
    
    def execute(self, a) -> List:
        a_data = self._get_data(a)
        return [math.tan(x) for x in a_data]


# =============================================================================
# OPERATION REGISTRY
# =============================================================================

# Singleton instances for common operations
ADD = AddOp()
SUB = SubOp()
MUL = MulOp()
DIV = DivOp()
FLOORDIV = FloorDivOp()
POW = PowOp()
MOD = ModOp()
NEG = NegOp()
ABS = AbsOp()
SQRT = SqrtOp()
EXP = ExpOp()
LOG = LogOp()
SIN = SinOp()
COS = CosOp()
TAN = TanOp()

