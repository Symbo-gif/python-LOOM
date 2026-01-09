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
loom Base Tensor Class.

This module defines the core Tensor class that unifies:
- Dense numerical arrays (NumPy replacement)
- Symbolic expressions (SymPy replacement)
- Sparse matrices (SciPy replacement)
- Field tensors (ASTRA architecture)

ESTABLISHED FACTS (Phase 1):
- Tensor uses lazy evaluation via computation DAG
- Tensor is the single unified type for all data
- Shape is inferred at construction time
- Pure Python implementation
- Arithmetic operations (+, -, *, /, **, //, %) are fully implemented
- Broadcasting is supported for all binary operations

DESIGN DECISIONS:
- Multiple representations can coexist (dense + symbolic)
- Operations return new Tensor with DAG reference
- compute() triggers actual evaluation

REFERENCE DOCUMENTATION:
- loom-native-complete.md Section 2.1 (Main Tensor Class)
- loom-week-by-week.md Week 2 (Tensor Data Initialization)

PHASE STATUS: Phase 1 - IMPLEMENTED (arithmetic operations working)
"""

from typing import Union, Tuple, Optional, List, Any, Iterator, Callable
from loom.core.shape import Shape, broadcast_shapes
from loom.core.dtype import DType, parse_dtype
from loom.symbolic import Expression, SymbolExpr, differentiate, simplify as simplify_expr
from loom.numeric.storage import NumericBuffer
import loom.config as config


class Tensor:
    """
    Master tensor class: unifies dense, sparse, symbolic, field, and hybrid modes.
    
    Design Philosophy:
    - Lazy evaluation via computation DAG
    - Optional symbolic expression tree
    - Optional sparse representation
    - Optional field tensor (spatial coherence)
    - Pure Python core + optional backends
    
    Attributes:
        name: Human-readable name for debugging
        dtype: Data type (float32, symbolic, etc.)
        shape: Tensor shape (immutable)
        requires_grad: Whether to track gradients
    
    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
        >>> c = a + b  # Creates DAG node
        >>> result = c.compute()  # Evaluates: [6, 8, 10, 12]
    
    PHASE STATUS: Phase 1 - IMPLEMENTED
    """
    
    # Class-level unique ID counter
    _uid_counter = 0
    
    def __init__(
        self,
        data: Optional[Union[list, tuple, float, int, complex, 'Tensor']] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Union[str, DType] = None,
        name: Optional[str] = None,
        backend: str = "cpu",
        requires_grad: bool = False,
        symbolic_expr: Optional[Expression] = None,
        _op=None,
        _args=(),
    ):
        """
        Initialize tensor.
        
        Args:
            data: Numerical data (list, tuple, scalar, or Tensor)
            shape: Explicit shape (overrides inferred shape)
            dtype: Data type (string or DType enum)
            name: Human-readable name
            backend: Computation backend ("cpu", "cython", "cuda")
            requires_grad: Whether to track gradients
            _op: Internal - operation that produced this tensor
            _args: Internal - arguments to the operation
        
        Raises:
            ValueError: If shape cannot be inferred
            TypeError: If data type is not supported
        """
        # Generate unique ID
        Tensor._uid_counter += 1
        self._id = Tensor._uid_counter
        
        # Basic properties
        self.name = name or f"tensor_{self._id}"
        self.dtype = parse_dtype(dtype if dtype is not None else config.DEFAULT_DTYPE)
        # print(f"DEBUG: Tensor init input dtype={dtype}, parsed={self.dtype}")
        self.backend = backend
        self.requires_grad = requires_grad
        
        # Multiple representations (can coexist)
        self._dense_data: Optional[List] = None
        self._symbolic_expr = symbolic_expr
        self._sparse_data = None    # Will be sparse format (Phase 5)
        self._field_data = None     # Will be FieldData type (Phase 7)
        self._shape: Optional[Shape] = None
        
        # DAG for lazy evaluation
        self._op = _op        # Operation node
        self._args: Tuple = _args  # Argument tensors
        self._cached_result = None
        
        # Gradient tracking
        self._grad: Optional['Tensor'] = None
        
        # Initialize from data
        if data is not None:
            self._initialize_from_data(data)
        
        # Set explicit shape if provided (overrides inferred)
        if shape is not None:
            self._shape = Shape(tuple(shape))
        
        # Infer shape from operation if needed
        if self._shape is None and self._op is not None:
            self._shape = self._op.infer_shape(*self._args)
            
        # Infer shape from symbolic expression
        if self._shape is None and self._symbolic_expr is not None:
             self._shape = self._symbolic_expr.infer_shape()
        
        # Validate we have a shape
        if self._shape is None:
            raise ValueError(
                "Cannot infer shape from data. "
                "Provide either data or explicit shape."
            )
    
    def _initialize_from_data(self, data: Any) -> None:
        """Initialize tensor from data (list, tuple, scalar, or Tensor)."""
        if isinstance(data, Tensor):
            # Copy from another tensor (force computation if needed)
            dense = data.compute()
            if isinstance(dense, NumericBuffer):
                self._dense_data = dense.copy()
            else:
                self._dense_data = NumericBuffer(dense, self.dtype)
            self._shape = data._shape
            self.dtype = data.dtype
        elif isinstance(data, (list, tuple)):
            # Convert nested list/tuple to internal format
            flat_data, inferred_shape = self._flatten_and_shape(data)
            self._dense_data = NumericBuffer(flat_data, self.dtype)
            # Sync dtype in case buffer auto-upgraded (e.g., to complex)
            self.dtype = self._dense_data.dtype
            if inferred_shape:
                self._shape = Shape(inferred_shape)
            else:
                self._shape = Shape((len(flat_data),))
        elif isinstance(data, (int, float, complex)):
            # Scalar
            self._dense_data = NumericBuffer([data], self.dtype)
            # Sync dtype in case buffer auto-upgraded
            self.dtype = self._dense_data.dtype
            self._shape = Shape(())
        elif isinstance(data, NumericBuffer) or type(data).__name__ == 'NumericBuffer':
            # Fallback for mismatched module imports
            if hasattr(data, 'copy'):
                self._dense_data = data.copy()
            else:
                 self._dense_data = NumericBuffer(data, self.dtype)

            if self._shape is None:
                self._shape = Shape((len(data),))
        else:
            raise TypeError(f"Cannot initialize tensor from {type(data).__name__}")
    
    def _flatten_and_shape(
        self, data: Union[list, tuple], depth: int = 0
    ) -> Tuple[List, Tuple[int, ...]]:
        """
        Recursively flatten nested list/tuple and infer shape.
        
        Returns:
            (flat_list, shape_tuple)
        
        Raises:
            ValueError: If nested structure is inconsistent
        """
        if not isinstance(data, (list, tuple)):
            # Base case: scalar
            return [data], ()
        
        if len(data) == 0:
            return [], (0,)
        
        # Recursive case
        sub_results = [
            self._flatten_and_shape(item, depth + 1) 
            for item in data
        ]
        
        flattened: List = []
        shapes: List[Tuple] = []
        
        for flat, sub_shape in sub_results:
            flattened.extend(flat)
            shapes.append(sub_shape)
        
        # Check consistency
        if shapes:
            first_shape = shapes[0]
            if not all(s == first_shape for s in shapes):
                raise ValueError(
                    f"Inconsistent nested structure at depth {depth}: "
                    f"expected all sub-shapes to be {first_shape}, got {shapes}"
                )
        
        result_shape = (len(data),) + (shapes[0] if shapes else ())
        return flattened, result_shape
    
    def _deep_copy(self, data: Any) -> Any:
        """Deep copy data structure."""
        if isinstance(data, list):
            return [self._deep_copy(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._deep_copy(item) for item in data)
        else:
            return data
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def shape(self) -> Shape:
        """Return tensor shape."""
        if self._shape is None:
            self._shape = self._infer_shape()
        return self._shape
    
    def _infer_shape(self) -> Shape:
        """Infer shape from operation DAG or data."""
        if self._op is not None:
            return self._op.infer_shape(*self._args)
        if self._symbolic_expr is not None:
            return self._symbolic_expr.infer_shape()
        return Shape(())  # Scalar
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.shape.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.shape.size
    
    @property
    def is_symbolic(self) -> bool:
        """True if contains unevaluated symbolic expressions."""
        return self._symbolic_expr is not None
    
    @property
    def symbolic_expr(self) -> Optional['Expression']:
        """Return the internal symbolic AST expression."""
        return self._symbolic_expr
    
    @property
    def data(self) -> Any:
        """
        Direct access to the underlying data buffer.
        Triggers computation if necessary.
        """
        return self.compute()
    
    @property
    def is_numeric(self) -> bool:
        """True if contains numerical data."""
        return self._dense_data is not None or self._op is not None or self._cached_result is not None
    
    @property
    def is_sparse(self) -> bool:
        """True if sparse representation."""
        return self._sparse_data is not None
    
    @property
    def is_field(self) -> bool:
        """True if field tensor."""
        return self._field_data is not None
    
    @property
    def is_computed(self) -> bool:
        """True if tensor has been evaluated."""
        return self._cached_result is not None or self._dense_data is not None
    
    # =========================================================================
    # COMPUTATION - PHASE 1 IMPLEMENTED
    # =========================================================================
    
    def compute(self) -> Union[List, NumericBuffer]:
        """
        Evaluate tensor synchronously.
        
        Triggers computation of the entire DAG if necessary.
        Uses iterative approach to avoid recursion depth limits.
        
        Returns:
            Computed values (NumericBuffer or list for symbolic)
        """
        # Check cache
        if self._cached_result is not None:
            return self._cached_result
        
        # Return if already has data
        if self._dense_data is not None and self._op is None:
            return self._dense_data
        
        if self._sparse_data is not None:
            return self._sparse_data
        
        if self._field_data is not None:
            return self._field_data
        
        # Symbolic: Phase 3 Implemented
        if self.is_symbolic and self._op is None:
            return self._symbolic_expr.simplify()
        
        # Evaluate DAG iteratively using explicit stack
        if self._op is not None:
            # Collect all tensors that need computation (topological order)
            to_compute = []
            visited = set()
            stack = [self]
            
            while stack:
                current = stack[-1]
                if id(current) in visited:
                    stack.pop()
                    continue
                
                # Check if all dependencies are computed
                all_deps_ready = True
                if current._op is not None:
                    for arg in current._args:
                        if isinstance(arg, Tensor) and id(arg) not in visited:
                            if arg._cached_result is None and arg._dense_data is None:
                                if arg._op is not None:
                                    stack.append(arg)
                                    all_deps_ready = False
                
                if all_deps_ready:
                    visited.add(id(current))
                    stack.pop()
                    to_compute.append(current)
            
            # Execute in topological order
            for tensor in to_compute:
                if tensor._cached_result is not None:
                    continue
                if tensor._dense_data is not None and tensor._op is None:
                    continue
                if tensor._op is not None:
                    result = tensor._op.execute(*tensor._args)
                    if isinstance(result, list) and not tensor.is_symbolic:
                        result = NumericBuffer(result, tensor.dtype)
                    tensor._cached_result = result
            
            return self._cached_result
        
        return []
    
    def tolist(self) -> Any:
        """Convert tensor to nested Python list."""
        flat = self.compute()
        return self._unflatten(flat, self.shape.dims)
        
    def __len__(self) -> int:
        """Return the size of the first dimension."""
        if self.ndim == 0:
            # Consistent with NumPy: len() on scalar is TypeError
            raise TypeError("len() of unsized object")
        return self.shape.dims[0]

    def __iter__(self) -> Iterator:
        """Iterate over the first dimension of the tensor."""
        if self.ndim == 0:
            raise TypeError("Cannot iterate over a scalar tensor")
        data = self.tolist()
        return iter(data)
    
    def _unflatten(self, flat: List, dims: Tuple[int, ...]) -> Any:
        """Convert flat list back to nested structure."""
        if not dims:
            return flat[0] if flat else 0
        
        if len(dims) == 1:
            res = flat[:dims[0]]
            # Ensure we return a list, not array.array
            if hasattr(res, 'tolist') and not isinstance(res, list):
                return res.tolist()
            return res
        
        chunk_size = 1
        for d in dims[1:]:
            chunk_size *= d
        
        result = []
        for i in range(dims[0]):
            start = i * chunk_size
            end = start + chunk_size
            sub = self._unflatten(flat[start:end], dims[1:])
            result.append(sub)
        return result

    def __bool__(self) -> bool:
        """Boolean value of scalar tensor."""
        if self.ndim != 0:
            if self.shape.size != 1:
                raise ValueError("The truth value of an array with more than one element is ambiguous")
        return bool(self.item())

    def __float__(self) -> float:
        """Float value of scalar tensor."""
        return float(self.item())
        
    def __int__(self) -> int:
        """Int value of scalar tensor."""
        return int(self.item())
        
    def __complex__(self) -> complex:
        """Complex value of scalar tensor."""
        return complex(self.item())
        
    # =========================================================================
    # HELPER FOR OPERATIONS
    # =========================================================================
    
    # =========================================================================
    # SYMBOLIC METHODS - PHASE 3 IMPLEMENTED
    # =========================================================================

    def simplify(self) -> 'Tensor':
        """Simplify symbolic tensor."""
        if not self.is_symbolic:
            return self
        return Tensor(symbolic_expr=simplify_expr(self._symbolic_expr), shape=self.shape, dtype=self.dtype, name=f"{self.name}_simp")
    

        
    def diff(self, var: 'Tensor') -> 'Tensor':
        """Differentiate tensor with respect to a variable."""
        if not self.is_symbolic:
             # Derivative of constant is 0
             from loom.core.tensor import zeros
             return zeros(self.shape, dtype=self.dtype)
             
        if not isinstance(var, Tensor) or not var.is_symbolic:
             raise TypeError("Differentiation variable must be a symbolic Tensor")
             
        # Extract symbol name. Assuming var is a simple Symbol.
        if hasattr(var, 'symbol_name'):
             sym_name = var.symbol_name
        else:
             # Try to extract from expr if it's a SymbolExpr
             if isinstance(var._symbolic_expr, SymbolExpr):
                 sym_name = var._symbolic_expr.name
             else:
                 raise ValueError("Differentiation variable must be a single symbol")

        return Tensor(symbolic_expr=differentiate(self._symbolic_expr, sym_name), dtype=self.dtype, name=f"d{self.name}/d{sym_name}")

    def differentiate(self, var: 'Tensor') -> 'Tensor':
        """Alias for diff."""
        return self.diff(var)

    def item(self):
        """Return scalar value of 1-element tensor."""
        vals = self.compute()
        if hasattr(vals, '__len__') and len(vals) == 1:
            return vals[0]
        if hasattr(vals, '__len__'):
             if len(vals) == 0: raise ValueError("Tensor is empty")
             raise ValueError("Tensor is not scalar")
        return vals 

    def cast(self, dtype: Union[str, DType]) -> 'Tensor':
        """Cast tensor to new dtype."""
        return Tensor(self.compute(), shape=self.shape, dtype=dtype)

    def astype(self, dtype: Union[str, DType]) -> 'Tensor':
        """Alias for cast."""
        return self.cast(dtype)

    def _symbolic_op(self, other: 'Tensor', op: str) -> 'Tensor':
        from loom.symbolic import NumericExpr
        
        def to_expr(t):
            if t.is_symbolic: return t._symbolic_expr
            # Allow scalar tensors
            if t.size == 1:
                val = t.item()
                # If it's a whole number, keep as int for cleaner symbolic strings
                if isinstance(val, (float, int)) and val == int(val):
                    val = int(val)
                return NumericExpr(val)
            raise NotImplementedError("Symbolic operations on non-scalar tensors not yet fully supported (Phase 3 limitation)")

        lhs = to_expr(self)
        rhs = to_expr(other)
        
        if op == "+": res = lhs + rhs
        elif op == "-": res = lhs - rhs
        elif op == "*": res = lhs * rhs
        elif op == "/": res = lhs / rhs
        elif op == "**": res = lhs ** rhs
        else: raise ValueError(f"Unknown symbolic op {op}")
        
        return Tensor(symbolic_expr=res, dtype=DType.SYMBOLIC)

    def _symbolic_unary(self, func_name: str) -> 'Tensor':
        from loom.symbolic import FuncExpr, NumericExpr
        
        if self.is_symbolic:
            arg = self._symbolic_expr
        elif self.size == 1:
            arg = NumericExpr(self.item())
        else:
             raise NotImplementedError("Symbolic operations on non-scalar tensors not yet fully supported")

        if func_name == "neg":
             res = -arg
        else:
             res = FuncExpr(func_name, arg)
             
        return Tensor(symbolic_expr=res, dtype=DType.SYMBOLIC)

    def _ensure_tensor(self, other) -> 'Tensor':
        """Convert scalar to tensor if needed."""
        if isinstance(other, Tensor):
            return other
        return Tensor(other)
    
    # =========================================================================
    # INDEXING - PHASE 1 WEEK 7 IMPLEMENTED
    # =========================================================================
    
    def __getitem__(self, indices) -> 'Tensor':
        """
        Get tensor elements via indexing.
        
        Supports:
        - Integer indexing: t[0], t[1, 2]
        - Slice indexing: t[1:3], t[:, 0:2]
        - Boolean indexing: t[t > 0]
        - Fancy indexing: t[[0, 2, 4]]
        
        Args:
            indices: Index specification
        
        Returns:
            Indexed tensor
        """
        from loom.ops.indexing import create_index_op
        op = create_index_op(indices)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    # =========================================================================
    # ARITHMETIC OPERATORS - PHASE 1 IMPLEMENTED
    # =========================================================================
    
    def __add__(self, other) -> 'Tensor':
        """a + b element-wise with broadcasting."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
            return self._symbolic_op(other, "+")

        from loom.ops.arithmetic import ADD
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=ADD,
            _args=(self, other),
        )

    def __radd__(self, other) -> 'Tensor':
        """other + self"""
        return self.__add__(other)
    
    def __sub__(self, other) -> 'Tensor':
        """a - b element-wise with broadcasting."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
            return self._symbolic_op(other, "-")

        from loom.ops.arithmetic import SUB
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=SUB,
            _args=(self, other),
        )
    
    def __rsub__(self, other) -> 'Tensor':
        """other - self"""
        other = self._ensure_tensor(other)
        return other.__sub__(self)
    
    def __mul__(self, other) -> 'Tensor':
        """a * b element-wise with broadcasting."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
            return self._symbolic_op(other, "*")

        from loom.ops.arithmetic import MUL
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=MUL,
            _args=(self, other),
        )
    
    def __rmul__(self, other) -> 'Tensor':
        """other * self"""
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'Tensor':
        """a / b element-wise with broadcasting."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
            return self._symbolic_op(other, "/")

        from loom.ops.arithmetic import DIV
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=DIV,
            _args=(self, other),
        )
    
    def __rtruediv__(self, other) -> 'Tensor':
        """other / self"""
        other = self._ensure_tensor(other)
        return other.__truediv__(self)
    
    def __floordiv__(self, other) -> 'Tensor':
        """a // b element-wise with broadcasting."""
        from loom.ops.arithmetic import FLOORDIV
        other = self._ensure_tensor(other)
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=FLOORDIV,
            _args=(self, other),
        )
    
    def __rfloordiv__(self, other) -> 'Tensor':
        """other // self"""
        other = self._ensure_tensor(other)
        return other.__floordiv__(self)
    
    def __pow__(self, other) -> 'Tensor':
        """a ** b element-wise with broadcasting."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
            return self._symbolic_op(other, "**")

        from loom.ops.arithmetic import POW
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=POW,
            _args=(self, other),
        )

    def pow(self, other) -> 'Tensor':
        """a ** b alias."""
        return self.__pow__(other)
    
    def __rpow__(self, other) -> 'Tensor':
        """other ** self"""
        other = self._ensure_tensor(other)
        return other.__pow__(self)
    
    def __mod__(self, other) -> 'Tensor':
        """a % b element-wise with broadcasting."""
        from loom.ops.arithmetic import MOD
        other = self._ensure_tensor(other)
        result_shape = broadcast_shapes(self.shape, other.shape)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=MOD,
            _args=(self, other),
        )
    
    def __rmod__(self, other) -> 'Tensor':
        """other % self"""
        other = self._ensure_tensor(other)
        return other.__mod__(self)
    
    def __neg__(self) -> 'Tensor':
        """-a element-wise."""
        if self.is_symbolic: return self._symbolic_unary("neg")
        from loom.ops.arithmetic import NEG
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=NEG,
            _args=(self,),
        )
    
    def __pos__(self) -> 'Tensor':
        """+a (identity)."""
        return self
    
    def __abs__(self) -> 'Tensor':
        """abs(a) element-wise."""
        if self.is_symbolic: return self._symbolic_unary("abs")
        from loom.ops.arithmetic import ABS
        from loom.core.dtype import DType
        
        dtype = self.dtype
        if dtype == DType.COMPLEX128:
            dtype = DType.FLOAT64
        elif dtype == DType.COMPLEX64:
            dtype = DType.FLOAT32
            
        return Tensor(
            shape=self.shape.dims,
            dtype=dtype,
            _op=ABS,
            _args=(self,),
        )

    def abs(self) -> 'Tensor':
        """abs(a) element-wise."""
        return self.__abs__()
    
    def sqrt(self) -> 'Tensor':
        """Square root element-wise."""
        if self.is_symbolic: return self._symbolic_unary("sqrt")
        from loom.ops.arithmetic import SQRT
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=SQRT,
            _args=(self,),
        )
    
    def exp(self) -> 'Tensor':
        """Exponential element-wise."""
        if self.is_symbolic: return self._symbolic_unary("exp")
        from loom.ops.arithmetic import EXP
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=EXP,
            _args=(self,),
        )
    
    def log(self) -> 'Tensor':
        """Natural logarithm element-wise."""
        if self.is_symbolic: return self._symbolic_unary("log")
        from loom.ops.arithmetic import LOG
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=LOG,
            _args=(self,),
        )
    
    def sin(self) -> 'Tensor':
        """Sine element-wise."""
        if self.is_symbolic: return self._symbolic_unary("sin")
        from loom.ops.arithmetic import SIN
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=SIN,
            _args=(self,),
        )
    
    def cos(self) -> 'Tensor':
        """Cosine element-wise."""
        if self.is_symbolic: return self._symbolic_unary("cos")
        from loom.ops.arithmetic import COS
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=COS,
            _args=(self,),
        )

    def tan(self) -> 'Tensor':
        """Tangent element-wise."""
        if self.is_symbolic: return self._symbolic_unary("tan")
        from loom.ops.arithmetic import TAN
        return Tensor(
            shape=self.shape.dims,
            dtype=self.dtype,
            _op=TAN,
            _args=(self,),
        )

    @property
    def real(self) -> 'Tensor':
        """Return real part of tensor."""
        if self.is_symbolic: raise NotImplementedError("Symbolic real part not implemented")
        # For now, simplistic implementation: if not complex, return self
        if not (self.dtype == DType.COMPLEX64 or self.dtype == DType.COMPLEX128):
            return self
        
        # If complex, extract real part
        # Implementation via list for now until we have a proper ops backend
        data = self.tolist()
        
        def extract_real(d):
            if isinstance(d, list):
                return [extract_real(x) for x in d]
            if isinstance(d, complex):
                return d.real
            return d
            
        real_data = extract_real(data)
        # Determine strict dtype
        new_dtype = DType.FLOAT32 if self.dtype == DType.COMPLEX64 else DType.FLOAT64
        return Tensor(real_data, dtype=new_dtype)

    @property
    def imag(self) -> 'Tensor':
        """Return imaginary part of tensor."""
        if self.is_symbolic: raise NotImplementedError("Symbolic imag part not implemented")
        if not (self.dtype == DType.COMPLEX64 or self.dtype == DType.COMPLEX128):
             # Return zeros of same shape
             from loom.core.tensor import zeros
             return zeros(self.shape, dtype=self.dtype) # Should prob be float...
             
        data = self.tolist()
        
        def extract_imag(d):
            if isinstance(d, list):
                return [extract_imag(x) for x in d]
            if isinstance(d, complex):
                return d.imag
            return 0.0
            
        imag_data = extract_imag(data)
        new_dtype = DType.FLOAT32 if self.dtype == DType.COMPLEX64 else DType.FLOAT64
        return Tensor(imag_data, dtype=new_dtype)
    
    def transpose(self, axes=None) -> 'Tensor':
        """Permute dimensions."""
        from loom.ops.manipulation import create_transpose_op
        op = create_transpose_op(self, axes=axes)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
        
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        """Change tensor shape without changing data."""
        from loom.ops.manipulation import create_reshape_op
        op = create_reshape_op(shape)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
        
    def flatten(self) -> 'Tensor':
        """Flatten tensor to 1D."""
        return self.reshape((-1,))
    @property
    def T(self) -> 'Tensor':
        """Transpose (reverse dimensions)."""
        return self.transpose()

    def squeeze(self, axis=None) -> 'Tensor':
        """
        Remove single-dimensional entries from the shape.
        
        Args:
            axis: Selects a subset of the single-dimensional entries in the shape.
                  If an axis is selected with shape entry greater than one, an error is raised.
                  
        Returns:
            The input tensor, but with all or a subset of the dimensions of length 1 removed.
        """
        shape = list(self.shape.dims)
        if axis is None:
            new_shape = [s for s in shape if s != 1]
        else:
            if isinstance(axis, int):
                axis = (axis,)
            new_shape = []
            for i, s in enumerate(shape):
                if i in axis:
                    if s != 1:
                        raise ValueError(f"cannot select an axis to squeeze out which has size not equal to one")
                    continue
                new_shape.append(s)
        
        if not new_shape: # Squeezing scalar
             new_shape = [] # Rank 0

        return self.reshape(tuple(new_shape))

    def unsqueeze(self, axis: int) -> 'Tensor':
        """
        Insert a new axis at the specified position.
        
        Args:
            axis: The position at which to insert the new axis.
        
        Returns:
            The input tensor with one more dimension.
        """
        shape = list(self.shape.dims)
        ndim = len(shape)
        if axis < 0:
            axis += ndim + 1
        
        if axis > ndim:
             raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim-1}, {ndim}], but got {axis})")

        shape.insert(axis, 1)
        return self.reshape(tuple(shape))

    
    def __matmul__(self, other):
        """a @ b matrix multiplication."""
        from loom.ops.matmul import create_matmul_op
        other = self._ensure_tensor(other)
        op = create_matmul_op(self, other)
        
        result_shape = op.infer_shape(self, other)
        
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,  # Note: Type promotion uses left operand's dtype
            _op=op,
            _args=(self, other),
        )
    
    def matmul(self, other):
        """Alias for @ operator."""
        return self.__matmul__(other)
        
    def dot(self, other):
        """Dot product. For 1D-1D equals inner product, for 2D-2D equals matmul."""
        return self.__matmul__(other)
    
    # =========================================================================
    # COMPARISON OPERATORS (return boolean tensors)
    # =========================================================================
    
    def __eq__(self, other) -> 'Tensor':
        """a == b element-wise."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
             if self.is_symbolic and other.is_symbolic:
                 return Tensor(int(self._symbolic_expr == other._symbolic_expr))
             return Tensor(0)

        a_data = self.compute()
        b_data = other.compute()
        result = [1 if a == b else 0 for a, b in zip(a_data, b_data)]
        return Tensor(result, shape=self.shape.dims, dtype="bool")
    
    def __ne__(self, other) -> 'Tensor':
        """a != b element-wise."""
        other = self._ensure_tensor(other)
        if self.is_symbolic or other.is_symbolic:
             if self.is_symbolic and other.is_symbolic:
                 return Tensor(int(self._symbolic_expr != other._symbolic_expr))
             return Tensor(1)

        a_data = self.compute()
        b_data = other.compute()
        result = [1 if a != b else 0 for a, b in zip(a_data, b_data)]
        return Tensor(result, shape=self.shape.dims, dtype="bool")

    def subs(self, name_or_dict, value=None) -> 'Tensor':
        """
        Substitute symbolic variables.
        
        Can be called as:
            - subs(name, value) - substitute single variable
            - subs(dict) - substitute multiple variables (legacy)
        
        Args:
            name_or_dict: Variable name (str) or dict of substitutions
            value: Value to substitute (if name is str)
        
        Returns:
            Tensor with substituted values
        """
        if not self.is_symbolic:
            return self
        
        # Handle both call styles: subs(name, value) or subs(dict)
        if isinstance(name_or_dict, dict):
            substitutions = name_or_dict
        else:
            substitutions = {name_or_dict: value}
        
        expr = self._symbolic_expr
        for name, val in substitutions.items():
            expr = expr.substitute(name, val)
            
        simplified = expr.simplify()
        from loom.symbolic import NumericExpr
        if isinstance(simplified, NumericExpr):
            return Tensor(simplified.value)
        return Tensor(symbolic_expr=simplified, dtype=DType.SYMBOLIC)
    
    def __lt__(self, other) -> 'Tensor':
        """a < b element-wise."""
        other = self._ensure_tensor(other)
        a_data = self.compute()
        b_data = other.compute()
        result = [1 if a < b else 0 for a, b in zip(a_data, b_data)]
        return Tensor(result, shape=self.shape.dims, dtype="bool")
    
    def __le__(self, other) -> 'Tensor':
        """a <= b element-wise."""
        other = self._ensure_tensor(other)
        a_data = self.compute()
        b_data = other.compute()
        result = [1 if a <= b else 0 for a, b in zip(a_data, b_data)]
        return Tensor(result, shape=self.shape.dims, dtype="bool")
    
    def __gt__(self, other) -> 'Tensor':
        """a > b element-wise."""
        other = self._ensure_tensor(other)
        a_data = self.compute()
        b_data = other.compute()
        result = [1 if a > b else 0 for a, b in zip(a_data, b_data)]
        return Tensor(result, shape=self.shape.dims, dtype="bool")
    
    def __ge__(self, other) -> 'Tensor':
        """a >= b element-wise."""
        other = self._ensure_tensor(other)
        a_data = self.compute()
        b_data = other.compute()
        result = [1 if a >= b else 0 for a, b in zip(a_data, b_data)]
        return Tensor(result, shape=self.shape.dims, dtype="bool")
    
    def __repr__(self) -> str:
        """String representation."""
        desc = f"Tensor(shape={self.shape}, dtype={self.dtype.value}"
        if self._dense_data is not None:
            desc += ", numeric"
        if self._op is not None:
            desc += f", op={self._op.name}"
        if self.is_symbolic:
            desc += ", symbolic"
        if self.is_sparse:
            desc += ", sparse"
        if self.is_field:
            desc += ", field"
        desc += ")"
        return desc
    
    # =========================================================================
    # REDUCTION METHODS - PHASE 1 WEEK 6 IMPLEMENTED
    # =========================================================================
    
    def sum(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """
        Sum of tensor elements along axis.
        
        Args:
            axis: Axis or axes along which to sum. None = sum all elements.
            keepdims: If True, reduced axes are kept with size 1.
        
        Returns:
            Tensor with summed values
        """
        if self.is_symbolic and self.size == 1: return self

        from loom.ops.reduction import create_sum_op
        op = create_sum_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def mean(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """
        Mean of tensor elements along axis.
        
        Args:
            axis: Axis or axes along which to compute mean. None = mean of all.
            keepdims: If True, reduced axes are kept with size 1.
        
        Returns:
            Tensor with mean values
        """
        if self.is_symbolic and self.size == 1: return self

        from loom.ops.reduction import create_mean_op
        op = create_mean_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def max(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """
        Maximum of tensor elements along axis.
        
        Args:
            axis: Axis or axes along which to find max. None = max of all.
            keepdims: If True, reduced axes are kept with size 1.
        
        Returns:
            Tensor with max values
        """
        if self.is_symbolic and self.size == 1: return self

        from loom.ops.reduction import create_max_op
        op = create_max_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def min(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """
        Minimum of tensor elements along axis.
        
        Args:
            axis: Axis or axes along which to find min. None = min of all.
            keepdims: If True, reduced axes are kept with size 1.
        
        Returns:
            Tensor with min values
        """
        if self.is_symbolic and self.size == 1: return self

        from loom.ops.reduction import create_min_op
        op = create_min_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def sum(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """Sum of tensor elements."""
        from loom.ops.reduction import create_sum_op
        op = create_sum_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )

    def mean(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """Average of tensor elements."""
        from loom.ops.reduction import create_mean_op
        op = create_mean_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def max(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """Maximum of tensor elements."""
        from loom.ops.reduction import create_max_op
        op = create_max_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )

    def min(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """Minimum of tensor elements."""
        from loom.ops.reduction import create_min_op
        op = create_min_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )

    def prod(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """
        Product of tensor elements along axis.
        
        Args:
            axis: Axis or axes along which to compute product. None = all.
            keepdims: If True, reduced axes are kept with size 1.
        
        Returns:
            Tensor with product values
        """
        from loom.ops.reduction import create_prod_op
        op = create_prod_op(axis=axis, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def argmax(self, axis=None) -> 'Tensor':
        """Index of maximum value along axis."""
        from loom.ops.reduction import create_argmax_op
        op = create_argmax_op(axis=axis)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype="int64",
            _op=op,
            _args=(self,),
        )
    
    def argmin(self, axis=None) -> 'Tensor':
        """Index of minimum value along axis."""
        from loom.ops.reduction import create_argmin_op
        op = create_argmin_op(axis=axis)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            _op=op,
            _args=(self,),
        )
    
    def std(self, axis=None, ddof: int = 0, keepdims: bool = False) -> 'Tensor':
        """
        Compute standard deviation.
        """
        from loom.ops.reduction import create_std_op
        op = create_std_op(axis=axis, ddof=ddof, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
        
    def var(self, axis=None, ddof: int = 0, keepdims: bool = False) -> 'Tensor':
        """
        Compute variance.
        """
        from loom.ops.reduction import create_var_op
        op = create_var_op(axis=axis, ddof=ddof, keepdims=keepdims)
        result_shape = op.infer_shape(self)
        return Tensor(
            shape=result_shape.dims,
            dtype=self.dtype,
            _op=op,
            _args=(self,),
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        if self.is_symbolic:
            return str(self._symbolic_expr)
        data = self.compute()
        if data:
            nested = self.tolist()
            return str(nested)
        return repr(self)


class Symbol(Tensor):
    """
    Symbolic variable for symbolic algebra.
    
    A Symbol is a Tensor with dtype=SYMBOLIC and a named variable.
    Used for symbolic differentiation, simplification, etc.
    
    Example (Phase 3):
        >>> x = Symbol('x')
        >>> expr = x**2 + 2*x + 1
        >>> df = differentiate(expr, x)  # Returns 2*x + 2
    
    PHASE STATUS: Skeleton (Phase 0) - Full implementation in Phase 3
    NOTE: Symbolic operations are deferred to Phase 3. Current implementation
    creates a placeholder Symbol but symbolic math operations are not functional.
    """
    
    def __init__(self, name: str):
        """
        Create a symbolic variable.
        
        Args:
            name: Variable name (e.g., 'x', 'theta')
        """
        expr = SymbolExpr(name)
        super().__init__(
            shape=(1,),
            dtype=DType.SYMBOLIC,
            name=name,
            symbolic_expr=expr
        )
        self._symbol_name = name
        # Placeholder data to satisfy compute()
        self._dense_data = None
    
    @property
    def symbol_name(self) -> str:
        """Return the symbolic variable name."""
        return self._symbol_name
    
    def __repr__(self) -> str:
        return f"Symbol('{self._symbol_name}')"


# =============================================================================
# TENSOR CREATION FACTORIES - PHASE 1 IMPLEMENTED
# =============================================================================

def array(data: Any, dtype: Union[str, DType] = None, name: Optional[str] = None) -> 'Tensor':
    """
    Create a tensor from data.
    
    Args:
        data: List, tuple, or scalar data
        dtype: Data type
        name: Optional name
    
    Returns:
        New Tensor
    """
    return Tensor(data, dtype=dtype if dtype is not None else config.DEFAULT_DTYPE, name=name)


def zeros(shape: Tuple[int, ...], dtype: Union[str, DType] = None, name: Optional[str] = None) -> 'Tensor':
    """
    Create a tensor filled with zeros.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        name: Optional name
    
    Returns:
        New Tensor filled with zeros
    """
    from loom.core.shape import Shape
    size = Shape(shape).size
    data = [0] * size
    return Tensor(data, shape=shape, dtype=dtype if dtype is not None else config.DEFAULT_DTYPE, name=name)


def ones(shape: Tuple[int, ...], dtype: Union[str, DType] = None, name: Optional[str] = None) -> 'Tensor':
    """
    Create a tensor filled with ones.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        name: Optional name
    
    Returns:
        New Tensor filled with ones
    """
    from loom.core.shape import Shape
    size = Shape(shape).size
    data = [1] * size
    return Tensor(data, shape=shape, dtype=dtype if dtype is not None else config.DEFAULT_DTYPE, name=name)


def full(shape: Tuple[int, ...], fill_value: Any, dtype: Union[str, DType] = None, name: Optional[str] = None) -> 'Tensor':
    """
    Create a tensor filled with a constant value.
    
    Args:
        shape: Tensor shape
        fill_value: Value to fill with
        dtype: Data type (inferred from fill_value if None)
        name: Optional name
    
    Returns:
        New Tensor filled with fill_value
    """
    from loom.core.shape import Shape
    size = Shape(shape).size
    data = [fill_value] * size
    return Tensor(data, shape=shape, dtype=dtype if dtype is not None else config.DEFAULT_DTYPE, name=name)


def eye(n: int, m: Optional[int] = None, dtype: Union[str, DType] = None, name: Optional[str] = None) -> 'Tensor':
    """
    Create a 2D identity tensor.
    
    Args:
        n: Number of rows
        m: Number of columns (equals n if None)
        dtype: Data type
        name: Optional name
    
    Returns:
        2D identity Tensor
    """
    if m is None:
        m = n
    
    data = []
    for i in range(n):
        for j in range(m):
            data.append(1 if i == j else 0)
    
    return Tensor(data, shape=(n, m), dtype=dtype if dtype is not None else config.DEFAULT_DTYPE, name=name)


