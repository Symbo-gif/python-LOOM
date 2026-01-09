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
Basic linear algebra operations.

PHASE STATUS: Phase 2 Week 11 Implementation
"""

from typing import Union, Optional, Tuple, Literal
from loom.core.tensor import Tensor, array
import loom as tf
import math

# =============================================================================
# MATRIX PRODUCTS
# =============================================================================

def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Matrix product of two arrays.
    
    Args:
        x1: Input tensor
        x2: Input tensor
    
    Returns:
        Tensor result of matrix multiplication
    """
    return x1 @ x2


def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Dot product of two arrays.
    
    - If both 1D: inner product of vectors (without complex conjugation)
    - If both 2D: matrix multiplication
    - If either 0D: scalar multiplication
    - If a is N-D, b is 1D: sum product over last axis of a and b
    - If a is N-D, b is M-D: sum product over last axis of a and second-to-last of b
    """
    # For now, simplistic implementation mapping to matmul for >=1D
    # Note: Full N-D matmul logic is handled by matmul operation in ops/matmul.py
    return a @ b


def vdot(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the dot product of two vectors, handling complex conjugation.
    conjugate(a) . b
    """
    from loom.ops.complex_ops import conj
    if a.dtype in ('complex64', 'complex128'):
        a = conj(a)
    
    # Flatten both
    # Note: Flatten approach is sufficient for correctness; optimize if profiling shows need
    # But vdot in numpy technically flattens inputs first if they are not 1D
    # "If a and b are non-scalar arrays, they are flattened to 1-D vectors first."
    
    # Simple list comprehension implementation for now to ensure correctness
    # This avoids the MatmulOp overhead for this specific scalar result op
    # But wait, we want a Tensor output.
    
    # Let's use element-wise mul + sum
    # Requires reshape to 1D?
    # tf.sum(a * b)
    
    # Wait, flatten?
    # We implemented .tolist(), but not a .flatten() tensor op yet.
    # But we can assume inputs are 1D valid for now or rely on broadcasting if compatible.
    # NumPy vdot flattens. We should verify shape logic.
    
    # For Phase 2 Week 11, let's implement for 1D tensors first.
    return (a * b).sum()


def inner(a: Tensor, b: Tensor) -> Tensor:
    """
    Inner product of two arrays.
    Sum product over the last dimensions.
    """
    # Simply sum(a * b, axis=-1) ???
    # If 1D, yes.
    # If 2D (M, K), (N, K) -> (M, N) via broadcasting?
    # NumPy: "Ordinary inner product of vectors for 1-D arrays... Sum product over last axes"
    
    # It requires broadcasting on all but last dimension? No.
    # It's flexible.
    # Let's defer full implementation until we have better broadcasting tools.
    # For 1D, it's dot.
    return dot(a, b)


def outer(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the outer product of two vectors.
    """
    # a (M,) -> (M, 1)
    # b (N,) -> (1, N)
    # (M, 1) * (1, N) -> (M, N)
    
    # Direct implementation for 1D vectors
    a_flat = a.flatten().tolist()
    b_flat = b.flatten().tolist()
    
    m = len(a_flat)
    n = len(b_flat)
    
    # Construct m×n matrix where result[i][j] = a[i] * b[j]
    result = [[a_flat[i] * b_flat[j] for j in range(n)] for i in range(m)]
    
    return array(result)


# =============================================================================
# NORMS AND PROPERTIES
# =============================================================================

def trace(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """
    Return the sum along diagonals of the array.
    """
    # Extract diagonal and sum.
    # requires 'diag' or advanced indexing.
    # a[i, i]
    # We can do this with basic iteration if we compute().
    # But we want to return a Tensor op.
    
    # Implementing a TraceOp would be best.
    # For now, eager evaluation implementation:
    data = a.compute()
    shape = a.shape.dims
    if len(shape) != 2:
        raise ValueError("Trace currently only supports 2D matrices")
    
    n = min(shape[0], shape[1])
    # stride = shape[1] + 1 ?
    # 0, 1*cols+1, 2*cols+2...
    
    cols = shape[1]
    diag_sum = 0.0
    for i in range(n):
        idx = i * cols + i
        diag_sum += data[idx]
        
    return tf.array(diag_sum)


def norm(x: Tensor, ord: Union[int, str, None] = None, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Tensor:
    """
    Matrix or vector norm.
    """
    # Default: 2-norm (Euclidean) for vectors, Frobenius for matrices
    if ord is None:
        # Frobenius: sqrt(sum(abs(x)**2))
        return (abs(x)**2).sum(axis=axis, keepdims=keepdims).sqrt()
        
    raise NotImplementedError(f"Norm order {ord} not yet implemented")


def matrix_transpose(x: Tensor) -> Tensor:
    """
    Transpose a matrix (swap last two dimensions).
    """
    # If 2D, strictly equals .T
    # If >2D, swap last two.
    # We can use our new transpose method with explicit axes!
    dims = x.shape.dims
    ndim = len(dims)
    if ndim < 2:
        return x
        
    axes = list(range(ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return x.transpose(axes=tuple(axes))


