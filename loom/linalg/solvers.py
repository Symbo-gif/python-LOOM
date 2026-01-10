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
Linear solvers using decompositions.
"""

from typing import Tuple, Optional, Union
import loom as tf
from loom.core.tensor import Tensor
from loom.linalg.decompositions import lu
import loom.linalg as la

def solve(a: Tensor, b: Tensor) -> Tensor:
    """
    Solve linear system Ax = b for x.
    
    Uses LU decomposition:
    A = P @ L @ U
    P @ L @ U @ x = b
    L @ U @ x = P.T @ b  (Let y = U @ x)
    L @ y = P.T @ b      (Solve for y using forward sub)
    U @ x = y            (Solve for x using backward sub)
    """
    # 1. Validation
    # Square A
    rows, cols = a.shape.dims
    if rows != cols:
        raise ValueError("solve requires square matrix")
    
    # Validation b
    # b can be (N,) or (N, K)
    
    # 2. LU Decomp
    P, L, U = lu(a, pivot=True)
    
    # 3. Permute b: (P.T @ b)
    # P is orthogonal, inv(P) = P.T
    # If P @ L @ U = A
    # Then L @ U @ x = P.T @ b
    
    # If b is 1D, reshape effectively
    pb = P.T @ b
    
    # 4. Forward substitution: L @ y = pb
    # L is lower triangular with unit diagonal
    rows_l = rows
    
    pb_data = pb.compute()
    L_data = L.compute()
    
    # Handle b shape (N,) or (N, K)
    b_dims = pb.shape.dims
    n_rhs = 1
    if len(b_dims) > 1:
        n_rhs = b_dims[1]
        
    y_data = [0.0] * (rows * n_rhs)
    
    # Helper for indexing flat data
    def get_L(r, c): return L_data[r * rows + c]
    def get_pb(r, c=0): return pb_data[r * n_rhs + c]
    def set_y(r, c, v): y_data[r * n_rhs + c] = v
    def get_y(r, c=0): return y_data[r * n_rhs + c]
    
    # Forward sub
    for k in range(n_rhs):
        for i in range(rows):
            sum_val = 0.0
            for j in range(i):
                sum_val += get_L(i, j) * get_y(j, k)
            
            # L[i,i] is 1.0 by construction of our LU
            val = get_pb(i, k) - sum_val
            set_y(i, k, val)
            
    # 5. Backward substitution: U @ x = y
    # U is upper triangular
    U_data = U.compute()
    x_data = [0.0] * (rows * n_rhs)
    
    def get_U(r, c): return U_data[r * cols + c] # cols=rows
    def set_x(r, c, v): x_data[r * n_rhs + c] = v
    def get_x(r, c=0): return x_data[r * n_rhs + c]
    
    for k in range(n_rhs):
        for i in range(rows - 1, -1, -1):
            sum_val = 0.0
            for j in range(i + 1, rows):
                sum_val += get_U(i, j) * get_x(j, k)
            
            # U[i,i] is pivot
            pivot = get_U(i, i)
            # Add epsilon protection to prevent division by zero
            eps = 1e-15
            if abs(pivot) < eps:
                # Use small epsilon to avoid division by zero for near-singular matrices
                pivot = eps if pivot >= 0 else -eps
                
            val = (get_y(i, k) - sum_val) / pivot
            set_x(i, k, val)
            
    return tf.Tensor(x_data, shape=b_dims)


def inv(a: Tensor) -> Tensor:
    """
    Compute multiplicative inverse of a matrix.
    
    Solves A @ x = I
    """
    rows = a.shape.dims[0]
    I = tf.eye(rows)
    return solve(a, I)


def det(a: Tensor) -> Tensor:
    """
    Compute determinant of a matrix.
    
    For A = P @ L @ U:
    det(A) = det(P) * det(L) * det(U)
    det(L) = 1 (unit diagonal)
    det(U) = prod(diagonal(U))
    det(P) = (-1)^S where S is number of swaps.
    """
    P, L, U = lu(a, pivot=True)
    U_data = U.compute()
    rows = a.shape.dims[0]
    cols = rows
    
    # 1. Product of U diagonal
    prod_u = 1.0
    for i in range(rows):
        prod_u *= U_data[i*cols + i]
        
    # 2. det(P) = sign of permutation
    p_sign = _compute_p_sign(P)
    
    return tf.array(p_sign * prod_u)

def _compute_p_sign(P: Tensor) -> float:
    """Compute det(P) for permutation matrix P."""
    # Find permutation vector from matrix
    p_data = P.tolist()
    n = len(p_data)
    perm = [0] * n
    for i in range(n):
        for j in range(n):
            if p_data[i][j] == 1.0:
                perm[i] = j
                break
                
    # Count swaps to compute parity
    swaps = 0
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            # New cycle
            curr = i
            cycle_len = 0
            while not visited[curr]:
                visited[curr] = True
                curr = perm[curr]
                cycle_len += 1
            if cycle_len > 1:
                swaps += (cycle_len - 1)
                
    return 1.0 if swaps % 2 == 0 else -1.0


def matrix_rank(a: Tensor, tol: Optional[float] = None) -> Tensor:
    """
    Return matrix rank of array using SVD.
    """
    # Use SVD
    from loom.linalg.decompositions import svd
    
    # We might need to handle exceptions if SVD fails or is slow, but assuming it works.
    _, S, _ = svd(a)
    s_data = S.compute()
    
    if tol is None:
        # Standard tolerance: S.max() * max(M, N) * eps
        max_s = max(s_data) if s_data else 0.0
        dims = a.shape.dims
        max_dim = max(dims)
        eps = 1.19209e-07 # float32 eps approximately, or 2e-16 for float64
        tol = max_s * max_dim * eps
        
    rank = sum(1 for s in s_data if s > tol)
    return tf.array(rank)
    

def cond(x: Tensor, p: Union[int, str, None] = None) -> Tensor:
    """
    Compute condition number of a matrix.
    
    Current implementation supports p=None (2-norm condition number).
    Ratio of largest to smallest singular value.
    """
    from loom.linalg.decompositions import svd
    
    if p is None or p == 2:
        _, S, _ = svd(x)
        s_data = S.compute()
        if not s_data:
            return tf.array(0.0)
            
        max_s = max(s_data)
        min_s = min(s_data)
        
        if min_s == 0:
             return tf.array(float('inf'))
             
        return tf.array(max_s / min_s)
        
    raise NotImplementedError(f"Condition number for norm {p} not implemented")

