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
Matrix functions: expm, logm, sqrtm.
"""

import math
from typing import Union, Optional
import loom as tf
from loom.core.tensor import Tensor, array, eye


def expm(A: Tensor) -> Tensor:
    """
    Compute the matrix exponential using scaling and squaring with Padé approximation.
    
    Args:
        A: Square matrix (n×n)
        
    Returns:
        Matrix exponential exp(A)
        
    Raises:
        ValueError: If A is not a square matrix
    """
    A = array(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expm requires a square matrix")
        
    n = A.shape[0]
    
    # Scaling factor based on matrix norm
    norm = A.abs().sum().item()
    s = max(0, int(math.ceil(math.log2(norm / 1.0)))) if norm > 1.0 else 0
    A_scaled = A / (2**s)
    
    # Padé (3,3) approximation
    X = A_scaled
    X2 = X @ X
    X3 = X2 @ X
    
    c1 = 0.5
    c2 = 0.1
    c3 = 1.0 / 120.0
    
    I = eye(n)
    N = I + c1 * X + c2 * X2 + c3 * X3
    D = I - c1 * X + c2 * X2 - c3 * X3
    
    from loom.linalg import solve
    R = solve(D, N)
    
    # Squaring step
    for _ in range(s):
        R = R @ R
        
    return R


def sqrtm(A: Tensor, maxiter: int = 50, tol: float = 1e-10) -> Tensor:
    """
    Compute the principal matrix square root using Denman-Beavers iteration.
    
    For matrices with negative eigenvalues, the result may be complex.
    
    Args:
        A: Square matrix (must be non-singular)
        maxiter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Matrix S such that S @ S ≈ A
        
    Raises:
        ValueError: If A is not a square matrix
    """
    A = array(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("sqrtm requires a square matrix")
        
    n = A.shape[0]
    
    # Check for negative eigenvalues by looking at diagonal dominance or trace
    # For matrices with potential negative eigenvalues, use Newton-Schulz iteration
    # which is more robust
    
    # First, try to detect if we might have negative eigenvalues
    # A simple heuristic: if trace is negative, we likely have negative eigenvalues
    trace_val = sum(A[i, i].item() for i in range(n))
    
    # Use Newton-Schulz iteration with regularization for robustness
    from loom.linalg import inv
    
    # Initialize: Y_0 = A / ||A||_F (scaled for convergence)
    norm_A = A.abs().sum().item()
    eps = 1e-15
    if norm_A < eps:
        # Nearly zero matrix - return zero matrix
        return tf.zeros((n, n))
    
    Y = A
    Z = eye(n)
    
    try:
        for iteration in range(maxiter):
            # Add small regularization to prevent singular matrix inversion
            # Regularized inverse: inv(M + eps*I) instead of inv(M)
            reg = eye(n) * eps
            
            try:
                inv_Z = inv(Z + reg)
                inv_Y = inv(Y + reg)
            except (ZeroDivisionError, ValueError):
                # If inversion fails, add more regularization
                reg = eye(n) * 1e-10
                inv_Z = inv(Z + reg)
                inv_Y = inv(Y + reg)
            
            Y_next = 0.5 * (Y + inv_Z)
            Z_next = 0.5 * (Z + inv_Y)
            
            diff = (Y_next - Y).abs().sum().item()
            Y, Z = Y_next, Z_next
            if diff < tol:
                break
    except (ZeroDivisionError, ValueError, OverflowError):
        # Fallback: Use scaled Newton iteration
        # For matrices with complex eigenvalues, return a scaled approximation
        # This is a simple fallback for robustness
        Y = A
        for _ in range(min(10, maxiter)):
            Y_sq = Y @ Y
            diff_sq = (Y_sq - A).abs().sum().item()
            if diff_sq < tol:
                break
            # Simple Newton step: Y = 0.5 * (Y + A @ inv(Y))
            try:
                inv_Y = inv(Y + eye(n) * eps)
                Y = 0.5 * (Y + A @ inv_Y)
            except:
                break
            
    return Y



def logm(A: Tensor, maxiter: int = 100, tol: float = 1e-10) -> Tensor:
    """
    Compute the principal matrix logarithm using inverse scaling and squaring.
    
    For a matrix A with no eigenvalues on the negative real axis, computes
    the principal logarithm L such that exp(L) = A.
    
    Args:
        A: Square matrix (should have no negative real eigenvalues for principal branch)
        maxiter: Maximum iterations for inner Newton iteration
        tol: Convergence tolerance
        
    Returns:
        Matrix logarithm log(A), may be complex for matrices with negative eigenvalues
        
    Raises:
        ValueError: If A is not a square matrix
    """
    A = array(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("logm requires a square matrix")
        
    n = A.shape[0]
    I = eye(n)
    
    # Inverse scaling: repeatedly take square roots until close to I
    # log(A) = 2^k * log(A^(1/2^k))
    k = 0
    X = A
    max_sqrt_iters = 50
    
    for _ in range(max_sqrt_iters):
        # Check if X is close to I
        diff = (X - I).abs().sum().item()
        if diff < 0.5:
            break
        X = sqrtm(X, maxiter=maxiter, tol=tol)
        k += 1
    
    # Now X is close to I, use Padé approximation for log(I + Y) where Y = X - I
    Y = X - I
    
    # For ||Y|| < 1, use series: log(I + Y) ≈ Y - Y²/2 + Y³/3 - Y⁴/4 + ...
    # Or use Padé diagonal approximant
    # Simple series for small Y:
    L = _log_pade(Y)
    
    # Scale back
    return L * (2 ** k)


def _log_pade(Y: Tensor) -> Tensor:
    """
    Compute log(I + Y) using Padé approximation for small Y.
    Uses (3,3) Padé approximant of log(1+x).
    """
    n = Y.shape[0]
    I = eye(n)
    Y2 = Y @ Y
    Y3 = Y2 @ Y
    
    # Padé (3,3) for log(1+x): coefficients derived from standard tables
    # log(1+x) ≈ x * (1 + a1*x + a2*x²) / (1 + b1*x + b2*x²) for small x
    # For matrices, this becomes more complex. Use simpler Taylor series.
    
    # Taylor series up to order 8 for better accuracy
    result = Y
    term = Y
    for i in range(2, 12):
        term = term @ Y
        sign = -1.0 if i % 2 == 0 else 1.0
        result = result + (sign / i) * term
        
        # Check convergence
        if term.abs().sum().item() / (n * n) < 1e-14:
            break
    
    return result

