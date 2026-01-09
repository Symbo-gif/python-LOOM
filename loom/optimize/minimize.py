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
Minimization algorithms.
"""

from typing import Callable, Union, Optional, Tuple, List
import loom as tf
from loom.core.tensor import Tensor, array
import math

def minimize(fun: Callable, x0: Union[List, Tensor, float], method: str = 'BFGS', 
             tol: float = 1e-6, maxiter: Optional[int] = None, 
             args: Tuple = (), options: dict = None) -> 'OptimizeResult':
    """
    Minimization of scalar function of one or more variables.
    """
    if options is not None:
        if maxiter is None:
            maxiter = options.get('maxiter')
            
    # Wrap function to handle args
    if args:
        original_fun = fun
        fun = lambda x: original_fun(x, *args)

    if isinstance(x0, (float, int)):
        x0 = array([x0])
    elif isinstance(x0, (list, tuple)):
        x0 = array(x0)
        
    if method.upper() == 'BFGS':
        return _minimize_bfgs(fun, x0, tol, maxiter)
    elif method.upper() == 'NELDER-MEAD':
        return _minimize_nelder_mead(fun, x0, tol, maxiter)
    else:
        raise ValueError(f"Unknown method {method}")

class OptimizeResult(dict):
    """Represents the optimization result."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

def _minimize_bfgs(fun: Callable, x0: Tensor, tol: float, maxiter: Optional[int]) -> OptimizeResult:
    """BFGS quasi-Newton method."""
    x = x0
    n = x.size
    
    # Ensure x is a proper array for operations
    if not isinstance(x, Tensor):
        x = array(x)
    
    I = tf.eye(n)
    H = I  # Initialize Hessian inverse as identity
    
    if maxiter is None:
        maxiter = n * 200
        
    def get_grad(x):
        # Implementation of numerical gradient
        h = 1e-4
        grad = []
        x_list = x.tolist()  # Get current x values
        for i in range(n):
            x_plus = list(x_list)
            x_minus = list(x_list)
            x_plus[i] += h
            x_minus[i] -= h
            grad.append((fun(array(x_plus)) - fun(array(x_minus))) / (2 * h))
        return array(grad)

    g = get_grad(x)
    
    for i in range(maxiter):
        gnorm = (g**2).sum().sqrt().item()
        if gnorm < tol:
            return OptimizeResult(x=x, success=True, nit=i, fun=fun(x), message="Optimization terminated successfully.", status=0)
            
        # Search direction p = -H * g
        # (N,N) @ (N,1) -> (N,1)
        g_reshaped = g.reshape((-1, 1))
        H_g = H @ g_reshaped
        p = -H_g.flatten()
        
        # Line search (very simple backtracking)
        alpha = 1.0
        f0 = fun(x)
        c1 = 1e-4
        for _ in range(20):
            x_trial = x + alpha * p
            if fun(x_trial) <= f0 + c1 * alpha * (g * p).sum().item():
                break
            alpha *= 0.5
            
        s = alpha * p
        x_next = x + s
        g_next = get_grad(x_next)
        y = g_next - g
        
        # Update H using BFGS formula
        # rho = 1 / (y^T * s)
        ys = (y * s).sum().item()
        if abs(ys) < 1e-15:
            H = I # Reset if numerical issues
        else:
            rho = 1.0 / ys
            # H_next = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
            s_col = s.reshape((-1, 1))
            y_col = y.reshape((-1, 1))
            
            V = I - rho * (s_col @ y_col.transpose())
            H = (V @ H @ V.transpose()) + rho * (s_col @ s_col.transpose())
        
        # Update x and g - use tensors directly (no need to compute/wrap)
        x = x_next
        g = g_next
        
    return OptimizeResult(x=x, success=False, nit=maxiter, fun=fun(x), message="Maximum iterations reached.", status=1)

def _minimize_nelder_mead(fun: Callable, x0: Tensor, tol: float, maxiter: Optional[int]) -> OptimizeResult:
    """Nelder-Mead simplex algorithm."""
    # Derivative-free
    x = x0.tolist()
    n = len(x)
    if maxiter is None:
        maxiter = n * 200
        
    # Create initial simplex
    simplex = [x]
    for i in range(n):
        point = list(x)
        point[i] = point[i] + 0.05 if point[i] != 0 else 0.00025
        simplex.append(point)
        
    # Map points to values
    values = [fun(array(p)) for p in simplex]
    
    # Coefficients
    rho = 1.0   # Reflection
    chi = 2.0   # Expansion
    psi = 0.5   # Contraction
    sigma = 0.5 # Shrinkage
    
    for it in range(maxiter):
        # Sort by values
        combined = sorted(zip(values, simplex), key=lambda x: x[0])
        values, simplex = zip(*combined)
        values = list(values)
        simplex = list(simplex)
        
        # Check convergence
        if (array(values).max() - array(values).min()).item() < tol:
            return OptimizeResult(x=array(simplex[0]), success=True, nit=it, fun=values[0], message="Optimization terminated successfully.", status=0)
            
        # Centroid (all except the worst)
        centroid = [sum(pt[i] for pt in simplex[:-1]) / n for i in range(n)]
        
        # Reflection
        xr = [centroid[i] + rho * (centroid[i] - simplex[-1][i]) for i in range(n)]
        fr = fun(array(xr))
        
        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        elif fr < values[0]:
            # Expansion
            xe = [centroid[i] + chi * (xr[i] - centroid[i]) for i in range(n)]
            fe = fun(array(xe))
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        else:
            # Contraction
            if values[-2] <= fr < values[-1]:
                # Outside contraction
                xc = [centroid[i] + psi * (xr[i] - centroid[i]) for i in range(n)]
                fc = fun(array(xc))
                if fc <= fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        values[i] = fun(array(simplex[i]))
            else:
                # Inside contraction
                xic = [centroid[i] - psi * (centroid[i] - simplex[-1][i]) for i in range(n)]
                fic = fun(array(xic))
                if fic < values[-1]:
                    simplex[-1] = xic
                    values[-1] = fic
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        values[i] = fun(array(simplex[i]))
                        
    return OptimizeResult(x=array(simplex[0]), success=False, nit=maxiter, fun=values[0], message="Maximum iterations reached.", status=1)

