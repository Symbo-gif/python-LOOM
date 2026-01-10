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
ODE Solvers.
"""

from typing import Callable, Union, Optional, Tuple, List
import loom as tf
from loom.core.tensor import Tensor, array
import math

class ODEResult(dict):
    """Represents the ODE solution result."""
    def __getattr__(self, name):
        return self[name]

def solve_ivp(fun: Callable, t_span: Tuple[float, float], y0: Union[List, float, Tensor], 
              method: str = 'RK45', t_eval: Optional[List[float]] = None) -> ODEResult:
    """
    Solve an initial value problem for a system of ODEs.
    
    dy/dt = f(t, y)
    y(t0) = y0
    """
    if not isinstance(y0, (list, tuple, Tensor)):
        y0 = [y0]
    
    if method.upper() == 'RK4':
        return _solve_rk4(fun, t_span, y0, t_eval)
    elif method.upper() == 'RK45':
        return _solve_rk45(fun, t_span, y0, t_eval)
    else:
        raise ValueError(f"Unknown method {method}")

def _solve_rk4(fun: Callable, t_span: Tuple[float, float], y0: Union[List, Tensor], 
               t_eval: Optional[List[float]] = None) -> ODEResult:
    """Classical 4th order Runge-Kutta with fixed step size."""
    t0, tf_ = t_span
    y = array(y0)
    # Ensure initial y is computed and stored as NumericBuffer
    y_data = y.compute()
    y = array(y_data)
    
    if t_eval is not None:
        times = t_eval
    else:
        # Default step size for 100 steps
        step = (tf_ - t0) / 100
        times = [t0 + i * step for i in range(101)]
        
    ts = []
    ys = []
    
    t = t0
    ts.append(t)
    ys.append(y_data)
    
    for next_t in times:
        if next_t <= t: continue
        
        while t < next_t:
            h = min(next_t - t, 0.1) # Max step size limit
            
            # Use Tensors for component-wise ops if fun returns Tensors
            k1 = array(fun(t, y))
            k2 = array(fun(t + h/2, y + h/2 * k1))
            k3 = array(fun(t + h/2, y + h/2 * k2))
            k4 = array(fun(t + h, y + h * k3))
            
            y_old = y  # Keep Ref
            y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
            # Force computation (Phase 4 requirement)
            y_data = y.compute()
            y = array(y_data)
            
        ts.append(t)
        ys.append(list(y_data))  # Convert NumericBuffer to list for consistency
        
    # Transpose ys from (Time, Comp) to (Comp, Time) to match SciPy's format
    # SciPy returns y with shape (n, t) where n=components, t=time points
    if not ys:
        y_transposed = []
    else:
        y_transposed = list(map(list, zip(*ys)))
        
    return ODEResult(t=ts, y=y_transposed, success=True, message="Optimization terminated successfully.", t_events=None, y_events=None)

def _solve_rk45(fun: Callable, t_span: Tuple[float, float], y0: Union[List, Tensor], 
                t_eval: Optional[List[float]] = None) -> ODEResult:
    """Adaptive step-size Dormand-Prince (RK45)."""
    t0, tf_ = t_span
    y = array(y0)
    
    # Dormand-Prince coefficients
    a = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    b = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    # RK4 coefficients (for error estimation)
    c = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    # RK5 coefficients
    chat = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    t = t0
    h = 0.01 # Initial step
    atol = 1e-6
    rtol = 1e-3
    
    # Ensure initial y is computed
    y_data = y.compute()
    y = array(y_data)
    
    ts = [t]
    ys = [list(y_data)]  # Convert to list for consistency
    
    while t < tf_:
        if t + h > tf_:
            h = tf_ - t
            
        k = []
        k.append(array(fun(t, y)))
        
        for i in range(1, 7):
            dy = sum(b[i][j] * k[j] for j in range(i))
            k.append(array(fun(t + a[i]*h, y + h*dy)))
            
        # Error estimation
        y_next = y + h * sum(c[i] * k[i] for i in range(7))
        y_hat = y + h * sum(chat[i] * k[i] for i in range(7))
        
        error_vec = y_next - y_hat
        # Check if error_vec is scalar or vector
        if error_vec.size == 1:
             err = abs(error_vec.item())
        else:
             err = max(error_vec.abs().tolist())
             
        # Tolerance
        tol = atol + rtol * max(y.abs().tolist() if y.size > 1 else [abs(y.item())])
        
        if err <= tol or h < 1e-12:
            # Step accepted
            t += h
            y = y_next
            # Force computation and clear DAG to prevent recursion depth issues (Phase 4 requirement)
            y_data = y.compute()
            y = array(y_data) 
            
            # Store all steps
            ts.append(t)
            ys.append(list(y_data))  # Convert to list for consistency
            
            # Adjust step size
            if err > 0:
                h *= min(5, max(0.1, 0.9 * (tol / err)**0.2))
            else:
                h *= 5
        else:
            # Step rejected
            h *= max(0.1, 0.9 * (tol / err)**0.2)
            
    # Post-process results
    if t_eval is not None:
        # Linear interpolation for requested points
        ts_out = t_eval
        
        # We need to find valid segments. Assumes t_eval is sorted.
        curr_idx = 0
        n_steps = len(ts)
        
        interpolated_ys = []
        for target_t in t_eval:
            # Find segment [ts[i], ts[i+1]] containing target_t
            # Optimization: continue from last idx
            while curr_idx < n_steps - 1 and ts[curr_idx+1] < target_t:
                curr_idx += 1
            
            if curr_idx >= n_steps - 1:
                # End of range, use last point
                interpolated_ys.append(ys[-1])
                continue
                
            t_prev = ts[curr_idx]
            t_next = ts[curr_idx+1]
            y_prev = ys[curr_idx]
            y_next = ys[curr_idx+1]
            
            if t_next == t_prev:
                 frac = 0.0
            else:
                 frac = (target_t - t_prev) / (t_next - t_prev)
            
            # Interpolate each component - y_prev/y_next are lists
            y_interp = []
            for k in range(len(y_prev)):
                val = y_prev[k] + frac * (y_next[k] - y_prev[k])
                y_interp.append(val)
            
            interpolated_ys.append(y_interp)
            
        ts = ts_out
        ys = interpolated_ys

    # Convert NumericBuffers to lists
    ys_as_lists = [list(y) if hasattr(y, '__iter__') else [y] for y in ys]
    
    # Transpose from (Time, Comp) to (Comp, Time) to match SciPy's format
    if not ys_as_lists:
        y_transposed = []
    else:
        y_transposed = list(map(list, zip(*ys_as_lists)))

    return ODEResult(t=ts, y=y_transposed, success=True, message="Optimization terminated successfully.", t_events=None, y_events=None)

