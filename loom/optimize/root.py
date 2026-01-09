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
Root finding algorithms.
"""

from typing import Callable, Union, Optional, Tuple
import loom as tf
from loom.core.tensor import Tensor, Symbol, array
import math

def bisect(f: Callable, a: float, b: float, xtol: float = 1e-6, maxiter: int = 100) -> float:
    """
    Find root of f(x) = 0 in interval [a, b] using bisection.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    if fa == 0: return a
    if fb == 0: return b
    
    for _ in range(maxiter):
        mid = (a + b) / 2
        fmid = f(mid)
        if abs(fmid) < 1e-12 or (b - a) / 2 < xtol:
            return mid
        if fmid * fa < 0:
            b = mid
        else:
            a = mid
            fa = fmid
    return (a + b) / 2

def newton(f: Callable, x0: Union[float, Tensor], fprime: Optional[Callable] = None, 
           tol: float = 1e-6, maxiter: int = 100) -> float:
    """
    Find root of f(x) = 0 using Newton-Raphson.
    
    If fprime is None, attempt to use symbolic differentiation if f(x) returns a symbolic Tensor.
    """
    x = x0
    for _ in range(maxiter):
        fx = f(x)
        
        # If fx is a Tensor, we might be able to get derivative
        if fprime is None:
            if isinstance(fx, Tensor) and fx.is_symbolic:
                # We need a Symbol to differentiate with respect to
                if isinstance(x, Tensor) and x.is_symbolic:
                    df = fx.diff(x)
                    # To evaluate, we need to substitute current value
                    # This is tricky because 'x' is the Symbol.
                    # For Newton to work with Symbols, we'd need to keep the Symbol separate.
                    raise NotImplementedError("Auto-symbolic differentiation in newton() requires a Symbol x.")
                else:
                    # Generic numeric fallback for derivative if fprime missing?
                    # Let's just require fprime or use a numeric approximation
                    pass
        
        if fprime is None:
            # Numeric approximation of derivative: (f(x+h) - f(x-h)) / 2h
            h = 1e-8
            dfx = (f(x + h) - f(x - h)) / (2 * h)
        else:
            dfx = fprime(x)
            
        if abs(dfx) < 1e-12:
            raise RuntimeWarning("Derivative is too small, Newton may not converge.")
            break
            
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
        
    return x

def brentq(f: Callable, a: float, b: float, xtol: float = 1e-6, maxiter: int = 100) -> float:
    """
    Find root of f(x) = 0 in interval [a, b] using Brent's method.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    mflag = True
    d = 0 # Placeholder for first iteration
    
    for i in range(maxiter):
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
            
        # Conditions for bisection fallback
        cond1 = not ((3*a + b)/4 <= s <= b)
        cond2 = mflag and (abs(s - b) >= abs(b - c)/2)
        cond3 = (not mflag) and (abs(s - b) >= abs(c - d)/2)
        cond4 = mflag and (abs(b - c) < xtol)
        cond5 = (not mflag) and (abs(c - d) < xtol)
        
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
            
        fs = f(s)
        d = c
        c = b
        fc = fb
        
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
            
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        if fs == 0 or abs(b - a) < xtol:
            return b
            
    return b

