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
Symbolic Integration.
"""

from typing import Union, Any
from loom.symbolic.ast import NumericExpr, SymbolExpr, BinOpExpr, FuncExpr, Expression

def integrate(expr: Any, var: Any) -> Expression:
    """
    Perform symbolic integration of an expression with respect to a variable.
    """
    if hasattr(expr, 'symbolic_expr'):
        expr = expr.symbolic_expr
    if hasattr(var, 'symbolic_expr'):
        var = var.symbolic_expr
        
    # Handle primitive types
    if isinstance(expr, (int, float, complex)):
        expr = NumericExpr(expr)

    res = _integrate_recursive(expr, var)
    if hasattr(res, 'simplify'):
        return res.simplify()
    return res

def _integrate_recursive(expr: Any, var: Any) -> Expression:
    if isinstance(expr, NumericExpr):
        # Integral of 0 dx = 0 (ignoring constant C)
        if expr.value == 0:
            return NumericExpr(0)
        # Integral of c dx = c*x
        return BinOpExpr(expr, var, '*')
        
    if isinstance(expr, SymbolExpr):
        if expr.name == var.name:
            # Integral of x dx = 1/2 * x^2
            half = BinOpExpr(NumericExpr(1), NumericExpr(2), '/')
            x_sq = BinOpExpr(var, NumericExpr(2), '**')
            return BinOpExpr(half, x_sq, '*')
        else:
            # Integral of y dx = y*x
            return BinOpExpr(expr, var, '*')
            
    if isinstance(expr, BinOpExpr):
        if expr.op == '+':
            return BinOpExpr(_integrate_recursive(expr.left, var), _integrate_recursive(expr.right, var), '+')
        if expr.op == '-':
            return BinOpExpr(_integrate_recursive(expr.left, var), _integrate_recursive(expr.right, var), '-')
        if expr.op == '*':
            # Handle c * f(x)
            if isinstance(expr.left, NumericExpr):
                return BinOpExpr(expr.left, _integrate_recursive(expr.right, var), '*')
            if isinstance(expr.right, NumericExpr):
                return BinOpExpr(expr.right, _integrate_recursive(expr.left, var), '*')
        if expr.op == '**':
            # Integral of x^n dx = x^(n+1) / (n+1)
            if expr.left == var and isinstance(expr.right, NumericExpr):
                n = expr.right.value
                if n == -1:
                    # Integral 1/x = log(x)
                    return FuncExpr('log', var)
                return BinOpExpr(BinOpExpr(var, NumericExpr(n + 1), '**'), NumericExpr(n + 1), '/')
                
    if isinstance(expr, FuncExpr):
        if expr.func_name == 'sin':
            # Integral sin(x) = -cos(x)
            if expr.arg == var:
                return BinOpExpr(NumericExpr(-1.0), FuncExpr('cos', var), '*')
        if expr.func_name == 'cos':
            # Integral cos(x) = sin(x)
            if expr.arg == var:
                return FuncExpr('sin', var)
        if expr.func_name == 'exp':
            # Integral exp(x) = exp(x)
            if expr.arg == var:
                return FuncExpr('exp', var)
                
    # Fallback or complex cases
    raise NotImplementedError(f"Integration of {expr} not yet supported")

