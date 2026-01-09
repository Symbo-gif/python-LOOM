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
Automatic differentiation engine for symbolic expressions.
"""

from loom.symbolic.ast import Expression, SymbolExpr, NumericExpr, BinOpExpr, FuncExpr

def differentiate(expr: Expression, var: str) -> Expression:
    """
    Compute the derivative of an expression with respect to a variable.
    
    Args:
        expr: The symbolic expression to differentiate.
        var: The name of the variable to differentiate by.
        
    Returns:
        The derivative expression.
    """
    # Handle Tensor inputs
    if hasattr(expr, '_symbolic_expr'): # Check for Tensor without importing
        # If expr is Tensor, delegate to its differentiate method
        # But wait, Tensor.diff returns Tensor. If we return Tensor here, type hint might look wrong but it works in Python.
        # Alternatively, unwrap here.
        # Let's delegate to Tensor.diff if available, which handles Tensor var.
        if hasattr(expr, 'diff'):
             return expr.diff(var)
        # If no diff method (e.g. mock), fallback to unwrap
        expr = expr._symbolic_expr
        
    if hasattr(var, 'symbol_name'): # Check for Tensor/Symbol
        var = var.symbol_name
    elif hasattr(var, 'name'): # Check for SymbolExpr
        var = var.name

    res = _differentiate_recursive(expr, var)
    if hasattr(res, 'simplify'):
        return res.simplify()
    return res

def _differentiate_recursive(expr: Expression, var: str) -> Expression:
    # Base cases
    if isinstance(expr, NumericExpr):
        return NumericExpr(0)
    
    if isinstance(expr, SymbolExpr):
        if expr.name == var:
            return NumericExpr(1)
        else:
            return NumericExpr(0) # Treat other symbols as constants for partial diff
            
    # Recursive step
    if isinstance(expr, BinOpExpr):
        u = expr.left
        v = expr.right
        du = _differentiate_recursive(u, var)
        dv = _differentiate_recursive(v, var)
        
        if expr.op == "+":
            return du + dv
        
        if expr.op == "-":
            return du - dv
        
        if expr.op == "*":
            # Product rule: u'v + uv'
            return (du * v) + (u * dv)
        
        if expr.op == "/":
            # Quotient rule: (u'v - uv') / v^2
            numerator = (du * v) - (u * dv)
            denominator = v ** NumericExpr(2)
            return numerator / denominator
            
        if expr.op == "**":
            # Power rule (generalized f(x)^g(x))
            # d/dx (u^v) = u^v * (v' * ln(u) + v * u' / u)
            # Simplified for constant exponent: d/dx(u^n) = n * u^(n-1) * u'
            if isinstance(v, NumericExpr):
                n = v
                return n * (u ** NumericExpr(n.value - 1)) * du
            else:
                # Full generalized power rule: u^v * (v'*ln(u) + v*u'/u)
                # Requires 'ln' function support
                ln_u = FuncExpr('log', u)
                term1 = dv * ln_u
                term2 = (v * du) / u
                return (u ** v) * (term1 + term2)
                
    if isinstance(expr, FuncExpr):
        arg = expr.arg
        d_arg = _differentiate_recursive(arg, var) # Chain rule: f'(g(x)) * g'(x)
        
        if expr.func_name == 'sin':
            # d/dx sin(u) = cos(u) * u'
            return FuncExpr('cos', arg) * d_arg
            
        if expr.func_name == 'cos':
            # d/dx cos(u) = -sin(u) * u'
            return -FuncExpr('sin', arg) * d_arg
            
        if expr.func_name == 'tan':
            # d/dx tan(u) = sec^2(u) * u' = (1/cos(u)^2) * u'
            sec_sq = NumericExpr(1) / (FuncExpr('cos', arg) ** NumericExpr(2))
            return sec_sq * d_arg
        
        if expr.func_name == 'exp':
            # d/dx exp(u) = exp(u) * u'
            return expr * d_arg
            
        if expr.func_name == 'log':
            # d/dx log(u) = 1/u * u'
            return (NumericExpr(1) / arg) * d_arg
            
        if expr.func_name == 'sqrt':
            # d/dx sqrt(u) = 1/(2*sqrt(u)) * u'
            return (NumericExpr(1) / (NumericExpr(2) * expr)) * d_arg

        if expr.func_name == 'erf':
            # d/dx erf(u) = (2/sqrt(pi)) * exp(-u^2) * u'
            import math
            two_over_sqrt_pi = NumericExpr(2.0 / math.sqrt(math.pi))
            exp_neg_u_sq = FuncExpr('exp', -(arg ** NumericExpr(2)))
            return two_over_sqrt_pi * exp_neg_u_sq * d_arg

    raise NotImplementedError(f"Differentiation not implemented for {expr}")

