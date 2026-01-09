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
Tests for symbolic integration and solving.
"""

import pytest
import loom as tf
from loom.core.tensor import Symbol
from loom.symbolic import integrate, solve, simplify
from loom.symbolic.ast import FuncExpr

def test_symbolic_integrate_polynomial():
    x = Symbol('x')
    # Integral of x^2 + 2x + 1 dx = 1/3 x^3 + x^2 + x
    expr = x**2 + 2*x + 1
    integral = integrate(expr, x)
    
    # Value at x=1: (1/3 + 1 + 1) = 2.333...
    # integral is an AST node, so we can use .evaluate() after .subs
    res = integral.substitute('x', 1.0).evaluate()
    assert abs(res - 7/3) < 1e-7

def test_symbolic_integrate_trig():
    x = Symbol('x')
    # Integral of sin(x) dx = -cos(x)
    expr = FuncExpr('sin', x.symbolic_expr)
    integral = integrate(expr, x)
    
    # Value at x=0: -cos(0) = -1
    res = integral.substitute('x', 0.0).evaluate()
    assert res == -1.0

def test_symbolic_solve_linear():
    x = Symbol('x')
    # 2x - 4 = 0 => x = 2
    expr = 2*x - 4
    solutions = solve(expr, x)
    assert len(solutions) == 1
    # Solution is x = 4/2 = 2
    assert solutions[0].evaluate() == 2.0

def test_symbolic_solve_quadratic():
    x = Symbol('x')
    # x^2 - 5x + 6 = 0 => x = 2, 3
    expr = x**2 - 5*x + 6
    solutions = solve(expr, x)
    assert len(solutions) == 2
    vals = sorted([s.evaluate() for s in solutions])
    assert vals == [2.0, 3.0]

