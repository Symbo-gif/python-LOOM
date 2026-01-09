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

import pytest
from loom.symbolic import Expression, SymbolExpr, NumericExpr, BinOpExpr, FuncExpr, simplify

def test_ast_construction():
    x = SymbolExpr("x")
    y = SymbolExpr("y")
    expr = x + y * 2
    assert isinstance(expr, BinOpExpr)
    assert expr.op == "+"
    assert expr.left == x
    assert isinstance(expr.right, BinOpExpr)
    assert expr.right.op == "*"
    
def test_evaluation():
    n1 = NumericExpr(10)
    n2 = NumericExpr(20)
    expr = n1 + n2
    assert expr.evaluate() == 30
    
    expr = n1 * n2
    assert expr.evaluate() == 200

def test_substitution():
    x = SymbolExpr("x")
    expr = x + 2
    
    # Subst x -> 3 (Direct Numeric) -- Wait, substitue returns Expression
    res = expr.substitute("x", NumericExpr(3))
    assert res.evaluate() == 5
    
    # Subst x -> y
    y = SymbolExpr("y")
    res = expr.substitute("x", y)
    assert res.free_symbols() == {"y"}

def test_simplification_rules():
    x = SymbolExpr("x")
    
    # x + 0 -> x
    expr = x + 0
    assert simplify(expr) == x
    
    # x * 1 -> x
    expr = x * 1
    assert simplify(expr) == x
    
    # x * 0 -> 0
    expr = x * 0
    res = simplify(expr)
    assert isinstance(res, NumericExpr)
    assert res.value == 0
    
    # Constant folding
    expr = NumericExpr(2) + NumericExpr(3)
    res = simplify(expr)
    assert isinstance(res, NumericExpr)
    assert res.value == 5
    
def test_func_evaluation():
    import math
    expr = FuncExpr("sin", NumericExpr(0))
    assert expr.evaluate() == 0
    
    expr = FuncExpr("exp", NumericExpr(1))
    assert expr.evaluate() == math.e

