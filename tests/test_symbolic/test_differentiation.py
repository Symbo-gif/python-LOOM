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
from loom.symbolic import SymbolExpr, NumericExpr, differentiate, simplify

def test_diff_constant():
    x = SymbolExpr("x")
    c = NumericExpr(5)
    assert differentiate(c, "x").evaluate() == 0

def test_diff_var():
    x = SymbolExpr("x")
    assert differentiate(x, "x").evaluate() == 1
    assert differentiate(x, "y").evaluate() == 0

def test_diff_polynomial():
    x = SymbolExpr("x")
    # x^2 + 2x + 1 => 2x + 2
    expr = x**NumericExpr(2) + x*NumericExpr(2) + NumericExpr(1)
    diff = differentiate(expr, "x")
    # Simplify to check structure or evaluate at points
    # 2*x + 2
    res = diff.substitute("x", NumericExpr(3)).evaluate()
    # 2*3 + 2 = 8
    assert res == 8

def test_diff_product():
    x = SymbolExpr("x")
    # x * x => 2x
    expr = x * x
    diff = differentiate(expr, "x")
    res = diff.substitute("x", NumericExpr(4)).evaluate()
    # 2*4 = 8
    # d(x*x) = 1*x + x*1 = 2x
    assert res == 8

def test_diff_trig():
    x = SymbolExpr("x")
    from loom.symbolic import FuncExpr
    # sin(x) => cos(x)
    expr = FuncExpr("sin", x)
    diff = differentiate(expr, "x")
    # evaluate at x=0 => cos(0) => 1
    res = diff.substitute("x", NumericExpr(0)).evaluate()
    assert res == 1

def test_chain_rule():
    x = SymbolExpr("x")
    from loom.symbolic import FuncExpr
    # sin(x^2) => cos(x^2) * 2x
    expr = FuncExpr("sin", x**NumericExpr(2))
    diff = differentiate(expr, "x")
    
    # Check value at sqrt(pi/2) approx 1.253? Too complex with floats
    # Check at 0: cos(0)*0 = 0
    res = diff.substitute("x", NumericExpr(0)).evaluate()
    assert res == 0

