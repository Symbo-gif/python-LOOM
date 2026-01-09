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
from loom.core.tensor import Tensor, Symbol, DType

def test_symbol_tensor_creation():
    x = Symbol("x")
    assert x.is_symbolic
    assert x.dtype == DType.SYMBOLIC
    assert str(x) == "x"

def test_tensor_symbolic_ops():
    x = Symbol("x")
    expr = x + 2
    assert isinstance(expr, Tensor)
    assert expr.is_symbolic
    assert str(expr.simplify()) == "(x + 2)"

def test_tensor_subs():
    x = Symbol("x")
    expr = x * 2
    res = expr.subs("x", 5)
    # Result is a stored symbolic expression, but constant.
    # .simplify() should evaluate it if it's all numeric
    # Tensor.compute() on symbolic simplified?
    # Symbolic Expression evaluate() returns float. simplify() returns NumericExpr.
    
    # Tensor wrapping a NumericExpr(10)
    assert "10" in str(res.simplify())
    
def test_tensor_diff():
    x = Symbol("x")
    # x^2
    expr = x ** Tensor(2) 
    # Tensor(2) might be treated as numeric tensor.
    # _symbolic_op handles Tensor(2) by taking item().
    
    diff = expr.diff(x)
    # 2 * x^(2-1) = 2*x
    
    res = diff.subs("x", 3).simplify()
    # 2*3 = 6
    assert "6" in str(res) # Loose check on repr

def test_tensor_multivariate_diff():
    x = Symbol("x")
    y = Symbol("y")
    expr = x * y
    
    dx = expr.diff(x) # y
    dy = expr.diff(y) # x
    
    assert str(dx.simplify()) == "y"
    assert str(dy.simplify()) == "x"

