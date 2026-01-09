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
Symbolic Expression AST for loom.

This module defines the node types for the symbolic expression tree.
"""

from abc import ABC, abstractmethod
from typing import Set, Union, Dict, Any, Optional
from dataclasses import dataclass
import math

class Expression(ABC):
    """Base class for all symbolic expressions."""
    
    @abstractmethod
    def free_symbols(self) -> Set[str]:
        """Return a set of free symbol names in the expression."""
        pass
    
    @abstractmethod
    def substitute(self, symbol: str, value: Union[float, int, 'Expression']) -> 'Expression':
        """Substitute a symbol with a value or another expression."""
        pass
    
    @abstractmethod
    def evaluate(self) -> Union[float, int, complex]:
        """Numerically evaluate the expression (must be constant)."""
        pass
    
    @abstractmethod
    def simplify(self) -> 'Expression':
        """Return a simplified version of the expression."""
        pass
    
    @abstractmethod
    def infer_shape(self):
        """Infer the shape of the expression result."""
        pass
    
    def has_trig(self) -> bool:
        """Check if expression contains trigonometric functions."""
        return False
        
    def has_log_exp(self) -> bool:
        """Check if expression contains log or exp functions."""
        return False

    def __add__(self, other):
        return BinOpExpr(self, _ensure_expr(other), "+")
    
    def __radd__(self, other):
        return BinOpExpr(_ensure_expr(other), self, "+")
    
    def __sub__(self, other):
        return BinOpExpr(self, _ensure_expr(other), "-")
    
    def __rsub__(self, other):
        return BinOpExpr(_ensure_expr(other), self, "-")
    
    def __mul__(self, other):
        return BinOpExpr(self, _ensure_expr(other), "*")
    
    def __rmul__(self, other):
        return BinOpExpr(_ensure_expr(other), self, "*")
    
    def __truediv__(self, other):
        return BinOpExpr(self, _ensure_expr(other), "/")
    
    def __rtruediv__(self, other):
        return BinOpExpr(_ensure_expr(other), self, "/")
    
    def __pow__(self, other):
        return BinOpExpr(self, _ensure_expr(other), "**")
    
    def __rpow__(self, other):
        return BinOpExpr(_ensure_expr(other), self, "**")
    
    def __neg__(self):
        return BinOpExpr(NumericExpr(-1), self, "*")

def _ensure_expr(val) -> Expression:
    if isinstance(val, Expression):
        return val
    return NumericExpr(val)

@dataclass(frozen=True)
class SymbolExpr(Expression):
    """Represents a symbolic variable (e.g., 'x')."""
    name: str
    
    def free_symbols(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, symbol: str, value: Union[float, int, Expression]) -> Expression:
        if self.name == symbol:
            return _ensure_expr(value)
        return self
    
    def evaluate(self) -> Union[float, int, complex]:
        raise ValueError(f"Cannot evaluate expression with free symbol: {self.name}")
    
    def simplify(self) -> Expression:
        return self
    
    def infer_shape(self):
        from loom.core.shape import Shape
        return Shape((1,)) # Default to scalar/broadcastable
    
    def __repr__(self) -> str:
        return self.name

@dataclass(frozen=True)
class NumericExpr(Expression):
    """Represents a numeric constant."""
    value: Union[float, int, complex]
    
    def free_symbols(self) -> Set[str]:
        return set()
    
    def substitute(self, symbol: str, value: Union[float, int, Expression]) -> Expression:
        return self
    
    def evaluate(self) -> Union[float, int, complex]:
        return self.value
    
    def simplify(self) -> Expression:
        return self
    
    def infer_shape(self):
        from loom.core.shape import Shape
        return Shape(())
    
    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, NumericExpr):
            return self.value == other.value
        if isinstance(other, (int, float, complex)):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

@dataclass(frozen=True)
class BinOpExpr(Expression):
    """Represents a binary operation between two expressions."""
    left: Expression
    right: Expression
    op: str
    
    def free_symbols(self) -> Set[str]:
        return self.left.free_symbols() | self.right.free_symbols()
    
    def substitute(self, symbol: str, value: Union[float, int, Expression]) -> Expression:
        return BinOpExpr(
            self.left.substitute(symbol, value),
            self.right.substitute(symbol, value),
            self.op
        )
    
    def evaluate(self) -> Union[float, int, complex]:
        l = self.left.evaluate()
        r = self.right.evaluate()
        if self.op == "+": return l + r
        if self.op == "-": return l - r
        if self.op == "*": return l * r
        if self.op == "/": return l / r
        if self.op == "**": return l ** r
        raise ValueError(f"Unknown operator: {self.op}")
    
    def simplify(self) -> Expression:
        l = self.left.simplify()
        r = self.right.simplify()
        
        # Constant folding
        if isinstance(l, NumericExpr) and isinstance(r, NumericExpr):
            lv, rv = l.value, r.value
            try:
                if self.op == "+": val = lv + rv
                elif self.op == "-": val = lv - rv
                elif self.op == "*": val = lv * rv
                elif self.op == "/": 
                    if rv == 0: return BinOpExpr(l, r, "/") # Don't fold div by zero
                    if isinstance(lv, int) and isinstance(rv, int) and lv % rv == 0:
                        val = lv // rv
                    elif isinstance(lv, int) and isinstance(rv, int):
                        return BinOpExpr(l, r, "/") # Keep fraction
                    else:
                        val = lv / rv
                elif self.op == "**": val = lv ** rv
                else: val = 0 
                
                # Normalize result
                if isinstance(val, float) and val == int(val):
                    val = int(val)
                elif isinstance(val, complex) and val.imag == 0:
                    real_val = val.real
                    if real_val == int(real_val): val = int(real_val)
                    else: val = real_val
                
                return NumericExpr(val)
            except Exception:
                pass # Fallback to symbolic if evaluation fails
        
        # Identity rules
        if self.op == "+":
            if isinstance(r, NumericExpr) and r.value == 0: return l
            if isinstance(l, NumericExpr) and l.value == 0: return r
            if repr(l) == repr(r):
                # x + x -> (2 * x)
                return (NumericExpr(2) * l).simplify()
            
            # (a*x) + x -> (a+1)*x
            if isinstance(l, BinOpExpr) and l.op == "*" and isinstance(l.left, NumericExpr) and repr(l.right) == repr(r):
                return (NumericExpr(l.left.value + 1) * r).simplify()
            # x + (a*x) -> (a+1)*x
            if isinstance(r, BinOpExpr) and r.op == "*" and isinstance(r.left, NumericExpr) and repr(r.right) == repr(l):
                return (NumericExpr(r.left.value + 1) * l).simplify()
            # (a*x) + (b*x) -> (a+b)*x
            if isinstance(l, BinOpExpr) and l.op == "*" and isinstance(l.left, NumericExpr) and \
               isinstance(r, BinOpExpr) and r.op == "*" and isinstance(r.left, NumericExpr) and \
               repr(l.right) == repr(r.right):
                return (NumericExpr(l.left.value + r.left.value) * l.right).simplify()

            if isinstance(l, BinOpExpr) and l.op == "+" and isinstance(l.right, NumericExpr) and isinstance(r, NumericExpr):
                 return (l.left + (l.right + r)).simplify()
            if isinstance(r, BinOpExpr) and r.op == "+" and isinstance(r.left, NumericExpr) and isinstance(l, NumericExpr):
                 return ((l + r.left) + r.right).simplify()
        
        if self.op == "*":
            if isinstance(r, NumericExpr):
                if r.value == 1: return l
                if r.value == 0: return NumericExpr(0)
            if isinstance(l, NumericExpr):
                if l.value == 1: return r
                if l.value == 0: return NumericExpr(0)
                if l.value == -1:
                     if isinstance(r, BinOpExpr) and r.op == '*' and isinstance(r.left, NumericExpr) and r.left.value == -1:
                         return r.right.simplify()

            if repr(l) == repr(r):
                return (l ** NumericExpr(2)).simplify()

            if isinstance(l, NumericExpr) and isinstance(r, BinOpExpr) and r.op == "+":
                 return (r.left * l + r.right * l).simplify()
            if isinstance(r, NumericExpr) and isinstance(l, BinOpExpr) and l.op == "+":
                 return (l.left * r + l.right * r).simplify()
            
            if isinstance(l, BinOpExpr) and l.op == "*" and isinstance(l.right, NumericExpr) and isinstance(r, NumericExpr):
                 return (l.left * (l.right * r)).simplify()
                
        if self.op == "-":
            if isinstance(r, NumericExpr) and r.value == 0: return l
            if repr(l) == repr(r):
                return NumericExpr(0)
            # x - (a*x) -> (1-a)*x
            if isinstance(r, BinOpExpr) and r.op == "*" and isinstance(r.left, NumericExpr) and repr(r.right) == repr(l):
                return (NumericExpr(1 - r.left.value) * l).simplify()

        if self.op == "/":
            if repr(l) == repr(r):
                return NumericExpr(1)
            if isinstance(r, NumericExpr) and r.value == 1:
                return l

        if self.op == "**":
            if isinstance(r, NumericExpr):
                if r.value == 0: return NumericExpr(1)
                if r.value == 1: return l
        
        return BinOpExpr(l, r, self.op)
    
    def infer_shape(self):
        from loom.core.shape import broadcast_shapes
        return broadcast_shapes(self.left.infer_shape(), self.right.infer_shape())

    def has_trig(self) -> bool:
        return self.left.has_trig() or self.right.has_trig()

    def has_log_exp(self) -> bool:
        return self.left.has_log_exp() or self.right.has_log_exp()
        
    def __repr__(self) -> str:
        if self.op == '*' and isinstance(self.left, NumericExpr) and self.left.value == -1:
             return f"(-{self.right})"
        op_str = "^" if self.op == "**" else self.op
        if self.op == "/":
            if isinstance(self.left, NumericExpr) and isinstance(self.right, NumericExpr):
                return f"({self.left}/{self.right})"
            return f"({self.left} / {self.right})"
        return f"({self.left} {op_str} {self.right})"

@dataclass(frozen=True)
class FuncExpr(Expression):
    """Represents a function call (e.g., sin(x))."""
    func_name: str
    arg: Expression
    
    def free_symbols(self) -> Set[str]:
        return self.arg.free_symbols()
    
    def substitute(self, symbol: str, value: Union[float, int, Expression]) -> Expression:
        return FuncExpr(self.func_name, self.arg.substitute(symbol, value))
    
    def evaluate(self) -> Union[float, int, complex]:
        val = self.arg.evaluate()
        if self.func_name == 'sin': return math.sin(val)
        if self.func_name == 'cos': return math.cos(val)
        if self.func_name == 'tan': return math.tan(val)
        if self.func_name == 'exp': return math.exp(val)
        if self.func_name == 'log': return math.log(val)
        if self.func_name == 'sqrt': return math.sqrt(val)
        if self.func_name == 'erf':
            from loom.special.erf import erf
            return erf(val)
        if self.func_name == 'gamma':
            from loom.special.gamma import gamma
            return gamma(val)
        raise ValueError(f"Unknown function: {self.func_name}")
    
    def simplify(self) -> Expression:
        arg = self.arg.simplify()
        if isinstance(arg, NumericExpr):
            return NumericExpr(self.evaluate())
        return FuncExpr(self.func_name, arg)
    
    def infer_shape(self):
        return self.arg.infer_shape()

    def has_trig(self) -> bool:
        return self.func_name in {'sin', 'cos', 'tan'} or self.arg.has_trig()

    def has_log_exp(self) -> bool:
        return self.func_name in {'exp', 'log'} or self.arg.has_log_exp()
        
    def __repr__(self) -> str:
        return f"{self.func_name}({self.arg})"

