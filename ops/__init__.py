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
loom Operations Module.

Element-wise operations, reductions, indexing, and manipulation.

PHASE STATUS: Phase 1 - Arithmetic & Reductions COMPLETE

Modules:
- arithmetic.py: +, -, *, /, **, //, %, abs, sqrt, exp, log, sin, cos, tan (50 tests)
- reduction.py: sum, mean, min, max, prod, argmax, argmin with axis (40 tests)
- indexing.py: advanced indexing - Week 7
"""

from loom.ops.arithmetic import (
    Operation,
    BinaryOp,
    UnaryOp,
    AddOp, SubOp, MulOp, DivOp, FloorDivOp, PowOp, ModOp,
    NegOp, AbsOp, SqrtOp, ExpOp, LogOp, SinOp, CosOp, TanOp,
    ADD, SUB, MUL, DIV, FLOORDIV, POW, MOD,
    NEG, ABS, SQRT, EXP, LOG, SIN, COS, TAN,
)

from loom.ops.reduction import (
    ReductionOp,
    SumOp, MeanOp, MaxOp, MinOp, ProdOp, ArgMaxOp, ArgMinOp,
    create_sum_op, create_mean_op, create_max_op, create_min_op, create_prod_op,
    create_argmax_op, create_argmin_op,
)

__all__ = [
    # Base classes
    "Operation", "BinaryOp", "UnaryOp", "ReductionOp",
    # Arithmetic Op classes
    "AddOp", "SubOp", "MulOp", "DivOp", "FloorDivOp", "PowOp", "ModOp",
    "NegOp", "AbsOp", "SqrtOp", "ExpOp", "LogOp", "SinOp", "CosOp", "TanOp",
    # Reduction Op classes
    "SumOp", "MeanOp", "MaxOp", "MinOp", "ProdOp", "ArgMaxOp", "ArgMinOp",
    # Singleton instances
    "ADD", "SUB", "MUL", "DIV", "FLOORDIV", "POW", "MOD",
    "NEG", "ABS", "SQRT", "EXP", "LOG", "SIN", "COS", "TAN",
    # Factory functions
    "create_sum_op", "create_mean_op", "create_max_op", "create_min_op", 
    "create_prod_op", "create_argmax_op", "create_argmin_op",
    # Matmul
    "MatmulOp", "create_matmul_op",
    # Manipulation
    "TransposeOp", "create_transpose_op",
]

from loom.ops.matmul import MatmulOp, create_matmul_op
from loom.ops.manipulation import TransposeOp, create_transpose_op

