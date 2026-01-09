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
Linear Algebra module.

PHASE STATUS: Phase 2 - In Progress
"""

from loom.linalg.basics import (
    matmul,
    dot,
    vdot,
    inner,
    outer,
    trace,
    norm,
    matrix_transpose
)

from loom.linalg.decompositions import lu, qr, cholesky, svd, eig, eigh
from loom.linalg.solvers import solve, inv, det, matrix_rank, cond
from loom.linalg.matfuncs import expm, sqrtm, logm

__all__ = [
    "matmul",
    "dot",
    "vdot",
    "inner",
    "outer",
    "trace",
    "norm",
    "matrix_transpose",
    # Decompositions
    "lu", "qr", "cholesky", "svd", "eig", "eigh",
    # Solvers
    "solve", "inv", "det", "matrix_rank", "cond",
    # Matrix Functions
    "expm", "sqrtm", "logm"
]

