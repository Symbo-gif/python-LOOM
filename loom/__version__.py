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
loom version information.

This file is the single source of truth for the package version.
"""

__version__ = "0.9.1"
__version_info__ = (0, 9, 1)

# Version history (established facts):
# 0.9.1 - Bug Fixes & Test Stabilization (January 2026)
#         - Added create_std_op and create_var_op for std/var reduction operations
#         - Fixed ODE solver integration loop and output format (SciPy-compatible)
#         - Added epsilon protection to division operations in linalg solvers
#         - Fixed window normalization in signal module
#         - Fixed IIR filter coefficient normalization
#         - Added complex eigenvalue support to sqrtm with regularization
#         - Fixed precision issues in scalar operations (dtype inheritance)
#         - Added conj, real, imag, angle, polar to top-level exports
#         - Fixed special case for Poisson PMF with lambda=0
#         - All 1556 tests passing
# 0.9.0 - Phase 9: Production Quality (January 2026)
#         - logm, outer product, rectangular LU, N-D field sampling
#         - Chi-square proper p-value via gammainc
#         - Accelerated backends (CPU, Numba, Cython, CUDA)
#         - All TODOs removed, 293 tests passing
# 0.8.0 - Phase 8: Numeric Optimization & Full Parity (January 2026)
# 0.7.0 - Phase 7: Field Tensors & Agent Orchestration (January 2026)
# 0.6.0 - Phase 6: Sparse & Spatial Algorithms (January 2026)
# 0.5.0 - Phase 5: Signal & Special Functions (January 2026)
# 0.4.0 - Phase 4: Optimization & Integration (January 2026)
# 0.3.0 - Phase 3: Symbolic Core (January 2026)
# 0.2.0 - Phase 2: Linear Algebra (January 2026)
# 0.1.0 - Phase 1: Core Tensors (January 2026)
# 0.0.1 - Initial scaffolding (January 2026)

