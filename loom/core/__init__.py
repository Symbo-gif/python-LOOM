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
loom Core Module

Foundation classes for tensor representation and computation.

This module provides:
- Tensor: Base tensor class with lazy evaluation
- Shape: Immutable shape representation
- DType: Data type enumeration

ESTABLISHED FACTS:
- Core module is the foundation for all loom functionality
- Tensor class supports multiple representations (dense, symbolic, etc.)
- Shape is immutable for safety

PHASE STATUS: Phase 0 (Skeleton)
"""

from loom.core.dtype import DType
from loom.core.shape import Shape
from loom.core.tensor import Tensor, Symbol

__all__ = [
    "Tensor",
    "Symbol",
    "Shape",
    "DType",
]

