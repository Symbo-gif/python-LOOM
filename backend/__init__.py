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
loom Backend Module.

Provides computation backend abstraction with automatic fallback:
- CPU: Pure Python (always available)
- Numba: JIT compilation (optional, install numba)
- Cython: C extensions (optional, install loom[cython])
- CUDA: GPU acceleration (optional, install cupy)
"""

from loom.backend.base import Backend
from loom.backend.cpu import CPUBackend, get_cpu_backend
from loom.backend.manager import (
    BackendManager,
    get_backend_manager,
    get_backend,
    set_backend,
    available_backends,
)

__all__ = [
    "Backend",
    "CPUBackend",
    "BackendManager",
    "get_cpu_backend",
    "get_backend_manager",
    "get_backend",
    "set_backend",
    "available_backends",
]

