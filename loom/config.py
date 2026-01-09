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
loom global configuration.

This module defines global settings that control loom behavior.
All settings have sensible defaults and can be modified at runtime.

ESTABLISHED FACTS:
- Default backend is "cpu" (pure Python, always available)
- Default dtype is "float32" for memory efficiency
- Configuration can be modified at runtime

GOALS (not yet implemented):
- Backend auto-detection (CPU → Cython → CUDA)
- Persistent configuration via config file
"""

from typing import Literal

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Default computation backend
# Options: "cpu" (always available), "cython" (optional), "cuda" (optional)
DEFAULT_BACKEND: Literal["cpu", "cython", "cuda"] = "cpu"

# Backend availability flags (set at import time by backend/manager.py)
CYTHON_AVAILABLE: bool = False
CUDA_AVAILABLE: bool = False

# =============================================================================
# PRECISION CONFIGURATION
# =============================================================================

# Default data type for new tensors
DEFAULT_DTYPE: str = "float32"

# Numerical precision threshold
EPSILON: float = 1e-10

# =============================================================================
# MEMORY CONFIGURATION
# =============================================================================

# Maximum cache size for computed results (in MB)
CACHE_SIZE_MB: int = 512

# Enable/disable result caching
ENABLE_CACHING: bool = True

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Number of threads for parallel operations (future use)
NUM_THREADS: int = 4

# Enable Cython-optimized operations when available
USE_CYTHON: bool = False

# Enable CUDA acceleration when available
USE_CUDA: bool = False

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Enable verbose output
VERBOSE: bool = False

# Enable debug mode (extra checks, slower)
DEBUG: bool = False


# =============================================================================
# RUNTIME CONFIGURATION FUNCTIONS
# =============================================================================

def set_backend(backend: Literal["cpu", "cython", "cuda"]) -> None:
    """
    Set the default computation backend.
    
    Args:
        backend: One of "cpu", "cython", "cuda"
    
    Raises:
        ValueError: If backend is not available
    
    Example:
        >>> tf.config.set_backend("cuda")  # Use GPU
    """
    global DEFAULT_BACKEND
    
    if backend == "cython" and not CYTHON_AVAILABLE:
        raise ValueError("Cython backend not available. Install with: pip install loom[cython]")
    if backend == "cuda" and not CUDA_AVAILABLE:
        raise ValueError("CUDA backend not available. Install with: pip install loom[cuda]")
    
    DEFAULT_BACKEND = backend


def get_backend() -> str:
    """Return the current default backend."""
    return DEFAULT_BACKEND


def set_dtype(dtype: str) -> None:
    """
    Set the default data type for new tensors.
    
    Args:
        dtype: One of "float32", "float64", "int32", "int64", etc.
    """
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype


def get_dtype() -> str:
    """Return the current default dtype."""
    return DEFAULT_DTYPE

