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
- Backend auto-detection prioritizes: numba → cpu

SUPPORTED BACKENDS:
- "cpu": Pure Python (always available)
- "numba": Numba JIT acceleration (optional, install numba)
- "cython": Cython acceleration (optional, install loom[cython])
- "cuda": GPU acceleration (optional, install cupy)
- "auto": Automatically select best available backend
"""

from typing import Literal, Dict, Any, List

# Type alias for backend names
BackendType = Literal["auto", "cpu", "numba", "cython", "cuda"]

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Default computation backend
# Options: "cpu" (always available), "numba" (optional), "cython" (optional), "cuda" (optional)
DEFAULT_BACKEND: str = "cpu"

# Backend availability flags (set at import time by backend/manager.py)
NUMBA_AVAILABLE: bool = False
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

def set_backend(backend: BackendType = "auto") -> str:
    """
    Set the default computation backend.
    
    Args:
        backend: One of "auto", "cpu", "numba", "cython", "cuda"
            "auto": Automatically select best available backend
                    Priority order: numba → cpu
            "cpu": Force pure Python (no acceleration, always available)
            "numba": Force Numba JIT (raises if unavailable)
            "cython": Force Cython (raises if unavailable)
            "cuda": Force CUDA GPU (raises if unavailable)
    
    Returns:
        Name of the activated backend
    
    Raises:
        ValueError: If the specified backend is not available
    
    Example:
        >>> import loom
        >>> loom.config.set_backend("auto")
        'numba'  # If Numba installed
        >>> loom.config.set_backend("cpu")
        'cpu'  # Force pure Python
    """
    global DEFAULT_BACKEND
    
    from loom.backend import (
        get_backend_manager,
        set_backend as _set_backend,
        available_backends as _available_backends,
    )
    
    manager = get_backend_manager()
    available = _available_backends()
    
    if backend == "auto":
        # Priority order for auto-selection: numba → cpu
        for preferred in ["numba", "cpu"]:
            if preferred in available:
                backend = preferred
                break
        else:
            backend = "cpu"  # Fallback
    
    if backend not in available and backend != "cpu":
        raise ValueError(
            f"Backend '{backend}' is not available. "
            f"Available backends: {available}. "
            f"Install numba with: pip install numba"
        )
    
    result = _set_backend(backend)
    if result:
        DEFAULT_BACKEND = backend
        return backend
    else:
        # Fallback to CPU
        DEFAULT_BACKEND = "cpu"
        return "cpu"


def get_backend() -> str:
    """
    Return the current default backend name.
    
    Returns:
        Name of the active backend (e.g., 'cpu', 'numba')
    """
    from loom.backend import get_backend as _get_backend
    return _get_backend().name


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the active backend.
    
    Returns:
        Dictionary containing:
            - name: Backend name
            - available: Whether the backend is currently available
            - all_available: List of all available backends
    
    Example:
        >>> import loom
        >>> loom.config.get_backend_info()
        {'name': 'cpu', 'available': True, 'all_available': ['cpu', 'numba']}
    """
    from loom.backend import (
        get_backend as _get_backend,
        available_backends as _available_backends,
    )
    
    backend = _get_backend()
    return {
        "name": backend.name,
        "available": backend.is_available,
        "all_available": _available_backends(),
    }


def list_backends() -> List[str]:
    """
    List all available backends on this system.
    
    Returns:
        List of backend names that are available
    
    Example:
        >>> import loom
        >>> loom.config.list_backends()
        ['cpu', 'numba']  # If Numba is installed
    """
    from loom.backend import available_backends as _available_backends
    return _available_backends()


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

