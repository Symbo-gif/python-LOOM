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
Backend Manager.

Handles backend selection, registration, and automatic fallback.
"""

from typing import Dict, Optional, List
from loom.backend.base import Backend
from loom.backend.cpu import CPUBackend, get_cpu_backend


class BackendManager:
    """
    Manages available backends and provides automatic selection and fallback.
    
    Usage:
        manager = get_backend_manager()
        manager.set_backend('numba')  # Try to use Numba
        backend = manager.get_backend()  # Returns Numba if available, else CPU
    """
    
    _instance = None
    
    def __init__(self):
        self._backends: Dict[str, Backend] = {}
        self._current_backend_name: str = 'cpu'
        
        # Register CPU backend (always available)
        self._backends['cpu'] = get_cpu_backend()
        
        # Try to register optional backends
        self._try_register_optional_backends()
    
    def _try_register_optional_backends(self):
        """Attempt to register optional accelerated backends."""
        try:
            from loom.backend.numba_backend import get_numba_backend
            backend = get_numba_backend()
            if backend.is_available:
                self._backends['numba'] = backend
        except Exception:
            pass
        
        try:
            from loom.backend.cython_backend import get_cython_backend
            backend = get_cython_backend()
            if backend.is_available:
                self._backends['cython'] = backend
        except Exception:
            pass
        
        try:
            from loom.backend.cuda_backend import get_cuda_backend
            backend = get_cuda_backend()
            if backend.is_available:
                self._backends['cuda'] = backend
        except Exception:
            pass
    
    def available_backends(self) -> List[str]:
        """Return list of available backend names."""
        return list(self._backends.keys())
    
    def set_backend(self, name: str) -> bool:
        """
        Set the current backend by name.
        
        Args:
            name: Backend name ('cpu', 'numba', 'cython', 'cuda')
            
        Returns:
            True if backend was set successfully, False if not available (falls back to CPU)
        """
        if name in self._backends:
            self._current_backend_name = name
            return True
        else:
            # Fallback to CPU
            self._current_backend_name = 'cpu'
            return False
    
    def get_backend(self) -> Backend:
        """Get the current active backend."""
        return self._backends.get(self._current_backend_name, self._backends['cpu'])
    
    def get_backend_name(self) -> str:
        """Get the name of the current active backend."""
        return self._current_backend_name
    
    def register_backend(self, name: str, backend: Backend):
        """
        Register a custom backend.
        
        Args:
            name: Unique name for the backend
            backend: Backend instance implementing the Backend interface
        """
        if not isinstance(backend, Backend):
            raise TypeError("backend must be an instance of Backend")
        self._backends[name] = backend


# Singleton accessor
_manager_instance: Optional[BackendManager] = None

def get_backend_manager() -> BackendManager:
    """Get the global BackendManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = BackendManager()
    return _manager_instance


def set_backend(name: str) -> bool:
    """
    Set the global computation backend.
    
    Args:
        name: Backend name ('cpu', 'numba', 'cython', 'cuda')
        
    Returns:
        True if successful, False if backend unavailable (falls back to CPU)
        
    Example:
        >>> import loom as tf
        >>> tf.set_backend('cuda')  # Use GPU if available
        True
    """
    return get_backend_manager().set_backend(name)


def get_backend() -> Backend:
    """Get the current active backend."""
    return get_backend_manager().get_backend()


def available_backends() -> List[str]:
    """
    List available backends on this system.
    
    Returns:
        List of backend names that are available
        
    Example:
        >>> import loom as tf
        >>> tf.available_backends()
        ['cpu', 'numba']  # Numba available, CUDA not installed
    """
    return get_backend_manager().available_backends()

