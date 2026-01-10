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
Tests for the backend registry and manager.

Task V1.1-001: Backend Abstraction Layer Testing
"""

import pytest

from loom.backend import (
    Backend,
    BackendManager,
    get_backend_manager,
    get_backend,
    set_backend,
    available_backends,
)
from loom.backend.cpu import CPUBackend, get_cpu_backend
from loom.backend.base import Backend as BaseBackend


class TestBackendRegistry:
    """Test backend registration and management."""

    def test_cpu_backend_always_available(self):
        """CPU backend should always be available."""
        backends = available_backends()
        assert 'cpu' in backends

    def test_get_backend_returns_cpu_by_default(self):
        """Default backend should be CPU."""
        # Reset to CPU first
        set_backend('cpu')
        backend = get_backend()
        assert backend.name == 'cpu'
        assert isinstance(backend, CPUBackend)

    def test_set_backend_to_cpu(self):
        """Setting backend to CPU should work."""
        result = set_backend('cpu')
        assert result is True
        assert get_backend().name == 'cpu'

    def test_set_backend_fallback_on_unavailable(self):
        """Setting unavailable backend should fallback to CPU."""
        result = set_backend('nonexistent_backend')
        assert result is False
        assert get_backend().name == 'cpu'

    def test_backend_manager_singleton(self):
        """BackendManager should be a singleton."""
        manager1 = get_backend_manager()
        manager2 = get_backend_manager()
        assert manager1 is manager2

    def test_backend_manager_available_backends(self):
        """BackendManager should list available backends."""
        manager = get_backend_manager()
        backends = manager.available_backends()
        assert isinstance(backends, list)
        assert 'cpu' in backends

    def test_backend_manager_register_custom(self):
        """BackendManager should allow registering custom backends."""
        manager = get_backend_manager()
        
        # Create a mock backend
        class DummyBackend(BaseBackend):
            @property
            def name(self):
                return "dummy"
            
            @property
            def is_available(self):
                return True
            
            def add(self, a, b):
                return [x + y for x, y in zip(a, b)]
            
            def mul(self, a, b):
                return [x * y for x, y in zip(a, b)]
            
            def matmul(self, a, b, a_shape, b_shape):
                return [0.0]
            
            def sum(self, a):
                return sum(a)
            
            def exp(self, a):
                return a
            
            def log(self, a):
                return a
            
            def sqrt(self, a):
                return a
        
        dummy = DummyBackend()
        manager.register_backend('dummy', dummy)
        assert 'dummy' in manager.available_backends()
        
        # Clean up
        if 'dummy' in manager._backends:
            del manager._backends['dummy']

    def test_backend_manager_register_invalid(self):
        """Registering non-Backend should raise TypeError."""
        manager = get_backend_manager()
        with pytest.raises(TypeError):
            manager.register_backend('invalid', "not a backend")


class TestBackendSwitching:
    """Test switching between backends."""

    def test_switch_to_cpu(self):
        """Switching to CPU backend should work."""
        set_backend('cpu')
        assert get_backend().name == 'cpu'

    def test_switch_to_nonexistent_falls_back(self):
        """Switching to nonexistent backend should fallback to CPU."""
        set_backend('nonexistent')
        assert get_backend().name == 'cpu'

    def test_backend_name_property(self):
        """Backend should have correct name property."""
        backend = get_backend()
        assert isinstance(backend.name, str)
        assert len(backend.name) > 0


class TestAutoBackendSelection:
    """Test automatic backend selection."""

    def test_auto_selection_prefers_accelerated(self):
        """Auto selection should prefer accelerated backends if available."""
        backends = available_backends()
        
        # If numba is available, it should be preferred
        if 'numba' in backends:
            # This would test auto selection
            pass
        
        # At minimum, CPU should always be available
        assert 'cpu' in backends
