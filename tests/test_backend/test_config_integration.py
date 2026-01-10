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
Tests for the config module backend integration.

Task V1.1-003: Backend Configuration Integration Testing
"""

import pytest

import loom


class TestConfigBackendIntegration:
    """Test config module backend integration."""

    def test_config_set_backend_cpu(self):
        """Setting backend to CPU through config should work."""
        result = loom.config.set_backend("cpu")
        assert result == "cpu"
        assert loom.config.get_backend() == "cpu"

    def test_config_set_backend_auto(self):
        """Auto backend selection should return available backend."""
        result = loom.config.set_backend("auto")
        # Should return either cpu or numba (if available)
        assert result in ["cpu", "numba"]
        assert loom.config.get_backend() == result

    def test_config_get_backend_info(self):
        """get_backend_info should return correct info structure."""
        info = loom.config.get_backend_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "available" in info
        assert "all_available" in info
        assert isinstance(info["name"], str)
        assert isinstance(info["available"], bool)
        assert isinstance(info["all_available"], list)

    def test_config_list_backends(self):
        """list_backends should return available backends."""
        backends = loom.config.list_backends()
        assert isinstance(backends, list)
        assert "cpu" in backends

    def test_config_set_backend_unavailable_raises(self):
        """Setting unavailable backend should raise ValueError."""
        with pytest.raises(ValueError):
            loom.config.set_backend("nonexistent_backend")

    def test_config_auto_prefers_accelerated(self):
        """Auto selection should prefer numba if available."""
        backends = loom.config.list_backends()
        result = loom.config.set_backend("auto")
        
        if "numba" in backends:
            assert result == "numba"
        else:
            assert result == "cpu"


class TestConfigBackendType:
    """Test BackendType type alias."""

    def test_backend_type_exists(self):
        """BackendType should be exported from config module."""
        assert hasattr(loom.config, "BackendType")


class TestConfigBackendFlags:
    """Test backend availability flags."""

    def test_numba_available_flag(self):
        """NUMBA_AVAILABLE flag should exist."""
        assert hasattr(loom.config, "NUMBA_AVAILABLE")
        assert isinstance(loom.config.NUMBA_AVAILABLE, bool)

    def test_cuda_available_flag(self):
        """CUDA_AVAILABLE flag should exist."""
        assert hasattr(loom.config, "CUDA_AVAILABLE")
        assert isinstance(loom.config.CUDA_AVAILABLE, bool)

    def test_cython_available_flag(self):
        """CYTHON_AVAILABLE flag should exist."""
        assert hasattr(loom.config, "CYTHON_AVAILABLE")
        assert isinstance(loom.config.CYTHON_AVAILABLE, bool)
