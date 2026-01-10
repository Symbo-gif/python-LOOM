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
Tests for backend configuration via top-level loom API.

Task V1.1-003: Backend Configuration Testing
"""

import pytest
import loom


def test_set_backend_python():
    """Test setting Python (cpu) backend explicitly."""
    result = loom.set_backend('cpu')
    assert result == 'cpu'
    assert loom.config.get_backend() == 'cpu'


def test_set_backend_auto():
    """Test auto-detection."""
    result = loom.set_backend('auto')
    assert result in ['cpu', 'numba']


def test_get_backend_info():
    """Test getting backend information."""
    info = loom.get_backend_info()
    assert 'name' in info
    assert 'available' in info
    assert info['available'] is True


def test_operations_use_selected_backend():
    """Test that operations actually use the selected backend."""
    loom.set_backend('cpu')
    
    a = loom.array([[1, 2], [3, 4]])
    b = loom.array([[5, 6], [7, 8]])
    c = loom.matmul(a, b)
    
    assert c.tolist() == [[19, 22], [43, 50]]


def test_top_level_set_backend_available():
    """Test that set_backend is available at top level."""
    assert hasattr(loom, 'set_backend')
    assert callable(loom.set_backend)


def test_top_level_get_backend_info_available():
    """Test that get_backend_info is available at top level."""
    assert hasattr(loom, 'get_backend_info')
    assert callable(loom.get_backend_info)


def test_top_level_matmul_available():
    """Test that matmul is available at top level."""
    assert hasattr(loom, 'matmul')
    assert callable(loom.matmul)
