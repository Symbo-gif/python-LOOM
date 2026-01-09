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
Pytest configuration and fixtures for loom tests.

This file is automatically loaded by pytest.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def tf():
    """Import loom module."""
    import loom as tf_module
    return tf_module


@pytest.fixture
def sample_tensor(tf):
    """Create a sample tensor for testing."""
    from loom.core import Tensor
    return Tensor([[1, 2, 3], [4, 5, 6]], shape=(2, 3))


@pytest.fixture  
def sample_shape():
    """Create a sample shape for testing."""
    from loom.core import Shape
    return Shape((2, 3, 4))

