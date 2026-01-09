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
loom Backend Base Interface.

All backends must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Tuple


class Backend(ABC):
    """
    Abstract backend interface for loom operations.
    
    All computation backends (CPU, Cython, Numba, CUDA) must implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name (e.g., 'cpu', 'numba', 'cuda')."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def add(self, a: List[float], b: List[float]) -> List[float]:
        """Element-wise addition."""
        pass
    
    @abstractmethod
    def mul(self, a: List[float], b: List[float]) -> List[float]:
        """Element-wise multiplication."""
        pass
    
    @abstractmethod
    def matmul(self, a: List[float], b: List[float], 
               a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> List[float]:
        """Matrix multiplication."""
        pass
    
    @abstractmethod
    def sum(self, a: List[float]) -> float:
        """Sum all elements."""
        pass
    
    @abstractmethod
    def exp(self, a: List[float]) -> List[float]:
        """Element-wise exponential."""
        pass
    
    @abstractmethod
    def log(self, a: List[float]) -> List[float]:
        """Element-wise natural logarithm."""
        pass
    
    @abstractmethod
    def sqrt(self, a: List[float]) -> List[float]:
        """Element-wise square root."""
        pass

