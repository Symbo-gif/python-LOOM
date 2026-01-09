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
Windowing functions.
"""

import math
from typing import Union, List, Optional
import loom as tf
from loom.core.tensor import Tensor, array

def boxcar(n: int) -> Tensor:
    """Rectangular window."""
    return tf.ones((n,))

def hamming(n: int) -> Tensor:
    """Hamming window."""
    if n == 1: return tf.ones((1,))
    indices = array(list(range(n)))
    return 0.54 - 0.46 * (2 * math.pi * indices / (n - 1)).cos()

def hanning(n: int) -> Tensor:
    """Hanning window."""
    if n == 1: return tf.ones((1,))
    indices = array(list(range(n)))
    return 0.5 * (1 - (2 * math.pi * indices / (n - 1)).cos())

def blackman(n: int) -> Tensor:
    """Blackman window."""
    if n == 1: return tf.ones((1,))
    indices = array(list(range(n)))
    alpha = 0.16
    a0 = (1 - alpha) / 2
    a1 = 0.5
    a2 = alpha / 2
    
    phi = 2 * math.pi * indices / (n - 1)
    return a0 - a1 * phi.cos() + a2 * (2 * phi).cos()

def bartlett(n: int) -> Tensor:
    """Bartlett (triangular) window."""
    if n == 1: return tf.ones((1,))
    indices = array(list(range(n)))
    return (2.0 / (n - 1)) * ( (n - 1)/2.0 - (indices - (n - 1)/2.0).abs() )

