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
Input/Output operations for loom.
"""

import json
import base64
import struct
from typing import Union
from loom.core.tensor import Tensor, array

def save(tensor: Tensor, filename: str):
    """
    Save a tensor to a file in .tfdata format (JSON-based).
    """
    data = {
        "shape": tensor.shape.dims,
        "dtype": tensor.dtype.name,
        "values": tensor.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load(filename: str) -> Tensor:
    """
    Load a tensor from a .tfdata file.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
        
    return array(
        data["values"],
        dtype=data["dtype"]
    )

