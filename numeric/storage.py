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
Optimized dense array storage for loom.
"""

import array
from typing import List, Union, Any, Tuple
from loom.core.dtype import DType

class NumericBuffer:
    """
    Optimized storage for tensor data using array.array for primitives.
    """
    _dtype_map = {
        DType.FLOAT32.value: 'f',
        DType.FLOAT64.value: 'd',
        DType.INT32.value: 'i',
        DType.INT64.value: 'q',
    }

    def __init__(self, data: Union[List, Any, int], dtype: DType):
        # print(f"DEBUG: NumericBuffer init dtype={dtype}")
        self.dtype = dtype
        
        # Check for complex numbers if real dtype was provided
        if not (dtype == DType.COMPLEX64 or dtype == DType.COMPLEX128):
            if not isinstance(data, int):
                # We need to flatten to check, but let's be pragmatic
                # If it's a list, check first element at least, or better, use a helper
                if self._check_is_complex(data):
                    self.dtype = DType.COMPLEX128 # Default upgrade
        
        self._is_complex = self.dtype in (DType.COMPLEX64, DType.COMPLEX128)
        type_code = self._dtype_map.get(self.dtype.value, 'd') if not self._is_complex else ('d' if self.dtype == DType.COMPLEX128 else 'f')

        if isinstance(data, int):
            # Pre-allocate with size
            if self._is_complex:
                self._real = array.array(type_code, [0.0] * data)
                self._imag = array.array(type_code, [0.0] * data)
            else:
                self._buffer = array.array(type_code, [0] * data if 'i' in type_code or 'q' in type_code else [0.0] * data)
            return

        if self._is_complex:
            # Flatten if multi-dimensional
            flat_data = self._flatten(data)
            self._real = array.array(type_code, [float(v.real if hasattr(v, 'real') else v) for v in flat_data])
            self._imag = array.array(type_code, [float(v.imag if hasattr(v, 'imag') else 0.0) for v in flat_data])
        else:
            flat_data = self._flatten(data)
            cast_func = int if 'i' in type_code or 'q' in type_code else float
            self._buffer = array.array(type_code, [cast_func(v) for v in flat_data])

    def _check_is_complex(self, data):
        if isinstance(data, complex): return True
        if isinstance(data, (list, tuple)) and len(data) > 0:
            # Deep check? or just first?
            # To be safe and efficient, let's check recursively but limit depth?
            # Or just check if any in flat?
            # For now, let's check if the first element is complex or contains complex
            item = data[0]
            if isinstance(item, complex): return True
            if isinstance(item, (list, tuple)): return self._check_is_complex(item)
        return False

    def _flatten(self, data):
        if not isinstance(data, (list, tuple)):
            if hasattr(data, 'tolist'): # Handle tensors/buffers
                return data.tolist()
            return [data]
        if len(data) == 0:
            return []
        if not isinstance(data[0], (list, tuple)):
            return data
        # Deep flatten
        res = []
        for item in data:
            res.extend(self._flatten(item))
        return res

    def tolist(self) -> List[Any]:
        if self._is_complex:
            return [complex(r, i) for r, i in zip(self._real, self._imag)]
        return self._buffer.tolist()

    def __len__(self):
        return len(self._real if self._is_complex else self._buffer)

    def __getitem__(self, idx):
        if self._is_complex:
            if isinstance(idx, slice):
                return [complex(r, i) for r, i in zip(self._real[idx], self._imag[idx])]
            return complex(self._real[idx], self._imag[idx])
        if isinstance(idx, slice):
            return self._buffer[idx].tolist()
        return self._buffer[idx]

    def __setitem__(self, idx, value):
        if self._is_complex:
            val = complex(value)
            if isinstance(idx, slice):
                # Calculate slice length
                start, stop, step = idx.indices(len(self._real))
                length = len(range(start, stop, step))
                self._real[idx] = array.array(self._real.typecode, [val.real] * length)
                self._imag[idx] = array.array(self._imag.typecode, [val.imag] * length)
            else:
                self._real[idx] = val.real
                self._imag[idx] = val.imag
        else:
            cast_func = int if 'i' in self._buffer.typecode or 'q' in self._buffer.typecode else float
            val = cast_func(value)
            if isinstance(idx, slice):
                start, stop, step = idx.indices(len(self._buffer))
                length = len(range(start, stop, step))
                self._buffer[idx] = array.array(self._buffer.typecode, [val] * length)
            else:
                self._buffer[idx] = val

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return self.tolist() == list(other)
        if isinstance(other, NumericBuffer):
            return self.tolist() == other.tolist()
        return False

    def __repr__(self):
        return f"NumericBuffer({self.tolist()}, dtype={self.dtype})"

    def copy(self) -> 'NumericBuffer':
        new_buf = object.__new__(NumericBuffer)
        new_buf.dtype = self.dtype
        new_buf._is_complex = self._is_complex
        if self._is_complex:
            new_buf._real = array.array(self._real.typecode, self._real)
            new_buf._imag = array.array(self._imag.typecode, self._imag)
        else:
            new_buf._buffer = array.array(self._buffer.typecode, self._buffer)
        return new_buf

