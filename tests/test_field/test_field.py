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

import pytest
import loom as tf
from loom.field import FieldTensor, compress_eigen, decompress_eigen

def test_field_tensor_sampling_1d():
    t = tf.array([0, 10, 20, 30])
    ft = FieldTensor(t)
    
    # Midpoint
    assert ft.sample([0.5]) == 5.0
    assert ft.sample([1.5]) == 15.0

def test_field_tensor_sampling_2d():
    # [[0, 1], [2, 3]]
    t = tf.array([[0, 1], [1, 2]])
    ft = FieldTensor(t)
    
    # Center (0.5, 0.5)
    # Average of 0, 1, 1, 2 is 1.0
    assert ft.sample([0.5, 0.5]) == 1.0
    
    # (0, 0.5) -> average of 0 and 1 is 0.5
    assert ft.sample([0.0, 0.5]) == 0.5

def test_eigen_compression():
    # Rank 1 matrix
    t = tf.array([[1, 2], [2, 4]])
    
    compressed = compress_eigen(t, rank=1)
    assert compressed["rank"] == 1
    
    decompressed = decompress_eigen(compressed)
    
    # Should be exact for rank 1
    for row_orig, row_dec in zip(t.tolist(), decompressed.tolist()):
        for v1, v2 in zip(row_orig, row_dec):
            assert abs(v1 - v2) < 1e-10

def test_eigen_compression_lossy():
    # Rank 2 matrix
    t = tf.array([[1, 0.1], [0.1, 1]])
    
    # Compress to rank 1
    compressed = compress_eigen(t, rank=1)
    decompressed = decompress_eigen(compressed)
    
    # Capture some energy, but not all
    # Magnitude should be roughly correct
    orig_sum = t.sum().item()
    dec_sum = decompressed.sum().item()
    assert abs(orig_sum - dec_sum) < 0.5

