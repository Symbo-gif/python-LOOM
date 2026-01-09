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
from loom.spatial import distance, KDTree, convex_hull

def test_distances():
    u = [0, 0]
    v = [3, 4]
    assert distance.euclidean(u, v) == 5.0
    assert distance.manhattan(u, v) == 7.0
    assert distance.minkowski(u, v, p=3) == (3**3 + 4**3)**(1/3)

def test_kdtree_basic():
    points = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    tree = KDTree(points)
    
    # Query point near [9, 6]
    dist, idx = tree.query([9, 5])
    nearest = points[idx]
    assert nearest == [9, 6]
    assert dist == 1.0
    
    # Query point near [2, 3]
    dist, idx = tree.query([1, 2])
    nearest = points[idx]
    assert abs(distance.euclidean([1,2], [2,3]) - dist) < 1e-12

def test_convex_hull_2d():
    points = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0.5)]
    hull = convex_hull(points)
    
    # Square points should be in hull
    expected = {(0, 0), (2, 0), (2, 2), (0, 2)}
    assert set(hull) == expected
    # (1, 1) and (1, 0.5) are inside
    assert (1, 1) not in hull
    assert (1, 0.5) not in hull

def test_kdtree_stress():
    # Large number of points in 3D
    import random
    random.seed(42)
    points = [[random.random() for _ in range(3)] for _ in range(1000)]
    tree = KDTree(points)
    
    query_point = [0.5, 0.5, 0.5]
    dist, idx = tree.query(query_point)
    nearest = points[idx]
    
    # Benchmarking against linear search
    min_dist = float('inf')
    best_p = None
    for p in points:
        d = distance.euclidean(p, query_point)
        if d < min_dist:
            min_dist = d
            best_p = p
            
    assert nearest == best_p
    assert abs(dist - min_dist) < 1e-12

