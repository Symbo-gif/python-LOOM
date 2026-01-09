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
Convex Hull algorithms.
"""

from typing import List, Tuple

def convex_hull(points: List[Tuple[float, float]]) -> 'ConvexHullResult':
    """
    Compute the convex hull of a set of 2D points using Jarvis March (Gift Wrapping).
    """
    n = len(points)
    if n < 3:
        return ConvexHullResult(points)
        
    # Find the leftmost point
    start_point = min(points, key=lambda p: p[0])
    hull = []
    
    p = points.index(start_point)
    l = p
    
    while True:
        hull.append(points[p])
        
        # Next point 'q' such that all points are counter-clockwise from 'p-q'
        q = (p + 1) % n
        for i in range(n):
            if _orientation(points[p], points[i], points[q]) == 2:
                q = i
        
        p = q
        if p == l:
            break
            
    return ConvexHullResult(hull)

class ConvexHullResult:
    def __init__(self, vertices):
        self.vertices = vertices

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self):
        return len(self.vertices)
    
    def __getitem__(self, idx):
        return self.vertices[idx]

    def __repr__(self):
        return f"ConvexHullResult(vertices={self.vertices})"

def _orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
    """
    Find orientation of triplet (p, q, r).
    0 -> p, q, r are collinear
    1 -> Clockwise
    2 -> Counter-clockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

