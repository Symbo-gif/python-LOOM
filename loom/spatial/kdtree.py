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
K-Dimensional Tree for fast spatial queries.
"""

from typing import List, Tuple, Union, Optional
from loom.spatial.distance import euclidean

class KDNode:
    def __init__(self, point: List[float], index: int, left=None, right=None, axis=0):
        self.point = point
        self.index = index
        self.left = left
        self.right = right
        self.axis = axis

class KDTree:
    """
    KDTree implementation for nearest neighbor search.
    """
    def __init__(self, data: List[List[float]]):
        if not data:
            self.root = None
            return
        self.ndim = len(data[0])
        self.root = self._build(data, 0)

    def _build(self, points: List[List[float]], depth: int) -> Optional[KDNode]:
        if not points:
            return None
        
        # Points with indices
        if isinstance(points[0], list):
             # First call, wrap with indices
             points = list(enumerate(points))
             # points is now [(idx, [x, y]), ...]
        
        axis = depth % self.ndim
        # Sort and pick median by value
        points.sort(key=lambda x: x[1][axis])
        median = len(points) // 2
        
        idx, pt = points[median]
        
        return KDNode(
            point=pt,
            index=idx,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1:], depth + 1),
            axis=axis
        )

    def query(self, point: List[float]) -> Tuple[float, int]:
        """
        Query the nearest neighbor.
        Returns (distance, index).
        """
        if self.root is None:
             raise ValueError("Tree is empty")
             
        best_pt, best_dist, best_idx = self._search(self.root, point, float('inf'), None, -1)
        return best_dist, best_idx

    def _search(self, node: Optional[KDNode], target: List[float], 
                best_dist: float, best_point: Optional[List[float]], best_idx: int) -> Tuple[List[float], float, int]:
        if node is None:
            return best_point, best_dist, best_idx
            
        dist = euclidean(node.point, target)
        if dist < best_dist:
            best_dist = dist
            best_point = node.point
            best_idx = node.index
            
        axis = node.axis
        diff = target[axis] - node.point[axis]
        
        # Decide which side to search first
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        
        best_point, best_dist, best_idx = self._search(near, target, best_dist, best_point, best_idx)
        
        # Check if we need to search the other side
        if abs(diff) < best_dist:
            best_point, best_dist, best_idx = self._search(far, target, best_dist, best_point, best_idx)
            
        return best_point, best_dist, best_idx

