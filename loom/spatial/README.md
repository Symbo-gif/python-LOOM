# loom Spatial Module

**KDTree, ConvexHull, and distance metrics**

## Status: ✅ Production-ready (v1.1.0)

_Validated by KDTree, convex hull, and distance metric tests in the v1.1.0 release suite._

## Features

- **KDTree**:
    - Fast Nearest-Neighbor Search (NNS).
    - Recursive construction for balanced trees.
    - `query()` method for finding the closest point in $O(\log n)$.
- **Convex Hull**:
    - 2D Convex Hull implementation using the Gift Wrapping (Jarvis March) algorithm.
- **Distance Metrics**:
    - `euclidean`, `manhattan`, `minkowski`.

## Usage Example

```python
from loom.spatial import KDTree

points = [[0, 0], [1, 1], [2, 2]]
tree = KDTree(points)

nearest, dist = tree.query([0.1, 0.1]) # ([0.0, 0.0], 0.141...)
```
