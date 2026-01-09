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

"""loom Spatial Module."""

from loom.spatial.distance import minkowski, euclidean, manhattan
from loom.spatial.kdtree import KDTree
from loom.spatial.hull import convex_hull

__all__ = [
    "minkowski",
    "euclidean",
    "manhattan",
    "KDTree",
    "convex_hull",
]

