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
Simplification engine for symbolic expressions.
"""

from loom.symbolic.ast import Expression

def simplify(expr: Expression) -> Expression:
    """
    Simplify a symbolic expression.
    
    Args:
        expr: The expression to simplify.
        
    Returns:
        A simplified expression.
    """
    if hasattr(expr, 'simplify'):
        return expr.simplify()
    return expr

