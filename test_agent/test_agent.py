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
from loom.agent import ComputationDaemon, Supervisor
import time

def test_daemon_async_add():
    daemon = ComputationDaemon()
    daemon.start()
    
    a = tf.array([1, 2])
    b = tf.array([3, 4])
    
    tid = daemon.submit(lambda x, y: x + y, a, b)
    result = daemon.get_result(tid)
    
    assert result.tolist() == [4.0, 6.0]
    daemon.stop()

def test_supervisor_recipe():
    superv = Supervisor()
    
    a = tf.array([1, 2])
    b = tf.array([3, 4])
    c = tf.array([1, 1])
    
    recipe = {
        "steps": [
            {"name": "add1", "op": "add", "args": [a, b]},
            {"name": "final", "op": "mul", "args": ["$add1", c]}
        ]
    }
    
    results = superv.run_recipe(recipe)
    assert "final" in results
    assert results["final"].tolist() == [4.0, 6.0]
    
    superv.shutdown()

