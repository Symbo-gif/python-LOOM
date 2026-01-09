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
Orchestration for multi-step tensor tasks.
"""

from typing import List, Dict, Any, Optional
from loom.agent.daemon import ComputationDaemon

class Supervisor:
    """
    Manages computation daemons and orchestrates multi-step execution.
    """
    def __init__(self, num_workers: int = 1):
        self.workers = [ComputationDaemon(f"Worker-{i}") for i in range(num_workers)]
        self.tasks = []
        for w in self.workers:
            w.start()

    def run_recipe(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a multi-step recipe.
        Format:
        {
            "steps": [
                {"name": "s1", "op": "add", "args": [A, B]},
                {"name": "s2", "op": "mul", "args": ["$s1", C]}
            ]
        }
        """
        results = {}
        for step in recipe.get("steps", []):
            name = step["name"]
            op_name = step["op"]
            args = step["args"]
            
            # Resolve references
            real_args = []
            for arg in args:
                if isinstance(arg, str) and arg.startswith("$"):
                    real_args.append(results[arg[1:]])
                else:
                    real_args.append(arg)
            
            # Find operation (simplified for now)
            import loom as tf
            if op_name == "add": op = lambda x, y: x + y
            elif op_name == "mul": op = lambda x, y: x * y
            elif op_name == "matmul": op = lambda x, y: x @ y
            else: raise ValueError(f"Unknown op in recipe: {op_name}")
            
            # Submit to first available worker
            worker = self.workers[0]
            tid = worker.submit(op, *real_args)
            results[name] = worker.get_result(tid)
            
        return results

    def shutdown(self):
        """Shut down all workers."""
        for w in self.workers:
            w.stop()

