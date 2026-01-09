# loom Agent Module

**Daemon-driven async computation and task orchestration**

## Status: ✅ COMPLETE (Phase 7)

## Features

- **ComputationDaemon**:
    - Background worker thread for executing tensor operations.
    - Thread-safe task submission and result retrieval.
- **Supervisor Orchestration**:
    - Manages a pool of workers.
    - **Recipe Execution**: Run multi-step computation DAGs defined via nested dictionaries.
    - Dependency resolution: Steps can refer to previous results using `$step_name`.

## Usage Example

```python
from loom.agent import ComputationDaemon, Supervisor
import loom as tf

# Async execution
daemon = ComputationDaemon()
daemon.start()
tid = daemon.submit(lambda x, y: x + y, lm.ones(5), lm.ones(5))
res = daemon.get_result(tid)

# Recipe
superv = Supervisor()
recipe = {
    "steps": [
        {"name": "s1", "op": "add", "args": [lm.array([1]), lm.array([2])]},
        {"name": "s2", "op": "mul", "args": ["$s1", lm.array([10])]}
    ]
}
results = superv.run_recipe(recipe) # {'s1': 3, 's2': 30}
```
