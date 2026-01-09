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
Daemon-driven asynchronous computation for loom.
"""

import threading
import queue
import time
from typing import Callable, Any, Dict, Optional

class ComputationDaemon:
    """
    A background worker that executes tensor operations asynchronously.
    """
    def __init__(self, name: str = "TF-Daemon"):
        self.name = name
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.results = {}
        self._task_id_counter = 0

    def start(self):
        """Start the background worker thread."""
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        """Stop the background worker."""
        self._stop_event.set()
        self.queue.put(None) # Wake up the thread to exit
        if self._thread.is_alive():
            self._thread.join()

    def submit(self, func: Callable, *args, **kwargs) -> int:
        """Submit a task to the daemon."""
        task_id = self._task_id_counter
        self._task_id_counter += 1
        self.queue.put((task_id, func, args, kwargs))
        return task_id

    def get_result(self, task_id: int, wait: bool = True) -> Any:
        """Retrieve a task result."""
        while wait and task_id not in self.results:
            if self._stop_event.is_set():
                break
            time.sleep(0.01)
        return self.results.get(task_id)

    def _run(self):
        while not self._stop_event.is_set():
            item = self.queue.get()
            if item is None:
                break
            
            task_id, func, args, kwargs = item
            try:
                # Force computation if result is a lazy Tensor
                result = func(*args, **kwargs)
                if hasattr(result, 'compute'):
                    result.compute()
                self.results[task_id] = result
            except Exception as e:
                self.results[task_id] = e
            finally:
                self.queue.task_done()

