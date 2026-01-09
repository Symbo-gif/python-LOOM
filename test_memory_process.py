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

"""Memory leak and process termination tests."""

import pytest
import gc
import time
import threading
import loom as tf
import loom.linalg as la


class TestMemoryLeaks:
    """Memory leak tests - verify proper cleanup."""
    
    def test_repeated_tensor_allocation(self):
        """Create and destroy 1000 tensors, verify memory cleanup."""
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for _ in range(1000):
            t = tf.zeros((100, 100))
            result = t.sum()
            del t, result
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Allow reasonable growth but not proportional to iterations
        growth = final_objects - initial_objects
        assert growth < 5000, f"Memory leak detected: {growth} new objects"
    
    def test_large_tensor_gc(self):
        """Large tensor properly garbage collected."""
        gc.collect()
        
        # Create large tensor
        t = tf.zeros((1000, 1000))
        assert t.size == 1_000_000
        
        # Delete and collect
        del t
        gc.collect()
        
        # Should be able to allocate again
        t2 = tf.zeros((1000, 1000))
        assert t2.size == 1_000_000
    
    def test_dag_computation_cleanup(self):
        """DAG nodes cleaned up after computation."""
        gc.collect()
        initial = len(gc.get_objects())
        
        for _ in range(100):
            a = tf.array([1, 2, 3])
            b = a + 1
            c = b * 2
            d = c - 1
            e = d / 2
            result = e.sum().item()
            del a, b, c, d, e, result
        
        gc.collect()
        final = len(gc.get_objects())
        
        growth = final - initial
        assert growth < 2000, f"DAG memory leak: {growth} new objects"
    
    def test_linalg_operation_cleanup(self):
        """Linear algebra operations don't leak."""
        gc.collect()
        initial = len(gc.get_objects())
        
        for _ in range(50):
            A = tf.array([[1, 2], [3, 4]])
            P, L, U = la.lu(A)
            d = la.det(A)
            del A, P, L, U, d
        
        gc.collect()
        final = len(gc.get_objects())
        
        growth = final - initial
        assert growth < 1000, f"Linalg memory leak: {growth} new objects"
    
    def test_backend_switching_isolation(self):
        """Backend switching doesn't leak memory."""
        from loom.backend import set_backend, available_backends
        
        gc.collect()
        initial = len(gc.get_objects())
        
        backends = available_backends()
        for _ in range(20):
            for name in backends:
                set_backend(name)
                t = tf.zeros((10, 10))
                del t
        
        set_backend('cpu')
        gc.collect()
        final = len(gc.get_objects())
        
        growth = final - initial
        assert growth < 500, f"Backend switch memory leak: {growth} new objects"


class TestProcessTermination:
    """Tests for proper process/thread termination."""
    
    def test_thread_completion(self):
        """Worker threads complete properly."""
        results = []
        errors = []
        
        def worker(x):
            try:
                t = tf.array([x, x+1, x+2])
                results.append(t.sum().item())
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread didn't complete"
        
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
    
    def test_computation_terminates(self):
        """Long computations complete in reasonable time."""
        start = time.time()
        
        # Moderately complex computation
        a = tf.zeros((100, 100))
        for _ in range(10):
            a = a + 1
        result = a.sum().item()
        
        elapsed = time.time() - start
        assert elapsed < 10.0, f"Computation took too long: {elapsed}s"
        assert result == 100 * 100 * 10
    
    def test_exception_cleanup(self):
        """Exceptions don't leave resources hanging."""
        gc.collect()
        initial = len(gc.get_objects())
        
        for _ in range(50):
            try:
                t = tf.array([[1, 2], [3]])  # Invalid - inconsistent
            except ValueError:
                pass
        
        gc.collect()
        final = len(gc.get_objects())
        
        growth = final - initial
        assert growth < 500, f"Exception handling leak: {growth} new objects"
    
    def test_nested_operations_complete(self):
        """Deeply nested operations complete without hanging."""
        start = time.time()
        
        a = tf.array([1.0])
        for _ in range(200):
            a = a + 0.001
        
        result = a.item()
        elapsed = time.time() - start
        
        assert elapsed < 5.0, "Nested operations took too long"
        assert 1.1 < result < 1.3


class TestResourceCleanup:
    """Verify resources are properly released."""
    
    def test_buffer_reuse(self):
        """Buffers can be reused after deletion."""
        for _ in range(10):
            # Create, use, delete
            t = tf.randn((500, 500))
            _ = t.sum()
            del t
            gc.collect()
            
            # Should be able to create again
            t2 = tf.randn((500, 500))
            assert t2.size == 250000
            del t2
    
    def test_symbolic_cleanup(self):
        """Symbolic expressions cleaned up."""
        gc.collect()
        initial = len(gc.get_objects())
        
        for _ in range(100):
            x = tf.Symbol('x')
            expr = x + 1
            del x, expr
        
        gc.collect()
        final = len(gc.get_objects())
        
        growth = final - initial
        assert growth < 1000, f"Symbolic memory leak: {growth} new objects"

