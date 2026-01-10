#!/usr/bin/env python3
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
Performance benchmarking suite for LOOM backends.

Run with: python benchmarks/benchmark_suite.py
"""

import time
import json
from typing import List, Dict, Any
from datetime import datetime


class Benchmark:
    """Single benchmark test case."""
    
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.results: Dict[str, float] = {}
    
    def run(self, backend: str, func, *args, **kwargs) -> float:
        """
        Run benchmark and return execution time.
        
        Returns:
            Execution time in seconds
        """
        import loom
        loom.set_backend(backend)
        
        # Warm-up (for JIT compilation)
        for _ in range(3):
            _ = func(*args, **kwargs)
        
        # Timed runs
        times: List[float] = []
        for _ in range(5):
            start = time.perf_counter()
            _ = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        # Return median time
        times.sort()
        median_time = times[len(times) // 2]
        
        self.results[backend] = median_time
        return median_time
    
    def compute_speedup(self, baseline: str = 'cpu') -> Dict[str, float]:
        """Compute speedup relative to baseline."""
        if baseline not in self.results:
            return {}
        
        baseline_time = self.results[baseline]
        speedups: Dict[str, float] = {}
        
        for backend, time_val in self.results.items():
            if backend != baseline:
                speedups[backend] = baseline_time / time_val
        
        return speedups
    
    def report(self) -> None:
        """Print benchmark results."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"Size: {self.size}")
        print(f"{'='*60}")
        
        for backend, time_val in sorted(self.results.items()):
            print(f"  {backend:15s}: {time_val*1000:8.2f} ms")
        
        speedups = self.compute_speedup()
        if speedups:
            print(f"\nSpeedup vs CPU:")
            for backend, speedup in sorted(speedups.items()):
                print(f"  {backend:15s}: {speedup:6.2f}x")


class BenchmarkSuite:
    """Collection of benchmarks."""
    
    def __init__(self):
        self.benchmarks: List[Benchmark] = []
    
    def add_matmul_benchmark(self, size: int) -> Benchmark:
        """Add matrix multiplication benchmark."""
        import random
        import loom
        
        random.seed(42)
        
        a = [[random.random() for _ in range(size)] for _ in range(size)]
        b = [[random.random() for _ in range(size)] for _ in range(size)]
        
        benchmark = Benchmark(f"MatMul {size}x{size}", size)
        
        def run_matmul():
            a_tensor = loom.array(a)
            b_tensor = loom.array(b)
            return loom.matmul(a_tensor, b_tensor)
        
        # Test on available backends
        available = loom.config.list_backends()
        for backend in available:
            try:
                benchmark.run(backend, run_matmul)
            except Exception as e:
                print(f"Warning: {backend} failed: {e}")
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def add_elementwise_benchmark(self, size: int) -> Benchmark:
        """Add element-wise operations benchmark."""
        import random
        import loom
        
        random.seed(42)
        
        a = [[random.random() for _ in range(size)] for _ in range(size)]
        b = [[random.random() for _ in range(size)] for _ in range(size)]
        
        benchmark = Benchmark(f"ElementWise {size}x{size}", size)
        
        def run_elementwise():
            a_tensor = loom.array(a)
            b_tensor = loom.array(b)
            return (a_tensor + b_tensor) * 2.0 - b_tensor
        
        available = loom.config.list_backends()
        for backend in available:
            try:
                benchmark.run(backend, run_elementwise)
            except Exception as e:
                print(f"Warning: {backend} failed: {e}")
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def run_all(self) -> None:
        """Run all benchmarks and display results."""
        import loom
        
        print("="*60)
        print("LOOM Performance Benchmark Suite")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Available backends: {loom.config.list_backends()}")
        print("="*60)
        
        for benchmark in self.benchmarks:
            benchmark.report()
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print summary of all benchmarks."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        # Collect average speedups
        backend_speedups: Dict[str, List[float]] = {}
        
        for benchmark in self.benchmarks:
            speedups = benchmark.compute_speedup()
            for backend, speedup in speedups.items():
                if backend not in backend_speedups:
                    backend_speedups[backend] = []
                backend_speedups[backend].append(speedup)
        
        # Print average speedups
        for backend, speedups in sorted(backend_speedups.items()):
            avg_speedup = sum(speedups) / len(speedups)
            min_speedup = min(speedups)
            max_speedup = max(speedups)
            
            print(f"{backend:15s}: {avg_speedup:6.2f}x avg "
                  f"(range: {min_speedup:.2f}x - {max_speedup:.2f}x)")
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save results to JSON file."""
        import loom
        
        results: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'backends': loom.config.list_backends(),
            'benchmarks': []
        }
        
        for benchmark in self.benchmarks:
            results['benchmarks'].append({
                'name': benchmark.name,
                'size': benchmark.size,
                'times': benchmark.results,
                'speedups': benchmark.compute_speedup()
            })
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main() -> None:
    """Run benchmark suite."""
    suite = BenchmarkSuite()
    
    # Small matrices (overhead visible)
    print("\nRunning small matrix benchmarks...")
    suite.add_matmul_benchmark(10)
    suite.add_elementwise_benchmark(10)
    
    # Medium matrices (JIT benefits visible)
    print("\nRunning medium matrix benchmarks...")
    suite.add_matmul_benchmark(50)
    suite.add_elementwise_benchmark(50)
    
    # Run and display
    suite.run_all()
    
    # Save to file
    suite.save_results()


if __name__ == '__main__':
    main()
