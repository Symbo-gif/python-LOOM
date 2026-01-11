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
Backend validation script for LOOM.

Validates numerical equivalence between Python and Numba backends.
Run with: python scripts/validate_backends.py
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple


# Default backend name used as baseline for comparison
DEFAULT_BACKEND = 'cpu'


def run_validation() -> Dict[str, Any]:
    """
    Run validation tests to ensure backend equivalence.
    
    Returns:
        Dictionary containing validation results
    """
    import loom
    
    results: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'backends_tested': [],
        'tests': [],
        'passed': 0,
        'failed': 0,
        'skipped': 0,
    }
    
    # Get available backends
    try:
        available = loom.backend.available_backends()
    except Exception:
        available = [DEFAULT_BACKEND]
    
    results['backends_tested'] = available
    
    # Define test cases: (name, function, tolerance)
    test_cases: List[Tuple[str, Any, float]] = []
    
    # Test 1: Matrix multiplication
    def test_matmul():
        a = loom.array([[1.0, 2.0], [3.0, 4.0]])
        b = loom.array([[5.0, 6.0], [7.0, 8.0]])
        return loom.matmul(a, b)
    
    test_cases.append(('matmul', test_matmul, 1e-10))
    
    # Test 2: Element-wise operations
    def test_elementwise():
        a = loom.array([1.0, 2.0, 3.0, 4.0])
        b = loom.array([2.0, 3.0, 4.0, 5.0])
        return a + b * 2.0 - 1.0
    
    test_cases.append(('elementwise', test_elementwise, 1e-10))
    
    # Test 3: Sum reduction
    def test_sum():
        a = loom.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return a.sum()
    
    test_cases.append(('sum', test_sum, 1e-10))
    
    # Test 4: Transpose
    def test_transpose():
        a = loom.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return a.T
    
    test_cases.append(('transpose', test_transpose, 1e-10))
    
    # Run tests comparing backends
    if len(available) < 2:
        # Only one backend available, just run tests to verify they work
        results['skipped'] = len(test_cases)
        for name, func, tol in test_cases:
            try:
                loom.set_backend(DEFAULT_BACKEND)
                _ = func()
                results['tests'].append({
                    'name': name,
                    'status': 'skipped',
                    'reason': 'Only one backend available'
                })
            except Exception as e:
                results['tests'].append({
                    'name': name,
                    'status': 'error',
                    'error': str(e)
                })
                results['failed'] += 1
        return results
    
    # Compare all backends against CPU baseline
    for name, func, tol in test_cases:
        test_result: Dict[str, Any] = {
            'name': name,
            'tolerance': tol,
            'backend_results': {}
        }
        
        try:
            # Get baseline from CPU backend
            loom.set_backend(DEFAULT_BACKEND)
            baseline = func()
            baseline_data = baseline.tolist() if hasattr(baseline, 'tolist') else baseline
            test_result['backend_results'][DEFAULT_BACKEND] = baseline_data
            
            all_match = True
            
            # Compare with other backends
            for backend in available:
                if backend == DEFAULT_BACKEND:
                    continue
                
                try:
                    loom.set_backend(backend)
                    result = func()
                    result_data = result.tolist() if hasattr(result, 'tolist') else result
                    test_result['backend_results'][backend] = result_data
                    
                    # Check equivalence
                    if not _compare_results(baseline_data, result_data, tol):
                        all_match = False
                        test_result['mismatch'] = f'{backend} differs from {DEFAULT_BACKEND}'
                except Exception as e:
                    test_result['backend_results'][backend] = f'Error: {e}'
                    all_match = False
            
            if all_match:
                test_result['status'] = 'passed'
                results['passed'] += 1
            else:
                test_result['status'] = 'failed'
                results['failed'] += 1
                
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            results['failed'] += 1
        
        results['tests'].append(test_result)
    
    # Reset to default backend
    loom.set_backend(DEFAULT_BACKEND)
    
    return results


def _compare_results(a: Any, b: Any, tol: float) -> bool:
    """Compare two results for numerical equivalence."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_compare_results(x, y, tol) for x, y in zip(a, b))
    
    return a == b


def main() -> int:
    """
    Main entry point for validation script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("LOOM Backend Validation")
    print("=" * 60)
    
    results = run_validation()
    
    # Print summary
    print(f"\nBackends tested: {results['backends_tested']}")
    print(f"Tests passed: {results['passed']}")
    print(f"Tests failed: {results['failed']}")
    print(f"Tests skipped: {results['skipped']}")
    
    # Print details
    for test in results['tests']:
        status = test.get('status', 'unknown')
        name = test.get('name', 'unknown')
        if status == 'passed':
            print(f"  ✓ {name}")
        elif status == 'skipped':
            print(f"  - {name} (skipped)")
        else:
            print(f"  ✗ {name}")
            if 'error' in test:
                print(f"    Error: {test['error']}")
            if 'mismatch' in test:
                print(f"    {test['mismatch']}")
    
    # Save results
    with open('backend_validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to backend_validation_report.json")
    
    # Return exit code
    if results['failed'] > 0:
        print("\n❌ VALIDATION FAILED")
        return 1
    else:
        print("\n✓ VALIDATION PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
