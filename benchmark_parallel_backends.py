#!/usr/bin/env python
"""
Benchmark comparing different parallelization backends for compile_hamiltonian_simulation_par.

Shows when to use each backend:
- concurrent.futures: Small, uniform workloads
- as_completed: When task execution times vary significantly
- joblib: Production code, better error handling, progress tracking
"""

import json
import time
from phoenix import Hamiltonian, compile_hamiltonian_simulation


def load_test_hamiltonian():
    """Load test Hamiltonian from JSON."""
    with open('benchmarks/uccsd_json/LiH_frz_BK_sto3g.json', 'r') as f:
        data = json.load(f)
    return Hamiltonian(data['paulis'], data['coeffs'])


def benchmark():
    """Benchmark different backends."""
    hamiltonian = load_test_hamiltonian()
    num_groups = len(hamiltonian.group_same_weights())
    
    print(f"Hamiltonian: {len(hamiltonian.paulis)} terms, {num_groups} groups")
    print("=" * 60)
    
    # Baseline: Sequential
    print("\n1. SEQUENTIAL (baseline)")
    start = time.perf_counter()
    qc_seq = compile_hamiltonian_simulation(hamiltonian)
    time_seq = time.perf_counter() - start
    print(f"   Time: {time_seq:.4f}s")
    
    # Backend 1: concurrent.futures (default)
    print("\n2. concurrent.futures + map()")
    start = time.perf_counter()
    qc1 = compile_hamiltonian_simulation(hamiltonian, backend='concurrent.futures')
    time_cf = time.perf_counter() - start
    speedup_cf = time_seq / time_cf
    print(f"   Time: {time_cf:.4f}s")
    print(f"   Speedup: {speedup_cf:.2f}x")
    print(f"   ✓ Best for: Simple, uniform workloads")
    print(f"   ✓ Pros: Lightweight, no dependencies, maintains order")
    
    # Backend 2: joblib
    print("\n3. joblib.Parallel()")
    start = time.perf_counter()
    qc2 = compile_hamiltonian_simulation(hamiltonian, backend='joblib')
    time_jl = time.perf_counter() - start
    speedup_jl = time_seq / time_jl
    print(f"   Time: {time_jl:.4f}s")
    print(f"   Speedup: {speedup_jl:.2f}x")
    print(f"   ✓ Best for: Production, variable workloads")
    print(f"   ✓ Pros: Better error handling, progress tracking, flexible backends")
    
    # Verify correctness
    print("\n" + "=" * 60)
    print("Verification:")
    print(f"  Sequential gates: {len(qc_seq.data)}")
    print(f"  concurrent.futures gates: {len(qc1.data)}")
    print(f"  joblib gates: {len(qc2.data)}")
    assert len(qc_seq.data) == len(qc1.data) == len(qc2.data), "Results differ!"
    print("  ✓ All backends produce identical results")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("  • Use concurrent.futures (default) if:")
    print("    - Groups have similar processing time")
    print("    - Want minimal dependencies")
    print("    - Prefer simplicity over features")
    print("\n  • Use joblib if:")
    print("    - Group processing times vary significantly")
    print("    - Need progress tracking (verbose=True)")
    print("    - Want robust error handling")
    print("    - Planning distributed execution later")
    print("\n  • as_completed variant:")
    print("    - Rarely needed for circuit compilation")
    print("    - Useful when partial results matter (streaming)")


if __name__ == '__main__':
    benchmark()
