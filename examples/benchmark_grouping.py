#!/usr/bin/env python
"""
Benchmark comparing grouping='holistic' vs grouping='support' for compile_hamiltonian_simulation.
"""

import json
import time
from phoenix import Hamiltonian, compile_hamiltonian_simulation
from phoenix.utils import print_circ_info


def load_hamiltonian(path):
    with open(path, "r") as f:
        data = json.load(f)
    return Hamiltonian(data["paulis"], data["coeffs"])


def benchmark():
    ham = load_hamiltonian("../benchmarks/uccsd/ucc_10e_7o_BK.json")
    # ham = load_hamiltonian('benchmarks/qaoa_json/qaoa_rand_16.json')
    print(f"Hamiltonian: {len(ham.paulis)} Pauli terms, {ham.paulis.num_qubits} qubits")
    print(f"Groups (same-weight): {len(ham.group_same_weights())}")
    print("=" * 60)

    print("\n1. Original circuit")
    print_circ_info(ham.generate_circuit(), title="Original circuit")

    results = {}
    for i, mode in enumerate(["holistic", "support"], start=2):
        print(f"\n{i}. grouping='{mode}'")
        start = time.perf_counter()
        qc = compile_hamiltonian_simulation(ham, grouping=mode)
        elapsed = time.perf_counter() - start
        print_circ_info(qc, title=f"grouping='{mode}'")
        print(f"   Compilation time: {elapsed:.4f}s")
        results[mode] = (elapsed, qc)

    print("\n" + "=" * 60)
    print("Summary:")
    for mode, (elapsed, qc) in results.items():
        print(f"  grouping='{mode}'{' ' * (8 - len(mode))}: {elapsed:.4f}s, {qc.num_nonlocal_gates()} CX gates, depth {qc.depth()}")
    print(f"  Speedup (time) : {results['support'][0] / results['holistic'][0]:.2f}x")


if __name__ == "__main__":
    benchmark()
