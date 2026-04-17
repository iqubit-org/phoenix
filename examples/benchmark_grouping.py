#!/usr/bin/env python
"""
Benchmark comparing grouping=True vs grouping=False for compile_hamiltonian_simulation.
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
    ham = load_hamiltonian("benchmarks/uccsd_json/LiH_frz_BK_sto3g.json")
    # ham = load_hamiltonian('benchmarks/qaoa_json/qaoa_rand_16.json')
    print(f"Hamiltonian: {len(ham.paulis)} Pauli terms, {ham.paulis.num_qubits} qubits")
    print(f"Groups (same-weight): {len(ham.group_same_weights())}")
    print("=" * 60)

    print("\n1. Original circuit")
    print_circ_info(ham.generate_circuit(), title="Original circuit")

    # grouping=True
    print("\n1. grouping=True")
    start = time.perf_counter()
    qc_grouped = compile_hamiltonian_simulation(ham, grouping=True)
    t_grouped = time.perf_counter() - start
    print_circ_info(qc_grouped, title="grouping=True")
    print(f"   Compilation time: {t_grouped:.4f}s")

    # grouping=False
    print("\n2. grouping=False")
    start = time.perf_counter()
    qc_ungrouped = compile_hamiltonian_simulation(ham, grouping=False)
    t_ungrouped = time.perf_counter() - start
    print_circ_info(qc_ungrouped, title="grouping=False")
    print(f"   Compilation time: {t_ungrouped:.4f}s")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(
        f"  grouping=True  : {t_grouped:.4f}s, {qc_grouped.num_nonlocal_gates()} CX gates, depth {qc_grouped.depth()}"
    )
    print(
        f"  grouping=False : {t_ungrouped:.4f}s, {qc_ungrouped.num_nonlocal_gates()} CX gates, depth {qc_ungrouped.depth()}"
    )
    print(f"  Speedup (time) : {t_ungrouped / t_grouped:.2f}x")


if __name__ == "__main__":
    benchmark()
