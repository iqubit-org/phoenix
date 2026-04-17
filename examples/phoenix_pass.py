#!/usr/bin/env python
"""
Test entry-point for the Phoenix compiler.

Loads a Hamiltonian (Pauli strings + coefficients) from a JSON benchmark file,
maps it to a target topology, runs ``phoenix.compile_hamiltonian_simulation``,
and prints original / optimized circuit statistics.

Usage:
    ./phoenix_pass.py path/to/benchmark.json [-d {all2all,chain,hhex,square}]
                                             [--backend BACKEND]
                                             [-o OUTPUT]
                                             [--no-grouping]
                                             [--O3]
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import warnings
import phoenix
from qiskit import qasm2
from qiskit.quantum_info import Operator

warnings.filterwarnings("ignore")

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Phoenix compiler test entry-point")
    parser.add_argument("filename", type=str, help="Benchmark JSON file (with `paulis`, `coeffs`, `num_qubits` fields)")
    parser.add_argument(
        "-d",
        "--device",
        default="all2all",
        type=str,
        choices=["all2all", "chain", "hhex", "square"],
        help="Device topology (default: all2all)",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="sequential",
        type=str,
        choices=["sequential", "joblib", "concurrent.futures"],
        help="Execution backend, i.e., whether use parallel (default: sequential)",
    )
    parser.add_argument("-o", "--output", type=str, help="Write the compiled circuit to the specified .qasm file")
    parser.add_argument("--O3", action="store_true", help="Apply Qiskit O3 post-optimization (default: False)")
    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable the internal optimize pass inside Phoenix (default: False)"
    )
    parser.add_argument(
        "--no-grouping", action="store_true", help="Disable the grouping of Pauli strings (default: False)"
    )
    parser.add_argument(
        "--parallel-search",
        action="store_true",
        help="Enable parallel search for the best Clifford gate (default: False)",
    )
    args = parser.parse_args()

    console.rule("Phoenix compiling {}".format(args.filename))
    console.print(args)

    with open(args.filename, "r") as f:
        data = json.load(f)

    paulis_orig = [p[::-1] for p in data["paulis"]]  # for fair "Original circuit" view
    ham = phoenix.Hamiltonian(paulis_orig, data["coeffs"])
    circ = ham.generate_circuit()
    if ham.num_qubits < 10:
        u = ham.unitary_evolution()
    else:
        u = None

    phoenix.utils.print_circ_info(circ, title="Original circuit")

    import time

    t0 = time.perf_counter()
    circ_opt = phoenix.compile_hamiltonian_simulation(
        ham,
        backend=args.backend,
        grouping=not args.no_grouping,
        parallel_search=args.parallel_search,
    )
    if args.O3:
        circ_opt = phoenix.utils.qiskit_O3_all2all(circ_opt)
    elapsed = time.perf_counter() - t0

    # print(circ_opt)

    phoenix.utils.print_circ_info(
        circ_opt,
        title="Optimized circuit (O3={}, {:.2f}s)".format(args.O3, elapsed),
    )
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qasm2.dump(circ_opt, output_path)
        print("Saved compiled circuit to {}".format(output_path))
    if u is not None:
        infidelity = phoenix.utils.infidelity(u, Operator(circ_opt).to_matrix())
        print("Infidelity: {:.2e}".format(infidelity))


if __name__ == "__main__":
    main()
