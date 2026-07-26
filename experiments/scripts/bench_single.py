#!/usr/bin/env python
import argparse
import json
import sys
import time
import warnings

sys.path.append("../..")

import bench_utils

import phoenix
import phoenix.utils
from qiskit.quantum_info import Operator

warnings.filterwarnings("ignore")

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark")
    parser.add_argument(
        "filename", type=str, help="Filename of the benchmark (a JSON file containing Pauli strings and coefficients)"
    )
    parser.add_argument(
        "-d",
        "--device",
        default="all2all",
        type=str,
        help="Device topology (default: all2all) (options: all2all, chain, hhex, square)",
    )
    parser.add_argument(
        "--O3",
        action="store_true",
        help="With Qiskit O3 for further local optimization in Phoenix compiler (default: False)",
    )
    parser.add_argument("--tket-greedy", action="store_true", help="Use tket GreedyPauliSimp pass (default: False)")
    parser.add_argument("-c", "--compiler", default="phoenixpp", type=str, help="Compiler (default: phoenixpp)")
    args = parser.parse_args()

    console.rule(f"Benchmarking on {args.filename}")
    console.print(args)

    with open(args.filename) as f:
        data = json.load(f)

    if args.device == "all2all":
        coupling_map = phoenix.utils.gene_all2all_coupling_map(data["num_qubits"])
    elif args.device == "chain":
        coupling_map = phoenix.utils.gene_chain_coupling_map(data["num_qubits"])
    elif args.device == "hhex":
        coupling_map = phoenix.utils.gene_hhex_coupling_map(data["num_qubits"])
    elif args.device == "square":
        coupling_map = phoenix.utils.gene_square_coupling_map(data["num_qubits"])
    else:
        raise ValueError("Unsupported device")

    ham = phoenix.Hamiltonian(data["paulis"], data["coeffs"])
    circ = ham.generate_circuit()
    u = ham.unitary_evolution() if ham.num_qubits < 10 else None
    phoenix.utils.print_circ_info(circ, title="Original circuit")

    t0 = time.perf_counter()
    if args.compiler == "tket":
        circ_opt = bench_utils.tket_pass(
            data["paulis"], data["coeffs"], greedy=args.tket_greedy, coupling_map=coupling_map
        )
    elif args.compiler == "qiskit":
        circ_opt = bench_utils.qiskit_pass(data["paulis"], data["coeffs"], coupling_map=coupling_map)
    elif args.compiler == "paulihedral":
        circ_opt = bench_utils.paulihedral_pass(
            data["paulis"], data["coeffs"], coupling_map=coupling_map
        )
    elif args.compiler == "tetris":
        circ_opt = bench_utils.tetris_pass(data["paulis"], data["coeffs"], coupling_map=coupling_map)
    elif args.compiler == "pauliopt":
        circ_opt = bench_utils.pauliopt_pass(data["paulis"], data["coeffs"], coupling_map=coupling_map)
    elif args.compiler == "quclear":
        circ_opt = bench_utils.quclear_pass(data["paulis"], data["coeffs"], coupling_map=coupling_map)
    elif args.compiler == "phoenix":
        circ_opt = bench_utils.phoenix_pass(
            data["paulis"], data["coeffs"], grouping="support", coupling_map=coupling_map
        )
    elif args.compiler == "phoenixpp":
        circ_opt = bench_utils.phoenix_pass(
            data["paulis"], data["coeffs"], grouping="holistic", coupling_map=coupling_map
        )
    else:
        raise ValueError("Unsupported compiler")
    elapsed = time.perf_counter() - t0

    phoenix.utils.print_circ_info(
        circ_opt,
        title=f"Optimized circuit (compiler={args.compiler}, O3={args.O3}, {elapsed:.2f}s)",
    )
    if u is not None:
        infidelity = phoenix.utils.infidelity(u, Operator(circ_opt).reverse_qargs().to_matrix())
        print(f"Infidelity: {infidelity:.2e}")


if __name__ == "__main__":
    main()
