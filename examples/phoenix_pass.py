#!/usr/bin/env python
"""
Test entry-point for the Phoenix compiler.

Loads a Hamiltonian (Pauli strings + coefficients) from a JSON benchmark file,
maps it to a target topology, runs ``phoenix.compile_hamiltonian_simulation``,
and prints original / optimized circuit statistics.

Usage:
    ./phoenix_pass.py path/to/benchmark.json [-d {all2all,chain,hhex,square}]
                                             [--order-method METHOD]
                                             [--O3]
                                             [--no-optimize]
                                             [--backend {concurrent.futures,joblib,sequential}]
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import warnings
import phoenix
import phoenix.utils

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()


ORDER_METHODS = ['trivial', 'greedy', 'greedy_multistart',
                 'tsp', 'tsp_2opt', 'mcts', 'beam']

BACKENDS = ['concurrent.futures', 'joblib', 'sequential']


def gene_coupling_map(device: str, num_qubits: int):
    if device == 'all2all':
        return phoenix.utils.gene_all2all_coupling_map(num_qubits)
    if device == 'chain':
        return phoenix.utils.gene_chain_coupling_map(num_qubits)
    if device == 'hhex':
        return phoenix.utils.gene_hhex_coupling_map(num_qubits)
    if device == 'square':
        return phoenix.utils.gene_square_coupling_map(num_qubits)
    raise ValueError('Unsupported device: {}'.format(device))



def main():
    parser = argparse.ArgumentParser(description='Phoenix compiler test entry-point')
    parser.add_argument('filename', type=str,
                        help='Benchmark JSON file (with `paulis`, `coeffs`, `num_qubits` fields)')
    parser.add_argument('-d', '--device', default='all2all', type=str,
                        choices=['all2all', 'chain', 'hhex', 'square'],
                        help='Device topology (default: all2all)')
    parser.add_argument('--order-method', default='trivial', type=str,
                        choices=ORDER_METHODS,
                        help='Block ordering method passed to compile_hamiltonian_simulation '
                             '(default: trivial)')
    parser.add_argument('--backend', default='concurrent.futures', type=str,
                        choices=BACKENDS,
                        help='Parallel backend for same-weight group processing '
                             '(default: concurrent.futures)')
    parser.add_argument('--O3', action='store_true',
                        help='Apply Qiskit O3 post-optimization (default: False)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Disable the internal optimize pass inside Phoenix (default: False)')
    args = parser.parse_args()

    console.rule('Phoenix compiling {}'.format(args.filename))
    console.print(args)

    with open(args.filename, 'r') as f:
        data = json.load(f)

    coupling_map = gene_coupling_map(args.device, data['num_qubits'])

    paulis_orig = [p[::-1] for p in data['paulis']]  # for fair "Original circuit" view
    ham = phoenix.Hamiltonian(paulis_orig, data['coeffs'])
    circ = ham.generate_circuit()
    phoenix.utils.print_circ_info(circ, title='Original circuit')

    import time
    t0 = time.perf_counter()
    circ_opt = phoenix.compile_hamiltonian_simulation(
        ham,
        order_method=args.order_method,
        backend=args.backend,
        optimize=args.O3,
    )
    elapsed = time.perf_counter() - t0

    print(circ_opt)

    phoenix.utils.print_circ_info(
        circ_opt,
        title='Optimized circuit (order={}, O3={}, {:.2f}s)'.format(
            args.order_method, args.O3, elapsed,
        ),
    )


if __name__ == '__main__':
    main()
