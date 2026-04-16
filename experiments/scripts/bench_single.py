#!/usr/bin/env python
import sys

sys.path.append('../..')

import json
import argparse
import warnings
import phoenix
import phoenix.utils
import bench_utils

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Run a single benchmark')
    parser.add_argument('filename', type=str,
                        help='Filename of the benchmark (a JSON file containing Pauli strings and coefficients)')
    parser.add_argument('-d', '--device', default='all2all', type=str,
                        help='Device topology (default: all2all) (options: all2all, chain, hhex, square)')
    parser.add_argument('--O3', action='store_true',
                        help='With Qiskit O3 for further local optimization in Phoenix compiler (default: False)')
    parser.add_argument('--tket-greedy', action='store_true',
                        help='Use tket GreedyPauliSimp pass (default: False)')
    parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                        help='Compiler (default: phoenix)')
    args = parser.parse_args()

    console.rule('Benchmarking on {}'.format(args.filename))
    console.print(args)

    with open(args.filename, 'r') as f:
        data = json.load(f)

    if args.device == 'all2all':
        coupling_map = phoenix.utils.gene_all2all_coupling_map(data['num_qubits'])
    elif args.device == 'chain':
        coupling_map = phoenix.utils.gene_chain_coupling_map(data['num_qubits'])
    elif args.device == 'hhex':
        coupling_map = phoenix.utils.gene_hhex_coupling_map(data['num_qubits'])
    elif args.device == 'square':
        coupling_map = phoenix.utils.gene_square_coupling_map(data['num_qubits'])
    else:
        raise ValueError('Unsupported device')

    ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
    circ = ham.generate_circuit()
    phoenix.utils.print_circ_info(circ, title='Original circuit')

    if args.compiler == 'tket':
        circ_opt = bench_utils.tket_pass(data['paulis'], data['coeffs'], greedy=args.tket_greedy, coupling_map=coupling_map, with_O3=args.O3)
    elif args.compiler == 'qiskit':
        circ_opt = bench_utils.qiskit_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
    elif args.compiler == 'paulihedral':
        circ_opt = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
    elif args.compiler == 'tetris':
        circ_opt = bench_utils.tetris_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
    elif args.compiler == 'pauliopt':
        circ_opt = bench_utils.pauliopt_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
    elif args.compiler == 'quclear':
        circ_opt = bench_utils.quclear_pass(data['paulis'], data['coeffs'],
                                            coupling_map=coupling_map,
                                            with_O3=args.O3)
    elif args.compiler == 'phoenix':
        circ_opt = bench_utils.phoenix_pass(data['paulis'], data['coeffs'],
                                            coupling_map=coupling_map,
                                            with_O3=args.O3)
    elif args.compiler == 'phoenix+':
        circ_opt = bench_utils.phoenix_pass(data['paulis'], data['coeffs'],
                                            grouping=False,
                                            coupling_map=coupling_map,
                                            with_O3=args.O3)
    else:
        raise ValueError('Unsupported compiler')

    phoenix.utils.print_circ_info(circ_opt, title='Optimized circuit')


if __name__ == '__main__':
    main()
