#!/usr/bin/env python
"""
Test entry-point for the old Phoenix compiler (phoenix_old).

Loads a Hamiltonian (Pauli strings + coefficients) from a JSON benchmark file,
runs ``HamiltonianModel.phoenix_circuit()``, and prints original / optimized
circuit statistics.

Usage:
    ./phoenix_old_pass.py path/to/benchmark.json [--no-order]
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import warnings
import time

warnings.filterwarnings('ignore')

from phoenix_old.models.hamiltonians import HamiltonianModel
from phoenix_old.utils.render import print_circ_info
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description='Phoenix (old) compiler test entry-point')
    parser.add_argument('filename', type=str,
                        help='Benchmark JSON file (with `paulis`, `coeffs`, `num_qubits` fields)')
    parser.add_argument('--no-order', action='store_true',
                        help='Disable block ordering (default: False)')
    args = parser.parse_args()

    console.rule('Phoenix-old compiling {}'.format(args.filename))
    console.print(args)

    with open(args.filename, 'r') as f:
        data = json.load(f)

    ham = HamiltonianModel(data['paulis'], data['coeffs'])

    # Original circuit (naive Trotter)
    circ_orig = ham.generate_circuit(order=1)
    print_circ_info(circ_orig, title='Original circuit')

    # Phoenix-old optimized circuit
    t0 = time.perf_counter()
    circ_opt = ham.phoenix_circuit(order_blocks=not args.no_order)
    elapsed = time.perf_counter() - t0

    print_circ_info(
        circ_opt,
        title='Optimized circuit (order={}, {:.2f}s)'.format(
            not args.no_order, elapsed,
        ),
    )


if __name__ == '__main__':
    main()
