#!/usr/bin/env python
"""
Benchmarking on Hamlib benchmarks with given "compiler" and given "category" in Hamlib programs.
"""

import sys

sys.path.append('../..')

import os
import json
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import qiskit
import qiskit.qasm2
from natsort import natsorted
import phoenix
import bench_utils

from rich.console import Console

console = Console()

INPUT_QASM_DPATH = '../../benchmarks/hamlib_qasm'
INPUT_JSON_DPATH = '../../benchmarks/hamlib_json'
OUTPUT_DPATH = '../output_hamlib'

CATEGORIES = ['binaryoptimization', 'chemistry', 'condensedmatter', 'discreteoptimization']

COMPILER_PASSES = {
    'phoenix': bench_utils.phoenix_pass,
    'paulihedral': bench_utils.paulihedral_pass,
    'tetris': bench_utils.tetris_pass,
    'pauliopt': bench_utils.pauliopt_pass,
    'quclear': bench_utils.quclear_pass,
    'tket': bench_utils.tket_pass,
    'qiskit': bench_utils.qiskit_pass,
}

# Compiler-specific per-benchmark size cap on num_nonlocal_gates in the origin circuit
# (None means no cap). Matches the skip logic in the original serial script.
COMPILER_NONLOCAL_CAPS = {
    'phoenix': 3000,
}

# Extra keyword arguments per compiler (beyond with_O3=True)
COMPILER_EXTRA_KWARGS = {
    'tket': {'greedy': True},
}


def process_one(fname, compiler, output_dpath):
    """Compile a single Hamlib benchmark and dump the result. Returns (fname, status, circ_or_msg)."""
    import warnings as _warnings
    _warnings.filterwarnings('ignore')

    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))

    # if os.path.exists(output_fname):
    #     return fname, 'cached', output_fname

    cap = COMPILER_NONLOCAL_CAPS.get(compiler)
    if cap is not None:
        qasm_file = fname.replace('json', 'qasm')
        if qiskit.QuantumCircuit.from_qasm_file(qasm_file).num_nonlocal_gates() > cap:
            return fname, 'skipped', 'nonlocal_gates > {}'.format(cap)

    with open(fname, 'r') as f:
        data = json.load(f)

    kwargs = {'with_O3': True}
    kwargs.update(COMPILER_EXTRA_KWARGS.get(compiler, {}))

    compiler_pass = COMPILER_PASSES[compiler]
    circ = compiler_pass(data['paulis'], data['coeffs'], **kwargs)

    qiskit.qasm2.dump(circ, output_fname)
    return fname, 'ok', circ


def main():
    parser = argparse.ArgumentParser(description='Benchmarking on hamlib100 with Phoenix compiler')
    parser.add_argument('-t', '--type', type=str,
                        help='Type of benchmarks (binaryoptimization, chemistry, condensedmatter, discreteoptimization)')
    parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                        help='Compiler (default: phoenix)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of parallel worker processes (default: os.cpu_count())')
    args = parser.parse_args()

    if args.compiler not in COMPILER_PASSES:
        raise ValueError('Unsupported compiler: {}'.format(args.compiler))
    if args.type not in CATEGORIES:
        raise ValueError('Unsupported category: {}'.format(args.type))

    json_fnames = [os.path.join(INPUT_JSON_DPATH, args.type, fname)
                   for fname in natsorted(os.listdir(os.path.join(INPUT_JSON_DPATH, args.type)),
                                          reverse=True)]

    output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.type)

    if not os.path.exists(output_dpath):
        os.makedirs(output_dpath)

    console.print('program type: {}'.format(args.type))
    console.print('compiler: {}'.format(args.compiler))
    console.print('output directory: {}'.format(output_dpath))
    console.print('parallel jobs: {}'.format(args.jobs))

    warnings.filterwarnings('ignore')

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_one, fname, args.compiler, output_dpath): fname
            for fname in json_fnames
        }
        for future in as_completed(futures):
            fname = futures[future]
            try:
                _, status, payload = future.result()
                if status == 'ok':
                    console.print('Done', fname)
                    phoenix.utils.print_circ_info(payload)
                elif status == 'skipped':
                    console.print('[yellow]Skipped[/yellow] {} ({})'.format(fname, payload))
                elif status == 'cached':
                    console.print('[cyan]Cached[/cyan] {} -> {}'.format(fname, payload))
            except Exception as e:
                console.print('[red]Failed[/red] {}: {}'.format(fname, e))


if __name__ == '__main__':
    main()
