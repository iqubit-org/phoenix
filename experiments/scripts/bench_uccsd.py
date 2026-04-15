#!/usr/bin/env python
"""
Benchmarking on UCCSD benchmarks with given "compiler" and given "topology".
"""

import sys

sys.path.append('../..')

import os
import json
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import qiskit.qasm2
import pytket.qasm
from natsort import natsorted
import bench_utils
import phoenix

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()

INPUT_JSON_DPATH = '../../benchmarks/uccsd_json'
OUTPUT_DPATH = '../output_uccsd'

COMPILER_PASSES = {
    'tket': bench_utils.tket_pass,
    'qiskit': bench_utils.qiskit_pass,
    'paulihedral': bench_utils.paulihedral_pass,
    'tetris': bench_utils.tetris_pass,
    'pauliopt': bench_utils.pauliopt_pass,
    'quclear': bench_utils.quclear_pass,
    'phoenix': bench_utils.phoenix_pass,
}

COUPLING_MAP_GENS = {
    'all2all': 'gene_all2all_coupling_map',
    'chain': 'gene_chain_coupling_map',
    'hhex': 'gene_hhex_coupling_map',
    'square': 'gene_square_coupling_map',
}


def process_one(fname, compiler, device, with_O3, output_dpath):
    """Compile a single UCCSD benchmark and dump the result."""
    import warnings as _warnings
    _warnings.filterwarnings('ignore')

    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))
    with open(fname, 'r') as f:
        data = json.load(f)

    gene_coupling_map = getattr(phoenix.utils, COUPLING_MAP_GENS[device])
    coupling_map = gene_coupling_map(data['num_qubits'])

    compiler_pass = COMPILER_PASSES[compiler]
    circ = compiler_pass(data['paulis'], data['coeffs'],
                         coupling_map=coupling_map, with_O3=with_O3)

    qiskit.qasm2.dump(circ, output_fname)
    return fname, circ


def main():
    parser = argparse.ArgumentParser(description='Benchmarking on UCCSD chemistry benchmarks')
    parser.add_argument('-d', '--device', default='all2all', type=str,
                        help='Device topology (default: all2all) (options: all2all, hhex, square)')
    parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                        help='Compiler (default: phoenix)')
    parser.add_argument('--O3', action='store_true',
                        help='With Qisit O3 for further local optimization in Phoenix compiler (default: False)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of parallel worker processes (default: os.cpu_count())')
    args = parser.parse_args()

    if args.compiler not in COMPILER_PASSES:
        raise ValueError('Unsupported compiler: {}'.format(args.compiler))
    if args.device not in COUPLING_MAP_GENS:
        raise ValueError('Unsupported device: {}'.format(args.device))

    json_fnames = [os.path.join(INPUT_JSON_DPATH, fname)
                   for fname in natsorted(os.listdir(INPUT_JSON_DPATH))]

    if args.device == 'all2all' and args.O3:
        output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, 'all2all_opt')
    else:
        output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.device)

    if not os.path.exists(output_dpath):
        os.makedirs(output_dpath)

    console.print('topology: {}'.format(args.device))
    console.print('compiler: {}'.format(args.compiler))
    console.print('output directory: {}'.format(output_dpath))
    console.print('parallel jobs: {}'.format(args.jobs))

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_one, fname, args.compiler, args.device,
                            args.O3, output_dpath): fname
            for fname in json_fnames
        }
        for future in as_completed(futures):
            fname = futures[future]
            try:
                _, circ = future.result()
                console.print('Done', fname)
                phoenix.utils.print_circ_info(circ)
            except Exception as e:
                console.print('[red]Failed[/red] {}: {}'.format(fname, e))


if __name__ == '__main__':
    main()
