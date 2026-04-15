#!/usr/bin/env python
"""
All-to-all compilation results (especially for TKet, PauliOpt, Phoenix) for UCCSD benchmarks
 to limited connectivity topology (Manhattan, Sycamore)
 for UCCSD benchmarks
"""
import sys

sys.path.append('../..')

import os
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import phoenix
from natsort import natsorted
import qiskit
import qiskit.qasm2

from rich.console import Console

console = Console()

warnings.filterwarnings('ignore')

COUPLING_MAP_GENS = {
    'hhex': 'gene_hhex_coupling_map',
    'square': 'gene_square_coupling_map',
}


def process_one(all2all_circ_file, limited_circ_file, device):
    import warnings as _warnings
    _warnings.filterwarnings('ignore')

    gene_coupling_map = getattr(phoenix.utils, COUPLING_MAP_GENS[device])
    circ = qiskit.QuantumCircuit.from_qasm_file(all2all_circ_file)
    circ = phoenix.utils.optimize_with_mapping(circ, gene_coupling_map(circ.num_qubits))
    qiskit.qasm2.dump(circ, limited_circ_file)
    return all2all_circ_file, limited_circ_file, circ


def main():
    parser = argparse.ArgumentParser(description='Map logical circuits to physical qubits with limited connectivity')
    parser.add_argument('-d', '--device', type=str,
                        help='Device topology (options: hhex, square)')
    parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                        help='Compiler (default: phoenix)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of parallel worker processes (default: os.cpu_count())')
    args = parser.parse_args()
    print(args)

    if args.device not in COUPLING_MAP_GENS:
        raise ValueError('Unsupported topology {}'.format(args.device))

    all2all_dpath = '../output_uccsd/{}/all2all'.format(args.compiler)
    limited_dpath = '../output_uccsd/{}/{}'.format(args.compiler, args.device)
    if not os.path.exists(limited_dpath):
        os.makedirs(limited_dpath)

    fnames = natsorted(os.listdir(all2all_dpath))
    tasks = [(os.path.join(all2all_dpath, fname),
              os.path.join(limited_dpath, fname)) for fname in fnames]

    console.print('parallel jobs: {}'.format(args.jobs))

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_one, src, dst, args.device): (src, dst)
            for src, dst in tasks
        }
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                _, _, circ = future.result()
                console.print('Converted {} -> {}'.format(src, dst))
                phoenix.utils.print_circ_info(circ)
            except Exception as e:
                console.print('[red]Failed[/red] {}: {}'.format(src, e))


if __name__ == '__main__':
    main()
