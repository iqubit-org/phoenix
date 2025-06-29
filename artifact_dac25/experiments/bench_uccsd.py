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
import qiskit.qasm2
import pytket.qasm
from natsort import natsorted
import bench_utils
from phoenix.utils.render import print_circ_info

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()

INPUT_JSON_DPATH = '../benchmarks/uccsd'
OUTPUT_DPATH = './output_uccsd'

parser = argparse.ArgumentParser(description='Benchmarking on UCCSD chemistry benchmarks')
parser.add_argument('-d', '--device', default='all2all', type=str,
                    help='Device topology (default: all2all) (options: all2all, hhex, square)')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

json_fnames = [os.path.join(INPUT_JSON_DPATH, fname) for fname in natsorted(os.listdir(INPUT_JSON_DPATH))]

output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.device)

if not os.path.exists(output_dpath):
    os.makedirs(output_dpath)

console.print('topology: {}'.format(args.device))
console.print('compiler: {}'.format(args.compiler))
console.print('output directory: {}'.format(output_dpath))

if args.device == 'all2all':
    coupling_map = bench_utils.All2all
elif args.device == 'hhex':
    coupling_map = bench_utils.Manhattan
elif args.device == 'square':
    coupling_map = bench_utils.Sycamore
else:
    raise ValueError('Unsupported device')

for fname in json_fnames:
    console.print('Processing', fname)
    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))
    with open(fname, 'r') as f:
        data = json.load(f)

    if args.compiler == 'phoenix':
        circ = bench_utils.phoenix_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map)
    elif args.compiler == 'paulihedral':
        circ = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'],
                                            coupling_map=coupling_map)
    elif args.compiler == 'tetris':
        circ = bench_utils.tetris_pass(data['paulis'], data['coeffs'],
                                       coupling_map=coupling_map)
    elif args.compiler == 'pauliopt':
        circ = bench_utils.pauliopt_pass(data['paulis'], data['coeffs'],
                                         coupling_map=coupling_map)
    elif args.compiler == 'quclear':
        circ = bench_utils.quclear_pass(data['paulis'], data['coeffs'],
                                        coupling_map=coupling_map)
    elif args.compiler == 'tket':
        circ = bench_utils.tket_pass(data['paulis'], data['coeffs'])
    else:
        raise ValueError('Unsupported compiler')

    print_circ_info(circ)
    if args.compiler == 'tket':
        pytket.qasm.circuit_to_qasm(circ, output_fname)
    else:
        qiskit.qasm2.dump(circ, output_fname)
