#!/usr/bin/env python
"""
Benchmarking on Hamlib benchmarks with given "compiler" and given "category" in Hamlib programs.
"""

import sys

sys.path.append('..')
sys.path.append('../..')

import os
import json
import argparse
import pytket.qasm
import qiskit.qasm2
from phoenix import gates
from natsort import natsorted
import bench_utils

from rich.console import Console

console = Console()

INPUT_QASM_DPATH = '../benchmarks/hamlib_qasm'
INPUT_JSON_DPATH = '../benchmarks/hamlib_json'
OUTPUT_DPATH = './output_hamlib'

CATEGORIES = ['binaryoptimization, chemistry, condensedmatter, discreteoptimization']

parser = argparse.ArgumentParser(description='Benchmarking on hamlib100 with Phoenix compiler')
parser.add_argument('-t', '--type', type=str,
                    help='Type of benchmarks (binaryoptimization, chemistry, condensedmatter, discreteoptimization')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

qasm_fnames = [os.path.join(INPUT_QASM_DPATH, args.type, fname) for fname in natsorted(os.listdir(INPUT_QASM_DPATH))]
json_fnames = [os.path.join(INPUT_JSON_DPATH, args.type, fname) for fname in natsorted(os.listdir(INPUT_JSON_DPATH))]

output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.type)

console.print('program type: {}'.format(args.type))
console.print('compiler: {}'.format(args.compiler))
console.print('output directory: {}'.format(output_dpath))

if args.compiler in ['phoenix', 'paulihedral', 'tetris', 'pauliopt']:
    for fname in json_fnames:
        output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))
        with open(fname, 'r') as f:
            data = json.load(f)
        pre_gates = [gates.X.on(q) for q in data['front_x_on']]

        if args.compiler == 'phoenix':
            circ = bench_utils.phoenix_pass(data['paulis'], data['coeffs'], pre_gates)
            pytket.qasm.circuit_to_qasm(circ, output_fname)
        elif args.compiler == 'paulihedral':
            circ = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'],
                                                pre_gates)  # TODO: do no return mappings?
            qiskit.qasm2.dump(circ, output_fname)
        elif args.compiler == 'tetris':
            circ = bench_utils.tetris_pass(data['paulis'], data['coeffs'], pre_gates)  # TODO: do no return mappings?
            qiskit.qasm2.dump(circ, output_fname)

        elif args.compiler == 'pauliopt':
            circ = bench_utils.pauliopt_pass(data['paulis'], data['coeffs'], pre_gates)  # TODO: do no return mappings?
            qiskit.qasm2.dump(circ, output_fname)
        else:
            raise ValueError('Unsupported compiler')
else:
    if args.compiler == 'tket':
        for fname in qasm_fnames:
            circ = pytket.qasm.circuit_from_qasm(fname)
            circ = bench_utils.tket_pass(circ)
            pytket.qasm.circuit_to_qasm(circ, os.path.join(output_dpath, os.path.basename(fname)))
    else:
        raise ValueError('Unsupported compiler')
#
#
#
# def bench_hamlib(type):
#     json_fnames = [fname for fname in os.listdir(os.path.join(BENCHMARK_DPATH, type)) if fname.endswith('.json')]
#     for json_fname in json_fnames:
#         output_fname = os.path.join(OUTPUT_DPATH, type, json_fname.replace('.json', '.qasm'))
#         if os.path.exists(output_fname):
#             continue
#         print('Processing', json_fname)
#         with open(os.path.join(BENCHMARK_DPATH, type, json_fname), 'r') as f:
#             data = json.load(f)
#         ham = HamiltonianModel(data['paulis'], data['coeffs'])
#         circ = ham.reconfigure_and_generate_circuit()
#         circ.to_qasm(output_fname)


# bench_hamlib(args.type)

#
# if args.type == 'binaryoptimization':
#     bench_hamlib100_type(args.type)
# elif args.type == 'chemistry':
#     bench_hamlib100_type(args.type)
#     # json_fnames = os.listdir(os.path.join(BENCHMARK_DPATH, args.type))
#     # for json_fname in json_fnames:
#     #     with open(os.path.join(BENCHMARK_DPATH, args.type, json_fname), 'r') as f:
#     #         data = json.load(f)
#     #     output_fname = os.path.join(OUTPUT_DPATH, args.type, json_fname.replace('.json', '.qasm'))
#     #     ham = HamiltonianModel(data['paulis'], data['coeffs'])
#     #     circ = ham.reconfigure_and_generate_circuit()
#     #     circ.to_qasm(output_fname)
#     #     print('Circuit saved to', output_fname)
# elif args.type == 'condensedmatter':
#     ...
# elif args.type == 'discreteoptimization':
#     ...
# else:
#     raise ValueError('Invalid benchmarks type (binaryoptimization, chemistry, condensedmatter, discreteoptimization)')
