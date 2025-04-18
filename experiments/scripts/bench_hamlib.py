#!/usr/bin/env python
"""
Benchmarking on Hamlib benchmarks with given "compiler" and given "category" in Hamlib programs.
"""

import sys

sys.path.append('../..')

import os
import json
import argparse
import pytket.qasm
import qiskit.qasm2
import rustworkx as rx
from qiskit.transpiler import CouplingMap
from phoenix.utils.display import print_circ_info
from natsort import natsorted
import bench_utils

from rich.console import Console

console = Console()

INPUT_QASM_DPATH = '../../benchmarks/hamlib_qasm'
INPUT_JSON_DPATH = '../../benchmarks/hamlib_json'
OUTPUT_DPATH = '../output_hamlib'

CATEGORIES = ['binaryoptimization, chemistry, condensedmatter, discreteoptimization']

parser = argparse.ArgumentParser(description='Benchmarking on hamlib100 with Phoenix compiler')
parser.add_argument('-t', '--type', type=str,
                    help='Type of benchmarks (binaryoptimization, chemistry, condensedmatter, discreteoptimization')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

qasm_fnames = [os.path.join(INPUT_QASM_DPATH, args.type, fname)
               for fname in natsorted(os.listdir(os.path.join(INPUT_QASM_DPATH, args.type)))]
json_fnames = [os.path.join(INPUT_JSON_DPATH, args.type, fname)
               for fname in natsorted(os.listdir(os.path.join(INPUT_JSON_DPATH, args.type)), reverse=True)]

output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.type)

if not os.path.exists(output_dpath):
    os.makedirs(output_dpath)

console.print('program type: {}'.format(args.type))
console.print('compiler: {}'.format(args.compiler))
console.print('output directory: {}'.format(output_dpath))

if args.compiler in ['phoenix', 'paulihedral', 'tetris', 'pauliopt']:
    for fname in json_fnames:

        console.print('Processing', fname)
        output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))

        # if os.path.exists(output_fname):
        #     console.print('Already processed', output_fname)
        #     continue

        with open(fname, 'r') as f:
            data = json.load(f)

        if args.compiler == 'phoenix':
            if qiskit.QuantumCircuit.from_qasm_file(fname.replace('json', 'qasm')).num_nonlocal_gates() > 3000:
                continue
                circ = bench_utils.phoenix_pass(data['paulis'], data['coeffs'], efficient=True)
            else:
                circ = bench_utils.phoenix_pass(data['paulis'], data['coeffs'], efficient=False)

            circ = qiskit.transpile(circ, optimization_level=2, basis_gates=['u1', 'u2', 'u3', 'cx'])
            circ = bench_utils.qiskit_O3_all2all(circ)
            print_circ_info(circ)
            qiskit.qasm2.dump(circ, output_fname)
        elif args.compiler == 'paulihedral':
            if qiskit.QuantumCircuit.from_qasm_file(fname.replace('json', 'qasm')).num_nonlocal_gates() > 10000:
                continue
            circ = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'], coupling_map=CouplingMap(
                rx.generators.complete_graph(data['num_qubits']).to_directed().edge_list()))
            print_circ_info(circ)
            qiskit.qasm2.dump(circ, output_fname)
        elif args.compiler == 'tetris':
            circ = bench_utils.tetris_pass(data['paulis'], data['coeffs'], coupling_map=CouplingMap(
                rx.generators.complete_graph(data['num_qubits']).to_directed().edge_list()))
            print_circ_info(circ)
            qiskit.qasm2.dump(circ, output_fname)
        elif args.compiler == 'pauliopt':
            circ = bench_utils.pauliopt_pass(data['paulis'], data['coeffs'])
            print_circ_info(circ)
            qiskit.qasm2.dump(circ, output_fname)
        else:
            raise ValueError('Unsupported compiler')
else:
    if args.compiler == 'tket':
        for fname in qasm_fnames:
            console.print('Processing', fname)

            # TODO: delete this line
            # if os.path.exists(os.path.join(output_dpath, os.path.basename(fname))):
            # continue

            circ = pytket.qasm.circuit_from_qasm(fname)
            circ = bench_utils.tket_pass(circ)
            print_circ_info(circ)

            # pytket.qasm.circuit_to_qasm(circ, os.path.join(output_dpath, os.path.basename(fname)))
    else:
        raise ValueError('Unsupported compiler')
