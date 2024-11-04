#!/usr/bin/env python
"""
Benchmarking on UCCSD benchmarks with given "compiler" and given "topology".
"""

import sys

sys.path.append('../..')

import os
import json
import argparse
import pytket.qasm
import qiskit.qasm2
from phoenix import gates
from natsort import natsorted
import bench_utils
from phoenix.utils.functions import infidelity

from rich.console import Console

console = Console()

INPUT_QASM_DPATH = '../../benchmarks/uccsd_qasm'
INPUT_JSON_DPATH = '../../benchmarks/uccsd_json'
OUTPUT_DPATH = '../output_uccsd'

parser = argparse.ArgumentParser(description='Benchmarking on UCCSD chemistry benchmarks')
parser.add_argument('-d', '--device', default='all2all', type=str,
                    help='Device topology (default: all2all) (options: all2all, manhattan, sycamore)')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

qasm_fnames = [os.path.join(INPUT_QASM_DPATH, fname) for fname in natsorted(os.listdir(INPUT_QASM_DPATH))]
json_fnames = [os.path.join(INPUT_JSON_DPATH, fname) for fname in natsorted(os.listdir(INPUT_JSON_DPATH), reverse=True)]

output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.device)

if not os.path.exists(output_dpath):
    os.makedirs(output_dpath)

console.print('topology: {}'.format(args.device))
console.print('compiler: {}'.format(args.compiler))
console.print('output directory: {}'.format(output_dpath))

if args.device == 'all2all':
    coupling_map = bench_utils.All2all  # TODO: make a all2all topology
elif args.device == 'manhattan':
    coupling_map = bench_utils.Manhattan
elif args.device == 'sycamore':
    coupling_map = bench_utils.Sycamore
else:
    raise ValueError('Unsupported device')

if args.compiler in ['phoenix', 'paulihedral', 'tetris', 'pauliopt']:
    for fname in json_fnames:

        # if 'LiH_frz' not in fname:
        #     continue

        console.print('Processing', fname)
        output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))
        with open(fname, 'r') as f:
            data = json.load(f)
        pre_gates = [gates.X.on(q) for q in data['front_x_on']]

        if args.compiler == 'phoenix':
            circ = bench_utils.phoenix_pass(data['paulis'], data['coeffs'], pre_gates)
            # pytket.qasm.circuit_to_qasm(circ, output_fname)
            qiskit.qasm2.dump(circ, output_fname)

            # circ_origin = qiskit.QuantumCircuit.from_qasm_file(
            #     os.path.join(INPUT_QASM_DPATH, os.path.basename(fname).replace('.json', '.qasm')))


            # console.print('Infidelity:',
            #               infidelity(bench_utils.qiskit_to_unitary(circ_origin),
            #                          bench_utils.qiskit_to_unitary(circ)))

        elif args.compiler == 'paulihedral':
            circ = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'], pre_gates,
                                                coupling_map=coupling_map)  # TODO: do no return mappings?

            circ_origin = qiskit.QuantumCircuit.from_qasm_file(
                os.path.join(INPUT_QASM_DPATH, os.path.basename(fname).replace('.json', '.qasm')))

            # if circ.num_qubits <= 10:

            console.print('Infidelity:',
                          infidelity(bench_utils.qiskit_to_unitary(circ_origin),
                                     bench_utils.qiskit_to_unitary(circ)))

            # qiskit.qasm2.dump(circ, output_fname)
        elif args.compiler == 'tetris':
            circ = bench_utils.tetris_pass(data['paulis'], data['coeffs'], pre_gates,
                                           coupling_map=coupling_map)  # TODO: do no return mappings?
            circ_origin = qiskit.QuantumCircuit.from_qasm_file(
                os.path.join(INPUT_QASM_DPATH, os.path.basename(fname).replace('.json', '.qasm')))

            # if circ.num_qubits <= 10:
            import sys
            sys.path.append('../..')
            from phoenix import Circuit
            c1 = Circuit.from_qiskit(circ)
            c2 = Circuit.from_qiskit(circ_origin)

            console.print('Infidelity:',
                          infidelity(c1.unitary(), c2.unitary()))
            # qiskit.qasm2.dump(circ, output_fname)
        elif args.compiler == 'pauliopt':
            circ = bench_utils.pauliopt_pass(data['paulis'], data['coeffs'], pre_gates,
                                             coupling_map=coupling_map)  # TODO: do no return mappings?
            qiskit.qasm2.dump(circ, output_fname)
        else:
            raise ValueError('Unsupported compiler')
else:
    if args.compiler == 'tket':
        for fname in qasm_fnames:
            console.print('Processing', fname)
            circ = pytket.qasm.circuit_from_qasm(fname)
            circ = bench_utils.tket_pass(circ)
            pytket.qasm.circuit_to_qasm(circ, os.path.join(output_dpath, os.path.basename(fname)))
    else:
        raise ValueError('Unsupported compiler')
