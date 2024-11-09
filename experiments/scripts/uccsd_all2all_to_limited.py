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
from natsort import natsorted
import qiskit.qasm2
import argparse
import bench_utils
from phoenix.utils.display import print_circ_info

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Map logical circuits to physical qubits with limited connectivity')
parser.add_argument('-d', '--device', type=str,
                    help='Device topology (options: manhattan, sycamore)')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

all2all_dpath = '../output_uccsd/{}/all2all'.format(args.compiler)
fnames = natsorted(os.listdir(all2all_dpath), reverse=True)

if args.device == 'manhattan':
    coupling_map = bench_utils.Manhattan
elif args.device == 'sycamore':
    coupling_map = bench_utils.Sycamore
else:
    raise ValueError('Unsupported topology')

limited_dpath = '../output_uccsd/{}/{}'.format(args.compiler, args.device)
if not os.path.exists(limited_dpath):
    os.makedirs(limited_dpath)

for fname in fnames:
    all2all_circ_file = os.path.join(all2all_dpath, fname)
    limited_circ_file = os.path.join(limited_dpath, fname)

    print('Converting {} to {}'.format(all2all_circ_file, limited_circ_file))

    circ = qiskit.QuantumCircuit.from_qasm_file(all2all_circ_file)
    circ = bench_utils.optimize_with_mapping(circ, coupling_map)
    print_circ_info(circ)
    qiskit.qasm2.dump(circ, limited_circ_file)
