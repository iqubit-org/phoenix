#!/usr/bin/env python
"""
Optimize logical circuits by Qiskit O3 for UCCSD benchmarks
"""
import sys

sys.path.append('../..')

import os
import warnings
import argparse
import qiskit.qasm2
import bench_utils
from natsort import natsorted
from phoenix.utils.display import print_circ_info

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()

parser = argparse.ArgumentParser(description='Optimize logical circuits by Qiskit O3 for UCCSD benchmarks')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='For which compiler (default: phoenix) (options: paulihedral, tetris, phoenix)')
args = parser.parse_args()

output_dpath = '../output_uccsd/'

all2all_dpath = os.path.join(output_dpath, args.compiler, 'all2all')
all2all_opt_dpath = os.path.join(output_dpath, args.compiler, 'all2all_opt')

if not os.path.exists(all2all_dpath):
    raise FileNotFoundError('Directory not found: {}'.format(all2all_dpath))
if not os.path.exists(all2all_opt_dpath):
    os.makedirs(all2all_opt_dpath)

console.rule('Applying Qiskit O3 for {}'.format(args.compiler))

for fname in natsorted(os.listdir(all2all_dpath)):
    all2all_fname = os.path.join(all2all_dpath, fname)
    all2all_opt_fname = os.path.join(all2all_opt_dpath, fname)
    console.print('Converting {} to {}'.format(all2all_fname, all2all_opt_fname))
    circ = qiskit.QuantumCircuit.from_qasm_file(all2all_fname)
    circ = bench_utils.qiskit_O3_all2all(circ)
    print_circ_info(circ)
    qiskit.qasm2.dump(circ, all2all_opt_fname)
