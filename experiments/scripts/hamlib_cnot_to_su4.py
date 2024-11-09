#!/usr/bin/env python
"""Convert CNOT-ISA circuits to SU(4)-ISA circuits for Hamlib benchmarks"""

import sys

sys.path.append('../..')

import os
import warnings
import argparse
import qiskit.qasm2
import bench_utils
from natsort import natsorted
from phoenix import Circuit
from phoenix.utils.display import print_circ_info
from phoenix.synthesis.utils import rebase_to_canonical

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()
parser = argparse.ArgumentParser(description='Convert CNOT-ISA circuits to SU(4)-ISA circuits for Hamlib benchmarks')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='For which compiler (default: phoenix) (options: paulihedral, tetris, phoenix)')
args = parser.parse_args()

output_dpath = '../output_hamlib'


for dir in ['binaryoptimization', 'discreteoptimization', 'chemistry', 'condensedmatter']:
    cnot_dpath = os.path.join(output_dpath, args.compiler, dir)
    su4_dpath = os.path.join(output_dpath, args.compiler, dir + '_su4')

    if not os.path.exists(cnot_dpath):
        raise FileNotFoundError('Directory not found: {}'.format(cnot_dpath))
    if not os.path.exists(su4_dpath):
        os.makedirs(su4_dpath)

    console.rule('{}: CNOT --> SU(4) ISA on hamlib-{}'.format(args.compiler, dir))

    for fname in natsorted(os.listdir(cnot_dpath)):
        cnot_fname = os.path.join(cnot_dpath, fname)
        su4_fname = os.path.join(su4_dpath, fname)
        if os.path.exists(su4_fname):  # TODO: remove this line later
            continue
        console.print('Converting {} to {}'.format(cnot_fname, su4_fname))
        circ = Circuit.from_qasm(fname=cnot_fname)
        circ_su4 = rebase_to_canonical(circ)
        print_circ_info(circ)
        print_circ_info(circ_su4)
        circ_su4.to_qasm(su4_fname)
