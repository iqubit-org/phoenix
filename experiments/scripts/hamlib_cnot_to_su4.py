#!/usr/bin/env python
"""Convert CNOT-ISA circuits to SU(4)-ISA circuits for Hamlib benchmarks"""

import sys

sys.path.append('../..')

import os
import warnings
import argparse
from natsort import natsorted
from phoenix import Circuit
from phoenix.utils.display import print_circ_info

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()
parser = argparse.ArgumentParser(description='Convert CNOT-ISA circuits to SU(4)-ISA circuits for Hamlib benchmarks')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='For which compiler (default: phoenix) (options: paulihedral, tetris, phoenix)')
args = parser.parse_args()

output_dpath = '../output_hamlib'

from phoenix import gates
from phoenix.synthesis.utils import unroll_su4, _normalize_canonical_coordinates, fuse_neighbor_u3
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner

bqskit_compiler = Compiler()

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
        # if os.path.exists(su4_fname):  # TODO: remove this line later
        #     continue
        console.print('Converting {} to {}'.format(cnot_fname, su4_fname))
        circ = Circuit.from_qasm(fname=cnot_fname)
        ##############################################
        # rebase to SU(4) ISA
        blocks = list(bqskit_compiler.compile(circ.to_bqskit(), QuickPartitioner(2)))
        fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])
        circ_su4 = unroll_su4(fused_2q, by='can')
        circ_su4 = _normalize_canonical_coordinates(circ_su4)
        circ_su4 = fuse_neighbor_u3(circ_su4)
        #############################################
        # circ_su4 = rebase_to_canonical(circ)
        print_circ_info(circ)
        print_circ_info(circ_su4)
        circ_su4.to_qasm(su4_fname)

bqskit_compiler.close()
