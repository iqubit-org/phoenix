"""
Statistics SU(4) circuits of results from UCCSD benchmarks.
"""

import sys

sys.path.append('..')

import os
import json
import argparse
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
from qiskit import QuantumCircuit
import bqskit
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from phoenix import Circuit, gates
from phoenix.utils.operations import is_tensor_prod

bqskit_compiler = Compiler()
workflow = QuickPartitioner(2)


def su4_circ_stats(qasm_fname):
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    circ = bqskit.Circuit.from_file(qasm_fname)
    blocks = list(bqskit_compiler.compile(circ, workflow))
    fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])
    circ_su4 = Circuit([g for g in fused_2q if not is_tensor_prod(g.data)])
    return circ_su4.num_gates, len(circ_su4.layer())


selected_benchmarks = [
    'CH2_cmplt_BK',
    'CH2_cmplt_JW',
    'CH2_frz_BK',
    'CH2_frz_JW',
    'H2O_cmplt_BK',
    'H2O_cmplt_JW',
    'H2O_frz_BK',
    'H2O_frz_JW'
]

BENCHMARK_DPATH = '../benchmarks/uccsd_qasm'
OUTPUT_DPATH = './output_hamlib/'

result = pd.DataFrame(columns=['program', 'num_su4(tket)', 'depth_su4(tket)',
                               'num_su4(paulihedral)', 'depth_su4(paulihedral)',
                               'num_su4(tetris)', 'depth_su4(tetris)',
                               'num_su4(phoenix)', 'depth_su4(phoenix)'])
result['program'] = selected_benchmarks

for compiler in ['tket', 'paulihedral', 'tetris', 'phoenix']:
    num_su4_list = []
    depth_su4_list = []
    output_dpath = os.path.join(OUTPUT_DPATH, compiler)
    for program_name in selected_benchmarks:
        fname = os.path.join(BENCHMARK_DPATH, program_name + '.qasm')
        num_su4, depth_su4 = su4_circ_stats(os.path.join(output_dpath, fname))
        num_su4_list.append(num_su4)
        depth_su4_list.append(depth_su4)

    result[f'num_su4({compiler})'] = num_su4_list
    result[f'depth_su4({compiler})'] = depth_su4_list

result.to_csv('su4_stats.csv', index=False)
print(result)
