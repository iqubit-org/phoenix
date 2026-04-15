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
from phoenix.utils.ops import is_tensor_prod

bqskit_compiler = Compiler()
workflow = QuickPartitioner(2)


def su4_circ_stats(qasm_fname):
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    circ = bqskit.Circuit.from_file(qasm_fname)
    blocks = list(bqskit_compiler.compile(circ, workflow))
    circ_su4 = Circuit(
        [gates.X.on(blk.location[0], blk.location[1]) for blk in blocks if not is_tensor_prod(blk.get_unitary().numpy)])
    return circ_su4.num_gates, circ_su4.depth

selected_benchmarks = [
    'CH2_cmplt_BK',
    'CH2_cmplt_JW',
    'CH2_frz_BK',
    'CH2_frz_JW',
    'H2O_cmplt_BK',
    'H2O_cmplt_JW',
    'H2O_frz_BK',
    'H2O_frz_JW',
    'LiH_cmplt_BK',
    'LiH_cmplt_JW',
    'LiH_frz_BK',
    'LiH_frz_JW',
    'NH_cmplt_BK',
    'NH_cmplt_JW',
    'NH_frz_BK',
    'NH_frz_JW'
]

OUTPUT_DPATH = './output_uccsd/'

result = pd.DataFrame(columns=['program', 'num_su4(tket)', 'depth_su4(tket)',
                               'num_su4(paulihedral)', 'depth_su4(paulihedral)',
                               'num_su4(tetris)', 'depth_su4(tetris)',
                               'num_su4(phoenix)', 'depth_su4(phoenix)'])
result['program'] = selected_benchmarks

for compiler in ['tket', 'paulihedral', 'tetris', 'phoenix']:
    num_su4_list = []
    depth_su4_list = []
    output_dpath = os.path.join(OUTPUT_DPATH, compiler, 'all2all_opt')
    print(f'Processing {output_dpath}...')
    for program_name in tqdm(selected_benchmarks):
        fname = os.path.join(output_dpath, program_name + '_sto3g' + '.qasm')
        num_su4, depth_su4 = su4_circ_stats(fname)
        num_su4_list.append(num_su4)
        depth_su4_list.append(depth_su4)

    result[f'num_su4({compiler})'] = num_su4_list
    result[f'depth_su4({compiler})'] = depth_su4_list

result.to_csv('./results/result_uccsd_su4_stats.csv', index=False)
print(result)

bqskit_compiler.close()
