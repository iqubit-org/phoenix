"""
Summarize UCCSD hardware-aware (all2all, manhattan, sycamore) benchmarking results from some compiler
"""

import sys

sys.path.append('..')

import os
import argparse
import pandas as pd
from natsort import natsorted
from qiskit import QuantumCircuit

# BENCHMARK_DPATH = '../benchmarks/uccsd_qasm'
OUTPUT_DPATH = './output_uccsd/'

parser = argparse.ArgumentParser(prog='Summarize compilation results (gate count and circuit depth statistics)')
parser.add_argument('-c', '--compiler', type=str, help='Compiler name')
args = parser.parse_args()

OUTPUT_DPATH = os.path.join(OUTPUT_DPATH, args.compiler)

if not os.path.exists(OUTPUT_DPATH):
    raise ValueError('{} deos not exist'.format(OUTPUT_DPATH))

result_fname = 'result_uccsd_{}.csv'.format(args.compiler)


result = pd.DataFrame(columns=['program',
                               'num_gates(all2all)', 'num_2q_gates(all2all)',
                               'depth(all2all)', 'depth_2q(all2all)',
                               'num_gates(manhattan)', 'num_2q_gates(manhattan)',
                               'depth(manhattan)', 'depth_2q(manhattan)',
                               'num_gates(sycamore)', 'num_2q_gates(sycamore)',
                               'depth(sycamore)', 'depth_2q(sycamore)'])

results = {
    'all2all': pd.DataFrame(columns=['program', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q']),
}


for device in ['all2all', 'manhattan', 'sycamore']:
    output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, device)
    if not os.path.exists(output_dpath):
        continue




for fname in natsorted(os.listdir(BENCHMARK_DPATH)):
    program_name = fname.split('.')[0]
    origin_circ_file = os.path.join(BENCHMARK_DPATH, fname)
    output_circ_file = os.path.join(OUTPUT_DPATH, fname)
    if not os.path.exists(output_circ_file):
        continue

    circ_origin = QuantumCircuit.from_qasm_file(origin_circ_file)
    circ_opt = QuantumCircuit.from_qasm_file(output_circ_file)

    result = pd.concat([result, pd.DataFrame({
        'program': program_name,
        'num_qubits': circ_origin.num_qubits,
        'num_gates': circ_origin.size(),
        'num_2q_gates': circ_origin.num_nonlocal_gates(),
        'depth': circ_origin.depth(),
        'depth_2q': circ_origin.depth(lambda instr: instr.operation.num_qubits > 1),
        'num_gates(opt)': circ_opt.size(),
        'num_2q_gates(opt)': circ_opt.num_nonlocal_gates(),
        'depth(opt)': circ_opt.depth(),
        'depth_2q(opt)': circ_opt.depth(lambda instr: instr.operation.num_qubits > 1)
    }, index=[0])], ignore_index=True)

result.to_csv(result_fname, index=False)
