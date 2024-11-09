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

BENCHMARK_DPATH = '../benchmarks/uccsd_qasm'
OUTPUT_DPATH = './output_uccsd/'

parser = argparse.ArgumentParser(prog='Summarize compilation results (gate count and circuit depth statistics)')
parser.add_argument('-c', '--compiler', type=str, help='Compiler name')
args = parser.parse_args()

OUTPUT_DPATH = os.path.join(OUTPUT_DPATH, args.compiler)

if not os.path.exists(OUTPUT_DPATH):
    raise ValueError('{} deos not exist'.format(OUTPUT_DPATH))

result_fname = './results/result_uccsd_{}.csv'.format(args.compiler)

results = {
    'all2all': pd.DataFrame(columns=['program', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q']),
    'all2all_opt': pd.DataFrame(columns=['program', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q']),
    'manhattan': pd.DataFrame(columns=['program', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q']),
    'sycamore': pd.DataFrame(columns=['program', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q'])
}


for dir in ['all2all', 'all2all_opt', 'manhattan', 'sycamore']:
    output_dpath = os.path.join(OUTPUT_DPATH, dir)
    print('Processing', output_dpath)
    for fname in natsorted(os.listdir(output_dpath)):
        program_name = fname.split('.')[0]
        program_name = program_name.replace('_sto3g', '')  # simplify the name
        output_circ_file = os.path.join(output_dpath, fname)
        circ = QuantumCircuit.from_qasm_file(output_circ_file)

        results[dir] = pd.concat([results[dir], pd.DataFrame({
            'program': program_name,
            'num_gates': circ.size(),
            'num_2q_gates': circ.num_nonlocal_gates(),
            'depth': circ.depth(),
            'depth_2q': circ.depth(lambda instr: instr.operation.num_qubits > 1)
        }, index=[0])], ignore_index=True)

# initially, this data frame contains information of original circuits
result = pd.DataFrame(columns=['program', 'num_qubits', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q'])
for fname in natsorted(os.listdir(BENCHMARK_DPATH)):
    program_name = fname.split('.')[0]
    program_name = program_name.replace('_sto3g', '')  # simplify the name

    qasm_file = os.path.join(BENCHMARK_DPATH, fname)
    circ = QuantumCircuit.from_qasm_file(qasm_file)

    result = pd.concat([result, pd.DataFrame({
        'program': program_name,
        'num_qubits': circ.num_qubits,
        'num_gates': circ.size(),
        'num_2q_gates': circ.num_nonlocal_gates(),
        'depth': circ.depth(),
        'depth_2q': circ.depth(lambda instr: instr.operation.num_qubits > 1)
    }, index=[0])], ignore_index=True)


result = pd.merge(result, results['all2all'], on='program', suffixes=('', '(all2all)'))
result = pd.merge(result, results['all2all_opt'], on='program', suffixes=('', '(all2all_opt)'))
result = pd.merge(result, results['manhattan'], on='program', suffixes=('', '(manhattan)'))
result = pd.merge(result, results['sycamore'], on='program', suffixes=('', '(sycamore)'))

result.to_csv(result_fname, index=False)
print('Saved to', result_fname)
