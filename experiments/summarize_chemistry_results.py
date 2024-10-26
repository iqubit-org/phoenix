import sys

sys.path.append('..')

import os
import argparse
import pandas as pd
from phoenix import Circuit
from qiskit import QuantumCircuit

BENCHMARK_DPATH = '../benchmarks/chem_qasm'
OUTPUT_DPATH = './output_chem/'

parser = argparse.ArgumentParser(prog='Summarize compilation results (gate count and circuit depth statistics)')
parser.add_argument('-c', '--compiler', type=str, help='Compiler name')
parser.add_argument('-d', '--device', type=str, help='Device topology')
args = parser.parse_args()

OUTPUT_DPATH = os.path.join(OUTPUT_DPATH, args.compiler, args.device)

if not os.path.exists(OUTPUT_DPATH):
    raise ValueError('{} error, or {} error'.format(args.compiler, args.device))

result_fname = 'result_chem_{}_{}.csv'.format(args.compiler, args.device)

# TODO: summarize all2all, chain, grid to only ONE file

result = pd.DataFrame(columns=['benchmark', 'num_qubits', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q',
                               'num_gates(opt)', 'num_2q_gates(opt)', 'depth(opt)', 'depth_2q(opt)'])

for fname in os.listdir(BENCHMARK_DPATH):
    bench_name = fname.split('.')[0]
    origin_circ_file = os.path.join(BENCHMARK_DPATH, fname)
    output_circ_file = os.path.join(OUTPUT_DPATH, fname)
    if not os.path.exists(output_circ_file):
        continue

    circ_origin = QuantumCircuit.from_qasm_file(origin_circ_file)
    circ_opt = QuantumCircuit.from_qasm_file(output_circ_file)

    result = pd.concat([result, pd.DataFrame({
        'benchmark': bench_name,
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
