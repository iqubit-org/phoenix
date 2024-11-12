"""
Summarize four-category Hamlib benchmarking results from some compiler
"""
import sys

sys.path.append('..')

import os
import json
import argparse
import pandas as pd
from natsort import natsorted
from qiskit import QuantumCircuit
import bqskit
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from phoenix import Circuit, gates
from phoenix.utils.operations import is_tensor_prod

BENCHMARK_DPATH = '../benchmarks/hamlib_qasm'
OUTPUT_DPATH = './output_hamlib/'

parser = argparse.ArgumentParser(prog='Summarize compilation results (gate count and circuit depth statistics)')
parser.add_argument('-c', '--compiler', type=str, help='Compiler name')
args = parser.parse_args()

OUTPUT_DPATH = os.path.join(OUTPUT_DPATH, args.compiler)

if not os.path.exists(OUTPUT_DPATH):
    raise ValueError('There is not compiled circuit output by {} compiler'.format(args.compiler))

result_fname = './results/result_hamlib_{}.csv'.format(args.compiler)

result = pd.DataFrame(columns=['category', 'program', 'num_qubits', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q',
                               'num_su4', 'depth_su4',
                               'num_gates(opt)', 'num_2q_gates(opt)', 'depth(opt)', 'depth_2q(opt)',
                               'num_su4(opt)', 'depth_su4(opt)'])

bqskit_compiler = Compiler()
workflow = QuickPartitioner(2)


def su4_circ_stats(qasm_fname):
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    circ = bqskit.Circuit.from_file(qasm_fname)
    blocks = list(bqskit_compiler.compile(circ, workflow))
    fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])
    circ_su4 = Circuit([g for g in fused_2q if not is_tensor_prod(g.data)])
    return circ_su4.num_gates, len(circ_su4.layer())


# pre-saved hamlib_su4_stats.json file
with open('../benchmarks/hamlib_su4_stats.json', 'r') as f:
    su4_stats_origin = json.load(f)

for dir in os.listdir(BENCHMARK_DPATH):
    print('Processing', os.path.join(BENCHMARK_DPATH, dir))
    fnames = natsorted(os.listdir(os.path.join(BENCHMARK_DPATH, dir)))
    for fname in fnames:
        program_name = fname.replace('.qasm', '')
        origin_circ_file = os.path.join(BENCHMARK_DPATH, dir, fname)
        output_circ_file = os.path.join(OUTPUT_DPATH, dir, fname)
        if not os.path.exists(output_circ_file):
            continue
        # if not os.path.exists(output_circ_su4_file):
        #     continue

        circ_origin = QuantumCircuit.from_qasm_file(origin_circ_file)
        circ_opt = QuantumCircuit.from_qasm_file(output_circ_file)

        if args.compiler == 'phoenix':  # update SU(4) circuit stats
            num_su4, depth_su4 = su4_circ_stats(output_circ_file)
        else:
            output_circ_su4_file = os.path.join(OUTPUT_DPATH, dir + '_su4', fname)
            circ_opt_su4 = QuantumCircuit.from_qasm_file(output_circ_su4_file)
            num_su4, depth_su4 = circ_opt_su4.size(), circ_opt_su4.depth(lambda instr: instr.operation.num_qubits > 1)

        result = pd.concat([result, pd.DataFrame({
            'category': dir,
            'program': program_name,
            'num_qubits': circ_origin.num_qubits,
            'num_gates': circ_origin.size(),
            'num_2q_gates': circ_origin.num_nonlocal_gates(),
            'depth': circ_origin.depth(),
            'depth_2q': circ_origin.depth(lambda instr: instr.operation.num_qubits > 1),
            'num_su4': su4_stats_origin[dir][program_name]['num_su4'],
            'depth_su4': su4_stats_origin[dir][program_name]['depth_su4'],
            'num_gates(opt)': circ_opt.size(),
            'num_2q_gates(opt)': circ_opt.num_nonlocal_gates(),
            'depth(opt)': circ_opt.depth(),
            'depth_2q(opt)': circ_opt.depth(lambda instr: instr.operation.num_qubits > 1),
            'num_su4(opt)': num_su4,
            'depth_su4(opt)': depth_su4
        }, index=[0])], ignore_index=True)

result.to_csv(result_fname, index=False)
print('Saved to', result_fname)

bqskit_compiler.close()
