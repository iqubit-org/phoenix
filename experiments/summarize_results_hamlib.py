"""
Summarize four-category Hamlib benchmarking results from some compiler
"""
import sys

sys.path.append('..')

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from natsort import natsorted
from qiskit import QuantumCircuit
# import canopus


BENCHMARK_DPATH = '../benchmarks/hamlib_qasm'
OUTPUT_DPATH = './output_hamlib/'


def bench_stats(args_tuple):
    category, program_name, origin_circ_file, output_circ_file = args_tuple
    circ_origin = QuantumCircuit.from_qasm_file(origin_circ_file)
    circ_opt = QuantumCircuit.from_qasm_file(output_circ_file)
    # circ_opt_su4 = canopus.rebase_to_canonical(circ_opt)

    # num_su4 = circ_opt_su4.count_ops().get('can', 0)
    # depth_su4 = circ_opt_su4.depth(lambda instr: instr.operation.name == 'can')

    return {
        'category': category,
        'program': program_name,
        'num_qubits': circ_origin.num_qubits,
        'num_gates': circ_origin.size(),
        'num_2q_gates': circ_origin.num_nonlocal_gates(),
        'depth': circ_origin.depth(),
        'depth_2q': circ_origin.depth(lambda instr: instr.operation.num_qubits > 1),
        'num_gates(opt)': circ_opt.size(),
        'num_2q_gates(opt)': circ_opt.num_nonlocal_gates(),
        'depth(opt)': circ_opt.depth(),
        'depth_2q(opt)': circ_opt.depth(lambda instr: instr.operation.num_qubits > 1),
        # 'num_su4(opt)': num_su4,
        # 'depth_su4(opt)': depth_su4,
    }


def main():
    parser = argparse.ArgumentParser(prog='Summarize compilation results (gate count and circuit depth statistics)')
    parser.add_argument('-c', '--compiler', type=str, help='Compiler name')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of parallel worker processes (default: os.cpu_count())')
    args = parser.parse_args()

    output_dpath = os.path.join(OUTPUT_DPATH, args.compiler)
    if not os.path.exists(output_dpath):
        raise ValueError('There is not compiled circuit output by {} compiler'.format(args.compiler))

    result_fname = './results/result_hamlib_{}.csv'.format(args.compiler)

    columns = ['category', 'program', 'num_qubits', 'num_gates', 'num_2q_gates', 'depth', 'depth_2q',
               'num_gates(opt)', 'num_2q_gates(opt)', 'depth(opt)', 'depth_2q(opt)',]
            #    'num_su4(opt)', 'depth_su4(opt)']

    tasks = []
    for category in os.listdir(BENCHMARK_DPATH):
        print('Collecting', os.path.join(BENCHMARK_DPATH, category))
        fnames = natsorted(os.listdir(os.path.join(BENCHMARK_DPATH, category)))
        for fname in fnames:
            program_name = fname.replace('.qasm', '')
            origin_circ_file = os.path.join(BENCHMARK_DPATH, category, fname)
            output_circ_file = os.path.join(output_dpath, category, fname)
            if not os.path.exists(output_circ_file):
                continue
            tasks.append((category, program_name, origin_circ_file, output_circ_file))

    print('Total benchmarks to process: {} (jobs={})'.format(len(tasks), args.jobs))

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(bench_stats, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as e:
                cat, prog, _, _ = futures[future]
                print('Failed {}/{}: {}'.format(cat, prog, e))

    result = pd.DataFrame(rows, columns=columns)
    result = result.sort_values(['category', 'program']).reset_index(drop=True)
    result.to_csv(result_fname, index=False)
    print('Saved to', result_fname)


if __name__ == '__main__':
    main()
