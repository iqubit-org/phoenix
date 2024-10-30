#!/usr/bin/env python
"""
Benchmarking on hamlib100 with Phoenix compiler
"""

import sys

sys.path.append('..')

import os
import json
import argparse
from phoenix.models import HamiltonianModel

BENCHMARK_DPATH = '../benchmarks/hamlib_json'
OUTPUT_DPATH = './output/phoenix'
# BINARY_OPTIMIZATION = 'binaryoptimization'
# CHEMISTRY = 'chemistry'
# CONDENSED_MATTER = 'condensedmatter'
# DISCRETE_OPTIMIZATION = 'discreteoptimization'
CATEGORIES = ['binaryoptimization, chemistry, condensedmatter, discreteoptimization']

parser = argparse.ArgumentParser(description='Benchmarking on hamlib100 with Phoenix compiler')
parser.add_argument('-t', '--type', type=str,
                    help='Type of benchmarks (binaryoptimization, chemistry, condensedmatter, discreteoptimization')
args = parser.parse_args()



def bench_hamlib100_type(type):
    json_fnames = [fname for fname in os.listdir(os.path.join(BENCHMARK_DPATH, type)) if fname.endswith('.json')]
    for json_fname in json_fnames:
        output_fname = os.path.join(OUTPUT_DPATH, type, json_fname.replace('.json', '.qasm'))
        if os.path.exists(output_fname):
            continue
        print('Processing', json_fname)
        with open(os.path.join(BENCHMARK_DPATH, type, json_fname), 'r') as f:
            data = json.load(f)
        ham = HamiltonianModel(data['paulis'], data['coeffs'])
        circ = ham.reconfigure_and_generate_circuit()
        circ.to_qasm(output_fname)

bench_hamlib100_type(args.type)

#
# if args.type == 'binaryoptimization':
#     bench_hamlib100_type(args.type)
# elif args.type == 'chemistry':
#     bench_hamlib100_type(args.type)
#     # json_fnames = os.listdir(os.path.join(BENCHMARK_DPATH, args.type))
#     # for json_fname in json_fnames:
#     #     with open(os.path.join(BENCHMARK_DPATH, args.type, json_fname), 'r') as f:
#     #         data = json.load(f)
#     #     output_fname = os.path.join(OUTPUT_DPATH, args.type, json_fname.replace('.json', '.qasm'))
#     #     ham = HamiltonianModel(data['paulis'], data['coeffs'])
#     #     circ = ham.reconfigure_and_generate_circuit()
#     #     circ.to_qasm(output_fname)
#     #     print('Circuit saved to', output_fname)
# elif args.type == 'condensedmatter':
#     ...
# elif args.type == 'discreteoptimization':
#     ...
# else:
#     raise ValueError('Invalid benchmarks type (binaryoptimization, chemistry, condensedmatter, discreteoptimization)')
