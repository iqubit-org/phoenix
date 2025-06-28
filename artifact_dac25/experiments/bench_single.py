#!/usr/bin/env python
import sys

sys.path.append('../..')

import json
import argparse
import warnings
from phoenix.utils.display import print_circ_info
from phoenix.models import HamiltonianModel
import bench_utils

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()

parser = argparse.ArgumentParser(description='Run a single benchmark')
parser.add_argument('filename', type=str,
                    help='Filename of the benchmark (a JSON file containing Pauli strings and coefficients)')
parser.add_argument('-d', '--device', default='all2all', type=str,
                    help='Device topology (default: all2all) (options: all2all, chain, manhattan, sycamore)')
parser.add_argument('--no-order', action='store_true',
                    help='Without IR group ordering procedure in Phoenix compiler (default: False)')
parser.add_argument('--O3', action='store_true',
                    help='With Qisit O3 for further local optimization in Phoenix compiler (default: False)')
parser.add_argument('--tket-greedy', action='store_true',
                    help='Use tket GreedyPauliSimp pass (default: False)')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

if args.device == 'all2all':
    coupling_map = bench_utils.All2all
elif args.device == 'chain':
    coupling_map = bench_utils.Chain
elif args.device == 'manhattan':
    coupling_map = bench_utils.Manhattan
elif args.device == 'sycamore':
    coupling_map = bench_utils.Sycamore
else:
    raise ValueError('Unsupported device')

console.rule('Benchmarking on {}'.format(args.filename))

with open(args.filename, 'r') as f:
    data = json.load(f)

circ = HamiltonianModel(data['paulis'], data['coeffs']).generate_circuit()
print_circ_info(circ, title='Original circuit')

if args.compiler == 'tket':
    circ_opt = bench_utils.tket_pass(data['paulis'], data['coeffs'],
                                     greedy=args.tket_greedy)
elif args.compiler == 'qiskit':
    circ_opt = bench_utils.qiskit_pass(data['paulis'], data['coeffs'], ...)
elif args.compiler == 'paulihedral':
    circ_opt = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
elif args.compiler == 'tetris':
    circ_opt = bench_utils.tetris_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
elif args.compiler == 'pauliopt':
    circ_opt = bench_utils.pauliopt_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map, with_O3=args.O3)
elif args.compiler == 'quclear':
    circ_opt = bench_utils.quclear_pass(data['paulis'], data['coeffs'],
                                        coupling_map=coupling_map,
                                        with_O3=args.O3)
elif args.compiler == 'phoenix':
    circ_opt = bench_utils.phoenix_pass(data['paulis'], data['coeffs'],
                                        order_blocks=not args.no_order,
                                        coupling_map=coupling_map,
                                        with_O3=args.O3)
else:
    raise ValueError('Unsupported compiler')

print_circ_info(circ_opt, title='Optimized circuit')
