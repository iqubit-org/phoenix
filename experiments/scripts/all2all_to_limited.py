"""
All-to-all compilation results (especially for TKet, PauliOpt, Phoenix) to limited connectivity topology (Manhattan, Sycamore)
"""
import sys

sys.path.append('../..')

import argparse
from experiments.scripts import bench_utils

bench_utils.optimize_with_mapping()

parser = argparse.ArgumentParser(description='Map logical circuits to physical qubits with limited connectivity')
parser.add_argument('-b', '--benchmark', type=str,
                    help='Designate the benchmark suite (hamlib or uccsd) )')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

'./'

ALL2ALL_DPATH = './output_{}'.format(args.benchmark)



