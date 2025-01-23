#!/usr/bin/env python
import sys

sys.path.append('../..')

import json
import qiskit
import pytket.qasm
import argparse
import warnings
from phoenix import Circuit, gates
from phoenix.utils.ops import is_tensor_prod
from phoenix.utils.display import print_circ_info, console
import bench_utils

warnings.filterwarnings('ignore')

from rich.console import Console

console = Console()

parser = argparse.ArgumentParser(description='Run a single benchmark')
parser.add_argument('filename', type=str, help='Filename of the benchmark')
parser.add_argument('-d', '--device', default='all2all', type=str,
                    help='Device topology (default: all2all) (options: all2all, chain, manhattan, sycamore)')
parser.add_argument('--no-order', action='store_true', help='Without IR group ordering procedure in Phoenix compiler (default: False)')
parser.add_argument('-c', '--compiler', default='phoenix', type=str,
                    help='Compiler (default: phoenix)')
args = parser.parse_args()

if 'json' in args.filename:
    json_fname = args.filename
    qasm_fname = args.filename.replace('json', 'qasm')
elif 'qasm' in args.filename:
    qasm_fname = args.filename
    json_fname = args.filename.replace('qasm', 'json')
else:
    raise ValueError('Unsupported file type {}'.format(args.filename))

# TODO: delete this
console.rule('Benchmarking on {}'.format(args.filename))

circ = qiskit.QuantumCircuit.from_qasm_file(qasm_fname)
with open(json_fname, 'r') as f:
    data = json.load(f)

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


def su4_circ_stats(circ):
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    from bqskit.compiler import Compiler
    from bqskit.passes import QuickPartitioner

    with Compiler() as compiler:
        blocks = list(compiler.compile(circ, QuickPartitioner(2)))

    fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])
    circ_su4 = Circuit([g for g in fused_2q if not is_tensor_prod(g.data)])
    return circ_su4.num_gates, len(circ_su4.layer())


if args.compiler == 'tket':
    circ = pytket.qasm.circuit_from_qasm(qasm_fname)
    circ_opt = bench_utils.tket_pass(circ)
    print_circ_info(circ, title='Original circuit')
    print_circ_info(circ_opt, title='Optimized circuit')
    console.print(su4_circ_stats(Circuit.from_tket(circ_opt).to_bqskit()))
elif args.compiler == 'paulihedral':
    circ = qiskit.QuantumCircuit.from_qasm_file(qasm_fname)
    circ_opt = bench_utils.paulihedral_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map)
    print_circ_info(circ, title='Original circuit')
    print_circ_info(circ_opt, title='Optimized circuit')
elif args.compiler == 'tetris':
    circ = qiskit.QuantumCircuit.from_qasm_file(qasm_fname)
    circ_opt = bench_utils.tetris_pass(data['paulis'], data['coeffs'], coupling_map=coupling_map)
    print_circ_info(circ, title='Original circuit')
    print_circ_info(circ_opt, title='Optimized circuit')
elif args.compiler == 'phoenix':
    circ = qiskit.QuantumCircuit.from_qasm_file(qasm_fname)
    circ_opt = bench_utils.phoenix_pass(data['paulis'], data['coeffs'], order=not args.no_order)
    print_circ_info(circ, title='Original circuit')
    print_circ_info(circ_opt, title='Optimized circuit')
else:
    raise ValueError('Unsupported compiler')
