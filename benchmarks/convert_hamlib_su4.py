"""
Convert CNOT-based circuits of Hamlib benchmarks to SU(4)-based circuits
"""
import sys

sys.path.append('..')

import os
import time
import qiskit
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from phoenix import Circuit, gates
from phoenix.synthesis.utils import unroll_su4, _normalize_canonical_coordinates, fuse_neighbor_u3
from phoenix.utils.display import print_circ_info

from rich.console import Console

console = Console()

bqskit_compiler = Compiler()

circ = Circuit.from_qasm(fname='./hamlib_qasm/chemistry/BH-JW-10.qasm')
# circ = Circuit.from_qasm(fname='./hamlib_qasm/chemistry/HF-BK12.qasm')
print_circ_info(circ, title='Original circuit')

start = time.process_time()
blocks = list(bqskit_compiler.compile(circ.to_bqskit(), QuickPartitioner(2)))
console.print('Partitioning time: {:.2f} s'.format(time.process_time() - start))

start = time.process_time()
fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])
console.print('Fusion time: {:.2f} s'.format(time.process_time() - start))

start = time.process_time()
circ_su4 = unroll_su4(fused_2q, by='can')
console.print('Unrolling time: {:.2f} s'.format(time.process_time() - start))

start = time.process_time()
circ_su4 = _normalize_canonical_coordinates(circ_su4)
console.print('Normalization time: {:.2f} s'.format(time.process_time() - start))

start = time.process_time()
circ_su4 = fuse_neighbor_u3(circ_su4)
console.print('Neighbor fusion time: {:.2f} s'.format(time.process_time() - start))

print_circ_info(circ_su4, title='SU(4) circuit')
