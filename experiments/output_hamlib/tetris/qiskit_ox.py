import qiskit
import os
from natsort import natsorted
import qiskit.qasm2
import sys

sys.path.append('../../..')
from phoenix.utils.display import print_circ_info

from rich.console import Console

console = Console()

output_dpath = './binaryoptimization'
output_dpath_new = output_dpath + '_new'

if not os.path.exists(output_dpath_new):
    os.mkdir(output_dpath_new)

for fname in natsorted(os.listdir(output_dpath)):
    console.rule(fname)
    fname = os.path.join(output_dpath, fname)
    qc = qiskit.QuantumCircuit.from_qasm_file(fname)
    print_circ_info(qc, title='Original circuit')
    qc_opt = qiskit.transpile(qc, optimization_level=2, basis_gates=['u1', 'u2', 'u3', 'cx'])
    print_circ_info(qc_opt, title='Optimized circuit')
    qiskit.qasm2.dump(qc_opt, os.path.join(output_dpath_new, os.path.basename(fname)))
