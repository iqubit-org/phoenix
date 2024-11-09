import os
import qiskit
import pytket
import pytket.qasm
import pytket.passes
import qiskit.qasm2
from natsort import natsorted
from qiskit.transpiler import CouplingMap

from experiments.scripts import bench_utils

input_dpath = './output_uccsd/phoenix/all2all'


def qiskit_post_optimize(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    circ = qiskit.transpile(circ, optimization_level=3,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=CouplingMap(bench_utils.Manhattan_coupling.edge_list()),
                            layout_method='sabre')
    return circ


def tket_post_optimize(circ: pytket.Circuit) -> pytket.Circuit:
    circ = circ.copy()

    pytket.passes.FullPeepholeOptimise().apply(circ)

    circ = bench_utils.tket_to_qiskit(circ)
    circ = qiskit.transpile(circ, optimization_level=2,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=CouplingMap(bench_utils.Manhattan_coupling.edge_list()),
                            layout_method='sabre')

    circ = bench_utils.qiskit_to_tket(circ)
    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)

    return circ


for fname in natsorted(os.listdir(input_dpath)):
    fname = os.path.join(input_dpath, fname)
    print('Processing', fname)

    circ_qiskit = qiskit.QuantumCircuit.from_qasm_file(fname)
    # if circ_qiskit.num_qubits > 10:
    #     continue
    # if not 'H2O_cmplt_JW' in fname:
    #     continue
    if not 'CH2_frz' in fname:
        continue

    circ_tket = pytket.qasm.circuit_from_qasm(fname)

    # circ_qiskit_opt = qiskit_post_optimize(circ_qiskit)
    circ_tket_opt = tket_post_optimize(circ_tket)

    # qiskit.qasm2.dump(circ_qiskit_opt, os.path.join('qiskit_post_opt_manhattan', fname.split('/')[-1]))
    pytket.qasm.circuit_to_qasm(circ_tket_opt, os.path.join('tket_post_opt_manhattan2', fname.split('/')[-1]))
