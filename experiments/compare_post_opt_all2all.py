import os
import cirq
import qiskit
import pytket
import pytket.qasm
import pytket.passes
import qiskit.qasm2
from qiskit.quantum_info import Operator
from natsort import natsorted

input_dpath = './output_uccsd/phoenix/all2all'

import sys

sys.path.append('..')
from phoenix.utils.functions import infidelity


def qiskit_post_optimize(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    from itertools import combinations
    for q0, q1 in combinations(range(circ.num_qubits), 2):
        circ.cx(q0, q1)
        circ.cx(q0, q1)

    circ = qiskit.transpile(circ, optimization_level=3, basis_gates=['u1', 'u2', 'u3', 'cx'])
    return circ


def tket_post_optimize(circ: pytket.Circuit) -> pytket.Circuit:
    circ = circ.copy()
    pytket.passes.FullPeepholeOptimise().apply(circ)
    return circ


for fname in natsorted(os.listdir(input_dpath)):
    fname = os.path.join(input_dpath, fname)
    print('Processing', fname)

    circ_qiskit = qiskit.QuantumCircuit.from_qasm_file(fname)
    # if circ_qiskit.num_qubits > 10:
    #     continue

    circ_tket = pytket.qasm.circuit_from_qasm(fname)
    circ_qiskit_opt = qiskit_post_optimize(circ_qiskit)
    circ_tket_opt = tket_post_optimize(circ_tket)

    qiskit.qasm2.dump(circ_qiskit_opt, os.path.join('qiskit_post_opt', fname.split('/')[-1]))
    pytket.qasm.circuit_to_qasm(circ_tket_opt, os.path.join('tket_post_opt', fname.split('/')[-1]))

    # if circ_qiskit.num_qubits <= 10:
    #     try:
    #         u = Operator(circ_qiskit.reverse_bits()).to_matrix()
    #         v = Operator(circ_qiskit_opt.reverse_bits()).to_matrix()
    #         print('Qiskit infidelity:', infidelity(u, v))
    #         cirq.testing.assert_allclose_up_to_global_phase(
    #             u, v,
    #             atol=1e-6
    #         )
    #     except:
    #         print('! Qiskit fails on', fname)
    #
    #     try:
    #         u = circ_tket.get_unitary()
    #         v = circ_tket_opt.get_unitary()
    #         print('Tket infidelity:', infidelity(u, v))
    #         cirq.testing.assert_allclose_up_to_global_phase(
    #             u, v,
    #             atol=1e-6
    #         )
    #     except:
    #         print('! Tket fails on', fname)
