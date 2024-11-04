import qiskit
import pytket.qasm

# from phoenix.utils.functions import infidelity
import cirq

from phoenix.utils.functions import infidelity

fname = './benchmarks/uccsd_qasm/LiH_frz_P_sto3g.qasm'

qc = qiskit.QuantumCircuit.from_qasm_file(fname)

print(qc.num_nonlocal_gates(), qc.size())

import qiskit_convert

circ = qiskit_convert.qiskit_to_tk(qc)
import pytket.passes

pytket.passes.FullPeepholeOptimise().apply(circ)


def tket_to_qiskit(circ: pytket.Circuit) -> qiskit.QuantumCircuit:
    return qiskit.QuantumCircuit.from_qasm_str(pytket.qasm.circuit_to_qasm_str(circ))


print(circ.n_2qb_gates(), circ.n_gates)

from qiskit.quantum_info import Operator



print(infidelity(
    # Operator(tket_to_qiskit(circ).reverse_bits()).to_matrix(),
    Operator(qiskit_convert.tk_to_qiskit(circ).reverse_bits()).to_matrix(),
    Operator(qc.reverse_bits()).to_matrix(),
))


cirq.testing.assert_allclose_up_to_global_phase(
    # circ.get_unitary(),

    # Operator(tket_to_qiskit(circ).reverse_bits()).to_matrix(),
    Operator(qiskit_convert.tk_to_qiskit(circ).reverse_bits()).to_matrix(),
    Operator(qc.reverse_bits()).to_matrix(),
    atol=1e-6
)
