from phoenix.utils.arch import gene_random_circuit
from phoenix import gates, Circuit
import cirq


def test_dump_load():
    circ = gene_random_circuit(4, depth=100)
    circ.append(gates.Can(1.1, 2.2, 3.3).on([0, 1]))  # Canonical gate
    circ.append(gates.RYY(1.1).on([1, 2]))  # YY rotation
    circ2 = Circuit.from_qasm(circ.to_qasm())
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ2.unitary(),
        atol=1e-7
    )


# def test_qasm_by_qiskit():
#     from qiskit_aer import UnitarySimulator
#     circ = gene_random_circuit(4, depth=100)
#     circ_qiskit = circ.to_qiskit().reverse_bits()
#     backend = UnitarySimulator()
#     job = qiskit.execute(circ_qiskit, backend)
#     result = job.result()
#     unitary = result.get_unitary().to_matrix()
#     cirq.testing.assert_allclose_up_to_global_phase(
#         circ.unitary(),
#         unitary,
#         atol=1e-7
#     )

def test_qasm_by_qiskit():
    from qiskit.quantum_info import Operator
    circ = gene_random_circuit(4, depth=100)
    circ_qiskit = circ.to_qiskit().reverse_bits()
    # unitary = result.get_unitary().to_matrix()
    unitary = Operator(circ_qiskit).to_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        unitary,
        atol=1e-7
    )


def test_qasm_by_bqskit():
    circ = gene_random_circuit(4, depth=100)
    circ_bqskit = circ.to_bqskit()
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_bqskit.get_unitary(),
        atol=1e-7
    )


def test_qasm_by_cirq():
    circ = gene_random_circuit(4, depth=100)
    circ_cirq = circ.to_cirq()
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        cirq.unitary(circ_cirq),
        atol=1e-7
    )


def test_qasm_by_tket():
    circ = gene_random_circuit(4, depth=100)
    circ_tket = circ.to_tket()
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_tket.get_unitary(),
        atol=1e-7
    )
