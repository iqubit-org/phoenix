"""Test some customized gate definitions"""
import cirq
import numpy as np
from phoenix import Circuit, gates


def test_ryy_by_qiskit():
    from qiskit.quantum_info import Operator
    rad = np.random.random()
    circ = Circuit([gates.RYY(rad).on([0, 1])])
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        Operator(circ.to_qiskit().reverse_bits()).to_matrix(),
        atol=1e-7
    )


def test_can_by_qiskit():
    from qiskit.quantum_info import Operator
    rads = np.random.random(3)
    circ = Circuit([gates.Can(*rads).on([0, 1])])
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        Operator(circ.to_qiskit().reverse_bits()).to_matrix(),
        atol=1e-7
    )


def test_clifford2q_by_qiskit():
    from qiskit.quantum_info import Operator
    from phoenix.utils import arch

    all_gates = arch.gene_random_circuit(8, 100).gates
    cliffs = []

    cliff_set = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    for _ in range(100):
        p0, p1 = np.random.choice(cliff_set)
        cliffs.append(gates.Clifford2QGate(p0, p1).on(np.random.choice(range(8), 2, replace=False).tolist()))

    all_gates += cliffs
    np.random.shuffle(all_gates)
    circ = Circuit(all_gates)

    circ_qiskit = circ.to_qiskit()
    circ1 = Circuit.from_qiskit(circ_qiskit)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        Operator(circ_qiskit.reverse_bits()).to_matrix(),
        atol=1e-7
    )

    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ1.unitary(),
        atol=1e-7
    )
