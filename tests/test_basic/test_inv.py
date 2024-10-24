"""Test gate inverse and circuit inverse"""

from phoenix import Circuit, gates
import cirq


def test_inverse_alu_v0_26():
    circ = Circuit.from_qasm(fname='../input/cx-basis/alu/alu-v0_26.qasm')

    circ.append(gates.Can(0.1, 0.2, 0.3).on([0, 1]))
    circ.append(gates.Can(-0.4, -0.5, 0.6).on([1, 2]))

    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ.inverse().unitary().conj().T,
        atol=1e-7
    )
