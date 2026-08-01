"""Tests for the custom RXY interaction gate."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Pauli

from phoenix.basics import RXYGate


def _expected_rxy(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * Pauli("YX").to_matrix()


@pytest.mark.parametrize("theta", [0.0, 0.37, -0.91, np.pi])
def test_rxy_gate_definition_matches_the_xy_rotation(theta):
    circuit = QuantumCircuit(2)
    circuit.append(RXYGate(theta), [0, 1])

    assert np.allclose(Operator(circuit.decompose()).data, _expected_rxy(theta))
    assert np.allclose(np.asarray(RXYGate(theta)), _expected_rxy(theta))


def test_rxy_gate_inverse_and_power():
    theta = 0.41

    assert np.allclose(
        Operator(RXYGate(theta).inverse()).data,
        _expected_rxy(-theta),
    )
    assert np.allclose(
        Operator(RXYGate(theta).power(3)).data,
        _expected_rxy(3 * theta),
    )


def test_rxy_gate_accepts_symbolic_angles():
    theta = Parameter("theta")
    gate = RXYGate(theta)

    assert gate.params == [theta]
    assert gate.definition.parameters == {theta}
