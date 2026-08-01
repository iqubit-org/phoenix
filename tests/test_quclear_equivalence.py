"""Functional regression tests for the QuCLEAR benchmark integration.

QuCLEAR takes RZ angles, while benchmark JSON files store Hamiltonian
coefficients for exp(-i * c * P).  These tests compare the complete QuCLEAR
wrapper, including its terminal Clifford synthesis and common post-pass, with
the corresponding first-order Pauli-evolution product.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "experiments" / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bench_utils  # noqa: E402, I001


CASES = (
    # One non-commuting one-qubit sequence exercises the order convention.
    (("X", "Z"), (0.31, -0.47)),
    # Mixed commuting/non-commuting terms exercise Clifford extraction.
    (("XX", "ZI", "IY"), (0.31, -0.47, 0.22)),
    (("XYZ", "ZIX", "IYY", "XXI"), (0.31, -0.47, 0.22, 0.13)),
)


def first_order_reference(paulis: tuple[str, ...], coeffs: tuple[float, ...]) -> QuantumCircuit:
    """Build \u220f_j exp(-i c_j P_j) in the JSON term order."""
    circuit = QuantumCircuit(len(paulis[0]))
    for pauli, coefficient in zip(paulis, coeffs):
        circuit.append(PauliEvolutionGate(SparsePauliOp([pauli], [coefficient])), circuit.qubits)
    return circuit


@pytest.mark.parametrize(("paulis", "coeffs"), CASES)
def test_quclear_matches_pauli_evolution(paulis: tuple[str, ...], coeffs: tuple[float, ...]) -> None:
    compiled = bench_utils.quclear_pass(list(paulis), list(coeffs))
    reference = first_order_reference(paulis, coeffs)
    assert Operator(compiled).equiv(Operator(reference))


def test_quclear_rejects_nonreal_coefficients() -> None:
    with pytest.raises(ValueError, match="only real"):
        bench_utils.quclear_pass(["Z"], [0.2 + 0.1j])
