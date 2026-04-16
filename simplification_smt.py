from itertools import combinations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford

from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian
from ..primitive.utils import SimplificationStep

# Define the set of 2-qubit Clifford gates used for simplification
# CLIFFORD_PAULI_PAIRS = [
#     ('X', 'X'), ('Y', 'Y'), ('Z', 'Z'),
#     ('X', 'Y'), ('Y', 'X'),
#     ('X', 'Z'), ('Z', 'X'),
#     ('Y', 'Z'), ('Z', 'Y')
# ]


# ! five elements is sufficient to generate the full 2-qubit Clifford group (11520 elements)
# CLIFFORD_PAULI_PAIRS = [
#     ('X', 'X'), ('Y', 'Y'), ('Z', 'Z'),
#     ('X', 'Z'), ('Z', 'X'),
# ]

# TODO: 为什么这个五个元素的效果比下面六个元素的效果要好
CLIFFORD_OPTIONS = [
    CNOTEquivCliffordGate('X', 'X'), CNOTEquivCliffordGate('Y', 'Y'), CNOTEquivCliffordGate('Z', 'Z'),
    CNOTEquivCliffordGate('X', 'Z'), CNOTEquivCliffordGate('Z', 'X'),
]


# CLIFFORD_OPTIONS = [
#     CNOTEquivCliffordGate("X", "X"), CNOTEquivCliffordGate("Y", "Y"), CNOTEquivCliffordGate("Z", "Z"),
#     CNOTEquivCliffordGate("X", "Y"), CNOTEquivCliffordGate("Y", "Z"), CNOTEquivCliffordGate("Z", "X"),
# ]

# TODO: 要么用五个元素，要么用下面九个元素
CLIFFORD_OPTIONS = [
    CNOTEquivCliffordGate('X', 'X'),
    CNOTEquivCliffordGate('Y', 'Y'),
    CNOTEquivCliffordGate('Z', 'Z'),
    CNOTEquivCliffordGate('X', 'Y'),
    CNOTEquivCliffordGate('Y', 'X'),
    CNOTEquivCliffordGate('X', 'Z'),
    CNOTEquivCliffordGate('Z', 'X'),
    CNOTEquivCliffordGate('Y', 'Z'),
    CNOTEquivCliffordGate('Z', 'Y'),
]


def simplify_hamiltonian_smt(ham: Hamiltonian) -> tuple[Hamiltonian, list[SimplificationStep]]:
    """
    Simplify a Hamiltonian (Pauli Tableau) using Clifford gates until weights are <= 2.
    
    Returns:
        The simplified Hamiltonian (remaining terms).
        A list of (CliffordGate, LocalHamiltonian) tuples, representing the operations applied
        and the local terms extracted at each step.
    """
