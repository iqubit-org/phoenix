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


def simplify_hamiltonian(ham: Hamiltonian) -> tuple[Hamiltonian, list[SimplificationStep]]:
    """
    Simplify a Hamiltonian (Pauli Tableau) using Clifford gates until weights are <= 2.
    
    Returns:
        The simplified Hamiltonian (remaining terms).
        A list of (CliffordGate, LocalHamiltonian) tuples, representing the operations applied
        and the local terms extracted at each step.
    """
    current_ham = ham
    simp_steps: list[SimplificationStep] = []
    avoid: tuple[int, int] = (-1, -1)

    while current_ham.total_weight > 2:
        local_ham, nonlocal_ham = current_ham.separate_local_nonlocal()

        best_ham, best_cliff, qubits = search_best_clifford(nonlocal_ham, avoid)

        simp_steps.append(SimplificationStep(
            clifford=best_cliff,
            local_hamiltonian=local_ham,
            qubits=qubits))

        # update current_ham and avoid qubit pair
        current_ham = best_ham
        avoid = qubits

    return current_ham, simp_steps

def search_best_clifford(ham: Hamiltonian, avoid: tuple[int, int]) -> tuple[
    Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]]:
    """
    Search for the best Clifford gate to apply.
    """
    n = ham.num_qubits

    qubit_pairs = sorted(combinations(ham.active_qubits, 2), key=lambda idx: (idx[0] % 2))
    qubit_pairs = [pair for pair in qubit_pairs if pair != avoid]

    def constr_clifford_operator(cliff, n, q0, q1):
        qc = QuantumCircuit(n)
        qc.append(cliff, [q0, q1])
        return Clifford(qc)

    # clifford_candidates = np.array([
    #     constr_clifford_operator(cliff, n, q0, q1) for cliff in CLIFFORD_OPTIONS for q0, q1 in qubit_pairs    
    # ],dtype=object)
    clifford_candidates = [
        constr_clifford_operator(cliff, n, q0, q1) for cliff in CLIFFORD_OPTIONS for q0, q1 in qubit_pairs    
    ]

    costs = []
    for clifford in clifford_candidates:
        new_paulis = ham.paulis.evolve(clifford, frame='s')
        costs.append(heuristic_bsf_cost(new_paulis.x, new_paulis.z))
        

    # def _eval_cliff_opr(clifford: Clifford):
    #     new_paulis = ham.paulis.evolve(clifford, frame='s')
    #     print('new_paulis:', new_paulis)
    #     return heuristic_bsf_cost(new_paulis.x, new_paulis.z)

    # eval_cliff_opr = np.vectorize(_eval_cliff_opr)
    # costs = eval_cliff_opr(clifford_candidates)

    argmin = np.argmin(costs)
    best_cliff = CLIFFORD_OPTIONS[argmin // len(qubit_pairs)]
    best_qubit_pair = qubit_pairs[argmin % len(qubit_pairs)]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    assert heuristic_bsf_cost(best_ham.paulis.x, best_ham.paulis.z) == costs[argmin]
    return best_ham, best_cliff, best_qubit_pair


def heuristic_bsf_cost(x: np.ndarray, z: np.ndarray) -> float:
    r"""
    Heuristic cost for a 3-qubit Pauli Tableau, the smaller the simpler.
    
    .. math::
        \mathrm{cost}_{\mathrm{bsf}} := \mathrm{total\_weight} * n_{\mathrm{nonlocal}}^2 
        + \sum_{\langle i,j \rangle} \lVert r_x^{(i)} \lor r_z^{(i)} \lor r_x^{(j)} \lor r_z^{(j)} \rVert  
        + \frac{1}{2} \sum_{\langle i,j \rangle} (\lVert r_x^{(i)} \lor r_x^{(j)} \rVert + \lVert r_z^{(i)} \lor r_z^{(j)} \rVert)

    """
    with_ops = np.logical_or(x, z)
    which_nonlocal_paulis = np.where(with_ops.sum(axis=1) > 1)[0]
    num_nonlocal_paulis = np.sum(with_ops.sum(axis=1) > 1)

    if not np.any(with_ops):
        total_weight = 0
    if not num_nonlocal_paulis:
        total_weight = 1
    total_weight = np.bitwise_or.reduce(with_ops[which_nonlocal_paulis], axis=0).sum()

    cost = 0.0
    if which_nonlocal_paulis.size > 1:
        row_combs = np.array(list(combinations(which_nonlocal_paulis, 2))).T
        cost += np.bitwise_or(with_ops[row_combs[0]], with_ops[row_combs[1]]).sum()
        cost += np.bitwise_or(x[row_combs[0]], x[row_combs[1]]).sum() * 0.5
        cost += np.bitwise_or(z[row_combs[0]], z[row_combs[1]]).sum() * 0.5

    cost += total_weight * num_nonlocal_paulis ** 2
    return cost
