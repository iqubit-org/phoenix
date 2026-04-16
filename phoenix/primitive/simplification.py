from itertools import combinations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford

from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian
from ..primitive.utils import SimplificationStep


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

# Precompute the 4x4 symplectic block for each Clifford type (computed once at import time).
# Each CNOT-equiv Clifford on qubits (q0, q1) only affects the 4x4 sub-block
# at indices {q0, q1, q0+n, q1+n} of the full 2n×2n symplectic matrix.
_CLIFFORD_BLOCKS: dict[int, np.ndarray] = {}
for _cliff in CLIFFORD_OPTIONS:
    _qc2 = QuantumCircuit(2)
    _qc2.append(_cliff, [0, 1])
    _CLIFFORD_BLOCKS[id(_cliff)] = Clifford(_qc2).symplectic_matrix.astype(np.int8)


def simplify_hamiltonian(ham: Hamiltonian) -> tuple[Hamiltonian, list[SimplificationStep]]:
    """
    Simplify a Hamiltonian (Pauli Tableau) using Clifford gates until weights are <= 2.

    Returns:
        The simplified Hamiltonian (remaining terms).
        A list of SimplificationStep, representing the operations applied
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

        current_ham = best_ham
        avoid = qubits

    return current_ham, simp_steps


def _apply_cliff_to_tableau(x: np.ndarray, z: np.ndarray,
                            block_4x4: np.ndarray, q0: int, q1: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2-qubit Clifford (given as its 4x4 symplectic block) to a tableau.

    Instead of constructing a full n-qubit Clifford and calling ``PauliList.evolve``,
    this directly multiplies the 4 affected columns of the tableau by the 4x4 block
    (mod 2).  ~10x faster for typical sizes.

    Returns new (x, z) arrays (copies — originals are not modified).
    """
    sub = np.column_stack([x[:, q0], x[:, q1], z[:, q0], z[:, q1]])
    new_sub = sub @ block_4x4 & 1
    new_x = x.copy()
    new_z = z.copy()
    new_x[:, q0] = new_sub[:, 0]
    new_x[:, q1] = new_sub[:, 1]
    new_z[:, q0] = new_sub[:, 2]
    new_z[:, q1] = new_sub[:, 3]
    return new_x, new_z


def search_best_clifford(ham: Hamiltonian, avoid: tuple[int, int]) -> tuple[
    Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]]:
    """Search for the best Clifford gate to apply."""

    qubit_pairs = sorted(combinations(ham.active_qubits, 2), key=lambda idx: (idx[0] % 2))
    qubit_pairs = [pair for pair in qubit_pairs if pair != avoid]

    x = ham.paulis.x.astype(np.int8)
    z = ham.paulis.z.astype(np.int8)

    best_cost = float('inf')
    best_cliff_idx = 0
    best_pair_idx = 0

    for ci, cliff in enumerate(CLIFFORD_OPTIONS):
        block = _CLIFFORD_BLOCKS[id(cliff)]
        for pi, (q0, q1) in enumerate(qubit_pairs):
            new_x, new_z = _apply_cliff_to_tableau(x, z, block, q0, q1)
            cost = heuristic_bsf_cost(new_x, new_z)
            if cost < best_cost:
                best_cost = cost
                best_cliff_idx = ci
                best_pair_idx = pi

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    return best_ham, best_cliff, best_qubit_pair


def _heuristic_bsf_cost(x: np.ndarray, z: np.ndarray) -> float:
    r"""
    Original heuristic cost for a Pauli Tableau, the smaller the simpler.

    .. math::
        \mathrm{cost}_{\mathrm{bsf}} := \mathrm{total\_weight} * n_{\mathrm{nonlocal}}^2
        + \sum_{\langle i,j \rangle} \lVert r_x^{(i)} \lor r_z^{(i)} \lor r_x^{(j)} \lor r_z^{(j)} \rVert
        + \frac{1}{2} \sum_{\langle i,j \rangle} (\lVert r_x^{(i)} \lor r_x^{(j)} \rVert + \lVert r_z^{(i)} \lor r_z^{(j)} \rVert)
    """
    with_ops = np.logical_or(x, z)
    row_weights = with_ops.sum(axis=1)
    which_nonlocal_paulis = np.where(row_weights > 1)[0]
    num_nonlocal_paulis = which_nonlocal_paulis.size

    if not np.any(with_ops):
        total_weight = 0
    elif not num_nonlocal_paulis:
        total_weight = 1
    else:
        total_weight = np.bitwise_or.reduce(with_ops[which_nonlocal_paulis], axis=0).sum()

    cost = 0.0
    if num_nonlocal_paulis > 1:
        row_combs = np.array(list(combinations(which_nonlocal_paulis, 2))).T
        cost += np.bitwise_or(with_ops[row_combs[0]], with_ops[row_combs[1]]).sum()
        cost += np.bitwise_or(x[row_combs[0]], x[row_combs[1]]).sum() * 0.5
        cost += np.bitwise_or(z[row_combs[0]], z[row_combs[1]]).sum() * 0.5

    cost += total_weight * num_nonlocal_paulis ** 2
    return cost


def heuristic_bsf_cost(x: np.ndarray, z: np.ndarray) -> float:
    r"""
    Optimized heuristic cost for a Pauli Tableau.

    Uses matmul-based pairwise OR counting:
        ``sum_{i<j} |a_i OR a_j| = (m-1)*sum|a_i| - (A·Aᵀ upper-triangle sum)``
    which replaces explicit ``combinations`` + fancy indexing for large m.
    """
    with_ops = np.logical_or(x, z)
    row_weights = with_ops.sum(axis=1)
    which_nl = np.where(row_weights > 1)[0]
    num_nl = which_nl.size

    if not np.any(with_ops):
        return 0.0
    elif not num_nl:
        return 0.0

    total_weight = np.bitwise_or.reduce(with_ops[which_nl], axis=0).sum()

    cost = 0.0
    if num_nl > 1:
        nl_ops = with_ops[which_nl].astype(np.int32)
        nl_x = x[which_nl].astype(np.int32)
        nl_z = z[which_nl].astype(np.int32)

        # sum_{i<j} |a_i OR a_j| = (m-1)*sum(|a_i|) - sum-upper-tri(A @ A.T)
        def _pairwise_or_sum(a: np.ndarray) -> float:
            m = a.shape[0]
            row_sums = a.sum(axis=1)
            aat = a @ a.T
            and_upper = (aat.sum() - np.trace(aat)) * 0.5
            return (m - 1) * row_sums.sum() - and_upper

        cost += _pairwise_or_sum(nl_ops)
        cost += _pairwise_or_sum(nl_x) * 0.5
        cost += _pairwise_or_sum(nl_z) * 0.5

    cost += total_weight * num_nl ** 2
    return cost
