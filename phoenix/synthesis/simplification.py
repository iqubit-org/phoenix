"""
Simplification of Pauli Tableau using Clifford gates.
"""
import numpy as np
from copy import deepcopy
from itertools import product, combinations
from typing import Tuple, List
from phoenix.models.paulis import BSF
from phoenix.models.cliffords import Clifford2Q


def simplify_bsf(bsf: BSF) -> Tuple[BSF, List[Tuple[Clifford2Q, BSF]]]:
    """Simplify a Pauli Tableau, until its weights are simultaneously 2."""
    bsf = deepcopy(bsf)
    cliffords_with_locals = []
    avoid = None
    while bsf.total_weight > 2:
        local_bsf = bsf.pop_local_paulis()
        # if local_bsf.total_weight > 0:
        #     console.print(local_bsf)
        #     console.print(local_bsf.paulilist)
        # local_paulis, local_coeffs = local_bsf.paulilist, local_bsf.coeffs
        t, c, cliff = search_cliffords(bsf, avoid)
        avoid = cliff.ctrl, cliff.targ
        # cliffords_with_locals.append((cliff, list(zip(local_paulis, local_coeffs))))
        cliffords_with_locals.append((cliff, local_bsf))
        # console.print('applied {} --> {} cost: {} (is_simplified: {})'.format(cliff, t.paulilist, c, t.total_weight <= 2))
        bsf = t
    # console.print('Now BSF: {}, cliff_with_locals: {}'.format(bsf, cliffords_with_locals))
    return bsf, cliffords_with_locals


def simplify_bsf_with_weight(bsf: BSF, q: int):
    """Simplify a q-qubit Pauli Tableau, until its weights are simultaneously q."""
    pass


def search_cliffords(bsf: BSF, avoid: Tuple[int, int] = None) -> Tuple[BSF, float, Clifford2Q]:
    bsfs = []
    costs = []
    clifford_candidates = []
    for pauli_0, pauli_1 in product(['X', 'Y', 'Z'], ['X', 'Y', 'Z']):
        cg = Clifford2Q(pauli_0, pauli_1)  # controlled gate
        for i, j in combinations(bsf.qubits_with_ops.tolist(), 2):
            if (i, j) == avoid or (j, i) == avoid:
                continue
            bsf_ = bsf.apply_clifford_2q(cg, i, j)
            cost = heuristic_bsf_cost(bsf_)
            bsfs.append(bsf_)
            clifford_candidates.append((cg.on(i, j)))
            costs.append(cost)
            # console.print('searching C({},{}).on({},{}) --> cost={}'.format(pauli_0, pauli_1, i, j, cost))

    # select the candidates with the minimum cost
    which_candidates = np.where(np.array(costs) == np.min(costs))[0]
    bsfs = [bsfs[i] for i in which_candidates]
    costs = [costs[i] for i in which_candidates]
    clifford_candidates = [clifford_candidates[i] for i in which_candidates]
    # console.print(bsfs[0], costs[0], clifford_candidates[0])
    return bsfs[0], costs[0], clifford_candidates[0]


def is_simplified(bsf: BSF, q: int) -> bool:
    """Check if a q-qubit Pauli Tableau is simplified, i.e. its weights are simultaneously q."""
    if bsf.total_weight <= q:
        return True
    return False


def heuristic_bsf_cost(bsf: BSF, init_score: float = 0.0) -> float:
    """Heuristic cost for a 3-qubit Pauli Tableau, the smaller the simpler."""
    cost = init_score
    # cost += np.linalg.norm(bsf.x, ord=1, axis=1).sum() * 0.25
    # cost += np.linalg.norm(bsf.z, ord=1, axis=1).sum() * 0.25
    # cost += np.linalg.norm(bsf.x | bsf.z, ord=1, axis=1).sum() * 0.5
    for i, j in combinations(bsf.which_nonlocal_paulis, 2):
        cost += np.linalg.norm(bsf.with_ops[i] | bsf.with_ops[j], ord=1)

    for i, j in combinations(bsf.which_nonlocal_paulis, 2):
        cost += np.linalg.norm(bsf.x[i] | bsf.x[j], ord=1) * 0.5
        cost += np.linalg.norm(bsf.z[i] | bsf.z[j], ord=1) * 0.5

    cost += bsf.total_weight * bsf.num_nonlocal_paulis ** 2

    return cost
