"""
Simplification of Pauli Tableau using Clifford gates.
"""
import numpy as np
from copy import deepcopy
from itertools import combinations
from typing import Tuple, List
from phoenix.models.paulis import BSF
from phoenix.models.cliffords import Clifford2Q, CLIFFORD_2Q_SET


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
    avoid = set(avoid) if avoid is not None else set()
    clifford_candidates = np.array([cg.on(qubits[0], qubits[1]) for cg in CLIFFORD_2Q_SET for qubits in
                                    np.array(list(combinations(bsf.qubits_with_ops, 2))) if set(qubits) != set(avoid)])

    # use numpy vectorization to accelerate computation
    def trans_bsf(cliff):
        return bsf.apply_clifford_2q(cliff, cliff.ctrl, cliff.targ)

    trans_bsf = np.vectorize(trans_bsf)

    bsfs = trans_bsf(clifford_candidates)
    costs = heuristic_bsf_cost(bsfs)

    # # select the candidates with the minimum cost
    # min_cost = np.min(costs)
    # which_candidates = np.where(np.array(costs) == min_cost)[0]
    # bsfs = bsfs[which_candidates]
    # costs = costs[which_candidates]
    # clifford_candidates = clifford_candidates[which_candidates]
    # return bsfs[0], costs[0], clifford_candidates[0]
    argmin = np.argmin(costs)
    return bsfs[argmin], costs[argmin], clifford_candidates[argmin]


def is_simplified(bsf: BSF, q: int) -> bool:
    """Check if a q-qubit Pauli Tableau is simplified, i.e. its weights are simultaneously q."""
    if bsf.total_weight <= q:
        return True
    return False


def _heuristic_bsf_cost_func(bsf: BSF) -> float:
    """Heuristic cost for a 3-qubit Pauli Tableau, the smaller the simpler."""
    # cost = init_score
    # cost += np.linalg.norm(bsf.x, ord=1, axis=1).sum() * 0.25
    # cost += np.linalg.norm(bsf.z, ord=1, axis=1).sum() * 0.25
    # cost += np.linalg.norm(bsf.x | bsf.z, ord=1, axis=1).sum() * 0.5
    cost = 0.0
    if bsf.which_nonlocal_paulis.size > 1:
        row_combs = np.array(list(combinations(bsf.which_nonlocal_paulis, 2))).T
        cost += np.linalg.norm(np.bitwise_or(bsf.with_ops[row_combs[0]],
                                             bsf.with_ops[row_combs[1]]), ord=1, axis=1).sum()
        cost += np.linalg.norm(np.bitwise_or(bsf.x[row_combs[0]], bsf.x[row_combs[1]]), ord=1, axis=1).sum() * 0.5
        cost += np.linalg.norm(np.bitwise_or(bsf.z[row_combs[0]], bsf.z[row_combs[1]]), ord=1, axis=1).sum() * 0.5
    cost += bsf.total_weight * bsf.num_nonlocal_paulis ** 2
    return cost


heuristic_bsf_cost = np.vectorize(_heuristic_bsf_cost_func)
