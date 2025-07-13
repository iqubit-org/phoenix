import numpy as np
from copy import deepcopy
from phoenix.models.cliffords import CLIFFORD_2Q_SET
from phoenix.holistic.utils import heuristic_bsf_cost
from itertools import combinations
from phoenix.models import BSF

from rich.console import Console

console = Console()


def simplify_bsf(bsf: BSF):
    bsf = deepcopy(bsf)
    cliffords_with_locals = []
    avoid = None
    # while bsf.total_weight > 2:
    while len(bsf.which_multi_nonlocal_paulis):
        one_local_bsf = bsf.pop_local_paulis()
        two_local_bsf = bsf.pop_two_local_paulis()
        console.print('popped local paulis: {}; two local paulis: {}'.format(one_local_bsf.paulis, two_local_bsf.paulis))
        t, c, cliff = search_cliffords(bsf, avoid)
        avoid = cliff.ctrl, cliff.targ
        local_bsf = two_local_bsf.concat(one_local_bsf)
        cliffords_with_locals.append((cliff, local_bsf))
        console.print('applied {} --> {} cost: {} (is_simplified: {})'.format(cliff, t.paulis, c, is_simplified(t)))
        bsf = t
    # console.print('Now BSF: {}, cliff_with_locals: {}'.format(bsf, cliffords_with_locals))
    bsf.resort_two_local_paulis()

    return bsf, cliffords_with_locals


def is_simplified(bsf: BSF) -> bool:
    """Check if a q-qubit Pauli Tableau is simplified, i.e. its weights are simultaneously q."""
    if len(bsf.which_multi_nonlocal_paulis) == 0:
        return True
    return False


def search_cliffords(bsf: BSF, avoid: tuple = None):
    avoid = set(avoid) if avoid is not None else set()
    qubit_pairs = sorted(combinations(bsf.qubits_with_ops, 2), key=lambda idx: (idx[0] % 2))

    clifford_candidates = np.array([cg.on(qubits[0], qubits[1]) for cg in CLIFFORD_2Q_SET for qubits in
                                    qubit_pairs if set(qubits) != set(avoid)])

    # print(clifford_candidates)

    # use numpy vectorization to accelerate computation
    def trans_bsf(cliff):
        return bsf.apply_clifford_2q(cliff, cliff.ctrl, cliff.targ)

    trans_bsf = np.vectorize(trans_bsf)

    bsfs = trans_bsf(clifford_candidates)
    costs = heuristic_bsf_cost(bsfs)

    # select the candidates with the minimum cost
    argmin = np.argmin(costs)
    return bsfs[argmin], costs[argmin], clifford_candidates[argmin]