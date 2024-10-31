"""
Transformations on Paulis.
"""
import numpy as np
from itertools import combinations, product
from copy import deepcopy
from functools import reduce
from operator import add
from typing import List, Tuple


from phoenix.basic import gates
from phoenix.basic.gates import Gate
from phoenix.basic.circuits import Circuit
from phoenix.models.paulis import BSF
from phoenix.models.cliffords import Clifford2Q


from rich.console import Console

console = Console()


################################################################
# Grouping
################################################################

def group_paulis(paulis: List[str]):
    """
    Group Pauli strings by their nontrivial parts.

    E.g.,

        ['XXIII', 'YYIII', 'ZZIII', 'IXXII', 'IYYII', 'IZZII', 'IIXXI', 'IIYYI', 'IIZZI', 'IIIXX', 'IIIYY', 'IIIZZ', 'ZIIII', 'IZIII', 'IIZII', 'IIIZI', 'IIIIZ']

    will be grouped as

         {(0, 1): ['XXIII', 'YYIII', 'ZZIII'],
          (2, 3): ['IIXXI', 'IIYYI', 'IIZZI'],
          (3, 4): ['IIIXX', 'IIIYY', 'IIIZZ'],
          (1, 2): ['IXXII', 'IYYII', 'IZZII'],
          (0,): ['ZIIII'],
          (1,): ['IZIII'],
          (2,): ['IIZII'],
          (3,): ['IIIZI'],
          (4,): ['IIIIZ']}
    """
    assert len(paulis) == len(np.unique(paulis)), 'Pauli strings must be unique'

    nontrivial = [tuple(np.where(np.array(list(pauli)) != 'I')[0]) for pauli in paulis]
    groups = {}
    for idx, pauli in zip(nontrivial, paulis):
        if idx not in groups:
            groups[idx] = [pauli]
        else:
            groups[idx].append(pauli)

    # sort "groups" according to the length of the keys and keys themselves
    groups = dict(sorted(groups.items(), key=lambda x: (-len(x[0]), x[0])))

    # reorder items to reduce overall length when organizing as circuit
    groups_on_length = {}  # {length: {idx: [paulis]}}
    for idx, paulis in groups.items():
        length = len(idx)
        if length not in groups_on_length:
            groups_on_length[length] = {idx: paulis}
        else:
            groups_on_length[length][idx] = paulis

    def least_overlap(indices, existing_indices):
        overlaps = []
        for idx in indices:
            overlap = 0
            for eidx in existing_indices:
                overlap += len(set(idx) & set(eidx))
            overlaps.append(overlap)
        return indices[np.argmin(overlaps)]

    groups.clear()
    for equal_len_groups in groups_on_length.values():
        selected_indices = []
        while equal_len_groups:
            idx = least_overlap(list(equal_len_groups.keys()), selected_indices)
            selected_indices.append(idx)
            groups[idx] = equal_len_groups.pop(idx)

    return groups


################################################################
# BSF Simplification
################################################################

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


# def group_paulis_and_coeffs(paulis: List[str], coeffs: List[float]):
#     """
#     Group Pauli strings by their nontrivial parts.
#
#     E.g.,
#
#         ['XXIII', 'YYIII', 'ZZIII', 'IXXII', 'IYYII', 'IZZII', 'IIXXI', 'IIYYI', 'IIZZI', 'IIIXX', 'IIIYY', 'IIIZZ', 'ZIIII', 'IZIII', 'IIZII', 'IIIZI', 'IIIIZ']
#
#     will be grouped as
#
#          {(0, 1): ['XXIII', 'YYIII', 'ZZIII'],
#           (2, 3): ['IIXXI', 'IIYYI', 'IIZZI'],
#           (3, 4): ['IIIXX', 'IIIYY', 'IIIZZ'],
#           (1, 2): ['IXXII', 'IYYII', 'IZZII'],
#           (0,): ['ZIIII'],
#           (1,): ['IZIII'],
#           (2,): ['IIZII'],
#           (3,): ['IIIZI'],
#           (4,): ['IIIIZ']}
#     """
#     # TODO: is it necessary to use `OrderedDict` instead of `dict`?
#
#     qubits_acted = [tuple(np.where(np.array(list(pauli)) != 'I')[0]) for pauli in paulis]
#     groups = {}
#     for qubits, pauli in zip(qubits_acted, paulis, coeffs):
#         if qubits not in groups:
#             groups[qubits] = [pauli]
#         else:
#             groups[qubits].append(pauli)
#
#     # sort "groups" according to the length of the keys and keys themselves
#     groups = dict(sorted(groups.items(), key=lambda x: (-len(x[0]), x[0])))
#
#     # reorder items to reduce overall length when organizing as circuit
#     groups_on_length = {}  # {length: {idx: [paulis]}}
#     for idx, paulis in groups.items():
#         length = len(idx)
#         if length not in groups_on_length:
#             groups_on_length[length] = {idx: paulis}
#         else:
#             groups_on_length[length][idx] = paulis
#
#     def least_overlap(indices, existing_indices):
#         overlaps = []
#         for idx in indices:
#             overlap = 0
#             for eidx in existing_indices:
#                 overlap += len(set(idx) & set(eidx))
#             overlaps.append(overlap)
#         return indices[np.argmin(overlaps)]
#
#     groups.clear()
#     for equal_len_groups in groups_on_length.values():
#         selected_indices = []
#         while equal_len_groups:
#             idx = least_overlap(list(equal_len_groups.keys()), selected_indices)
#             selected_indices.append(idx)
#             groups[idx] = equal_len_groups.pop(idx)
#
#     return groups


def pauli_rotation_gate_to_pauli_and_coeff(g: Gate, num_qubits: int) -> Tuple[str, float]:
    assert g.name.lower() in gates.PAULI_ROTATION_GATES, 'Not a Pauli rotation gate'
    assert g.name.startswith('R'), 'Not a Pauli rotation gate'
    pauli = 'I' * num_qubits
    for p, tq in zip(g.name[1:], g.tqs):
        pauli = pauli[:tq] + p + pauli[tq + 1:]
    return pauli, g.angle / 2


################################################################
# Ordering
################################################################
def order_blocks(blocks: Circuit) -> Circuit:
    def wire_width(circ):
        return sum([g for g in circ.gates if g.num_qregs > 1])

    def end_empty_layers(circ: Circuit):
        circ = Circuit([g for g in circ if g.num_qregs > 1])
        left_ends = {i: -1 for i in circ.qubits}
        right_end = {i: -1 for i in circ.qubits}
        for num_layer, layer in enumerate(circ.layer()):
            for q in reduce(add, [g.qregs for g in layer]):
                if left_ends[q] < 0:
                    left_ends[q] = num_layer
            if np.all(np.array(list(left_ends.values())) > 0):
                break
        for num_layer, layer in enumerate(circ.inverse().layer()):
            for q in reduce(add, [g.qregs for g in layer]):
                if right_end[q] < 0:
                    right_end[q] = num_layer
            if np.all(np.array(list(right_end.values())) > 0):
                break




        return left_ends, right_end





################################################################
# Other utils
################################################################
def unique_paulis_and_coeffs(paulis: List[str], coeffs: List[float]) -> Tuple[List[str], List[float]]:
    """Remove duplicates in Pauli strings and sum up their coefficients."""
    if len(np.unique(paulis)) == len(paulis):
        return paulis, coeffs
    unique_paulis = []
    unique_coeffs = []
    for pauli, coeff in zip(paulis, coeffs):
        if pauli in unique_paulis:
            idx = unique_paulis.index(pauli)
            unique_coeffs[idx] += coeff
        else:
            unique_paulis.append(pauli)
            unique_coeffs.append(coeff)
    return unique_paulis, unique_coeffs
