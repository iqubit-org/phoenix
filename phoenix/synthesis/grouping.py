"""
Pauli string IR grouping.
"""
import numpy as np
from typing import List, Tuple, Dict

from phoenix.basic import gates
from phoenix.basic.gates import Gate

from rich.console import Console

console = Console()


def group_paulis(paulis: List[str]) -> Dict[Tuple[int], List[str]]:
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


def group_paulis_and_coeffs(paulis: List[str], coeffs: List[float]) -> Dict[Tuple[int], Tuple[List[str], np.ndarray]]:
    """Group Pauli strings (with coefficients) by their nontrivial parts."""
    groups = {}
    for idx, pls in group_paulis(paulis).items():
        groups[idx] = pls, np.array([coeffs[paulis.index(p)] for p in pls])
    return groups

def pauli_rotation_gate_to_pauli_and_coeff(g: Gate, num_qubits: int) -> Tuple[str, float]:
    assert g.name.lower() in gates.PAULI_ROTATION_GATES, 'Not a Pauli rotation gate'
    assert g.name.startswith('R'), 'Not a Pauli rotation gate'
    pauli = 'I' * num_qubits
    for p, tq in zip(g.name[1:], g.tqs):
        pauli = pauli[:tq] + p + pauli[tq + 1:]
    return pauli, g.angle / 2
