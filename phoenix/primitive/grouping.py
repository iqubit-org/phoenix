import numpy as np


def group_paulis(paulis: list[str]) -> dict[tuple[int, ...], list[str]]:
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
    nontrivial = []
    for pauli in paulis:
        # Find indices where pauli is not 'I'
        # Note: qiskit Pauli strings are little-endian (qubit 0 is rightmost), 
        indices = tuple(np.where(np.array(list(pauli)) != 'I')[0])
        nontrivial.append(indices)

    groups: dict[tuple[int, ...], list[str]] = {}
    for idx, pauli in zip(nontrivial, paulis):
        if idx not in groups:
            groups[idx] = [pauli]
        else:
            groups[idx].append(pauli)

    # Sort groups by length of keys (descending) and then keys themselves
    groups = dict(sorted(groups.items(), key=lambda x: (-len(x[0]), x[0])))

    # Reorder items to reduce overall length when organizing as circuit
    groups_on_length: dict[int, dict[tuple[int, ...], list[str]]] = {}
    for idx, pls in groups.items():
        length = len(idx)
        if length not in groups_on_length:
            groups_on_length[length] = {idx: pls}
        else:
            groups_on_length[length][idx] = pls

    def least_overlap(indices: list[tuple[int, ...]], existing_indices: list[tuple[int, ...]]) -> tuple[int, ...]:
        overlaps = []
        for idx in indices:
            overlap = 0
            for eidx in existing_indices:
                overlap += len(set(idx) & set(eidx))
            overlaps.append(overlap)
        return indices[np.argmin(overlaps)]

    final_groups = {}
    for equal_len_groups in groups_on_length.values():
        selected_indices = []
        # We need to process keys of equal_len_groups
        keys = list(equal_len_groups.keys())
        while keys:
            idx = least_overlap(keys, selected_indices)
            selected_indices.append(idx)
            final_groups[idx] = equal_len_groups[idx]
            keys.remove(idx)

    return final_groups


def group_paulis_and_coeffs(paulis: list[str], coeffs: np.ndarray) -> dict[tuple[int, ...], tuple[list[str], np.ndarray]]:
    """Group Pauli strings (with coefficients) by their nontrivial parts."""
    groups = {}
    grouped_paulis = group_paulis(paulis)
    
    # We need to map back to coefficients. 
    # Since paulis might contain duplicates in general, we should be careful.
    # However, group_paulis returns lists of strings.
    # We assume the input `paulis` and `coeffs` are aligned.
    
    # Create a mapping from pauli string to list of indices in the original array
    # to handle duplicate strings if necessary.
    pauli_to_indices = {}
    for i, p in enumerate(paulis):
        if p not in pauli_to_indices:
            pauli_to_indices[p] = []
        pauli_to_indices[p].append(i)
        
    # Consume indices
    pauli_to_indices_iter = {p: iter(idxs) for p, idxs in pauli_to_indices.items()}

    for idx, pls in grouped_paulis.items():
        group_coeffs = []
        for p in pls:
            original_idx = next(pauli_to_indices_iter[p])
            group_coeffs.append(coeffs[original_idx])
        groups[idx] = (pls, np.array(group_coeffs))
        
    return groups

