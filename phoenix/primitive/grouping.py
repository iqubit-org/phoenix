from __future__ import annotations

import numpy as np


def _reorder_by_least_overlap(groups: dict[tuple[int, ...], list[str]]) -> dict[tuple[int, ...], list[str]]:
    """Reorder groups so that consecutive groups have minimal qubit overlap.

    Within each length class, greedily select the index tuple whose qubit-set
    has minimum overlap with the union of already-selected tuples. Ties are
    broken by original insertion order (same as the previous implementation).

    Complexity: for a class of k equal-length tuples over a qubit support of
    size s (s <= q), this runs in O(k² · s) numpy ops (single matmul per
    iteration instead of per-pair set intersection), vs the prior
    O(k³ · q) Python-level set operations.
    """
    groups_on_length: dict[int, dict[tuple[int, ...], list[str]]] = {}
    for idx, pls in groups.items():
        groups_on_length.setdefault(len(idx), {})[idx] = pls

    final: dict[tuple[int, ...], list[str]] = {}
    INT_MAX = np.iinfo(np.int64).max

    for equal_len_groups in groups_on_length.values():
        keys = list(equal_len_groups.keys())
        k = len(keys)
        if k == 1:
            final[keys[0]] = equal_len_groups[keys[0]]
            continue

        # Compact qubit support for this length class (union of used qubits)
        support = sorted({q for idx in keys for q in idx})
        q2col = {q: c for c, q in enumerate(support)}
        s = len(support)

        # Membership matrix M[r, c] = 1 iff qubit support[c] is in keys[r]
        M = np.zeros((k, s), dtype=np.int32)
        for r, idx in enumerate(keys):
            for q in idx:
                M[r, q2col[q]] = 1

        # use[c] = # of already-selected indices that touch qubit support[c]
        use = np.zeros(s, dtype=np.int64)
        alive = np.ones(k, dtype=bool)

        for _ in range(k):
            overlap = M @ use  # shape (k,)
            # Mask out already-selected rows so argmin only sees the alive ones.
            # np.argmin returns the smallest index on ties, matching the prior
            # list-based `indices[np.argmin(overlaps)]` tie-break exactly.
            overlap_masked = np.where(alive, overlap, INT_MAX)
            r = int(np.argmin(overlap_masked))
            final[keys[r]] = equal_len_groups[keys[r]]
            use += M[r]
            alive[r] = False
    return final


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
        indices = tuple(np.where(np.array(list(pauli)) != "I")[0])
        nontrivial.append(indices)

    groups: dict[tuple[int, ...], list[str]] = {}
    for idx, pauli in zip(nontrivial, paulis):
        if idx not in groups:
            groups[idx] = [pauli]
        else:
            groups[idx].append(pauli)

    groups = dict(sorted(groups.items(), key=lambda x: (-len(x[0]), x[0])))
    return _reorder_by_least_overlap(groups)


# def group_paulis_coarse(paulis: list[str]) -> dict[tuple[int, ...], list[str]]:
#     """Group Pauli strings with coarse multi-qubit merging.

#     - Weight-1 (single-qubit) Paulis: grouped by their nontrivial qubit index.
#     - Weight-2 (two-qubit) Paulis: grouped by their nontrivial qubit pair.
#     - Weight-3+ (multi-qubit) Paulis: **all merged into one group** keyed by the
#       union of their nontrivial qubit indices.

#     This avoids over-fragmentation of high-weight terms.
#     """
#     multi_group: list[str] = []
#     multi_qubits: set[int] = set()
#     low_weight_groups: dict[tuple[int, ...], list[str]] = {}

#     for pauli in paulis:
#         indices = tuple(np.where(np.array(list(pauli)) != 'I')[0])
#         if len(indices) >= 3:
#             multi_group.append(pauli)
#             multi_qubits.update(indices)
#         else:
#             low_weight_groups.setdefault(indices, []).append(pauli)

#     merged: dict[tuple[int, ...], list[str]] = {}
#     if multi_group:
#         merged[tuple(sorted(multi_qubits))] = multi_group
#     merged.update(dict(sorted(low_weight_groups.items(), key=lambda x: (-len(x[0]), x[0]))))
#     return _reorder_by_least_overlap(merged)


# def group_paulis_and_coeffs_coarse(paulis: list[str], coeffs: np.ndarray
#                                    ) -> dict[tuple[int, ...], tuple[list[str], np.ndarray]]:
#     """Like ``group_paulis_and_coeffs`` but using coarse multi-qubit merging."""
#     grouped = group_paulis_coarse(paulis)

#     pauli_to_indices: dict[str, list[int]] = {}
#     for i, p in enumerate(paulis):
#         pauli_to_indices.setdefault(p, []).append(i)
#     pauli_to_indices_iter = {p: iter(idxs) for p, idxs in pauli_to_indices.items()}

#     result = {}
#     for key, pls in grouped.items():
#         group_coeffs = [coeffs[next(pauli_to_indices_iter[p])] for p in pls]
#         result[key] = (pls, np.array(group_coeffs))
#     return result


def group_paulis_and_coeffs(
    paulis: list[str], coeffs: np.ndarray
) -> dict[tuple[int, ...], tuple[list[str], np.ndarray]]:
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
