from __future__ import annotations

import numpy as np


def _reorder_by_least_overlap(groups: dict[tuple[int, ...], list[str]]) -> dict[tuple[int, ...], list[str]]:
    """Reorder groups so that consecutive groups have minimal qubit overlap.

    Within each length class, greedily select the index tuple whose qubit-set
    has minimum overlap with the union of already-selected tuples (ties:
    original insertion order). O(k² · s) numpy ops per class of k tuples over
    a support of size s.
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
            # Mask out already-selected rows; argmin's smallest-index tie-break
            # preserves insertion order.
            overlap_masked = np.where(alive, overlap, INT_MAX)
            r = int(np.argmin(overlap_masked))
            final[keys[r]] = equal_len_groups[keys[r]]
            use += M[r]
            alive[r] = False
    return final


def _support_keys(paulis: list[str]) -> list[tuple[int, ...]]:
    """Non-identity support (qubit-index tuple) of each Pauli string."""
    keys: list[tuple[int, ...]] = []
    for pauli in paulis:
        key: list[int] = []
        for i, op in enumerate(pauli):
            if op == "I":
                continue
            if op not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli character {op!r} in {pauli!r}")
            key.append(i)
        keys.append(tuple(key))
    return keys


def group_paulis(paulis: list[str]) -> dict[tuple[int, ...], list[str]]:
    """Group Pauli strings by exact match of their nontrivial qubit indices
    (same-support grouping, the DAC'25 unit)."""
    support_keys = _support_keys(paulis)

    group_rows: dict[tuple[int, ...], list[int]] = {}
    for row, key in enumerate(support_keys):
        group_rows.setdefault(key, []).append(row)

    sorted_keys = sorted(group_rows.keys(), key=lambda k: (-len(k), k))
    groups = {k: [paulis[row] for row in group_rows[k]] for k in sorted_keys}

    return _reorder_by_least_overlap(groups)


def group_paulis_and_coeffs(
    paulis: list[str], coeffs: np.ndarray,
) -> dict[tuple[int, ...], tuple[list[str], np.ndarray]]:
    """Group Pauli strings (with coefficients) by their nontrivial parts."""
    grouped_paulis = group_paulis(paulis)

    # Map each string back to its original indices; duplicates are consumed
    # in order so coefficients stay aligned.
    pauli_to_indices: dict[str, list[int]] = {}
    for i, p in enumerate(paulis):
        pauli_to_indices.setdefault(p, []).append(i)
    pauli_to_indices_iter = {p: iter(idxs) for p, idxs in pauli_to_indices.items()}

    return {
        idx: (pls, np.array([coeffs[next(pauli_to_indices_iter[p])] for p in pls]))
        for idx, pls in grouped_paulis.items()
    }
