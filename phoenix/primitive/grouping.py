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


def _hamming(p: str, q: str) -> int:
    """Number of qubit positions where Pauli strings ``p`` and ``q`` differ."""
    return sum(1 for a, b in zip(p, q) if a != b)


def _encode_paulis(
    paulis: list[str],
) -> tuple[list[int], list[int], list[int], list[tuple[int, ...]]]:
    """Encode Pauli strings as X/Z bitsets plus non-identity support keys."""
    x_bits: list[int] = []
    z_bits: list[int] = []
    support_bits: list[int] = []
    support_keys: list[tuple[int, ...]] = []

    for pauli in paulis:
        x = 0
        z = 0
        support = 0
        key: list[int] = []
        for i, op in enumerate(pauli):
            if op == "I":
                continue

            bit = 1 << i
            if op == "X":
                x |= bit
            elif op == "Y":
                x |= bit
                z |= bit
            elif op == "Z":
                z |= bit
            else:
                raise ValueError(f"Invalid Pauli character {op!r} in {pauli!r}")

            support |= bit
            key.append(i)

        x_bits.append(x)
        z_bits.append(z)
        support_bits.append(support)
        support_keys.append(tuple(key))

    return x_bits, z_bits, support_bits, support_keys


def _hamming_bits(x_bits: list[int], z_bits: list[int], p: int, q: int) -> int:
    return ((x_bits[p] ^ x_bits[q]) | (z_bits[p] ^ z_bits[q])).bit_count()


def _support_key_from_bits(bits: int) -> tuple[int, ...]:
    indices = []
    while bits:
        low_bit = bits & -bits
        indices.append(low_bit.bit_length() - 1)
        bits ^= low_bit
    return tuple(indices)


def group_paulis(
    paulis: list[str],
    subset: bool = False,
) -> dict[tuple[int, ...], list[str]]:
    """
    Group Pauli strings by their nontrivial parts.

    When ``subset=False`` (default), Pauli strings are grouped by exact match
    of their nontrivial qubit indices.

    When ``subset=True``, a coarser "soft subset" grouping is used based on
    Hamming distance between the actual Pauli strings (not just their qubit
    supports). Exact-match groups are built first and processed in order of
    decreasing weight. A candidate key is absorbed into an existing cluster
    iff some Pauli in the candidate is, on average, Hamming-close to the
    cluster's members. Concretely (**min-mean** aggregation):

    * For each (candidate, cluster) pair, compute
      ``union = set(candidate_key) | cluster.support`` and
      ``score = min_{p in candidate} mean_{q in cluster} Hamming(p, q)`` --
      i.e. pick, among the candidate's Paulis, the one whose *average*
      distance to the whole cluster is smallest.
    * The candidate is eligible for the cluster iff
      ``score <= max(2, len(union) // 4)``.
    * Among eligible clusters, pick the one with the smallest ``score``
      (ties: earliest-inserted).
    * If no cluster is eligible, the candidate seeds a new cluster.

    The *mean* on the cluster side prevents a runaway cascade: a single
    close member of a diverse cluster is no longer enough to pull unrelated
    candidates in. The *min* on the candidate side gives the candidate a
    fair chance by letting its best-aligned representative speak for the
    whole exact-match group (e.g. ``{XXIII, YYIII, ZZIII}`` can join a
    cluster as long as any one of these aligns with the cluster's pattern).

    The merged cluster's key becomes the sorted union of absorbed supports,
    so a cluster's support can grow when it absorbs a "sibling" group whose
    qubit set is not strictly contained.

    Rationale: this rule only merges Pauli strings that *actually look alike*
    on their active qubits, so downstream BSF simplification can share
    Clifford scaffolding. Purely index-based subset grouping would happily
    merge ``IYZIII`` into the ``XXXIII`` group (same support {1, 2}) even
    though the two strings have no common Pauli letters, which pollutes BSF
    and hurts compilation.

    Examples (with the Hamming rule above)::

        group_paulis(['XXXIII', 'IXXIII'], subset=True)
        # -> {(0, 1, 2): ['XXXIII', 'IXXIII']}     # Hamming 1, len(union)=3 -> absorb

        group_paulis(['XXXIII', 'IYZIII'], subset=True)
        # -> {(0, 1, 2): ['XXXIII'], (1, 2): ['IYZIII']}  # Hamming 3 > 1 -> separate

        group_paulis(['XXXIII', 'XXIXII'], subset=True)
        # -> {(0, 1, 2, 3): ['XXXIII', 'XXIXII']}  # Hamming 2, len(union)=4 -> absorb
    """
    x_bits, z_bits, support_bits, support_keys = _encode_paulis(paulis)

    group_rows: dict[tuple[int, ...], list[int]] = {}
    key_to_bits: dict[tuple[int, ...], int] = {}
    for row, (bits, key) in enumerate(zip(support_bits, support_keys)):
        if key not in group_rows:
            group_rows[key] = [row]
            key_to_bits[key] = bits
        else:
            group_rows[key].append(row)

    sorted_keys = sorted(group_rows.keys(), key=lambda k: (-len(k), k))

    if subset:
        # Greedy soft-subset clustering using min-mean Hamming: for each
        # candidate Pauli, measure its mean distance to all cluster members,
        # then take the best (smallest) across candidate Paulis. Larger
        # groups seed clusters first so smaller sibling groups get absorbed
        # rather than starting redundant clusters.
        cluster_supports: list[int] = []
        cluster_rows: list[list[int]] = []
        for key in sorted_keys:
            key_rows = group_rows[key]
            key_bits = key_to_bits[key]

            best_idx = -1
            best_h: float | None = None
            for i, (supp, g_rows) in enumerate(zip(cluster_supports, cluster_rows)):
                union_size = (key_bits | supp).bit_count()
                threshold = max(2, union_size // 4)
                n_g = len(g_rows)
                score = min(
                    sum(_hamming_bits(x_bits, z_bits, p, q) for q in g_rows) / n_g
                    for p in key_rows
                )
                if score <= threshold and (best_h is None or score < best_h):
                    best_h = score
                    best_idx = i

            if best_idx >= 0:
                cluster_supports[best_idx] |= key_bits
                cluster_rows[best_idx].extend(key_rows)
            else:
                cluster_supports.append(key_bits)
                cluster_rows.append(list(key_rows))

        merged: dict[tuple[int, ...], list[str]] = {}
        for supp, rows in zip(cluster_supports, cluster_rows):
            ukey = _support_key_from_bits(supp)
            pls = [paulis[row] for row in rows]
            if ukey in merged:
                merged[ukey].extend(pls)
            else:
                merged[ukey] = pls
        groups = dict(sorted(merged.items(), key=lambda x: (-len(x[0]), x[0])))
    else:
        groups = {k: [paulis[row] for row in group_rows[k]] for k in sorted_keys}

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
    paulis: list[str], coeffs: np.ndarray,
    subset: bool = False,
) -> dict[tuple[int, ...], tuple[list[str], np.ndarray]]:
    """Group Pauli strings (with coefficients) by their nontrivial parts."""
    groups = {}
    grouped_paulis = group_paulis(paulis, subset=subset)

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
