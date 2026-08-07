from __future__ import annotations

from itertools import combinations

import numpy as np

from ..basics import _CLIFFORD_BLOCKS, CLIFFORD_OPTIONS, CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian
from ..primitive.utils import SimplificationStep


def _phi_record(ham: Hamiltonian) -> tuple[int, int, float]:
    """Lexicographic potential Φ = (N_nl, W, cost_bsf) of a tableau.

    ``N_nl`` (number of nonlocal rows) is non-increasing under the simplify
    loop (locals get extracted, Clifford conjugation never maps a non-identity
    row to identity), and ``W`` (total Pauli weight over nonlocal rows) is a
    bounded non-negative integer, so "strictly improve the best-ever record
    within a bounded patience" is a terminating discipline. ``cost_bsf`` only
    breaks ties.
    """
    x = ham.paulis.x
    z = ham.paulis.z
    wo = x | z
    rw = wo.sum(axis=1)
    nl = rw > 1
    return int(nl.sum()), int(rw[nl].sum()), heuristic_bsf_cost(x, z)


def _force_reduce_min_row(
    ham: Hamiltonian,
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]]:
    """Forced row elimination move (Phase D safety valve).

    Picks the minimum-weight nonlocal row ``r`` and applies the Clifford that
    strictly reduces ``w_r``, tie-breaking by the total-weight side effect ΔW
    on all rows. Always succeeds: for any two active qubits of ``r`` its
    2-qubit sub-pattern σ⊗τ is reducible to weight 1 by one of the 9 options
    (verified exhaustively). Consecutive forced moves strictly decrease the
    minimum nonlocal row weight, so a row is extracted after ≤ w_r − 1 moves.
    """
    x = ham.paulis.x.astype(np.int8)
    z = ham.paulis.z.astype(np.int8)
    wo = x | z
    rw = wo.sum(axis=1)
    nl = np.where(rw > 1)[0]
    r = int(nl[np.argmin(rw[nl])])
    support = np.where(wo[r])[0]

    pairs = list(combinations(support.tolist(), 2))
    pairs_arr = np.asarray(pairs, dtype=np.int64)
    q0p, q1p = pairs_arr[:, 0], pairs_arr[:, 1]

    # (P, m, 4) tableau bits at the affected columns
    x_i32 = x.astype(np.int32)
    z_i32 = z.astype(np.int32)
    sub_base = np.stack([x_i32[:, q0p].T, x_i32[:, q1p].T, z_i32[:, q0p].T, z_i32[:, q1p].T], axis=-1)
    wo_at = (sub_base[:, :, 0] | sub_base[:, :, 2]) + (sub_base[:, :, 1] | sub_base[:, :, 3])  # (P, m)

    best_key = None
    best_move = None
    for cliff in CLIFFORD_OPTIONS:
        block = _CLIFFORD_BLOCKS[id(cliff)].astype(np.int32, copy=False)
        new_sub = (sub_base @ block) & 1  # (P, m, 4)
        new_wo_at = (new_sub[:, :, 0] | new_sub[:, :, 2]) + (new_sub[:, :, 1] | new_sub[:, :, 3])
        d_rw = new_wo_at - wo_at  # (P, m)
        w_r_new = rw[r] + d_rw[:, r]  # (P,)
        d_W = d_rw.sum(axis=1)  # (P,) total-weight side effect

        valid = w_r_new < rw[r]
        if not np.any(valid):
            continue
        idx = np.where(valid)[0]
        order = np.lexsort((d_W[idx], w_r_new[idx]))
        pi = int(idx[order[0]])
        key = (int(w_r_new[pi]), int(d_W[pi]))
        if best_key is None or key < best_key:
            best_key = key
            best_move = (cliff, pairs[pi])

    assert best_move is not None, "forced row reduction must always find a move"
    cliff, (q0, q1) = best_move
    return ham.apply_clifford(cliff, q0, q1), cliff, (int(q0), int(q1))


def simplify_hamiltonian(
    ham: Hamiltonian,
    parallel: bool = False,
    patience: int | None = None,
) -> tuple[Hamiltonian, list[SimplificationStep]]:
    """
    Simplify a Hamiltonian (Pauli Tableau) using Clifford gates until weights are <= 2.

    The greedy BSF-cost search is guarded by a stall safety net: a best-ever
    record of Φ = (N_nl, W, cost) is kept, and ``patience`` applied moves
    without improving it — or a search that finds no unvisited candidate at
    all — triggers a forced row-elimination episode
    (:func:`_force_reduce_min_row`) until a row is extracted, after which the
    normal search resumes. This makes the loop terminate unconditionally.

    Returns:
        The simplified Hamiltonian (remaining terms).
        A list of SimplificationStep, representing the operations applied
        and the local terms extracted at each step.
    """
    current_ham = ham
    simp_steps: list[SimplificationStep] = []
    visited = set()

    if patience is None:
        patience = max(16, 2 * len(ham.active_qubits))

    best_record: tuple[int, int, float] | None = None
    stall_count = 0
    forced_mode = False

    while current_ham.total_weight > 2:
        local_ham, nonlocal_ham = current_ham.separate_local_nonlocal()
        visited.add(_tableau_key(nonlocal_ham.paulis.x, nonlocal_ham.paulis.z))

        if forced_mode and np.any(local_ham.with_ops):
            forced_mode = False  # a row was extracted; episode over

        if forced_mode:
            best_ham, best_cliff, qubits = _force_reduce_min_row(nonlocal_ham)
        else:
            if parallel:
                result = search_best_clifford_par(nonlocal_ham, visited)
            else:
                result = search_best_clifford(nonlocal_ham, visited)

            stalled = result is None
            if not stalled:
                best_ham, best_cliff, qubits = result
                record = _phi_record(best_ham)
                if best_record is None or record < best_record:
                    best_record = record
                    stall_count = 0
                else:
                    stall_count += 1
                    stalled = stall_count > patience

            if stalled:
                best_ham, best_cliff, qubits = _force_reduce_min_row(nonlocal_ham)
                forced_mode = True
                stall_count = 0

        simp_steps.append(SimplificationStep(clifford=best_cliff, local_hamiltonian=local_ham, qubits=qubits))

        current_ham = best_ham
        visited.add(_tableau_key(current_ham.paulis.x, current_ham.paulis.z))

    return current_ham, simp_steps


def _apply_cliff_to_tableau(
    x: np.ndarray, z: np.ndarray, block_4x4: np.ndarray, q0: int, q1: int
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


def search_best_clifford(
    ham: Hamiltonian, visited: set[bytes] | None = None
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]] | None:
    """Search for the best Clifford gate to apply.

    Returns ``None`` when every candidate is visited or a no-op (stall signal).
    """
    qubit_pairs = sorted(combinations(ham.active_qubits, 2), key=lambda idx: (idx[0] % 2))

    x = ham.paulis.x.astype(np.int8)
    z = ham.paulis.z.astype(np.int8)

    best_cost = float("inf")
    best_cliff_idx = 0
    best_pair_idx = 0
    for ci, cliff in enumerate(CLIFFORD_OPTIONS):
        block = _CLIFFORD_BLOCKS[id(cliff)]
        for pi, (q0, q1) in enumerate(qubit_pairs):
            new_x, new_z = _apply_cliff_to_tableau(x, z, block, q0, q1)

            if visited is not None and _tableau_key(new_x, new_z) in visited:
                continue
            if np.array_equal(new_x, x) and np.array_equal(new_z, z):
                continue

            cost = heuristic_bsf_cost(new_x, new_z)
            if cost < best_cost:
                best_cost = cost
                best_cliff_idx = ci
                best_pair_idx = pi

    if not np.isfinite(best_cost):
        # Every candidate is visited or a no-op: blindly applying
        # (cliff[0], pair[0]) here would re-enter a visited state and hard-cycle
        # the caller. Signal the stall instead.
        return None

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    return best_ham, best_cliff, best_qubit_pair


def _tableau_key(x: np.ndarray, z: np.ndarray) -> bytes:
    return np.packbits(np.hstack([x, z]).astype(np.uint8), axis=None).tobytes()


def _candidate_tableau_key(
    x: np.ndarray,
    z: np.ndarray,
    q0: int,
    q1: int,
    new_x_q0: np.ndarray,
    new_x_q1: np.ndarray,
    new_z_q0: np.ndarray,
    new_z_q1: np.ndarray,
) -> bytes:
    candidate_x = x.copy()
    candidate_z = z.copy()
    candidate_x[:, q0] = new_x_q0.astype(np.int8, copy=False)
    candidate_x[:, q1] = new_x_q1.astype(np.int8, copy=False)
    candidate_z[:, q0] = new_z_q0.astype(np.int8, copy=False)
    candidate_z[:, q1] = new_z_q1.astype(np.int8, copy=False)
    return _tableau_key(candidate_x, candidate_z)


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
    if num_nl == 0:
        return 0.0

    total_weight = np.bitwise_or.reduce(with_ops[which_nl], axis=0).sum()

    cost = 0.0
    if num_nl > 1:
        nl_ops = with_ops[which_nl].astype(np.int32)
        nl_x = x[which_nl].astype(np.int32)
        nl_z = z[which_nl].astype(np.int32)

        def _pairwise_or_sum(a: np.ndarray) -> float:
            m = a.shape[0]
            row_sums = a.sum(axis=1)
            aat = a @ a.T
            and_upper = (aat.sum() - np.trace(aat)) * 0.5
            return (m - 1) * row_sums.sum() - and_upper

        cost += _pairwise_or_sum(nl_ops)
        cost += _pairwise_or_sum(nl_x) * 0.5
        cost += _pairwise_or_sum(nl_z) * 0.5

    cost += total_weight * num_nl**2
    return cost


def search_best_clifford_par(
    ham: Hamiltonian,
    visited: set[bytes] | None = None,
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]] | None:
    """Compressed-representation vectorized search.

    Returns ``None`` when every candidate is visited or a no-op (stall signal).

    Never materializes the full (C, m, n) tableau tensor (which blows up to
    GB-scale for wide groups like condensed-matter parity encoding). Instead,
    exploits the fact that a 2-qubit Clifford only modifies 2 columns of x
    and 2 columns of z per candidate, decomposes the cost per column, and
    reduces all per-row work to lookups over an 80-state joint code
    (16 local tableau codes x 5 row-weight classes) histogrammed per pair.

    Memory: O(P·m + P·n) rather than O(9·P·m·n).

    Cost-formula decomposition (per candidate p, with modified cols q0, q1):
        For a 0/1 matrix M of shape (m, n),
            S = Σ_{i<j, both nl} |M_i OR M_j|
              = n · C(num_nl, 2) − Σ_k f(s_nl_k)
        where s_nl_k = Σ_i nl_mask_i · M_ik  and  f(s) = C(num_nl − s, 2).

        Σ_k f(s_nl_k) decomposes as:
            [Σ_k f(s_nl_k(ORIG))]
            − f(s_nl at col q0, ORIG) − f(s_nl at col q1, ORIG)
            + f(s_nl at col q0, NEW)  + f(s_nl at col q1, NEW).

    The per-candidate nonlocal-restricted column sums collapse to fixed
    (n,) lookups: at any column outside {q0, q1} the sum provably equals the
    original nonlocal column sum S_k for every candidate (rows can only
    enter/leave the nonlocal set when their support lies inside the pair),
    so Σ_k f(·) is evaluated from a per-ν table over the few distinct
    num_nl values.

    Tie-break matches the sequential implementation.
    """
    qubit_pairs = sorted(combinations(ham.active_qubits, 2), key=lambda idx: (idx[0] % 2))
    P = len(qubit_pairs)
    if P == 0:
        raise ValueError("search_best_clifford_par: no valid qubit pairs")

    x = ham.paulis.x.astype(np.int8)
    z = ham.paulis.z.astype(np.int8)
    m, n = x.shape

    pairs_arr = np.asarray(qubit_pairs, dtype=np.int64)  # (P, 2)
    q0p = pairs_arr[:, 0]
    q1p = pairs_arr[:, 1]

    row_weights = (x | z).sum(axis=1)
    if int(row_weights.min()) > 3:
        return _search_best_clifford_par_nonlocal_stable(ham, qubit_pairs, q0p, q1p, x, z, visited)

    # ---- Precomputed originals ----
    wo = (x | z).astype(np.int32)  # (m, n)
    rw_orig = wo.sum(axis=1)  # (m,)

    # Column sums over the *originally nonlocal* rows. Key invariant: for any
    # candidate (clifford, pair), the nl-restricted column sum at any column
    # k ∉ {q0, q1} equals S_k exactly — a row can enter/leave the nonlocal set
    # only if its support lies inside the pair (the symplectic block is
    # invertible and outside columns are untouched), and such rows contribute
    # 0 to every outside column. Hence the per-candidate (P, n) column-sum
    # recomputation collapses to lookups into these fixed (n,) arrays.
    nl_rows0 = rw_orig >= 2  # (m,)
    S_wo = wo[nl_rows0].sum(axis=0).astype(np.int64)  # (n,)
    S_x = x[nl_rows0].astype(np.int64).sum(axis=0)
    S_z = z[nl_rows0].astype(np.int64).sum(axis=0)
    NZ_wo_total = int((S_wo > 0).sum())

    # ---- Compact per-(pair, row) encoding --------------------------------
    # A 2q Clifford on (q0, q1) touches a row only through its 4 tableau bits
    # (x_q0, x_q1, z_q0, z_q1) =: code in [0, 16). Every per-row quantity the
    # cost needs (new bits, weight delta, nonlocal status) is a pure function
    # of (code, row weight), and the weight only matters through
    # wclass = min(rw, 4) since a 2q Clifford changes a row weight by at most
    # ±2. Joint code jc = code + 16*wclass in [0, 80). A per-pair histogram
    # over the 80 joint codes therefore replaces all O(P·m) per-Clifford
    # reductions with (P, 80) @ (80,) lookups.
    cq = x.astype(np.uint8) | (z.astype(np.uint8) << 1)  # (m, n) per-qubit 2-bit nibble
    code = cq[:, q0p].T | (cq[:, q1p].T << 2)  # (P, m) uint8
    wclass = np.minimum(rw_orig, 4).astype(np.uint8)  # (m,)
    jc = code + (wclass << 4)[None, :]  # (P, m) uint8, in [0, 80)

    offs = (np.arange(P, dtype=np.int64) * 80)[:, None]
    hist = np.bincount((jc.astype(np.int64) + offs).ravel(), minlength=80 * P).reshape(P, 80)

    # Bit decomposition of the 16 codes (original values at q0, q1):
    # bit0 = x_q0, bit1 = z_q0, bit2 = x_q1, bit3 = z_q1
    codes16 = np.arange(16, dtype=np.int64)
    bit_x0 = codes16 & 1
    bit_z0 = (codes16 >> 1) & 1
    bit_x1 = (codes16 >> 2) & 1
    bit_z1 = (codes16 >> 3) & 1
    wo_q0_16 = bit_x0 | bit_z0
    wo_q1_16 = bit_x1 | bit_z1
    w_rep = np.arange(5)[:, None]  # representative row weight per class; 4 = ">=4"

    def _tile80(v16: np.ndarray) -> np.ndarray:
        """Replicate a (16,) per-code table across the 5 weight classes."""
        return np.tile(v16, 5)

    # Per-pair gathers of the fixed nonlocal column sums (Clifford-independent)
    S_wo_q0 = S_wo[q0p]  # (P,)
    S_wo_q1 = S_wo[q1p]
    S_x_q0 = S_x[q0p]
    S_x_q1 = S_x[q1p]
    S_z_q0 = S_z[q0p]
    S_z_q1 = S_z[q1p]
    NZ_wo_outside = (
        NZ_wo_total - (S_wo_q0 > 0).astype(np.int64) - (S_wo_q1 > 0).astype(np.int64)
    )  # (P,)

    # Running global-best across 9 Cliffords
    best_cost = np.inf
    best_cliff_idx = 0
    best_pair_idx = 0

    def _f_diff(nnl: np.ndarray, s: np.ndarray) -> np.ndarray:
        """f(s) = C(num_nl - s, 2) = (num_nl - s)(num_nl - s - 1) / 2, integer."""
        d = nnl - s
        return d * (d - 1) // 2

    def _g_table(num_nl: np.ndarray, S: np.ndarray) -> np.ndarray:
        """g(ν)[p] = Σ_k f(ν_p − S_k), evaluated via the few distinct ν values."""
        uniq, inv = np.unique(num_nl, return_inverse=True)
        d = uniq[:, None] - S[None, :]  # (u, n)
        return (d * (d - 1) // 2).sum(axis=1)[inv]  # (P,)

    for ci, cliff in enumerate(CLIFFORD_OPTIONS):
        block = _CLIFFORD_BLOCKS[id(cliff)].astype(np.int64, copy=False)  # (4, 4)

        # Apply the symplectic block to each of the 16 codes (16x4 @ 4x4)
        vecs16 = np.stack([bit_x0, bit_x1, bit_z0, bit_z1], axis=1)  # (16, 4)
        new16 = (vecs16 @ block) & 1  # (16, 4)
        n_x0_16, n_x1_16, n_z0_16, n_z1_16 = new16.T  # each (16,)
        n_wo0_16 = n_x0_16 | n_z0_16
        n_wo1_16 = n_x1_16 | n_z1_16
        d_rw_16 = (n_wo0_16 + n_wo1_16) - (wo_q0_16 + wo_q1_16)  # (16,)
        changed_16 = (n_x0_16 != bit_x0) | (n_x1_16 != bit_x1) | (n_z0_16 != bit_z0) | (n_z1_16 != bit_z1)

        # (80,) lookup tables over the joint code (weight class major, code minor)
        nl80 = ((w_rep + d_rw_16[None, :]) > 1).astype(np.int64).ravel()  # row nonlocal after
        same_count = hist @ _tile80(changed_16.astype(np.int64))  # (P,)
        same_mask = same_count == 0

        num_nl = hist @ nl80  # (P,)

        # NEW s values at the affected columns — per candidate.
        s_n_wo_q0 = hist @ (nl80 * _tile80(n_wo0_16))
        s_n_wo_q1 = hist @ (nl80 * _tile80(n_wo1_16))
        s_n_x_q0 = hist @ (nl80 * _tile80(n_x0_16))
        s_n_x_q1 = hist @ (nl80 * _tile80(n_x1_16))
        s_n_z_q0 = hist @ (nl80 * _tile80(n_z0_16))
        s_n_z_q1 = hist @ (nl80 * _tile80(n_z1_16))

        # Σ_k f(ν − s_k) over all columns: outside columns contribute exactly
        # f(ν − S_k) (see the S_* invariant above); the two pair columns are
        # swapped from their (algebraic) f(ν − S_q) share to the candidate's
        # actual nl-restricted NEW values f(ν − s_n_q).
        sum_f_wo = (
            _g_table(num_nl, S_wo)
            - _f_diff(num_nl, S_wo_q0)
            - _f_diff(num_nl, S_wo_q1)
            + _f_diff(num_nl, s_n_wo_q0)
            + _f_diff(num_nl, s_n_wo_q1)
        )
        sum_f_x = (
            _g_table(num_nl, S_x)
            - _f_diff(num_nl, S_x_q0)
            - _f_diff(num_nl, S_x_q1)
            + _f_diff(num_nl, s_n_x_q0)
            + _f_diff(num_nl, s_n_x_q1)
        )
        sum_f_z = (
            _g_table(num_nl, S_z)
            - _f_diff(num_nl, S_z_q0)
            - _f_diff(num_nl, S_z_q1)
            + _f_diff(num_nl, s_n_z_q0)
            + _f_diff(num_nl, s_n_z_q1)
        )

        # pair_or_X[p] = n · C(num_nl, 2) − sum_f_X[p]
        n_C2 = n * num_nl * (num_nl - 1) // 2  # (P,)
        pair_or_wo = n_C2 - sum_f_wo
        pair_or_x = n_C2 - sum_f_x
        pair_or_z = n_C2 - sum_f_z

        # total_weight[p]: outside columns keep their S_k > 0 status; the two
        # pair columns use the candidate's nl-restricted NEW values.
        total_weight = (
            NZ_wo_outside
            + (s_n_wo_q0 > 0).astype(np.int64)
            + (s_n_wo_q1 > 0).astype(np.int64)
        )

        costs = (
            pair_or_wo.astype(np.float64)
            + 0.5 * pair_or_x.astype(np.float64)
            + 0.5 * pair_or_z.astype(np.float64)
            + total_weight.astype(np.float64) * num_nl.astype(np.float64) ** 2
        )

        # Mark same-as-input candidates as invalid.
        costs = np.where(same_mask, np.inf, costs)

        # --- Reduce within this Clifford, then fold into global best ------
        # Visited check is done lazily: only the running argmin candidate is
        # materialized and tested, instead of reconstructing all P candidate
        # tableaus (O(P·m·n) memory traffic per Clifford).
        while True:
            pi_best = int(np.argmin(costs))
            c_best = float(costs[pi_best])
            if not visited or not np.isfinite(c_best):
                break
            crow = code[pi_best]  # (m,) uint8 — winner's per-row codes
            key = _candidate_tableau_key(
                x,
                z,
                int(q0p[pi_best]),
                int(q1p[pi_best]),
                n_x0_16[crow],
                n_x1_16[crow],
                n_z0_16[crow],
                n_z1_16[crow],
            )
            if key not in visited:
                break
            costs[pi_best] = np.inf

        if c_best < best_cost:
            best_cost = c_best
            best_cliff_idx = ci
            best_pair_idx = pi_best

    if not np.isfinite(best_cost):
        return None  # all candidates visited or no-op — signal stall (see search_best_clifford)

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    return best_ham, best_cliff, best_qubit_pair


def _search_best_clifford_par_nonlocal_stable(
    ham: Hamiltonian,
    qubit_pairs: list,
    q0p: np.ndarray,
    q1p: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    visited: set[bytes] | None,
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]] | None:
    """Specialized search when all rows are guaranteed to stay nonlocal.

    A 2-qubit Clifford can remove support from at most two columns in any row.
    If every row starts with weight > 3, every candidate keeps every row
    nonlocal.  The heuristic's nonlocal row set is then fixed, so only the two
    touched columns can change the pairwise-OR and total-weight terms.
    """
    m = x.shape[0]

    wo = (x | z).astype(np.int32)
    x_i32 = x.astype(np.int32, copy=False)
    z_i32 = z.astype(np.int32, copy=False)

    x_at_q0 = x_i32[:, q0p].T
    x_at_q1 = x_i32[:, q1p].T
    z_at_q0 = z_i32[:, q0p].T
    z_at_q1 = z_i32[:, q1p].T
    sub_base = np.stack([x_at_q0, x_at_q1, z_at_q0, z_at_q1], axis=-1)

    def _pair_or_col_contrib(cols: np.ndarray) -> np.ndarray:
        zeros = m - cols.sum(axis=-1)
        return m * (m - 1) // 2 - zeros * (zeros - 1) // 2

    wo_col_contrib = _pair_or_col_contrib(wo.T)
    x_col_contrib = _pair_or_col_contrib(x_i32.T)
    z_col_contrib = _pair_or_col_contrib(z_i32.T)
    wo_col_nonzero = wo.sum(axis=0) > 0

    base_pair_wo = int(wo_col_contrib.sum())
    base_pair_x = int(x_col_contrib.sum())
    base_pair_z = int(z_col_contrib.sum())
    base_total_weight = int(wo_col_nonzero.sum())
    num_nl_sq = float(m * m)

    best_cost = np.inf
    best_cliff_idx = 0
    best_pair_idx = 0

    for ci, cliff in enumerate(CLIFFORD_OPTIONS):
        block = _CLIFFORD_BLOCKS[id(cliff)].astype(np.int32, copy=False)
        new_sub = (sub_base @ block) & 1

        new_x_q0 = new_sub[:, :, 0]
        new_x_q1 = new_sub[:, :, 1]
        new_z_q0 = new_sub[:, :, 2]
        new_z_q1 = new_sub[:, :, 3]
        new_wo_q0 = new_x_q0 | new_z_q0
        new_wo_q1 = new_x_q1 | new_z_q1

        pair_or_wo = (
            base_pair_wo
            - wo_col_contrib[q0p]
            - wo_col_contrib[q1p]
            + _pair_or_col_contrib(new_wo_q0)
            + _pair_or_col_contrib(new_wo_q1)
        )
        pair_or_x = (
            base_pair_x
            - x_col_contrib[q0p]
            - x_col_contrib[q1p]
            + _pair_or_col_contrib(new_x_q0)
            + _pair_or_col_contrib(new_x_q1)
        )
        pair_or_z = (
            base_pair_z
            - z_col_contrib[q0p]
            - z_col_contrib[q1p]
            + _pair_or_col_contrib(new_z_q0)
            + _pair_or_col_contrib(new_z_q1)
        )
        total_weight = (
            base_total_weight
            - wo_col_nonzero[q0p].astype(np.int64)
            - wo_col_nonzero[q1p].astype(np.int64)
            + (new_wo_q0.sum(axis=1) > 0).astype(np.int64)
            + (new_wo_q1.sum(axis=1) > 0).astype(np.int64)
        )

        costs = (
            pair_or_wo.astype(np.float64)
            + 0.5 * pair_or_x.astype(np.float64)
            + 0.5 * pair_or_z.astype(np.float64)
            + total_weight.astype(np.float64) * num_nl_sq
        )

        same_mask = np.all(new_sub == sub_base, axis=(1, 2))
        costs = np.where(same_mask, np.inf, costs)

        while True:
            pi_best = int(np.argmin(costs))
            c_best = float(costs[pi_best])
            if not visited or not np.isfinite(c_best):
                break

            q0 = int(q0p[pi_best])
            q1 = int(q1p[pi_best])
            key = _candidate_tableau_key(
                x,
                z,
                q0,
                q1,
                new_x_q0[pi_best],
                new_x_q1[pi_best],
                new_z_q0[pi_best],
                new_z_q1[pi_best],
            )
            if key not in visited:
                break
            costs[pi_best] = np.inf

        if c_best < best_cost:
            best_cost = c_best
            best_cliff_idx = ci
            best_pair_idx = pi_best

    if not np.isfinite(best_cost):
        return None  # all candidates visited or no-op — signal stall (see search_best_clifford)

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    return best_ham, best_cliff, best_qubit_pair
