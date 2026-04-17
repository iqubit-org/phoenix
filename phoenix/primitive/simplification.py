from __future__ import annotations

from itertools import combinations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford

from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian
from ..primitive.utils import SimplificationStep

CLIFFORD_OPTIONS = [
    CNOTEquivCliffordGate("X", "X"),
    CNOTEquivCliffordGate("Y", "Y"),
    CNOTEquivCliffordGate("Z", "Z"),
    CNOTEquivCliffordGate("X", "Y"),
    CNOTEquivCliffordGate("Y", "X"),
    CNOTEquivCliffordGate("X", "Z"),
    CNOTEquivCliffordGate("Z", "X"),
    CNOTEquivCliffordGate("Y", "Z"),
    CNOTEquivCliffordGate("Z", "Y"),
]

# Precompute the 4x4 symplectic block for each Clifford type (computed once at import time).
# Each CNOT-equiv Clifford on qubits (q0, q1) only affects the 4x4 sub-block
# at indices {q0, q1, q0+n, q1+n} of the full 2n×2n symplectic matrix.
_CLIFFORD_BLOCKS: dict[int, np.ndarray] = {}
for _cliff in CLIFFORD_OPTIONS:
    _qc2 = QuantumCircuit(2)
    _qc2.append(_cliff, [0, 1])
    _CLIFFORD_BLOCKS[id(_cliff)] = Clifford(_qc2).symplectic_matrix.astype(np.int8)


def simplify_hamiltonian(ham: Hamiltonian, parallel: bool = False) -> tuple[Hamiltonian, list[SimplificationStep]]:
    """
    Simplify a Hamiltonian (Pauli Tableau) using Clifford gates until weights are <= 2.

    Returns:
        The simplified Hamiltonian (remaining terms).
        A list of SimplificationStep, representing the operations applied
        and the local terms extracted at each step.
    """
    current_ham = ham
    simp_steps: list[SimplificationStep] = []
    visited = set()

    # last_layer = np.zeros(current_ham.num_qubits, dtype=np.int64)

    while current_ham.total_weight > 2:
        local_ham, nonlocal_ham = current_ham.separate_local_nonlocal()
        visited.add(_tableau_key(nonlocal_ham.paulis.x, nonlocal_ham.paulis.z))

        if parallel:
            best_ham, best_cliff, qubits = search_best_clifford_par(nonlocal_ham, visited)
        else:
            best_ham, best_cliff, qubits = search_best_clifford(nonlocal_ham, visited)

        simp_steps.append(SimplificationStep(clifford=best_cliff, local_hamiltonian=local_ham, qubits=qubits))

        # q0, q1 = qubits
        # new_layer = max(int(last_layer[q0]), int(last_layer[q1])) + 1
        # last_layer[q0] = new_layer
        # last_layer[q1] = new_layer

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
    ham: Hamiltonian, visited: set[bytes] = None
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]]:
    """Search for the best Clifford gate to apply."""
    # n = ham.num_qubits
    # if last_layer is None:
    #     last_layer = np.zeros(n, dtype=np.int64)
    # if depth_weight is None:
    #     depth_weight = np.sqrt(n)
    # current_depth = last_layer.max() if n > 0 else 0

    qubit_pairs = sorted(combinations(ham.active_qubits, 2), key=lambda idx: (idx[0] % 2))
    qubit_pairs = [pair for pair in qubit_pairs]

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
            # new_layer   = max(last_layer[q0], last_layer[q1]) + 1
            # delta_depth = max(0, new_layer - current_depth)
            # cost += delta_depth * depth_weight

            if cost < best_cost:
                best_cost = cost
                best_cliff_idx = ci
                best_pair_idx = pi

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    # print(f"Applied Clifford: {best_cliff.name} on qubits {best_qubit_pair}, total_weight: {ham.total_weight} → {best_ham.total_weight}, cost: {best_cost:.2f}")
    return best_ham, best_cliff, best_qubit_pair


def _tableau_key(x: np.ndarray, z: np.ndarray) -> bytes:
    return np.packbits(np.hstack([x, z]).astype(np.uint8), axis=None).tobytes()


def _heuristic_bsf_cost(x: np.ndarray, z: np.ndarray) -> float:
    r"""
    Original heuristic cost for a Pauli Tableau, the smaller the simpler.

    .. math::
        \mathrm{cost}_{\mathrm{bsf}} := \mathrm{total\_weight} * n_{\mathrm{nonlocal}}^2
        + \sum_{\langle i,j \rangle} \lVert r_x^{(i)} \lor r_z^{(i)} \lor r_x^{(j)} \lor r_z^{(j)} \rVert
        + \frac{1}{2} \sum_{\langle i,j \rangle} (\lVert r_x^{(i)} \lor r_x^{(j)} \rVert + \lVert r_z^{(i)} \lor r_z^{(j)} \rVert)
    """
    with_ops = np.logical_or(x, z)
    row_weights = with_ops.sum(axis=1)
    which_nonlocal_paulis = np.where(row_weights > 1)[0]
    num_nonlocal_paulis = which_nonlocal_paulis.size

    if not np.any(with_ops):
        total_weight = 0
    elif not num_nonlocal_paulis:
        total_weight = 1
    else:
        total_weight = np.bitwise_or.reduce(with_ops[which_nonlocal_paulis], axis=0).sum()

    cost = 0.0
    if num_nonlocal_paulis > 1:
        row_combs = np.array(list(combinations(which_nonlocal_paulis, 2))).T
        cost += np.bitwise_or(with_ops[row_combs[0]], with_ops[row_combs[1]]).sum()
        cost += np.bitwise_or(x[row_combs[0]], x[row_combs[1]]).sum() * 0.5
        cost += np.bitwise_or(z[row_combs[0]], z[row_combs[1]]).sum() * 0.5

    cost += total_weight * num_nonlocal_paulis**2
    return cost


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

    if not np.any(with_ops):
        return 0.0
    elif not num_nl:
        return 0.0

    total_weight = np.bitwise_or.reduce(with_ops[which_nl], axis=0).sum()

    cost = 0.0
    if num_nl > 1:
        nl_ops = with_ops[which_nl].astype(np.int32)
        nl_x = x[which_nl].astype(np.int32)
        nl_z = z[which_nl].astype(np.int32)

        # sum_{i<j} |a_i OR a_j| = (m-1)*sum(|a_i|) - sum-upper-tri(A @ A.T)
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
    col_chunk: int = 64,
    visited_chunk: int = 256,
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]]:
    """Compressed-representation vectorized search.

    Never materializes the full (C, m, n) tableau tensor (which blows up to
    GB-scale for wide groups like condensed-matter parity encoding). Instead,
    exploits the fact that a 2-qubit Clifford only modifies 2 columns of x
    and 2 columns of z per candidate, decomposes the cost per column, and
    streams the unchanged-column contributions in small K-wide chunks.

    Memory: O(P·m + P·K + chunk·m·n) rather than O(9·P·m·n).

    Cost-formula decomposition (per candidate p, with modified cols q0, q1):
        For a 0/1 matrix M of shape (m, n),
            S = Σ_{i<j, both nl} |M_i OR M_j|
              = n · C(num_nl, 2) − Σ_k f(s_nl_k)
        where s_nl_k = Σ_i nl_mask_i · M_ik  and  f(s) = C(num_nl − s, 2).

        Σ_k f(s_nl_k) decomposes as:
            [Σ_k f(s_nl_k(ORIG))]
            − f(s_nl at col q0, ORIG) − f(s_nl at col q1, ORIG)
            + f(s_nl at col q0, NEW)  + f(s_nl at col q1, NEW).

    Tie-break matches the sequential / prior par implementation.
    """
    qubit_pairs = sorted(combinations(ham.active_qubits, 2), key=lambda idx: (idx[0] % 2))
    qubit_pairs = list(qubit_pairs)
    P = len(qubit_pairs)
    if P == 0:
        raise ValueError("search_best_clifford_par: no valid qubit pairs")

    x = ham.paulis.x.astype(np.int8)
    z = ham.paulis.z.astype(np.int8)
    m, n = x.shape

    pairs_arr = np.asarray(qubit_pairs, dtype=np.int64)  # (P, 2)
    q0p = pairs_arr[:, 0]
    q1p = pairs_arr[:, 1]

    # ---- Fast path for m=1 --------------------------------------------------
    # With a single Pauli row (num_nl ∈ {0, 1}), the pairwise-OR sums vanish:
    #     f(s) = C(num_nl − s, 2) is 0 for num_nl ∈ {0, 1}, any s.
    # So cost collapses to `total_weight · num_nl² = rw_new · 1[rw_new > 1]`.
    # Skips the entire column-streaming loop; per-Clifford O(P), not O(P·n).
    if m == 1:
        return _search_best_clifford_par_m1(ham, qubit_pairs, q0p, q1p, x, z, n, visited, visited_chunk)

    # ---- Precomputed originals ----
    wo = (x | z).astype(np.int32)  # (m, n)
    x_i32 = x.astype(np.int32, copy=False)
    z_i32 = z.astype(np.int32, copy=False)
    rw_orig = wo.sum(axis=1)  # (m,)

    # Original values at affected columns, shape (P, m)
    wo_at_q0 = wo[:, q0p].T
    wo_at_q1 = wo[:, q1p].T
    x_at_q0 = x_i32[:, q0p].T
    x_at_q1 = x_i32[:, q1p].T
    z_at_q0 = z_i32[:, q0p].T
    z_at_q1 = z_i32[:, q1p].T

    # (P, m, 4) slab used as matmul input against each Clifford block
    sub_base = np.stack([x_at_q0, x_at_q1, z_at_q0, z_at_q1], axis=-1)  # (P, m, 4) int32

    # Running global-best across 9 Cliffords
    best_cost = np.inf
    best_cliff_idx = 0
    best_pair_idx = 0

    # Preallocate small (P,) scratchpads reused per Clifford
    # (allocated in loop for clarity)

    def _f_diff(nnl: np.ndarray, s: np.ndarray) -> np.ndarray:
        """f(s) = C(num_nl - s, 2) = (num_nl - s)(num_nl - s - 1) / 2, integer."""
        d = nnl - s
        return d * (d - 1) // 2

    for ci, cliff in enumerate(CLIFFORD_OPTIONS):
        block = _CLIFFORD_BLOCKS[id(cliff)].astype(np.int32, copy=False)  # (4, 4)
        new_sub = (sub_base @ block) & 1  # (P, m, 4) int32

        new_x_q0 = new_sub[:, :, 0]  # (P, m)
        new_x_q1 = new_sub[:, :, 1]
        new_z_q0 = new_sub[:, :, 2]
        new_z_q1 = new_sub[:, :, 3]
        new_wo_q0 = new_x_q0 | new_z_q0  # (P, m)
        new_wo_q1 = new_x_q1 | new_z_q1

        # Same-as-input mask (skip no-op Cliffords)
        same_mask = np.all(new_sub == sub_base, axis=(1, 2))  # (P,)

        # Candidate row weights and nonlocal mask
        d_rw = (new_wo_q0 - wo_at_q0) + (new_wo_q1 - wo_at_q1)  # (P, m) signed
        rw_new = rw_orig[None, :] + d_rw  # (P, m)
        nl_mask = (rw_new > 1).astype(np.int32)  # (P, m)
        num_nl = nl_mask.sum(axis=1).astype(np.int64)  # (P,)

        # Streaming pass over all n columns of original (wo, x, z) in chunks.
        # Produces:
        #   sum_f_wo_orig[p] = Σ_k f(s_wo_nl_orig[p, k])      (for pairwise_or on wo)
        #   sum_f_x_orig[p], sum_f_z_orig[p]                   (same for x, z)
        #   sum_nz_wo_orig[p] = # cols k with s_wo_orig[p, k] > 0  (for total_weight)
        sum_f_wo_orig = np.zeros(P, dtype=np.int64)
        sum_f_x_orig = np.zeros(P, dtype=np.int64)
        sum_f_z_orig = np.zeros(P, dtype=np.int64)
        sum_nz_wo_orig = np.zeros(P, dtype=np.int64)

        nnl_col = num_nl[:, None]  # (P, 1) int64
        for k_start in range(0, n, col_chunk):
            k_end = min(k_start + col_chunk, n)
            wo_chunk = wo[:, k_start:k_end]  # (m, K) int32
            x_chunk = x_i32[:, k_start:k_end]
            z_chunk = z_i32[:, k_start:k_end]

            s_wo = (nl_mask @ wo_chunk).astype(np.int64, copy=False)  # (P, K)
            s_x = (nl_mask @ x_chunk).astype(np.int64, copy=False)
            s_z = (nl_mask @ z_chunk).astype(np.int64, copy=False)

            d_wo = nnl_col - s_wo
            d_x = nnl_col - s_x
            d_z = nnl_col - s_z
            sum_f_wo_orig += (d_wo * (d_wo - 1) // 2).sum(axis=1)
            sum_f_x_orig += (d_x * (d_x - 1) // 2).sum(axis=1)
            sum_f_z_orig += (d_z * (d_z - 1) // 2).sum(axis=1)
            sum_nz_wo_orig += (s_wo > 0).sum(axis=1)

        # ORIGINAL s values at the affected columns — per candidate.
        s_o_wo_q0 = (nl_mask * wo_at_q0).sum(axis=1).astype(np.int64)  # (P,)
        s_o_wo_q1 = (nl_mask * wo_at_q1).sum(axis=1).astype(np.int64)
        s_o_x_q0 = (nl_mask * x_at_q0).sum(axis=1).astype(np.int64)
        s_o_x_q1 = (nl_mask * x_at_q1).sum(axis=1).astype(np.int64)
        s_o_z_q0 = (nl_mask * z_at_q0).sum(axis=1).astype(np.int64)
        s_o_z_q1 = (nl_mask * z_at_q1).sum(axis=1).astype(np.int64)

        # NEW s values at the affected columns — per candidate.
        s_n_wo_q0 = (nl_mask * new_wo_q0).sum(axis=1).astype(np.int64)
        s_n_wo_q1 = (nl_mask * new_wo_q1).sum(axis=1).astype(np.int64)
        s_n_x_q0 = (nl_mask * new_x_q0).sum(axis=1).astype(np.int64)
        s_n_x_q1 = (nl_mask * new_x_q1).sum(axis=1).astype(np.int64)
        s_n_z_q0 = (nl_mask * new_z_q0).sum(axis=1).astype(np.int64)
        s_n_z_q1 = (nl_mask * new_z_q1).sum(axis=1).astype(np.int64)

        # Corrections: replace orig-column contributions with new-column ones.
        corr_wo = (
            _f_diff(num_nl, s_n_wo_q0)
            + _f_diff(num_nl, s_n_wo_q1)
            - _f_diff(num_nl, s_o_wo_q0)
            - _f_diff(num_nl, s_o_wo_q1)
        )
        corr_x = (
            _f_diff(num_nl, s_n_x_q0)
            + _f_diff(num_nl, s_n_x_q1)
            - _f_diff(num_nl, s_o_x_q0)
            - _f_diff(num_nl, s_o_x_q1)
        )
        corr_z = (
            _f_diff(num_nl, s_n_z_q0)
            + _f_diff(num_nl, s_n_z_q1)
            - _f_diff(num_nl, s_o_z_q0)
            - _f_diff(num_nl, s_o_z_q1)
        )

        sum_f_wo = sum_f_wo_orig + corr_wo
        sum_f_x = sum_f_x_orig + corr_x
        sum_f_z = sum_f_z_orig + corr_z

        # pair_or_X[p] = n · C(num_nl, 2) − sum_f_X[p]
        n_C2 = n * num_nl * (num_nl - 1) // 2  # (P,)
        pair_or_wo = n_C2 - sum_f_wo
        pair_or_x = n_C2 - sum_f_x
        pair_or_z = n_C2 - sum_f_z

        # total_weight[p] = sum_nz_wo_orig - orig_nz_at_affected + new_nz_at_affected
        total_weight = (
            sum_nz_wo_orig
            - (s_o_wo_q0 > 0).astype(np.int64)
            - (s_o_wo_q1 > 0).astype(np.int64)
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

        # --- Visited check via chunked tableau reconstruction ---------------
        if visited:
            # Convert the (P, m) new-column arrays to int8 once.
            new_x_q0_i8 = new_x_q0.astype(np.int8, copy=False)
            new_x_q1_i8 = new_x_q1.astype(np.int8, copy=False)
            new_z_q0_i8 = new_z_q0.astype(np.int8, copy=False)
            new_z_q1_i8 = new_z_q1.astype(np.int8, copy=False)

            for p_start in range(0, P, visited_chunk):
                p_end = min(p_start + visited_chunk, P)
                k = p_end - p_start

                # Broadcast-build full (k, m, n) for this chunk; keeps mem bounded.
                chunk_x = np.broadcast_to(x, (k, m, n)).copy()
                chunk_z = np.broadcast_to(z, (k, m, n)).copy()
                idx = np.arange(k)
                chunk_x[idx, :, q0p[p_start:p_end]] = new_x_q0_i8[p_start:p_end]
                chunk_x[idx, :, q1p[p_start:p_end]] = new_x_q1_i8[p_start:p_end]
                chunk_z[idx, :, q0p[p_start:p_end]] = new_z_q0_i8[p_start:p_end]
                chunk_z[idx, :, q1p[p_start:p_end]] = new_z_q1_i8[p_start:p_end]

                xz = np.concatenate([chunk_x, chunk_z], axis=-1)  # (k, m, 2n)
                packed = np.packbits(xz.reshape(k, -1).astype(np.uint8, copy=False), axis=-1)
                row_nb = packed.shape[1]
                pb = packed.tobytes()
                for j in range(k):
                    if pb[j * row_nb : (j + 1) * row_nb] in visited:
                        costs[p_start + j] = np.inf

        # --- Reduce within this Clifford, then fold into global best ------
        pi_best = int(np.argmin(costs))
        c_best = float(costs[pi_best])
        if c_best < best_cost:
            best_cost = c_best
            best_cliff_idx = ci
            best_pair_idx = pi_best

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    # print(f"Applied Clifford: {best_cliff.name} on qubits {best_qubit_pair}, total_weight: {ham.total_weight} → {best_ham.total_weight}, cost: {best_cost:.2f}")
    return best_ham, best_cliff, best_qubit_pair


def _search_best_clifford_par_m1(
    ham: Hamiltonian,
    qubit_pairs: list,
    q0p: np.ndarray,
    q1p: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    n: int,
    visited: set[bytes] | None,
    visited_chunk: int,
) -> tuple[Hamiltonian, CNOTEquivCliffordGate | fSwapEquivCliffordGate, tuple[int, int]]:
    """Specialized par search for m == 1.

    For a single-Pauli tableau, pairwise-OR sums are trivially 0, and
        cost[p] = rw_new[p] if rw_new[p] > 1 else 0.
    Reduces each Clifford's work from O(P·n) column streaming to O(P).
    """
    P = len(qubit_pairs)
    # Original scalar values at the affected qubits (single row).
    x_row = x[0]  # (n,) int8
    z_row = z[0]  # (n,)
    wo_row = (x_row | z_row).astype(np.int32)  # (n,) int32
    rw_orig = int(wo_row.sum())  # scalar

    # Vector form at affected columns for each pair
    x_at_q0 = x_row[q0p].astype(np.int32)  # (P,)
    x_at_q1 = x_row[q1p].astype(np.int32)
    z_at_q0 = z_row[q0p].astype(np.int32)
    z_at_q1 = z_row[q1p].astype(np.int32)
    wo_at_q0 = wo_row[q0p]  # (P,)
    wo_at_q1 = wo_row[q1p]

    # (P, 1, 4) sub_base — keep 3D for consistency with block matmul.
    # Shape chosen so (sub_base @ block) is (P, 1, 4).
    sub_base = np.stack([x_at_q0, x_at_q1, z_at_q0, z_at_q1], axis=-1)  # (P, 4)
    sub_base_3d = sub_base[:, None, :].astype(np.int32, copy=False)  # (P, 1, 4)

    best_cost = np.inf
    best_cliff_idx = 0
    best_pair_idx = 0

    for ci, cliff in enumerate(CLIFFORD_OPTIONS):
        block = _CLIFFORD_BLOCKS[id(cliff)].astype(np.int32, copy=False)
        new_sub = ((sub_base_3d @ block) & 1)[:, 0, :]  # (P, 4)

        new_x_q0 = new_sub[:, 0]  # (P,)
        new_x_q1 = new_sub[:, 1]
        new_z_q0 = new_sub[:, 2]
        new_z_q1 = new_sub[:, 3]
        new_wo_q0 = new_x_q0 | new_z_q0  # (P,)
        new_wo_q1 = new_x_q1 | new_z_q1

        # Row-weight change for the single row
        d_rw = (new_wo_q0 - wo_at_q0) + (new_wo_q1 - wo_at_q1)  # (P,)
        rw_new = rw_orig + d_rw  # (P,)
        nl_mask = rw_new > 1  # (P,)

        # Cost collapses to rw_new where nl_mask, else 0
        costs = np.where(nl_mask, rw_new.astype(np.float64), 0.0)

        # Invalidate same-as-input (no-op Clifford).
        same_mask = np.all(new_sub == sub_base, axis=1)
        costs = np.where(same_mask, np.inf, costs)

        # Visited: reconstruct (chunk, 1, n) tableaux; matches _tableau_key.
        if visited:
            for p_start in range(0, P, visited_chunk):
                p_end = min(p_start + visited_chunk, P)
                k = p_end - p_start
                chunk_x = np.broadcast_to(x_row, (k, n)).copy()  # (k, n)
                chunk_z = np.broadcast_to(z_row, (k, n)).copy()
                idx = np.arange(k)
                chunk_x[idx, q0p[p_start:p_end]] = new_x_q0[p_start:p_end].astype(np.int8)
                chunk_x[idx, q1p[p_start:p_end]] = new_x_q1[p_start:p_end].astype(np.int8)
                chunk_z[idx, q0p[p_start:p_end]] = new_z_q0[p_start:p_end].astype(np.int8)
                chunk_z[idx, q1p[p_start:p_end]] = new_z_q1[p_start:p_end].astype(np.int8)
                xz = np.concatenate([chunk_x, chunk_z], axis=-1)  # (k, 2n)
                packed = np.packbits(xz.astype(np.uint8, copy=False), axis=-1)
                row_nb = packed.shape[1]
                pb = packed.tobytes()
                for j in range(k):
                    if pb[j * row_nb : (j + 1) * row_nb] in visited:
                        costs[p_start + j] = np.inf

        pi_best = int(np.argmin(costs))
        c_best = float(costs[pi_best])
        if c_best < best_cost:
            best_cost = c_best
            best_cliff_idx = ci
            best_pair_idx = pi_best

    best_cliff = CLIFFORD_OPTIONS[best_cliff_idx]
    best_qubit_pair = qubit_pairs[best_pair_idx]
    best_ham = ham.apply_clifford(best_cliff, *best_qubit_pair)
    return best_ham, best_cliff, best_qubit_pair
