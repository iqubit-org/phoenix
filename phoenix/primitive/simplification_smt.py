"""
SMT-based optimal BSF simplification for Pauli Hamiltonians.

Provides a holistic compilation flow:
    Pauli strings + coefficients  →  minimum-CNOT QuantumCircuit

The SMT solver finds the globally optimal Clifford conjugation (minimum CNOT
count) with free single-qubit Cliffords (H, S) that reduce all Pauli strings
to weight ≤ 2.

=== GF(2) Gate Actions on BSF (x, z) ===

Single-qubit Clifford F = [[α, β], [γ, δ]] ∈ Sp(2, GF(2)) on qubit j:
    x'_j = α · x_j  ⊕  β · z_j
    z'_j = γ · x_j  ⊕  δ · z_j
    Symplecticity: α·δ ⊕ β·γ = 1

The 6 elements of Sp(2, GF(2)) = ⟨H, S⟩  (name = operator order, circuit = reversed):
    I:   (1,0,0,1)  →  x'=x,     z'=z
    H:   (0,1,1,0)  →  x'=z,     z'=x
    S:   (1,0,1,1)  →  x'=x,     z'=x⊕z
    HS:  (1,1,1,0)  →  x'=x⊕z,  z'=x        [operator H·S, circuit: S then H]
    SH:  (0,1,1,1)  →  x'=z,     z'=x⊕z      [operator S·H, circuit: H then S]
    HSH: (1,1,0,1)  →  x'=x⊕z,  z'=z

CNOT(ctrl, targ):
    x'_targ = x_targ ⊕ x_ctrl
    z'_ctrl = z_ctrl ⊕ z_targ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from itertools import permutations

from z3 import And, Bool, If, Int, Optimize, Or, Solver, Sum, Xor, is_true, sat


# ═══════════════════════════════════════════════════════════════════
#  Lookup tables
# ═══════════════════════════════════════════════════════════════════

CLIFFORD_LOOKUP = {
    (True, False, False, True): "I",
    (False, True, True, False): "H",
    (True, False, True, True): "S",
    (True, True, True, False): "HS",
    (False, True, True, True): "SH",
    (True, True, False, True): "HSH",
}

# Forward: circuit gates (left-to-right) for each named Clifford
_SQ_FORWARD = {
    "I": [],
    "H": ["h"],
    "S": ["s"],
    "HS": ["s", "h"],       # HS = H·S, circuit order: S then H
    "SH": ["h", "s"],       # SH = S·H, circuit order: H then S
    "HSH": ["h", "s", "h"],
}

# Inverse: circuit gates (left-to-right) for the adjoint
_SQ_INVERSE = {
    "I": [],
    "H": ["h"],             # H† = H
    "S": ["sdg"],           # S† = Sdg
    "HS": ["h", "sdg"],     # (HS)† = S†H† = Sdg·H
    "SH": ["sdg", "h"],     # (SH)† = H†S† = H·Sdg
    "HSH": ["h", "sdg", "h"],
}


# ═══════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SMTResult:
    """Structured result from the SMT solver."""
    depth: int                                   # T = number of CNOTs
    cnots: list[tuple[int, int]]                 # (ctrl, targ) per step
    sq_cliffords: list[list[str]]                # per-step, per-qubit name
    tableaux_x: list[np.ndarray]                 # X at each step boundary (T+1)
    tableaux_z: list[np.ndarray]                 # Z at each step boundary (T+1)
    tableaux_active: list[np.ndarray]            # active (weight > 1) at each boundary
    peeled_at: list[list[int]] = field(default_factory=list)  # legacy, always empty


# ═══════════════════════════════════════════════════════════════════
#  SMT Solver
# ═══════════════════════════════════════════════════════════════════

def solve_min_cnots(
    x0: np.ndarray,
    z0: np.ndarray,
    min_depth: int = 1,
    max_depth: int = 20,
    timeout_ms: int = 60_000,
    verbose: bool = True,
    progress_callback: callable | None = None,
    optimize_total_cx: bool = True,
) -> Optional[SMTResult]:
    """Find the minimum number of CNOTs to reduce all Pauli strings to weight ≤ 2.

    When *optimize_total_cx* is True (default), after finding the minimum CNOT
    depth T the solver runs two additional Z3 Optimize passes:
      1. At depth T, minimize the number of qubit pairs with weight-2 strings
         (interaction-matrix-aware rotation CX cost), then total weight.
      2. At depth T+1, check if spending one extra CNOT yields lower or equal
         total CX cost — preferring fewer weight-2 rotations when tied.

    Args:
        progress_callback: Optional ``fn(depth: int, status: str)`` called after
            each depth is tried.  *status* is ``"sat"``, ``"unsat"``, ``"skip"``
            (for the trivial T=0 case), ``"optimizing"`` (weight optimization at T),
            or ``"trying_extra"`` (trying T+1 with weight optimization).
            When provided, plain-text ``verbose`` output is suppressed.
        optimize_total_cx: If True, run additional Optimize passes to minimize
            total CX cost (conjugation + rotation decomposition).
    """
    m, n = x0.shape
    assert z0.shape == (m, n)

    use_cb = progress_callback is not None
    show = verbose and not use_cb          # plain-text fallback

    weights = np.sum(x0 | z0, axis=1)
    if np.all(weights <= 2):
        if use_cb:
            progress_callback(0, "skip")
        elif show:
            print("T=0: already weight ≤ 2")
        active0 = weights > 1
        return SMTResult(
            depth=0, cnots=[], sq_cliffords=[],
            tableaux_x=[x0.copy()], tableaux_z=[z0.copy()],
            tableaux_active=[active0], peeled_at=[],
        )

    result = None
    T_opt = None
    for T in range(min_depth, max_depth + 1):
        if show:
            print(f"T={T}: ", end="", flush=True)
        result = _solve_for_depth(x0, z0, T, timeout_ms)
        if result is not None:
            T_opt = T
            if use_cb:
                progress_callback(T, "sat")
            elif show:
                print("SAT ✓")
            break
        if use_cb:
            progress_callback(T, "unsat")
        elif show:
            print("UNSAT")

    if result is None:
        return None

    if not optimize_total_cx:
        return result

    # Phase 2: weight-optimize at T
    if use_cb:
        progress_callback(T_opt, "optimizing")
    elif show:
        print(f"T={T_opt}: optimizing weight ... ", end="", flush=True)

    result_opt = _solve_for_depth_opt_weight(x0, z0, T_opt, timeout_ms)
    if result_opt is not None:
        result = result_opt
        if show:
            cx_cost = _compute_total_cx_cost(result)
            print(f"total CX = {cx_cost}")
    elif show:
        print("timeout (keeping original)")

    # Phase 3: try T+1 with weight optimization
    if T_opt + 1 <= max_depth:
        if use_cb:
            progress_callback(T_opt + 1, "trying_extra")
        elif show:
            print(f"T={T_opt + 1}: trying extra depth ... ", end="", flush=True)

        result_t1 = _solve_for_depth_opt_weight(x0, z0, T_opt + 1, timeout_ms)
        if result_t1 is not None and _compute_total_cx_cost(result_t1) <= _compute_total_cx_cost(result):
            # Use <= (not <): when total CX cost is equal, prefer the deeper
            # solution — extra scaffold depth gives the optimizer more freedom
            # to reduce weight-2 strings, and Qiskit's block consolidation
            # handles single-qubit rotations more efficiently.
            result = result_t1
            if show:
                cx_cost = _compute_total_cx_cost(result)
                print(f"better! total CX = {cx_cost}")
        elif show:
            print("no improvement")

    return result


def _solve_for_depth(x0, z0, T, timeout_ms):
    m, n = x0.shape
    s = Solver()
    s.set("timeout", timeout_ms)

    # ── state variables ───────────────────────────────────────
    X = [[[Bool(f"X_{t}_{i}_{j}") for j in range(n)]
          for i in range(m)] for t in range(T + 1)]
    Z = [[[Bool(f"Z_{t}_{i}_{j}") for j in range(n)]
          for i in range(m)] for t in range(T + 1)]

    # ── gate variables ────────────────────────────────────────
    ctrl = [Int(f"ctrl_{t}") for t in range(T)]
    targ = [Int(f"targ_{t}") for t in range(T)]
    sq_a = [[Bool(f"a_{t}_{j}") for j in range(n)] for t in range(T)]
    sq_b = [[Bool(f"b_{t}_{j}") for j in range(n)] for t in range(T)]
    sq_g = [[Bool(f"g_{t}_{j}") for j in range(n)] for t in range(T)]
    sq_d = [[Bool(f"d_{t}_{j}") for j in range(n)] for t in range(T)]

    # ── initialization ────────────────────────────────────────
    for i in range(m):
        for j in range(n):
            s.add(X[0][i][j] == bool(x0[i, j]))
            s.add(Z[0][i][j] == bool(z0[i, j]))

    # ── transitions (all strings evolve unconditionally) ─────
    for t in range(T):
        s.add(ctrl[t] >= 0, ctrl[t] < n, targ[t] >= 0, targ[t] < n)
        s.add(ctrl[t] != targ[t])

        for j in range(n):
            s.add(Xor(And(sq_a[t][j], sq_d[t][j]),
                       And(sq_b[t][j], sq_g[t][j])))

        for i in range(m):
            x_mid = [Xor(And(sq_a[t][j], X[t][i][j]),
                         And(sq_b[t][j], Z[t][i][j])) for j in range(n)]
            z_mid = [Xor(And(sq_g[t][j], X[t][i][j]),
                         And(sq_d[t][j], Z[t][i][j])) for j in range(n)]

            x_at_ctrl = _z3_select(x_mid, ctrl[t], n)
            z_at_targ = _z3_select(z_mid, targ[t], n)

            x_after = [If(targ[t] == j, Xor(x_mid[j], x_at_ctrl), x_mid[j])
                       for j in range(n)]
            z_after = [If(ctrl[t] == j, Xor(z_mid[j], z_at_targ), z_mid[j])
                       for j in range(n)]

            for j in range(n):
                s.add(X[t+1][i][j] == x_after[j])
                s.add(Z[t+1][i][j] == z_after[j])

    # ── terminal: ALL strings must be weight ≤ 2 ────────────
    for i in range(m):
        w = Sum([If(Or(X[T][i][j], Z[T][i][j]), 1, 0) for j in range(n)])
        s.add(w <= 2)

    # ── solve & extract ───────────────────────────────────────
    if s.check() == sat:
        return _extract(s.model(), T, n, m,
                        ctrl, targ, sq_a, sq_b, sq_g, sq_d, X, Z)
    return None


def _z3_select(arr, idx, n):
    result = arr[0]
    for k in range(1, n):
        result = If(idx == k, arr[k], result)
    return result


def _extract(model, T, n, m, ctrl, targ, sq_a, sq_b, sq_g, sq_d, X, Z):
    cnots = []
    sq_layers = []
    for t in range(T):
        cnots.append((model[ctrl[t]].as_long(), model[targ[t]].as_long()))
        layer = []
        for j in range(n):
            key = (is_true(model[sq_a[t][j]]), is_true(model[sq_b[t][j]]),
                   is_true(model[sq_g[t][j]]), is_true(model[sq_d[t][j]]))
            layer.append(CLIFFORD_LOOKUP.get(key, "?"))
        sq_layers.append(layer)

    # extract ALL intermediate tableaux; compute active from weights
    tab_x, tab_z, tab_act = [], [], []
    for t in range(T + 1):
        xt = np.zeros((m, n), dtype=bool)
        zt = np.zeros((m, n), dtype=bool)
        for i in range(m):
            for j in range(n):
                xt[i, j] = is_true(model[X[t][i][j]])
                zt[i, j] = is_true(model[Z[t][i][j]])
        # active = weight > 1 (derived, not from Z3 variable)
        weights = np.sum(xt | zt, axis=1)
        tab_x.append(xt)
        tab_z.append(zt)
        tab_act.append(weights > 1)

    return SMTResult(
        depth=T, cnots=cnots, sq_cliffords=sq_layers,
        tableaux_x=tab_x, tableaux_z=tab_z, tableaux_active=tab_act,
        peeled_at=[],
    )


def _solve_for_depth_opt_weight(x0, z0, T, timeout_ms):
    """Like _solve_for_depth, but minimizes rotation CX cost among all valid solutions.

    Uses z3.Optimize with a lexicographic objective:
      1. Primary: minimize number of distinct qubit pairs with weight-2 strings
         (each pair costs 2–3 CX for rotation synthesis, regardless of how many
         strings share it — see interaction-matrix theory).
      2. Secondary: minimize total final Pauli weight (tie-breaker: prefer
         weight-1 over weight-2 when pair count is equal).
    """
    m, n = x0.shape
    opt = Optimize()
    opt.set("timeout", timeout_ms)

    # ── state variables ───────────────────────────────────────
    X = [[[Bool(f"X_{t}_{i}_{j}") for j in range(n)]
          for i in range(m)] for t in range(T + 1)]
    Z = [[[Bool(f"Z_{t}_{i}_{j}") for j in range(n)]
          for i in range(m)] for t in range(T + 1)]

    # ── gate variables ────────────────────────────────────────
    ctrl = [Int(f"ctrl_{t}") for t in range(T)]
    targ = [Int(f"targ_{t}") for t in range(T)]
    sq_a = [[Bool(f"a_{t}_{j}") for j in range(n)] for t in range(T)]
    sq_b = [[Bool(f"b_{t}_{j}") for j in range(n)] for t in range(T)]
    sq_g = [[Bool(f"g_{t}_{j}") for j in range(n)] for t in range(T)]
    sq_d = [[Bool(f"d_{t}_{j}") for j in range(n)] for t in range(T)]

    # ── initialization ────────────────────────────────────────
    for i in range(m):
        for j in range(n):
            opt.add(X[0][i][j] == bool(x0[i, j]))
            opt.add(Z[0][i][j] == bool(z0[i, j]))

    # ── transitions (all strings evolve unconditionally) ─────
    for t in range(T):
        opt.add(ctrl[t] >= 0, ctrl[t] < n, targ[t] >= 0, targ[t] < n)
        opt.add(ctrl[t] != targ[t])

        for j in range(n):
            opt.add(Xor(And(sq_a[t][j], sq_d[t][j]),
                        And(sq_b[t][j], sq_g[t][j])))

        for i in range(m):
            x_mid = [Xor(And(sq_a[t][j], X[t][i][j]),
                         And(sq_b[t][j], Z[t][i][j])) for j in range(n)]
            z_mid = [Xor(And(sq_g[t][j], X[t][i][j]),
                         And(sq_d[t][j], Z[t][i][j])) for j in range(n)]

            x_at_ctrl = _z3_select(x_mid, ctrl[t], n)
            z_at_targ = _z3_select(z_mid, targ[t], n)

            x_after = [If(targ[t] == j, Xor(x_mid[j], x_at_ctrl), x_mid[j])
                       for j in range(n)]
            z_after = [If(ctrl[t] == j, Xor(z_mid[j], z_at_targ), z_mid[j])
                       for j in range(n)]

            for j in range(n):
                opt.add(X[t+1][i][j] == x_after[j])
                opt.add(Z[t+1][i][j] == z_after[j])

    # ── terminal: ALL strings must be weight ≤ 2 ────────────
    for i in range(m):
        w = Sum([If(Or(X[T][i][j], Z[T][i][j]), 1, 0) for j in range(n)])
        opt.add(w <= 2)

    # ── optimization objectives (lexicographic) ────────────────
    # Primary: minimize number of distinct qubit pairs with weight-2 strings.
    # Terminal constraint ensures all strings have weight ≤ 2, so a string
    # with non-identity at both j1 and j2 is exactly weight-2.
    pair_used = []
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            has_w2 = Or([And(Or(X[T][i][j1], Z[T][i][j1]),
                             Or(X[T][i][j2], Z[T][i][j2]))
                         for i in range(m)])
            pair_used.append(If(has_w2, 1, 0))
    opt.minimize(Sum(pair_used))

    # Secondary: minimize total weight (tie-breaker)
    total_final_weight = Sum([
        If(Or(X[T][i][j], Z[T][i][j]), 1, 0)
        for i in range(m) for j in range(n)
    ])
    opt.minimize(total_final_weight)

    # ── solve & extract ───────────────────────────────────────
    if opt.check() == sat:
        return _extract(opt.model(), T, n, m,
                        ctrl, targ, sq_a, sq_b, sq_g, sq_d, X, Z)
    return None


def _pauli_axis(x_bit: bool, z_bit: bool) -> int:
    """Map BSF bits to Pauli axis index: X=0, Y=1, Z=2.

    Caller must ensure the qubit is non-identity (x_bit or z_bit is True).
    """
    if x_bit and not z_bit:
        return 0   # X
    if x_bit and z_bit:
        return 1   # Y
    return 2       # Z


def _max_matching_size_3x3(edges: set[tuple[int, int]]) -> int:
    """Max matching of a bipartite graph on {0,1,2} x {0,1,2}.

    Returns a value in 0..3.  Used to determine the generic rank of the
    interaction matrix from its support pattern (see Proposition 1 in
    ``cnot_synthesis_cost_for_successive_pauli_rotations_paper.md``).
    """
    best = 0
    for size in range(1, 4):
        for left in permutations(range(3), size):
            for right in permutations(range(3), size):
                if all((left[k], right[k]) in edges for k in range(size)):
                    best = max(best, size)
    return best


def _rotation_cx_cost_for_group(labels: list[tuple[int, int]]) -> int:
    """CX cost for a group of weight-2 Pauli rotations on the same qubit pair.

    *labels* is a list of ``(left_axis, right_axis)`` tuples (axis ∈ {0,1,2}).
    Uses the interaction-matrix generic-rank criterion:
      - ≤ 2 distinct labels  → rank ≤ 2  → 2 CX
      - 3+ distinct labels with max matching ≤ 2 → 2 CX
      - 3+ distinct labels with perfect matching  → 3 CX (generically)
    """
    if not labels:
        return 0
    if len(labels) <= 2:
        return 2  # any pair of 2Q Pauli rotations → rank ≤ 2 → 2 CX
    edges = set(labels)
    generic_rank = _max_matching_size_3x3(edges)
    return 2 if generic_rank <= 2 else 3


def _compute_rotation_cx_cost(final_x, final_z, final_active) -> int:
    """Total rotation CX cost, grouping weight-2 strings by qubit pair.

    Uses interaction-matrix rank analysis per group instead of the naive
    ``2 * num_weight2_strings`` estimate.
    """
    m, n = final_x.shape
    pair_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(m):
        if not final_active[i]:
            continue
        support = final_x[i] | final_z[i]
        if np.sum(support) != 2:
            continue  # weight ≤ 1 → 0 CX
        j1, j2 = np.where(support)[0]
        label = (_pauli_axis(bool(final_x[i, j1]), bool(final_z[i, j1])),
                 _pauli_axis(bool(final_x[i, j2]), bool(final_z[i, j2])))
        pair_groups.setdefault((int(j1), int(j2)), []).append(label)
    return sum(_rotation_cx_cost_for_group(g) for g in pair_groups.values())


def _compute_total_cx_cost(result: SMTResult) -> int:
    """Total CX = 2*T (conjugation scaffold) + rotation CX (interaction-matrix-aware).

    Groups weight-2 strings by qubit pair and uses the generic rank of the
    3×3 interaction matrix to determine CX cost per group (2 or 3), instead
    of the naive 2 CX per weight-2 string.
    """
    T = result.depth
    rotation_cx = _compute_rotation_cx_cost(
        result.tableaux_x[-1], result.tableaux_z[-1], result.tableaux_active[-1])
    return 2 * T + rotation_cx


# ═══════════════════════════════════════════════════════════════════
#  Circuit Construction
# ═══════════════════════════════════════════════════════════════════

def _bsf_to_pauli_char(x_bit: bool, z_bit: bool) -> str:
    if not x_bit and not z_bit:
        return "I"
    if x_bit and not z_bit:
        return "X"
    if not x_bit and z_bit:
        return "Z"
    return "Y"


def _bsf_row_to_label(x_row: np.ndarray, z_row: np.ndarray) -> str:
    """Convert one BSF row to a Qiskit-convention Pauli label (big-endian: q_{n-1}...q_0)."""
    n = len(x_row)
    # Qiskit label is big-endian: label[0] = qubit n-1, label[n-1] = qubit 0
    return "".join(_bsf_to_pauli_char(x_row[n - 1 - j], z_row[n - 1 - j]) for j in range(n))


def _append_sq_forward(qc: QuantumCircuit, sq_names: list[str]):
    for j, name in enumerate(sq_names):
        for gate in _SQ_FORWARD[name]:
            getattr(qc, gate)(j)


def _append_sq_inverse(qc: QuantumCircuit, sq_names: list[str]):
    # Inverse of (SQ then CNOT): CNOT† then SQ† = CNOT then SQ_inv
    # But we only handle the SQ inverse part here; CNOT is separate.
    # The inverse of a sequence of single-qubit gates must be applied in reverse order.
    for j, name in enumerate(sq_names):
        for gate in _SQ_INVERSE[name]:
            getattr(qc, gate)(j)


def _append_pauli_rotation(qc: QuantumCircuit, x_row, z_row, coeff, n, time=1.0):
    """Append a single Pauli rotation for one string (any weight)."""
    label = _bsf_row_to_label(x_row, z_row)
    if label == "I" * n:
        return  # global phase, skip
    theta = 2.0 * np.real(coeff) * time
    op = SparsePauliOp(label, coeffs=[1.0])
    qc.append(PauliEvolutionGate(op, time=theta / 2), range(n))


def build_circuit(result: SMTResult, paulis, coeffs: np.ndarray,
                  num_qubits: int, time: float = 1.0) -> QuantumCircuit:
    """
    Build a QuantumCircuit from the SMT solution.

    Circuit structure: C → Π_i exp(-i c_i P'_i t) → C†
    where C is the forward Clifford and P'_i = C P_i C† (Schrödinger frame).
    The unitary is C† (Π_i exp(-i c_i P'_i t)) C = Π_i exp(-i c_i P_i t).

    Pauli phases are computed exactly via PauliList.evolve().
    """
    T = result.depth
    n = num_qubits
    m = len(coeffs)
    qc = QuantumCircuit(n)

    # ── build forward Clifford circuit ──────────────────────────
    qc_cliff = QuantumCircuit(n)
    for t in range(T):
        _append_sq_forward(qc_cliff, result.sq_cliffords[t])
        qc_cliff.cx(result.cnots[t][0], result.cnots[t][1])

    # ── evolve Paulis through the Clifford (Schrödinger picture: C P C†)
    #    This correctly tracks phases (signs) ────────────────────
    evolved = paulis.evolve(qc_cliff, frame='s')  # P' = C P C†

    # ── forward Clifford ────────────────────────────────────────
    qc.compose(qc_cliff, inplace=True)

    # ── all rotations at center with phase-corrected coefficients ─
    for i in range(m):
        label = evolved[i].to_label()
        # to_label() includes the sign (e.g. '-ZIIIIIYI'), so SparsePauliOp
        # already absorbs the Clifford phase.  Use coeffs[i] directly.
        if label.lstrip('-+i') == "I" * n:
            continue
        theta = 2.0 * np.real(coeffs[i]) * time
        op = SparsePauliOp(label, coeffs=[1.0])
        qc.append(PauliEvolutionGate(op, time=theta / 2), range(n))

    # ── reverse Clifford (adjoint) ──────────────────────────────
    qc.compose(qc_cliff.inverse(), inplace=True)

    return qc


# ═══════════════════════════════════════════════════════════════════
#  High-level API
# ═══════════════════════════════════════════════════════════════════

def compile_hamiltonian_smt(
    hamiltonian,
    time: float | Parameter = 1.0,
    min_depth: int = 1,
    max_depth: int = 20,
    timeout_ms: int = 60_000,
    verbose: bool = True,
    progress_callback: callable | None = None,
    optimize_total_cx: bool = True,
) -> QuantumCircuit:
    """
    Holistic SMT-based compilation: Pauli strings + coefficients → QuantumCircuit.

    Unlike the greedy pipeline (group → simplify → order), this takes ALL Pauli
    strings at once and finds the globally optimal Clifford conjugation sequence
    with minimum CNOT count.

    Args:
        hamiltonian: A Phoenix Hamiltonian (SparsePauliOp with Pauli strings + coefficients).
        time: Evolution time parameter.
        min_depth: Minimum CNOT depth to start searching from (skip T < min_depth).
        max_depth: Maximum number of CNOTs to try.
        timeout_ms: Z3 solver timeout per depth level (milliseconds).
        verbose: Print solver progress (ignored when *progress_callback* is set).
        progress_callback: Optional ``fn(depth, status)`` for custom progress display.
        optimize_total_cx: If True, run additional Optimize passes to minimize
            total CX cost (conjugation + rotation decomposition).

    Returns:
        A Qiskit QuantumCircuit implementing exp(-i H t).
    """
    x0 = hamiltonian.paulis.x.astype(bool)
    z0 = hamiltonian.paulis.z.astype(bool)

    result = solve_min_cnots(x0, z0, min_depth=min_depth, max_depth=max_depth,
                             timeout_ms=timeout_ms, verbose=verbose,
                             progress_callback=progress_callback,
                             optimize_total_cx=optimize_total_cx)
    if result is None:
        raise RuntimeError(f"No solution found within depth {max_depth}")

    qc = build_circuit(result, hamiltonian.paulis, hamiltonian.coeffs,
                       hamiltonian.num_qubits, time=float(time))

    # decompose PauliEvolution gates into primitive gates
    qc = qc.decompose("PauliEvolution")

    return qc
