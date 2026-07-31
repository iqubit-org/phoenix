"""Holistic compilation: forward-frame two-qubit peeling (no a-priori grouping).

Design: ``docs/peel_forward_design.md``. The holistic engine replaces the
support-grouped BSF search (greedy + tabu + stall machinery) with a single
guaranteed-descent loop over the whole table — grouping is fully emergent:

- every active row with weight <= 1 is *emitted* (frozen) immediately, while
  weight-2 rows are emitted only in the sparse active-tableau regime selected
  by a density threshold;
- a *target* row (min weight; ties by pattern popularity) is locked and
  reduced by 2-qubit Clifford conjugations restricted to pairs inside its own
  support and constrained to strictly decrease its weight (always possible —
  exhaustively verified lemma, cf. ``_force_reduce_min_row``);
- among guaranteed-descent candidates, the move with the best *whole-table*
  benefit wins, scored by exact integers (total weight change, #rows improved,
  #rows hurt); remaining ties resolve by enumeration order (a rank-J landing
  tie-break was ablated provably inert across all families — 46/46 programs
  gate-identical, penalty never nonzero — and removed; see design doc §3.2.1);
- Cliffords accumulate forward only; ONE terminal Clifford closes the frame
  (replayed, resynthesized, or absorbed into observables).

Termination: the potential (#active rows, target weight) strictly decreases
lexicographically at every move — at most m*(n-1) moves, no visited set, no
patience, no fission.  Emission has one calibrated density threshold.

Circuit identity (phoenix ``frame='s'`` convention, all 9 options self-inverse):

    U = E_0 . C_1 . E_1 . C_2 ... C_T . E_T . [C_T ... C_1]
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Clifford, Pauli

from ..basics import _CLIFFORD_BLOCKS, CLIFFORD_OPTIONS
from ..hamiltonian import Hamiltonian
from .utils import asap_order as _asap_order
from .utils import (
    cnot_equiv_commute,
    schedule_cnot_equiv_clifford,
)

# ---------------------------------------------------------------------------
# Precomputed per-(clifford, 2q-code) tables.
#
# A row's behaviour under a 2q Clifford at pair (a, b) is a pure function of
# its 4-bit code: bit0 = x_a, bit1 = z_a, bit2 = x_b, bit3 = z_b (same packing
# as ``search_best_clifford_par``). We tabulate, for each of the 9 options:
#   _NEWCODE16[c, code] — the code after conjugation,
#   _DELTA16[c, code]   — the at-pair weight change (in {-1, 0, +1}),
#   _SIGN16[c, code]    — the conjugation phase (+1/-1) of the Pauli row.
# ---------------------------------------------------------------------------

_codes16 = np.arange(16, dtype=np.int64)
_bit_x0 = _codes16 & 1
_bit_z0 = (_codes16 >> 1) & 1
_bit_x1 = (_codes16 >> 2) & 1
_bit_z1 = (_codes16 >> 3) & 1
_wo16 = (_bit_x0 | _bit_z0) + (_bit_x1 | _bit_z1)

_NEWCODE16 = np.zeros((9, 16), dtype=np.uint8)
_DELTA16 = np.zeros((9, 16), dtype=np.int64)
_SIGN16 = np.ones((9, 16), dtype=np.float64)

for _ci, _cliff in enumerate(CLIFFORD_OPTIONS):
    _block = _CLIFFORD_BLOCKS[id(_cliff)].astype(np.int64, copy=False)
    _vecs16 = np.stack([_bit_x0, _bit_x1, _bit_z0, _bit_z1], axis=1)  # (16, 4)
    _new = (_vecs16 @ _block) & 1
    _nx0, _nx1, _nz0, _nz1 = _new.T
    _NEWCODE16[_ci] = _nx0 | (_nz0 << 1) | (_nx1 << 2) | (_nz1 << 3)
    _DELTA16[_ci] = ((_nx0 | _nz0) + (_nx1 | _nz1)) - _wo16

    _qc2 = QuantumCircuit(2)
    _qc2.append(_cliff, [0, 1])
    _cl2 = Clifford(_qc2)
    for _code in range(16):
        _p = Pauli((np.array([(_code >> 1) & 1, (_code >> 3) & 1], dtype=bool),
                    np.array([_code & 1, (_code >> 2) & 1], dtype=bool)))
        _evolved = _p.evolve(_cl2, frame="s")
        # Conjugation of a Hermitian Pauli by a Clifford yields phase +/-1.
        _SIGN16[_ci, _code] = (-1.0) ** ((_evolved.phase % 4) // 2)
        # Cross-check the symplectic-block bit update against qiskit's evolve.
        _ec = (int(_evolved.x[0]) | (int(_evolved.z[0]) << 1)
               | (int(_evolved.x[1]) << 2) | (int(_evolved.z[1]) << 3))
        assert _ec == int(_NEWCODE16[_ci, _code]), "tableau/evolve convention mismatch"


@dataclass
class PeelResult:
    """Output of the holistic peeling engine.

    moves: applied Cliffords in order, ``(cliff, (q0, q1))``.
    emissions: ``(t, labels, coeffs)`` — rows frozen after ``t`` moves, with
        labels in the frame at time ``t`` and conjugation signs folded into
        the (real) coefficients.
    """

    moves: list
    emissions: list
    num_qubits: int


RHO_THRESHOLD = 0.35

def peel_forward(
    ham: Hamiltonian,
    verbose: bool = False,
    rho_threshold: float = None,
) -> PeelResult:
    """Run the deterministic holistic peeling engine.

    Weight-1 rows are always emitted.  Weight-2 rows are emitted only if the
    current active-tableau density

    ``mean(active row weight) / number of active qubits``

    is at most ``rho_threshold``.  Otherwise they remain active and are peeled
    to weight 1.  ``rho_threshold=0.0`` recovers the fixed weight-1 baseline,
    while ``rho_threshold=1.0`` recovers fixed aggressive weight-2 emission.

    The loop then locks the cheapest active row as the target and reduces it
    with guaranteed-descent 2-qubit Clifford conjugations until it emits.  The
    potential (#active rows, target weight) strictly decreases
    lexicographically at every move, giving the general bound m(n-1).

    (The retired v3 certified-holistic-search Tier-1 that used to gate this
    loop is archived in ``backup/peel_v3.py``; its full-matrix ablation,
    ``experiments/attic/ablate_v3.json``, showed it net-worse on UCCSD, so it
    was removed.)
    """
    if rho_threshold is None:
        rho_threshold = RHO_THRESHOLD
    if not np.isfinite(rho_threshold) or not 0.0 <= rho_threshold <= 1.0:
        raise ValueError("rho_threshold must be a finite number between 0 and 1.")

    x = np.asarray(ham.paulis.x, dtype=np.uint8).copy()
    z = np.asarray(ham.paulis.z, dtype=np.uint8).copy()
    coeffs = np.real(np.asarray(ham.coeffs)).astype(np.float64).copy()
    m, n = x.shape

    active = np.ones(m, dtype=bool)
    w = (x | z).sum(axis=1).astype(np.int64)
    moves: list = []
    emissions: list = []
    target = -1

    def _emit():
        nonlocal target
        rows = np.where(active & (w <= 1))[0]
        rows_weight_2 = np.where(active & (w == 2))[0]
        if rows_weight_2.size:
            active_rows = np.where(active)[0]
            active_qubits = int(
                np.any((x[active_rows] | z[active_rows]) != 0, axis=0).sum()
            )
            active_density = float(w[active_rows].mean()) / max(active_qubits, 1)
            emit_weight_2 = active_density <= rho_threshold
            if emit_weight_2:
                rows = np.concatenate((rows, rows_weight_2))
            if verbose:
                print(
                    f"  [emit-policy] rho={active_density:.4f}, "
                    f"threshold={rho_threshold:.4f}, emit_weight_2={emit_weight_2}"
                )
        if rows.size == 0:
            return
        # NOTE: ``str(Pauli)`` truncates labels beyond 50 qubits ("II...");
        # ``to_label()`` is the untruncated form (wide Hamlib programs).
        labels = [Pauli((z[r].astype(bool), x[r].astype(bool))).to_label() for r in rows]
        emissions.append((len(moves), labels, coeffs[rows].tolist()))
        active[rows] = False
        if target in rows:
            target = -1
        if verbose:
            print(f"  t={len(moves)}: emit {labels}")

    def _apply(ci: int, a: int, b: int, tag: str):
        act_idx = np.where(active)[0]
        cq = (x[act_idx, a] | (z[act_idx, a] << 1)
              | (x[act_idx, b] << 2) | (z[act_idx, b] << 3)).astype(np.int64)
        nc = _NEWCODE16[ci][cq]
        x[act_idx, a] = (nc & 1).astype(np.uint8)
        z[act_idx, a] = ((nc >> 1) & 1).astype(np.uint8)
        x[act_idx, b] = ((nc >> 2) & 1).astype(np.uint8)
        z[act_idx, b] = ((nc >> 3) & 1).astype(np.uint8)
        w[act_idx] += _DELTA16[ci][cq]
        coeffs[act_idx] *= _SIGN16[ci][cq]
        moves.append((CLIFFORD_OPTIONS[ci], (a, b)))
        if verbose:
            print(f"  move {len(moves)} [{tag}]: {CLIFFORD_OPTIONS[ci].name}@({a},{b})")
        _emit()

    _emit()
    while active.any():
        # target-descent episode: lock the cheapest active row and reduce it to
        # emission by guaranteed-descent 2q Cliffords, then repeat.
        if target < 0:
            # cheapest active row; ties by 1-local pattern popularity
            wa = np.where(active, w, np.iinfo(np.int64).max)
            tied = np.where(wa == wa.min())[0]
            if tied.size > 1:
                v = (x + 2 * z).astype(np.int64)
                va = v[active]
                freq = np.zeros((n, 4), dtype=np.int64)
                for q in range(n):
                    freq[q] = np.bincount(va[:, q], minlength=4)
                vt = v[tied]
                pop = (freq[np.arange(n)[None, :], vt] * (vt != 0)).sum(axis=1)
                tied = tied[np.lexsort((tied, -pop))]
            target = int(tied[0])
            if verbose:
                print(f"  [T2] target: row {target}, w={w[target]}")

        act_idx = np.where(active)[0]
        ma = act_idx.size
        supp = np.where((x[target] | z[target]) != 0)[0]
        pairs = np.asarray(list(combinations(supp.tolist(), 2)), dtype=np.int64)
        P = pairs.shape[0]
        cq = (x[act_idx] | (z[act_idx] << 1)).astype(np.int64)  # (ma, n) in [0, 4)
        code = cq[:, pairs[:, 0]].T | (cq[:, pairs[:, 1]].T << 2)  # (P, ma)
        offs = (np.arange(P, dtype=np.int64) * 16)[:, None]
        hist = np.bincount((code + offs).ravel(), minlength=16 * P).reshape(P, 16)
        ti = int(np.searchsorted(act_idx, target))
        tcode = code[:, ti]  # (P,)

        K = np.int64(ma + 1)
        best_key = np.int64(-1)
        best: list[tuple[int, int]] = []  # tied (ci, pi)
        for ci in range(9):
            d16 = _DELTA16[ci]
            valid = d16[tcode] == -1  # guaranteed descent on the target
            if not valid.any():
                continue
            dW = hist @ d16
            nben = hist @ (d16 < 0).astype(np.int64)
            nharm = hist @ (d16 > 0).astype(np.int64)
            key = ((ma - dW) * K + nben) * K + (ma - nharm)
            key = np.where(valid, key, np.int64(-1))
            kmax = key.max()
            if kmax > best_key:
                best_key = kmax
                best = [(ci, int(pi)) for pi in np.where(key == kmax)[0]]
            elif kmax == best_key:
                best.extend((ci, int(pi)) for pi in np.where(key == kmax)[0])

        assert best, "guaranteed-descent move must exist (verified lemma)"
        if len(best) > 1:
            best.sort()
        ci, pi = best[0]
        _apply(ci, int(pairs[pi, 0]), int(pairs[pi, 1]), "T2")

    return PeelResult(moves=moves, emissions=emissions, num_qubits=n)


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

# V2-A commutation-aware ASAP scheduling
SCHEDULE_ASAP = True

# Exact-commutation-aware scheduling with ASAP
SCHEDULE_ASAP_COMMUTE = SCHEDULE_ASAP and True


def _moves_commute(mi: dict, mj: dict) -> bool:
    return cnot_equiv_commute(mi["cliff"], mi["qubits"], mj["cliff"], mj["qubits"])


def _move_fixes_block(move_item: dict, block_item: dict, n: int) -> bool:
    """Does the move's Clifford fix EVERY rotation in the block bucket (so the
    block commutes with the move)? Block factor on the control leg must be in
    {I,P0}, on the target leg in {I,P1}. Label chars are big-endian: qubit q is
    ``label[n-1-q]``."""
    a, b = move_item["qubits"]
    p0, p1 = move_item["cliff"].pauli_0, move_item["cliff"].pauli_1
    for lbl in block_item["labels"]:
        if lbl[n - 1 - a] not in ("I", p0) or lbl[n - 1 - b] not in ("I", p1):
            return False
    return True


def _items_commute(x: dict, y: dict, n: int) -> bool:
    kx, ky = x["kind"], y["kind"]
    if kx == "block" and ky == "block":
        return True  # Trotter freedom
    if kx == "move" and ky == "move":
        return _moves_commute(x, y)
    move, block = (x, y) if kx == "move" else (y, x)
    return _move_fixes_block(move, block, n)


def _exact_commute_deps(items: list[dict], n: int) -> list[set]:
    """Dependency partial order keeping only genuine non-commutation edges: per
    qubit, an item links to EVERY earlier item on that qubit it does not commute
    with. Non-transitivity of commutation forbids any early stop."""
    on_q: list[list[int]] = [[] for _ in range(n)]
    deps: list[set] = [set() for _ in items]
    for i, it in enumerate(items):
        for q in it["qubits"]:
            col = on_q[q]
            for j in reversed(col):
                if not _items_commute(items[j], it, n):
                    deps[i].add(j)
            col.append(i)
    return deps


def peel_circuit(result: PeelResult, terminal: str = "auto", phase_exact: bool = False) -> QuantumCircuit:
    """Build the circuit  E_0 C_1 E_1 ... C_T E_T [terminal].

    Emissions are held in per-support pending buckets, flushed before a move
    touches their qubits (gates on disjoint supports commute exactly), so
    same-pair rotations sit adjacent and consolidate into a single 2q block
    downstream; with :data:`SCHEDULE_ASAP` the buckets and moves are packed
    into parallel layers by ASAP list scheduling over the exact commutation
    partial order.

    terminal:
      - ``"auto"`` (default): the cheaper (by 2q count) of replay and synth;
      - ``"replay"``: the T self-inverse gates replayed in reverse (phase-exact
        by construction); with :data:`SCHEDULE_ASAP_COMMUTE` the pure
        CNOT-equiv tail is post-processed by the operator-exact standalone
        pass :func:`schedule_cnot_equiv_clifford` (commutation-reachable
        self-inverse cancellation + exact-commutation ASAP depth packing);
      - ``"synth"``: ONE Clifford resynthesized at full width (greedy method,
        <= O(n^2/log n) 2q gates);
      - ``"absorb"``: omitted — QuCLEAR-style observable-absorption semantics,
        valid for expectation-value workloads.

    phase_exact: Clifford resynthesis is only defined up to a global phase
    (the qiskit ``Clifford`` tableau is phase-blind). With
    ``phase_exact=True`` the deficit — provably quantized to the pi/4 grid —
    is recovered exactly from one statevector amplitude
    ``<0| S . R^{-1} |0> = e^{i phi}`` (snapped, validated, graceful fallback
    to replay), at a compile-time cost that grows with 2^n (~8 s at 22q).
    The default ``False`` skips the recovery: the circuit is then correct up
    to a global phase — indistinguishable for standalone execution and
    expectation-value workloads, but NOT safe to use under control (e.g.
    inside QPE); pass ``phase_exact=True`` there. (Once an O(n^3) phase-aware
    stabilizer amplitude lands, the recovery becomes free and the default can
    flip back.)
    """
    n = result.num_qubits
    by_t: dict[int, tuple[list, list]] = {}
    for t, labels, cs in result.emissions:
        bucket = by_t.setdefault(t, ([], []))
        bucket[0].extend(labels)
        bucket[1].extend(cs)

    # ---- Pass 1: items (moves + emission buckets) with their exact
    # commutation dependencies, replayed from the engine timeline.
    # Constraints: a move orders after the previous move on each of its qubits
    # AND after every block emitted on those qubits since (a block must precede
    # EVERY later move touching any of its support qubits — the first toucher
    # alone is not enough, as later touchers of other support qubits may be
    # disjoint from it). A block orders after the last move on each support
    # qubit. Blocks carry no mutual order (Trotter freedom). ------------------
    items: list[dict] = []
    last_move: list[int | None] = [None] * n
    blocks_since: list[list[int]] = [[] for _ in range(n)]  # per qubit, since last move
    open_buckets: dict[tuple, int] = {}
    for t in range(len(result.moves) + 1):
        if t in by_t:
            labels, cs = by_t[t]
            for lbl, c in zip(labels, cs):
                p = Pauli(lbl)
                key = tuple(int(q) for q in np.where(np.asarray(p.x) | np.asarray(p.z))[0])
                if key in open_buckets:
                    it = items[open_buckets[key]]
                    it["labels"].append(lbl)
                    it["cs"].append(c)
                else:
                    deps = {last_move[q] for q in key if last_move[q] is not None}
                    bid = len(items)
                    open_buckets[key] = bid
                    items.append({"kind": "block", "qubits": key, "deps": deps,
                                  "labels": [lbl], "cs": [c]})
                    for q in key:
                        blocks_since[q].append(bid)
        if t < len(result.moves):
            cliff, (a, b) = result.moves[t]
            deps = {last_move[q] for q in (a, b) if last_move[q] is not None}
            deps.update(blocks_since[a])
            deps.update(blocks_since[b])
            blocks_since[a] = []
            blocks_since[b] = []
            for key in [k for k in open_buckets if a in k or b in k]:
                open_buckets.pop(key)  # accumulation stops once the frame moves
            mid = len(items)
            items.append({"kind": "move", "qubits": (a, b), "deps": deps, "cliff": cliff})
            last_move[a] = last_move[b] = mid

    if SCHEDULE_ASAP_COMMUTE:
        for it, d in zip(items, _exact_commute_deps(items, n)):
            it["deps"] = d
    order = _asap_order(items, n) if SCHEDULE_ASAP else _sequential_order(items)
    qc = QuantumCircuit(n)
    for i in order:
        it = items[i]
        if it["kind"] == "move":
            qc.append(it["cliff"], it["qubits"])
        else:
            qc.append(PauliEvolutionGate(Hamiltonian(it["labels"], it["cs"])), range(n))

    if terminal in ("auto", "replay", "synth") and result.moves:
        replay = QuantumCircuit(n)
        for cliff, (a, b) in reversed(result.moves):
            replay.append(cliff, (a, b))
        if SCHEDULE_ASAP_COMMUTE:
            replay = schedule_cnot_equiv_clifford(replay)
        tail = replay
        if terminal in ("auto", "synth"):
            synth = _synth_terminal(result.moves, n, phase_exact=phase_exact)
            if synth is not None:
                n2_synth = sum(1 for inst in synth.data if inst.operation.num_qubits == 2)
                if terminal == "synth" or n2_synth < replay.size():
                    tail = synth
        qc.compose(tail, inplace=True)
    elif terminal != "absorb" and terminal not in ("auto", "replay", "synth"):
        raise ValueError(
            f"Unknown terminal mode: {terminal!r}; options: 'auto', 'replay', 'synth', 'absorb'"
        )
    return qc.decompose("PauliEvolution")


def _sequential_order(items: list[dict]) -> list[int]:
    """The engine's original timeline order (ablation baseline): each bucket
    right before the move that seals it; never-sealed buckets at the end in
    creation order."""
    order: list[int] = []
    emitted = set()
    for i, it in enumerate(items):
        if it["kind"] == "move":
            for d in sorted(it["deps"]):
                if d not in emitted and items[d]["kind"] == "block":
                    order.append(d)
                    emitted.add(d)
            order.append(i)
            emitted.add(i)
    for i in range(len(items)):
        if i not in emitted:
            order.append(i)
            emitted.add(i)
    return order


def _synth_terminal(moves, n: int, phase_exact: bool = False) -> QuantumCircuit | None:
    """Full-width resynthesis of the terminal Clifford.

    Qiskit's Clifford abstraction is phase-blind: collecting the replay tail
    into one Clifford and resynthesizing it (greedy) yields S with
    ``Operator(S) = e^{i phi} Operator(R)``. With ``phase_exact=True``, the
    deficit phi — which lies on the pi/4 grid (the Clifford group's phase
    subgroup) — is recovered exactly from one statevector amplitude
    ``<0| S . R^{-1} |0> = e^{i phi}``; we snap to the grid, validate, and
    write ``-phi`` into ``S.global_phase``. Returns None (caller falls back
    to replay) if anything is off.
    """
    from qiskit.transpiler import PassManager, passes

    try:
        from qiskit.transpiler.passes.synthesis import HLSConfig
    except ImportError:
        return None

    replay = QuantumCircuit(n)
    for cliff, (a, b) in reversed(moves):
        replay.append(cliff, (a, b))
    try:
        pm = PassManager([
            passes.CollectCliffords(matrix_based=True, min_block_size=1, max_block_width=n),
            passes.HighLevelSynthesis(
                hls_config=HLSConfig(use_default_on_unspecified=False, clifford=["greedy"])
            ),
        ])
        synth = pm.run(replay)
    except Exception:
        return None

    if not phase_exact:
        return synth

    # Phase-deficit circuit D = S . R^{-1}; R^{-1} = the moves replayed forward
    # (every option is self-inverse). Operator(D) = e^{i phi} I.
    d = QuantumCircuit(n)
    for cliff, (a, b) in moves:
        d.append(cliff, (a, b))
    d.compose(synth, inplace=True)
    amp = _zero_amplitude(d)
    if amp is None or abs(abs(amp) - 1.0) > 1e-6:
        return None
    phi = float(np.angle(amp))
    snapped = round(phi / (np.pi / 4)) * (np.pi / 4)
    if abs(phi - snapped) > 1e-6:
        return None
    synth = synth.copy()
    synth.global_phase = float(synth.global_phase) - snapped
    return synth


def _zero_amplitude(circ: QuantumCircuit) -> complex | None:
    """<0...0| circ |0...0>, via Aer statevector when available."""
    try:
        from qiskit import transpile as _transpile
        from qiskit_aer import AerSimulator

        c = _transpile(circ, basis_gates=["cx", "u"], optimization_level=0)
        c.save_amplitudes([0])
        res = AerSimulator(method="statevector").run(c).result()
        return complex(res.data()["amplitudes"][0])
    except Exception:
        pass
    try:
        from qiskit.quantum_info import Statevector

        if circ.num_qubits > 20:
            return None
        return complex(Statevector.from_instruction(circ).data[0])
    except Exception:
        return None


def holistic_compile(
    ham: Hamiltonian,
    terminal: str = "auto",
    phase_exact: bool = False,
    verbose: bool = False,
    rho_threshold: float = None,
) -> QuantumCircuit:
    """Engine + circuit construction (pre-optimizer).

    See :func:`peel_circuit` for the ``terminal`` / ``phase_exact`` semantics
    (default output is correct up to a global phase; pass
    ``phase_exact=True`` for controlled/QPE use).
    """
    return peel_circuit(
        peel_forward(
            ham,
            verbose=verbose,
            rho_threshold=rho_threshold,
        ),
        terminal=terminal,
        phase_exact=phase_exact,
    )
