"""Tests for the holistic engine (phoenix/primitive/holistic.py).

Run directly (``python tests/test_holistic.py``) or via pytest.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import qiskit.quantum_info as qi

import phoenix
from phoenix.hamiltonian import Hamiltonian
from phoenix.primitive.holistic import peel_circuit, peel_forward

# ---------------------------------------------------------------------------
# The 152-row near-Cartesian-product UCCSD-JW group on which the legacy
# subset=True pipeline oscillated forever (10 qubits). The holistic engine must
# terminate by construction and compile it cleanly.
# ---------------------------------------------------------------------------
OSCILLATING_GROUP = [
    "YZIIIIIIII", "YIIIIIIIII", "YYYIIIIIII", "XYXIIIIIII", "YXIZIIIIII", "XYZIIIIIII",
    "YXIYYXIIII", "XXIYXXIIII", "IIIIIXYIII", "IIIIZYXIII", "IIIZIYIZII", "IIIIZYZIII",
    "IIIZIYIYYX", "IIIZZXIYXX", "IIIZIYIYIY", "IIIZZXIYZX", "XIIIIXYIII", "YZIIIXXIII",
    "YZIIZYYIII", "XIIIZYXIII", "XZIIIXYIII", "YIIIIXXIII", "YIIIZYYIII", "XZIIZYXIII",
    "XIIZIYIZII", "YZIIIXZIII", "YZIZZXIZII", "XIIIZYZIII", "XZIZIYIZII", "YIIIIXZIII",
    "YIIZZXIZII", "XZIIZYZIII", "XIIZIYIYYX", "YZIZIYIYXX", "YZIZZXIYYX", "XIIZZXIYXX",
    "XZIZIYIYYX", "YIIZIYIYXX", "YIIZZXIYYX", "XZIZZXIYXX", "XIIZIYIYIY", "YZIZIYIYZX",
    "YZIZZXIYIY", "XIIZZXIYZX", "XZIZIYIYIY", "YIIZIYIYZX", "YIIZZXIYIY", "XZIZZXIYZX",
    "YYXIIXYIII", "YYYIIXXIII", "YYYIZYYIII", "YYXIZYXIII", "XYYIIXYIII", "XYXIIXXIII",
    "XYXIZYYIII", "XYYIZYXIII", "YYXZIYIZII", "YYYIIXZIII", "YYYZZXIZII", "YYXIZYZIII",
    "XYYZIYIZII", "XYXIIXZIII", "XYXZZXIZII", "XYYIZYZIII", "YYXZIYIYYX", "YYYZIYIYXX",
    "YYYZZXIYYX", "YYXZZXIYXX", "XYYZIYIYYX", "XYXZIYIYXX", "XYXZZXIYYX", "XYYZZXIYXX",
    "YYXZIYIYIY", "YYYZIYIYZX", "YYYZZXIYIY", "YYXZZXIYZX", "XYYZIYIYIY", "XYXZIYIYZX",
    "XYXZZXIYIY", "XYYZZXIYZX", "YYZIIXYIII", "YXIZIXXIII", "YXIZZYYIII", "YYZIZYXIII",
    "XXIZIXYIII", "XYZIIXXIII", "XYZIZYYIII", "XXIZZYXIII", "YYZZIYIZII", "YXIZIXZIII",
    "YXIIZXIZII", "YYZIZYZIII", "XXIIIYIZII", "XYZIIXZIII", "XYZZZXIZII", "XXIZZYZIII",
    "YYZZIYIYYX", "YXIIIYIYXX", "YXIIZXIYYX", "YYZZZXIYXX", "XXIIIYIYYX", "XYZZIYIYXX",
    "XYZZZXIYYX", "XXIIZXIYXX", "YYZZIYIYIY", "YXIIIYIYZX", "YXIIZXIYIY", "YYZZZXIYZX",
    "XXIIIYIYIY", "XYZZIYIYZX", "XYZZZXIYIY", "XXIIZXIYZX", "YXIYXIYIII", "YXIYYIXIII",
    "YXIYXZYIII", "YXIYYZXIII", "XXIYYIYIII", "XXIYXIXIII", "XXIYYZYIII", "XXIYXZXIII",
    "YXIXXZIZII", "YXIYYIZIII", "YXIXXIIZII", "YXIYYZZIII", "XXIXYZIZII", "XXIYXIZIII",
    "XXIXYIIZII", "XXIYXZZIII", "YXIXXZIYYX", "YXIXYZIYXX", "YXIXXIIYYX", "YXIXYIIYXX",
    "XXIXYZIYYX", "XXIXXZIYXX", "XXIXYIIYYX", "XXIXXIIYXX", "YXIXXZIYIY", "YXIXYZIYZX",
    "YXIXXIIYIY", "YXIXYIIYZX", "XXIXYZIYIY", "XXIXXZIYZX", "XXIXYIIYIY", "XXIXXIIYZX",
]

# Six 8-qubit strings sharing a weight-4 common substring across two exact
# supports (the support-grouping blind spot; cf. exploration/peel_forward_demo.py).
DEMO_GROUP = ["IZYIZXXX", "IYYIZXXY", "IZYIZXXZ", "ZZYIZXXI", "YYYIZXXI", "YZYIZXXI"]


def _random_ham(n, m, seed):
    rng = np.random.default_rng(seed)
    paulis = set()
    while len(paulis) < m:
        p = "".join(rng.choice(list("IXYZ"), size=n))
        if p != "I" * n:
            paulis.add(p)
    return Hamiltonian(sorted(paulis), rng.normal(size=m))


def _h_eff(ham, eps=1e-2, terminal="replay", phase_exact=False):
    """Symmetric-difference effective Hamiltonian i(U(+eps)-U(-eps))/2eps —
    equals H exactly for any first-order product formula, independent of
    term ordering (times a common e^{i phi} if the terminal carries an
    unrecovered global phase)."""
    qp = peel_circuit(peel_forward(Hamiltonian(ham.paulis, ham.coeffs * eps)), terminal, phase_exact)
    qm = peel_circuit(peel_forward(Hamiltonian(ham.paulis, -ham.coeffs * eps)), terminal, phase_exact)
    return 1j * (qi.Operator(qp).data - qi.Operator(qm).data) / (2 * eps)


def _h_eff_err(ham, eps=1e-2, terminal="replay", phase_exact=False):
    return float(np.abs(_h_eff(ham, eps, terminal, phase_exact) - ham.to_matrix()).max())


def test_holistic_correctness_random():
    """Derivative check on random 4-5 qubit Hamiltonians, all terminal modes.

    ``synth``/``auto`` with ``phase_exact=True`` exercise the full-width
    Clifford resynthesis incl. the pi/4 global-phase recovery — the check is
    phase-sensitive, so a wrong phase fails loudly.
    """
    for seed, (n, m) in enumerate([(4, 8), (4, 12), (5, 10), (5, 16)]):
        ham = _random_ham(n, m, seed)
        err = _h_eff_err(ham, terminal="replay")
        assert err < 5e-3, (seed, "replay", err)
        for terminal in ["synth", "auto"]:
            err = _h_eff_err(ham, terminal=terminal, phase_exact=True)
            assert err < 5e-3, (seed, terminal, err)
        print(f"  random ({n}q, {m}P, seed={seed}): all terminals ok (err={err:.2e})")


def test_holistic_default_phase_free_equivalence():
    """Default mode (phase_exact=False) must be correct up to ONE common
    global phase: He = e^{i phi} H with phi on the pi/4 grid."""
    for seed in [11, 12]:
        ham = _random_ham(5, 12, seed)
        he = _h_eff(ham, terminal="synth", phase_exact=False)
        h = ham.to_matrix()
        tr = np.trace(h.conj().T @ he)
        phi = float(np.angle(tr))
        err = float(np.abs(he - np.exp(1j * phi) * h).max())
        assert err < 5e-3, (seed, err)
        snap = round(phi / (np.pi / 4)) * (np.pi / 4)
        assert abs(phi - snap) < 1e-3, (seed, phi)
        print(f"  phase-free (seed={seed}): err={err:.2e}, phi={phi / np.pi:.3f}*pi (on grid)")


def test_holistic_correctness_demo_group():
    """The cross-support shared-substring demo case."""
    ham = Hamiltonian(DEMO_GROUP, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    err = _h_eff_err(ham)
    assert err < 5e-3, err
    res = peel_forward(ham, rho_threshold=1.0)
    assert len(res.moves) == 4, "shared weight-4 core must peel in 4 moves"
    print(f"  demo group: 4 moves, err={err:.2e}")


def test_holistic_oscillating_group_compiles():
    """The 152-row oscillating group terminates and compiles end to end."""
    ham = Hamiltonian(OSCILLATING_GROUP, np.ones(len(OSCILLATING_GROUP)))
    qc = phoenix.compile_hamiltonian_simulation(ham, grouping="holistic")
    n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    assert n2 > 0
    print(f"  oscillating group: compiled, 2q={n2}")


def test_holistic_deterministic():
    """Same input must produce the identical move sequence and emissions."""
    ham = _random_ham(5, 14, 99)
    r1 = peel_forward(ham)
    r2 = peel_forward(ham)
    assert [(c.name, q) for c, q in r1.moves] == [(c.name, q) for c, q in r2.moves]
    assert r1.emissions == r2.emissions
    print(f"  determinism: {len(r1.moves)} moves identical across runs")


def test_holistic_atom_staging_correctness():
    """Atom-heavy program: two size-4 same-support families on 5 qubits.
    Exercises staging, protection-constraint moves, and cohort flush; the
    phase-sensitive derivative check validates the emission frames."""
    fam1 = ["XXYXI", "YYXXI", "XYXYI", "YXYYI"]
    fam2 = ["IXZZY", "IYXZX", "IZYZZ", "IXXZX"]
    ham = Hamiltonian(fam1 + fam2, np.linspace(0.1, 0.8, 8))
    err = _h_eff_err(ham)
    assert err < 5e-3, err
    res = peel_forward(ham)
    sizes = sorted(len(labels) for _, labels, _ in res.emissions)
    print(f"  atom staging: err={err:.2e}, cohort sizes={sizes}")


def test_holistic_wide_program_labels():
    """Programs beyond 50 qubits must round-trip emission labels untruncated
    (regression: ``str(Pauli)`` truncates at 50 qubits, ``to_label()`` not)."""
    rng = np.random.default_rng(0)
    n, m = 60, 12
    paulis = []
    for _ in range(m):
        p = ["I"] * n
        for q in rng.choice(n, size=4, replace=False):
            p[q] = rng.choice(list("XYZ"))
        paulis.append("".join(p))
    ham = Hamiltonian(paulis, rng.normal(size=m))
    qc = peel_circuit(peel_forward(ham))
    assert qc.num_qubits == n
    print(f"  wide program ({n}q): compiled without label truncation")


def test_holistic_termination_bound():
    """Adaptive emission must respect the m*(n-1) potential bound."""
    ham = _random_ham(6, 20, 5)
    res = peel_forward(ham)
    m, n = len(ham.paulis), ham.num_qubits
    assert len(res.moves) <= m * (n - 1), (len(res.moves), m, n)
    print(f"  termination: {len(res.moves)} moves <= {m * (n - 1)} bound")


# ---------------------------------------------------------------------------
# Exact-commutation-aware scheduling (SCHEDULE_EXACT_COMMUTE): relaxing the
# item DAG by dropping provably-commuting move<->move and move<->block ordering
# constraints must (a) preserve the unitary EXACTLY, (b) leave the 2q count
# unchanged (pure reorder), (c) never increase 2q depth, and (d) strictly cut
# depth on move-heavy programs. Commutativity is NOT transitive, so a correct
# implementation must keep every non-commuting pair transitively ordered.
# ---------------------------------------------------------------------------
from phoenix.primitive import holistic as _holistic

# seeds 2/6/7/11 broke a naive "remove commuting edges independently" prototype
# (wrong unitary) -> they are the regression guard for the transitivity bug.
_EXACT_SEEDS = [0, 1, 2, 3, 6, 7, 10, 11]


def _compile_both(ham, optimize=False, terminal="replay"):
    _holistic.SCHEDULE_EXACT_COMMUTE = False
    off = phoenix.compile_hamiltonian_simulation(ham, optimize=optimize, terminal=terminal)
    _holistic.SCHEDULE_EXACT_COMMUTE = True
    on = phoenix.compile_hamiltonian_simulation(ham, optimize=optimize, terminal=terminal)
    _holistic.SCHEDULE_EXACT_COMMUTE = False
    return off, on


def _n2_d2(qc):
    n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
    return n2, d2


def test_exact_commute_preserves_first_order_trotter():
    """The relaxation reorders only provably-commuting ops, so it must preserve
    the engine's first-order Trotter formula exactly (derivative-exact) -- the
    same contract SCHEDULE_ASAP already honours. A naive independent-edge-removal
    prototype reordered NON-commuting pairs (commutativity is not transitive);
    the derivative check blows up there (seeds 2, 11), so this is its guard."""
    for seed in _EXACT_SEEDS:
        ham = _random_ham(6, 16, seed)
        _holistic.SCHEDULE_EXACT_COMMUTE = True
        try:
            err = _h_eff_err(ham, terminal="replay")
        finally:
            _holistic.SCHEDULE_EXACT_COMMUTE = False
        assert err < 5e-3, (seed, err)
    print(f"  exact-commute: first-order Trotter preserved on {len(_EXACT_SEEDS)} seeds")


def test_exact_commute_count_neutral_and_reduces_total_depth():
    """Pure reorder: identical 2q count per program (rock-solid invariant). It
    cuts 2q depth on aggregate over move-heavy programs -- asserted on the total
    rather than per-program, since a greedy schedule over a relaxed partial order
    can regress an individual case (Graham-type anomaly)."""
    tot_off = tot_on = 0
    for seed in _EXACT_SEEDS:
        ham = _random_ham(6, 16, seed)
        off, on = _compile_both(ham)
        (n2o, d2o), (n2n, d2n) = _n2_d2(off), _n2_d2(on)
        assert n2n == n2o, (seed, n2o, n2n)  # pure reorder -> count invariant
        tot_off += d2o
        tot_on += d2n
    assert tot_on < tot_off, (tot_off, tot_on)
    print(f"  exact-commute: count-neutral, total 2q depth {tot_off}->{tot_on}")


# ---------------------------------------------------------------------------
# Terminal-Clifford commutation scheduling (SCHEDULE_TERMINAL_COMMUTE): the
# replay tail is a pure CNOT-equiv circuit, so the standalone operator-exact
# pass (cancellation + exact-commutation ASAP) applies to it as a unit. The
# full circuit must stay EXACTLY equal (body untouched, tail operator-equal),
# never gain 2q gates, and cut 2q depth on aggregate.
# ---------------------------------------------------------------------------


def _compile_terminal_both(ham, terminal="replay"):
    old_tc, old_ec = _holistic.SCHEDULE_TERMINAL_COMMUTE, _holistic.SCHEDULE_EXACT_COMMUTE
    try:
        _holistic.SCHEDULE_EXACT_COMMUTE = True
        _holistic.SCHEDULE_TERMINAL_COMMUTE = False
        off = phoenix.compile_hamiltonian_simulation(ham, optimize=False, terminal=terminal)
        _holistic.SCHEDULE_TERMINAL_COMMUTE = True
        on = phoenix.compile_hamiltonian_simulation(ham, optimize=False, terminal=terminal)
    finally:
        _holistic.SCHEDULE_TERMINAL_COMMUTE, _holistic.SCHEDULE_EXACT_COMMUTE = old_tc, old_ec
    return off, on


def test_terminal_commute_exact_unitary_equivalence():
    """Tail cancellation/reorder uses exact operator commutation only, so the
    WHOLE circuit (identical body + operator-equal tail) must match exactly —
    stronger than the derivative check, global phase included."""
    for seed in [0, 2, 7, 11]:
        ham = _random_ham(5, 12, seed)
        off, on = _compile_terminal_both(ham)
        assert np.allclose(qi.Operator(on).data, qi.Operator(off).data), seed
    print("  terminal-commute: exact unitary equality on 4 seeds")


def test_terminal_commute_no_count_increase_and_cuts_depth():
    """Cancellation can only shrink the 2q count; depth improves on aggregate
    (per-program regressions possible — greedy list-scheduling anomaly)."""
    tot_off = tot_on = 0
    for seed in _EXACT_SEEDS:
        ham = _random_ham(6, 16, seed)
        off, on = _compile_terminal_both(ham)
        (n2o, d2o), (n2n, d2n) = _n2_d2(off), _n2_d2(on)
        assert n2n <= n2o, (seed, n2o, n2n)
        tot_off += d2o
        tot_on += d2n
    assert tot_on < tot_off, (tot_off, tot_on)
    print(f"  terminal-commute: total 2q depth {tot_off}->{tot_on}")


if __name__ == "__main__":
    for fn in [
        test_holistic_correctness_random,
        test_holistic_default_phase_free_equivalence,
        test_holistic_correctness_demo_group,
        test_holistic_oscillating_group_compiles,
        test_holistic_deterministic,
        test_holistic_atom_staging_correctness,
        test_holistic_wide_program_labels,
        test_holistic_termination_bound,
        test_exact_commute_preserves_first_order_trotter,
        test_exact_commute_count_neutral_and_reduces_total_depth,
        test_terminal_commute_exact_unitary_equivalence,
        test_terminal_commute_no_count_increase_and_cuts_depth,
    ]:
        print(fn.__name__)
        fn()
    print("ALL OK")
