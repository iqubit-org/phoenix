"""Tests for the standalone CNOT-equivalent Clifford scheduler
(phoenix/primitive/clifford_schedule.py).

Run directly (``python tests/test_clifford_schedule.py``) or via pytest.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from phoenix.basics import CLIFFORD_OPTIONS, CNOTEquivCliffordGate
from phoenix.primitive.utils import (
    cnot_equiv_commute,
    schedule_cnot_equiv_clifford,
)


def _random_moves_circuit(n, ngates, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(ngates):
        a, b = map(int, rng.choice(n, size=2, replace=False))
        qc.append(CLIFFORD_OPTIONS[int(rng.integers(9))], [a, b])
    return qc


def _depth2q(qc):
    return qc.depth(lambda inst: inst.operation.num_qubits == 2)


def test_rule_matches_qiskit_commutation():
    """The O(1) shared-qubit-axes rule (clifford_pauli_commutation.md) must
    agree with qiskit's matrix-based CommutationChecker on every gate pair and
    relative placement (same pair, swapped, single overlap x4, disjoint)."""
    from qiskit.circuit.commutation_library import SessionCommutationChecker as scc

    placements = [
        ((0, 1), (0, 1)), ((0, 1), (1, 0)),
        ((0, 1), (1, 2)), ((0, 1), (2, 1)),
        ((1, 0), (1, 2)), ((1, 0), (2, 1)),
        ((0, 1), (2, 3)),
    ]
    checked = 0
    for gi in CLIFFORD_OPTIONS:
        for gj in CLIFFORD_OPTIONS:
            for qa, qb in placements:
                want = scc.commute(gi, qa, [], gj, qb, [])
                got = cnot_equiv_commute(gi, qa, gj, qb)
                assert got == want, (gi.name, qa, gj.name, qb, got, want)
                checked += 1
    print(f"  rule vs qiskit CommutationChecker: {checked} cases, all agree")


def test_schedule_exact_operator_equivalence():
    """Rescheduling + cancellation must preserve the operator EXACTLY
    (commutation-only reorders and C.C = I cancellations carry no phase)."""
    for seed, (n, g) in enumerate([(4, 20), (5, 40), (6, 60)]):
        qc = _random_moves_circuit(n, g, seed)
        out = schedule_cnot_equiv_clifford(qc)
        assert np.allclose(Operator(out).data, Operator(qc).data), seed
        assert out.size() <= qc.size(), seed
        assert _depth2q(out) <= _depth2q(qc), seed
        print(f"  seed={seed} ({n}q, {g} gates): exact-equal, "
              f"size {qc.size()}->{out.size()}, depth2q {_depth2q(qc)}->{_depth2q(out)}")


def test_schedule_reduces_depth_crafted():
    """cxx(0,1), cxz(1,2), czz(2,3): pairwise commuting chain (X matches on q1,
    Z matches on q2) that serializes to depth 3 as written but packs to 2."""
    qc = QuantumCircuit(4)
    qc.append(CNOTEquivCliffordGate("x", "x"), [0, 1])
    qc.append(CNOTEquivCliffordGate("x", "z"), [1, 2])
    qc.append(CNOTEquivCliffordGate("z", "z"), [2, 3])
    assert _depth2q(qc) == 3
    out = schedule_cnot_equiv_clifford(qc)
    assert _depth2q(out) == 2, _depth2q(out)
    assert np.allclose(Operator(out).data, Operator(qc).data)
    print(f"  crafted chain: depth2q 3 -> {_depth2q(out)}")


def test_schedule_cancels_self_inverse_pairs():
    """cxx(0,1), cxz(1,2), cxx(0,1): the middle gate commutes with the outer
    pair (X matches on q1), so the two cxx cancel."""
    qc = QuantumCircuit(3)
    qc.append(CNOTEquivCliffordGate("x", "x"), [0, 1])
    qc.append(CNOTEquivCliffordGate("x", "z"), [1, 2])
    qc.append(CNOTEquivCliffordGate("x", "x"), [0, 1])
    out = schedule_cnot_equiv_clifford(qc)
    assert out.size() == 1, out.size()
    assert np.allclose(Operator(out).data, Operator(qc).data)
    print(f"  cancellation: size 3 -> {out.size()}")


def test_schedule_deterministic_and_trivial_cases():
    qc = _random_moves_circuit(5, 30, 7)
    a = schedule_cnot_equiv_clifford(qc)
    b = schedule_cnot_equiv_clifford(qc)
    assert a == b
    empty = QuantumCircuit(3)
    assert schedule_cnot_equiv_clifford(empty).size() == 0
    single = QuantumCircuit(3)
    single.append(CLIFFORD_OPTIONS[0], [0, 2])
    assert schedule_cnot_equiv_clifford(single).size() == 1
    print("  deterministic + trivial cases ok")


def test_schedule_rejects_foreign_gates():
    qc = QuantumCircuit(2)
    qc.h(0)
    try:
        schedule_cnot_equiv_clifford(qc)
    except ValueError:
        print("  foreign gate rejected ok")
    else:
        raise AssertionError("expected ValueError for non-CNOT-equiv gate")


if __name__ == "__main__":
    for fn in [
        test_rule_matches_qiskit_commutation,
        test_schedule_exact_operator_equivalence,
        test_schedule_reduces_depth_crafted,
        test_schedule_cancels_self_inverse_pairs,
        test_schedule_deterministic_and_trivial_cases,
        test_schedule_rejects_foreign_gates,
    ]:
        print(fn.__name__)
        fn()
    print("ALL OK")
