"""Differential tests: the vectorized Clifford search (``parallel=True``) must
be bit-identical to the naive per-candidate reference (``parallel=False``).

``search_best_clifford_par`` only changes *how* the per-candidate heuristic
cost is computed (histogram/closed-form decomposition instead of materializing
each candidate tableau); it must select exactly the same sequence of
(Clifford, qubit-pair) operations and produce the same final tableau as the
sequential reference search.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import phoenix
from phoenix.primitive.simplification import simplify_hamiltonian

_PAULIS = "IXYZ"


def _random_ham(seed: int, m: int, n: int, min_weight: int, max_weight: int) -> phoenix.Hamiltonian:
    """Build a reproducible Hamiltonian with row weights in [min_weight, max_weight]."""
    rng = np.random.default_rng(seed)
    labels: set[str] = set()
    while len(labels) < m:
        w = int(rng.integers(min_weight, max_weight + 1))
        pos = rng.choice(n, size=w, replace=False)
        arr = np.zeros(n, dtype=int)
        arr[pos] = rng.integers(1, 4, size=w)
        labels.add("".join(_PAULIS[i] for i in arr))
    ordered = sorted(labels)
    coeffs = rng.random(len(ordered)) + 0.1
    return phoenix.Hamiltonian(ordered, coeffs)


def _steps_signature(steps) -> list[tuple[str, tuple[int, ...]]]:
    return [(s.clifford.name, tuple(int(q) for q in s.qubits)) for s in steps]


def _assert_identical(ham: phoenix.Hamiltonian) -> None:
    ref_ham, ref_steps = simplify_hamiltonian(ham, parallel=False)
    par_ham, par_steps = simplify_hamiltonian(ham, parallel=True)

    assert _steps_signature(par_steps) == _steps_signature(ref_steps)
    assert np.array_equal(par_ham.paulis.x, ref_ham.paulis.x)
    assert np.array_equal(par_ham.paulis.z, ref_ham.paulis.z)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_par_matches_reference_low_weight(seed: int) -> None:
    # Weight 2-4 rows exercise the general path (rows can drop to local).
    ham = _random_ham(seed=seed, m=12, n=8, min_weight=2, max_weight=4)
    _assert_identical(ham)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_par_matches_reference_with_local_rows(seed: int) -> None:
    # Weight-1 rows present from the start (local rows fed through grouping).
    ham = _random_ham(seed=seed, m=12, n=8, min_weight=1, max_weight=4)
    _assert_identical(ham)


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_par_matches_reference_nonlocal_stable(seed: int) -> None:
    # All rows weight >= 5: exercises the nonlocal-stable fast path dispatch.
    ham = _random_ham(seed=seed, m=8, n=12, min_weight=5, max_weight=8)
    _assert_identical(ham)


_GRAY = (
    "./benchmarks/hamlib/discreteoptimization/"
    "gray-color02-1-fullins_5_k-5-1-fullins_5.json"
)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_GRAY), reason="benchmark file not present")
def test_par_matches_reference_real_gray_subgroup() -> None:
    """A small subgroup of the slow gray-coloring benchmark (weight 3-4 rows)."""
    with open(_GRAY) as f:
        data = json.load(f)
    ham = phoenix.Hamiltonian(data["paulis"], data["coeffs"])
    groups = ham.group_same_weights(subset=True)
    # pick a mid-sized group so the reference search stays fast
    group = sorted(groups, key=lambda h: len(h.active_qubits))[-7]
    _assert_identical(group)
