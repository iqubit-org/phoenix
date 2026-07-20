#!/usr/bin/env python
"""Ablation: SCHEDULE_EXACT_COMMUTE on/off (holistic engine, terminal='auto' +
O3 post-optimization).

Measures the effect of exact-commutation-aware ASAP scheduling -- relaxing the
item DAG by dropping provably-commuting move<->move and move<->block ordering
constraints (docs/clifford_pauli_commutation.md). It is a pure reorder: 2q count
is (essentially) unchanged and it never increases 2q depth. The win is family-
dependent -- it needs Clifford *moves*, so diagonal/QAOA programs see 0.

Samples Hamlib categories across the size range + a UCCSD chemistry set + the
152-row oscillating stress group. Reports per-category geomean on/off ratios.
"""

import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import phoenix
import phoenix.primitive.holistic as holistic_mod
from phoenix.hamiltonian import Hamiltonian

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
UCCSD = os.path.join(REPO, "benchmarks", "uccsd")
SAMPLE_PER_CAT = 7


def _metrics(qc):
    n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
    return n2, d2


def run_one(name, ham):
    row = {"name": name, "qubits": ham.num_qubits, "paulis": len(ham.paulis)}
    for arm, flag in [("off", False), ("on", True)]:
        holistic_mod.SCHEDULE_EXACT_COMMUTE = flag
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            qc = phoenix.compile_hamiltonian_simulation(ham)  # optimize=True (O3)
        dt = time.perf_counter() - t0
        n2, d2 = _metrics(qc)
        row[arm] = {"num_2q": n2, "depth_2q": d2, "time": round(dt, 2)}
    holistic_mod.SCHEDULE_EXACT_COMMUTE = False
    dd = 100 * (row["on"]["depth_2q"] - row["off"]["depth_2q"]) / max(row["off"]["depth_2q"], 1)
    d2q = 100 * (row["on"]["num_2q"] - row["off"]["num_2q"]) / max(row["off"]["num_2q"], 1)
    print(f"{name:44s} 2q {row['off']['num_2q']:6d}->{row['on']['num_2q']:6d} ({d2q:+5.1f}%)  "
          f"d2q {row['off']['depth_2q']:6d}->{row['on']['depth_2q']:6d} ({dd:+6.1f}%)", flush=True)
    return row


def geomean(xs):
    xs = [x for x in xs if x and np.isfinite(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def main():
    groups = {}
    if os.path.isdir(HAMLIB):
        for cat in ["binaryoptimization", "condensedmatter", "discreteoptimization"]:
            d = os.path.join(HAMLIB, cat)
            if not os.path.isdir(d):
                continue
            progs = []
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    with open(os.path.join(d, f)) as fh:
                        data = json.load(fh)
                    progs.append((len(data["paulis"]) * len(data["paulis"][0]), f, data))
            progs.sort()
            idx = np.linspace(0, len(progs) - 1, SAMPLE_PER_CAT).astype(int)
            groups[cat] = [(progs[i][1], progs[i][2]) for i in sorted(set(idx))]

    groups["chemistry-uccsd"] = []
    for f in sorted(os.listdir(UCCSD)):
        if f.endswith(".json"):
            with open(os.path.join(UCCSD, f)) as fh:
                groups["chemistry-uccsd"].append((f, json.load(fh)))

    from test_holistic import OSCILLATING_GROUP

    results = {"misc": [run_one("oscillating-152",
                                Hamiltonian(OSCILLATING_GROUP, np.ones(len(OSCILLATING_GROUP))))]}
    for cat, progs in groups.items():
        results[cat] = []
        for fname, data in progs:
            ham = Hamiltonian(data["paulis"], data["coeffs"])
            results[cat].append(run_one(f"{cat[:12]}/{fname[:-5][:30]}", ham))

    print("\n=== geomean on/off per category (x<1.0 = exact-commute wins) ===")
    for cat, rs in results.items():
        g2 = geomean([r["on"]["num_2q"] / r["off"]["num_2q"] for r in rs if r["off"]["num_2q"]])
        gd = geomean([r["on"]["depth_2q"] / r["off"]["depth_2q"] for r in rs if r["off"]["depth_2q"]])
        print(f"  {cat:22s} ({len(rs)})  Num2Q x{g2:.3f}   Depth2Q x{gd:.3f}")

    out = os.path.join(REPO, "experiments", "ablate_commute.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
