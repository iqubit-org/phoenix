#!/usr/bin/env python
"""v3 certified-holistic-search ablation matrix:
arms = v2 (V3_HOLISTIC=False) / v3-strict / v3-relaxed,
suite = UCCSD 18 + osc-152 + N2-JW-22 + Na2-JW24.
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

import phoenix.primitive.holistic as peel_mod
from phoenix.compiler import optimize_phoenix_circuit_by_qiskit
from phoenix.hamiltonian import Hamiltonian
from phoenix.primitive.holistic import holistic_compile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARMS = [("v2", False, False), ("v3", True, False), ("v3rx", True, True)]


def run_one(name, ham):
    row = {"name": name}
    for arm, hol, rx in ARMS:
        peel_mod.V3_HOLISTIC = hol
        peel_mod.V3_RELAXED = rx
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            qc = optimize_phoenix_circuit_by_qiskit(holistic_compile(ham))
        dt = time.perf_counter() - t0
        n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
        row[arm] = {"num_2q": n2, "depth_2q": d2, "time": round(dt, 1)}
    peel_mod.V3_HOLISTIC = True
    peel_mod.V3_RELAXED = False
    print(f"{name:22s} " + "  ".join(
        f"{arm}: 2q={row[arm]['num_2q']:6d} d2q={row[arm]['depth_2q']:5d} ({row[arm]['time']:6.1f}s)"
        for arm, _, _ in ARMS), flush=True)
    return row


def main():
    results = []
    from test_holistic import OSCILLATING_GROUP

    results.append(run_one("oscillating-152",
                           Hamiltonian(OSCILLATING_GROUP, np.ones(len(OSCILLATING_GROUP)))))
    ud = os.path.join(REPO, "benchmarks", "uccsd")
    for fname in sorted(os.listdir(ud)):
        if fname.endswith(".json"):
            with open(os.path.join(ud, fname)) as f:
                d = json.load(f)
            results.append(run_one(fname[:-5], Hamiltonian(d["paulis"], d["coeffs"])))
    for big in ["N2-JW-22", "Na2-JW24"]:
        with open(os.path.join(REPO, "benchmarks", "hamlib", "chemistry", big + ".json")) as f:
            d = json.load(f)
        results.append(run_one(big, Hamiltonian(d["paulis"], d["coeffs"])))

    def geomean(xs):
        xs = [x for x in xs if x and np.isfinite(x)]
        return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")

    print("\n=== geomean vs v2 (UCCSD-18 only | all 21) ===")
    uccsd = [r for r in results if "sto3g" in r["name"]]
    for arm, _, _ in ARMS[1:]:
        for label, rs in [("UCCSD", uccsd), ("ALL", results)]:
            g2 = geomean([r[arm]["num_2q"] / r["v2"]["num_2q"] for r in rs])
            gd = geomean([r[arm]["depth_2q"] / r["v2"]["depth_2q"] for r in rs])
            w = sum(1 for r in rs if r[arm]["num_2q"] < r["v2"]["num_2q"])
            print(f"  {arm:5s} {label:6s} Num2Q x{g2:.4f}  Depth2Q x{gd:.4f}  (2q wins {w}/{len(rs)})")

    with open(os.path.join(REPO, "experiments", "ablate_v3.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved -> experiments/ablate_v3.json")


if __name__ == "__main__":
    main()
