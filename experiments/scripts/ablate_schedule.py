#!/usr/bin/env python
"""V2-A ablation: SCHEDULE_ASAP on/off (holistic terminal='auto' + standard
optimizer). Samples Hamlib categories across the size range + UCCSD + osc.
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
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
UCCSD = os.path.join(REPO, "benchmarks", "uccsd")
SAMPLE_PER_CAT = 7


def run_one(name, ham):
    row = {"name": name, "qubits": ham.num_qubits, "paulis": len(ham.paulis)}
    for arm, flag in [("off", False), ("on", True)]:
        peel_mod.SCHEDULE_ASAP = flag
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            qc = optimize_phoenix_circuit_by_qiskit(holistic_compile(ham))
        dt = time.perf_counter() - t0
        n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
        row[arm] = {"num_2q": n2, "depth_2q": d2, "time": round(dt, 2)}
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
    for cat in ["binaryoptimization", "condensedmatter", "discreteoptimization"]:
        d = os.path.join(HAMLIB, cat)
        progs = []
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                with open(os.path.join(d, f)) as fh:
                    data = json.load(fh)
                progs.append((len(data["paulis"]) * len(data["paulis"][0]), f, data))
        progs.sort()
        idx = np.linspace(0, len(progs) - 1, SAMPLE_PER_CAT).astype(int)
        groups[cat] = [(progs[i][1], progs[i][2]) for i in sorted(set(idx))]

    uccsd_picks = ["LiH_frz_JW_sto3g.json", "LiH_cmplt_BK_sto3g.json", "CH2_frz_P_sto3g.json",
                   "NH_cmplt_JW_sto3g.json", "CH2_cmplt_BK_sto3g.json"]
    groups["chemistry-uccsd"] = []
    for f in uccsd_picks:
        with open(os.path.join(UCCSD, f)) as fh:
            groups["chemistry-uccsd"].append((f, json.load(fh)))

    from test_holistic import OSCILLATING_GROUP

    results = {}
    rows = [run_one("oscillating-152", Hamiltonian(OSCILLATING_GROUP, np.ones(len(OSCILLATING_GROUP))))]
    results["misc"] = rows
    for cat, progs in groups.items():
        results[cat] = []
        for fname, data in progs:
            ham = Hamiltonian(data["paulis"], data["coeffs"])
            results[cat].append(run_one(f"{cat[:12]}/{fname[:-5][:30]}", ham))

    print("\n=== geomean on/off per category ===")
    for cat, rs in results.items():
        g2 = geomean([r["on"]["num_2q"] / r["off"]["num_2q"] for r in rs if r["off"]["num_2q"]])
        gd = geomean([r["on"]["depth_2q"] / r["off"]["depth_2q"] for r in rs if r["off"]["depth_2q"]])
        print(f"  {cat:22s} ({len(rs)})  Num2Q x{g2:.3f}   Depth2Q x{gd:.3f}")

    with open(os.path.join(REPO, "experiments", "ablate_schedule.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved -> experiments/ablate_schedule.json")


if __name__ == "__main__":
    main()
