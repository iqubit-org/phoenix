#!/usr/bin/env python
"""Benchmark holistic vs support on UCCSD + stress programs.

Usage: python bench_holistic.py [--uccsd] [--osc] [--big PROGRAM.json ...]
Default: --uccsd --osc
"""

import argparse
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
from phoenix.hamiltonian import Hamiltonian

UCCSD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "uccsd")
MODES = ["support", "holistic", "holistic-absorb"]


def metrics(qc):
    n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
    return n2, d2


def run_one(name, ham):
    print(f"--- {name} ({len(ham.paulis)}P, {ham.num_qubits}q) ---", flush=True)
    row = {"name": name, "paulis": len(ham.paulis), "qubits": ham.num_qubits}
    for mode in MODES:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            if mode == "holistic-absorb":
                from phoenix import optimize_phoenix_circuit_by_qiskit
                from phoenix.primitive.holistic import holistic_compile

                qc = optimize_phoenix_circuit_by_qiskit(holistic_compile(ham, terminal="absorb"))
            else:
                qc = phoenix.compile_hamiltonian_simulation(ham, grouping=mode, parallel_search=False)
        dt = time.perf_counter() - t0
        n2, d2 = metrics(qc)
        row[mode] = {"num_2q": n2, "depth_2q": d2, "time": round(dt, 2)}
        print(f"  {mode:9s} 2q={n2:6d}  depth2q={d2:6d}  t={dt:7.1f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uccsd", action="store_true")
    ap.add_argument("--osc", action="store_true")
    ap.add_argument("--big", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.uccsd or args.osc or args.big):
        args.uccsd = args.osc = True

    results = []
    if args.osc:
        from test_holistic import OSCILLATING_GROUP

        results.append(run_one(
            "oscillating-152", Hamiltonian(OSCILLATING_GROUP, np.ones(len(OSCILLATING_GROUP)))
        ))
    if args.uccsd:
        for fname in sorted(os.listdir(UCCSD_DIR)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(UCCSD_DIR, fname)) as f:
                data = json.load(f)
            results.append(run_one(fname[:-5], Hamiltonian(data["paulis"], data["coeffs"])))
    for path in args.big:
        with open(path) as f:
            data = json.load(f)
        results.append(run_one(os.path.basename(path)[:-5], Hamiltonian(data["paulis"], data["coeffs"])))

    # Summary: geomean ratios vs support
    def geomean(xs):
        xs = [x for x in xs if x and np.isfinite(x)]
        return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")

    print("\n=== geomean ratios vs support ===")
    for mode in ["holistic", "holistic-absorb"]:
        g2 = geomean([r[mode]["num_2q"] / r["support"]["num_2q"] for r in results if r["support"]["num_2q"]])
        gd = geomean([r[mode]["depth_2q"] / r["support"]["depth_2q"] for r in results if r["support"]["depth_2q"]])
        gt = geomean([r[mode]["time"] / r["support"]["time"] for r in results if r["support"]["time"]])
        print(f"  {mode:9s} 2q x{g2:.3f}  depth2q x{gd:.3f}  time x{gt:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
