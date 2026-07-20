#!/usr/bin/env python
"""Full ablation: SCHEDULE_EXACT_COMMUTE off vs on across ALL 100 Hamlib
benchmarks (binaryopt 15 / chemistry 35 / condensedmatter 35 / discreteopt 15).

For each program, compile with the holistic engine + O3 post-optimization under
both settings and record Num2Q and Depth2Q. Reports per-category and overall
geomean ratios, win/loss/tie counts, distribution, best wins / worst regressions,
and total compile cost. Result dumped to experiments/ablate_commute_full.json.
"""

import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import phoenix
import phoenix.primitive.holistic as holistic_mod
from phoenix.hamiltonian import Hamiltonian

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
CATS = ["binaryoptimization", "chemistry", "condensedmatter", "discreteoptimization"]


def _metrics(qc):
    n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
    return n2, d2


def compile_metrics(ham, flag):
    holistic_mod.SCHEDULE_EXACT_COMMUTE = flag
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            qc = phoenix.compile_hamiltonian_simulation(ham)  # optimize=True (O3)
        return _metrics(qc)
    finally:
        holistic_mod.SCHEDULE_EXACT_COMMUTE = False


def run_all():
    rows = []
    idx = 0
    for cat in CATS:
        d = os.path.join(HAMLIB, cat)
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            idx += 1
            data = json.load(open(os.path.join(d, fname)))
            ham = Hamiltonian(data["paulis"], data["coeffs"])
            t0 = time.perf_counter()
            try:
                n2o, d2o = compile_metrics(ham, False)
                n2n, d2n = compile_metrics(ham, True)
            except Exception as e:  # noqa
                print(f"[{idx:3d}] SKIP {cat[:8]}/{fname[:34]}: {type(e).__name__}: {e}", flush=True)
                continue
            dt = time.perf_counter() - t0
            row = {"cat": cat, "name": fname[:-5], "q": ham.num_qubits, "P": len(ham.paulis),
                   "n2_off": n2o, "n2_on": n2n, "d2_off": d2o, "d2_on": d2n, "sec": round(dt, 1)}
            rows.append(row)
            dd = 100 * (d2n - d2o) / d2o if d2o else 0.0
            dn = 100 * (n2n - n2o) / n2o if n2o else 0.0
            print(f"[{idx:3d}] {cat[:8]:8s}/{fname[:-5][:30]:30s} q={ham.num_qubits:3d} "
                  f"2q {n2o:6d}->{n2n:6d}({dn:+4.1f}%) d2q {d2o:6d}->{d2n:6d}({dd:+6.1f}%) {dt:5.1f}s",
                  flush=True)
    return rows


def geomean(xs):
    xs = [x for x in xs if x and np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def summarize(rows):
    def bucket(rs, key_on, key_off):
        ratios = [r[key_on] / r[key_off] for r in rs if r[key_off]]
        g = geomean(ratios)
        wins = sum(1 for x in ratios if x < 0.999)
        losses = sum(1 for x in ratios if x > 1.001)
        ties = len(ratios) - wins - losses
        return g, wins, losses, ties

    print("\n" + "=" * 78)
    print("SUMMARY  (on = SCHEDULE_EXACT_COMMUTE True; ratio<1 => exact-commute wins)")
    print("=" * 78)
    print(f"{'category':22s} {'N':>3s}  {'Depth2Q geomean':>15s}  {'W/L/T':>10s}  {'Num2Q geo':>10s}")
    for cat in CATS + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["cat"] == cat]
        if not rs:
            continue
        gd, wd, ld, td = bucket(rs, "d2_on", "d2_off")
        gn, _, _, _ = bucket(rs, "n2_on", "n2_off")
        print(f"{cat:22s} {len(rs):3d}  {'x%.4f' % gd:>15s}  {f'{wd}/{ld}/{td}':>10s}  {'x%.4f' % gn:>10s}")

    all_dr = [(r["d2_on"] / r["d2_off"], r) for r in rows if r["d2_off"]]
    all_dr.sort(key=lambda t: t[0])
    print("\nTop 8 depth WINS:")
    for ratio, r in all_dr[:8]:
        print(f"  {r['cat'][:8]:8s}/{r['name'][:34]:34s} d2q {r['d2_off']:6d}->{r['d2_on']:6d} ({100*(ratio-1):+6.1f}%)")
    print("Top 5 depth REGRESSIONS:")
    for ratio, r in [x for x in reversed(all_dr) if x[0] > 1.001][:5]:
        print(f"  {r['cat'][:8]:8s}/{r['name'][:34]:34s} d2q {r['d2_off']:6d}->{r['d2_on']:6d} ({100*(ratio-1):+6.1f}%)")

    depth_pct = [100 * (r["d2_on"] / r["d2_off"] - 1) for r in rows if r["d2_off"]]
    print(f"\nDepth2Q %change: mean {np.mean(depth_pct):+.2f}  median {np.median(depth_pct):+.2f}  "
          f"min {min(depth_pct):+.1f}  max {max(depth_pct):+.1f}")
    n2_changed = sum(1 for r in rows if r["n2_on"] != r["n2_off"])
    print(f"Num2Q changed on {n2_changed}/{len(rows)} programs (pure reorder -> count ~neutral)")
    print(f"Total compile wall time: {sum(r['sec'] for r in rows):.0f}s over {len(rows)} programs (x2 compiles each)")


if __name__ == "__main__":
    t0 = time.perf_counter()
    rows = run_all()
    out = os.path.join(REPO, "experiments", "ablate_commute_full.json")
    json.dump(rows, open(out, "w"), indent=1)  # save BEFORE summarize (crash-safe)
    summarize(rows)
    print(f"\nsaved -> {out}   (total {time.perf_counter()-t0:.0f}s)")
