#!/usr/bin/env python
"""Compare old (git HEAD = support+subset era) vs new (working tree = peel)
phoenix Hamlib outputs: per-category Num2Q / Depth2Q geomean ratios.
"""

import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit, qasm2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CATS = ["binaryoptimization", "condensedmatter", "discreteoptimization", "chemistry"]


def metrics(qc):
    n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    d2 = qc.depth(lambda inst: inst.operation.num_qubits == 2)
    return n2, d2


def geomean(xs):
    xs = [x for x in xs if x and np.isfinite(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def main():
    out = subprocess.run(
        ["git", "-C", REPO, "status", "--short", "experiments/output_hamlib/phoenix"],
        capture_output=True, text=True,
    ).stdout
    files = [line[3:].strip() for line in out.splitlines() if line.startswith(" M")]

    by_cat = {c: [] for c in CATS}
    for rel in files:
        cat = next((c for c in CATS if f"/{c}/" in rel), None)
        if cat is None:
            continue
        try:
            old_src = subprocess.run(
                ["git", "-C", REPO, "show", f"HEAD:{rel}"], capture_output=True, text=True
            ).stdout
            qc_old = qasm2.loads(old_src)
            qc_new = QuantumCircuit.from_qasm_file(os.path.join(REPO, rel))
        except Exception as e:
            print(f"  skip {rel}: {e}", flush=True)
            continue
        o2, od = metrics(qc_old)
        n2, nd = metrics(qc_new)
        by_cat[cat].append((os.path.basename(rel)[:-5], o2, od, n2, nd))
        print(f"  {cat[:6]} {os.path.basename(rel)[:40]:42s} "
              f"2q {o2:6d}->{n2:6d}  d2q {od:6d}->{nd:6d}", flush=True)

    print("\n=== per-category geomean (new/old; <1 = peel better) ===")
    for cat in CATS:
        rows = by_cat[cat]
        if not rows:
            continue
        g2 = geomean([n / o for _, o, _, n, _ in rows if o])
        gd = geomean([nd / od for _, _, od, _, nd in rows if od])
        print(f"  {cat:22s} ({len(rows):3d} files)  Num2Q x{g2:.3f}   Depth2Q x{gd:.3f}")


if __name__ == "__main__":
    main()
