#!/usr/bin/env python
"""Summarise the UCCSD scaling benchmark suite (UCC-10 ... UCC-35) for the paper.

Reads benchmarks/uccsd/uccsd_{n}.json, computes per-benchmark #qubits, #Pauli
terms, Pauli-weight (min/max/median over terms), and the naive (per-term)
synthesis two-qubit gate count and circuit depth. Emits a Markdown table and a
LaTeX table matching the HamLib suite table style.

Run:  python experiments/analysis/summarize_uccsd_suite.py
"""
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import phoenix
from phoenix.hamiltonian import Hamiltonian

REPO = Path(__file__).resolve().parents[2]
UCCSD = REPO / "benchmarks" / "uccsd"
OUT_MD = REPO / "docs" / "tables" / "uccsd_suite.md"
OUT_TEX = REPO / "docs" / "tables" / "uccsd_suite_summary.tex"

SIZES = [10, 15, 20, 25, 30, 35]


def build_rows():
    rows = []
    for s in SIZES:
        d = json.load(open(UCCSD / f"uccsd_{s}.json"))
        paulis = d["paulis"]
        w = np.array([sum(1 for c in p if c != "I") for p in paulis])
        qc = Hamiltonian(paulis, d["coeffs"]).generate_circuit()   # naive synthesis
        n2 = sum(1 for i in qc.data if i.operation.num_qubits == 2)
        d2 = qc.depth(lambda i: i.operation.num_qubits == 2)
        rows.append({
            "Benchmark": f"UCC-{s}",
            "\\#Qubit": len(paulis[0]),
            "\\#Pauli": len(paulis),
            "Pauli weight (min/max/median)": f"{w.min()}/{w.max()}/{int(np.median(w))}",
            "2Q gate count": n2,
            "2Q circuit depth": d2,
        })
    return pd.DataFrame(rows)


def to_latex(tbl):
    cols = list(tbl.columns)
    align = "|" + "|".join(["l"] + ["c"] * (len(cols) - 1)) + "|"
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline",
             " & ".join(cols) + " \\\\", "\\hline"]
    for _, r in tbl.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def to_markdown(tbl):
    cols = list(tbl.columns)
    hdr = [c.replace("\\#", "#") for c in cols]
    lines = ["| " + " | ".join(hdr) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in tbl.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    tbl = build_rows()
    print(to_markdown(tbl))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(to_markdown(tbl))
    OUT_TEX.write_text(to_latex(tbl))
    print(f"saved -> {os.path.relpath(OUT_MD, REPO)}")
    print(f"saved -> {os.path.relpath(OUT_TEX, REPO)}")


if __name__ == "__main__":
    main()
