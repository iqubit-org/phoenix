#!/usr/bin/env python
"""Summarise the HamLib benchmark suite (100 Hamiltonians, Benchpress selection)
for inclusion in the paper.

Reads benchmarks/description_hamlib.csv (per-program: qubits, #Pauli terms,
max Pauli weight, naive gate/depth counts) and emits a per-problem-class
summary as a Markdown table and a LaTeX table (booktabs).

Run:  python experiments/analysis/summarize_hamlib_suite.py
"""
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "benchmarks" / "description_hamlib.csv"
OUT_MD = REPO / "docs" / "tables" / "hamlib_suite.md"
OUT_TEX = REPO / "docs" / "tables" / "hamlib_suite_summary.tex"

# problem-class display names + order (matches the Benchpress description)
CLASSES = [
    ("chemistry", "Chemistry"),
    ("condensedmatter", "Condensed matter"),
    ("discreteoptimization", "Discrete optimization"),
    ("binaryoptimization", "Binary optimization"),
]


def rng(series):
    """min--max string, integer."""
    return f"{int(series.min())}--{int(series.max())}"


def med(series):
    return f"{int(round(series.median()))}"


def _row(name, g):
    return {
        "Category": name,
        "\\#Ham.": len(g),
        "\\#Qubit": rng(g["num_qubits"]),
        "\\#Pauli": rng(g["num_paulis"]),
        "Weight": rng(g["max_pauli_weight"]),
        "2Q gate count": rng(g["num_2q_gates"]),
        "2Q circuit depth": rng(g["depth_2q"]),
    }


def build_rows(df):
    rows = [_row(name, df[df["category"] == key]) for key, name in CLASSES]
    rows.append({**_row("\\emph{All}", df), "\\#Ham.": len(df)})
    return pd.DataFrame(rows)


def to_markdown(tbl):
    cols = list(tbl.columns)
    hdr = [c.replace("\\#", "#").replace("\\textbf{", "").replace("}", "") for c in cols]
    lines = ["| " + " | ".join(hdr) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in tbl.iterrows():
        cells = [str(r[c]).replace("\\textbf{", "**").replace("\\emph{", "**")
                 .replace("}", "**").replace("--", "–") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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


def main():
    df = pd.read_csv(CSV)
    assert len(df) == 100, f"expected 100 Hamiltonians, got {len(df)}"
    tbl = build_rows(df)

    md = to_markdown(tbl)
    tex = to_latex(tbl)
    print(md)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    OUT_TEX.write_text(tex)
    print(f"saved -> {os.path.relpath(OUT_MD, REPO)}")
    print(f"saved -> {os.path.relpath(OUT_TEX, REPO)}")


if __name__ == "__main__":
    main()
