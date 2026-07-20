#!/usr/bin/env python
"""Pareto scatter of Phoenix++ vs SOTA on HamLib (100 programs, all-to-all).

Each compiler is one point: x = 2-qubit *depth* optimization rate, y = 2-qubit
*gate-count* optimization rate, both = geomean over the 100 per-program ratios
(opt / naive), i.e. exactly the ``All (100)`` row of ``display_results_hamlib``.
Lower is better on both axes, so the best compiler sits at the bottom-left.
Phoenix++ (our holistic engine, CSV key ``phoenixpp``) is the sole point in the
region that Pareto-dominates every baseline; the old published Phoenix
(support-grouped DAC'25 version, CSV key ``phoenix``) is one of the baselines,
drawn automatically once its results CSV exists.

Run from ``experiments/``:  python plot_hamlib_effect.py
Output: figures/hamlib_pareto.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import gmean

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures", "hamlib_pareto.pdf")

HERO = "Phoenix++"
N_PROGRAMS = 100  # a compiler is plotted only if its CSV has the full suite
# display label -> results CSV key
COMPILERS = [
    ("Qiskit", "qiskit"),
    ("TKet", "tket"),
    ("Paulihedral", "paulihedral"),
    ("Tetris", "tetris"),
    ("QuCLEAR", "quclear"),
    ("Phoenix", "phoenix"),
    ("Phoenix++", "phoenixpp"),
]

# --- palette (dataviz reference instance, light surface) --------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
ACCENT = "#2a78d6"          # Phoenix (categorical slot 1 / blue)
FIELD = "#9a998f"           # baseline field (muted gray)
SHADE = "#cde2fb"           # dominated-region tint (blue-100)


def opt_rate(df, opt, orig):
    return float(gmean(df[opt] / df[orig]))


def load():
    """(label -> (depth_rate, count_rate)) for every compiler whose CSV exists,
    geomean over whatever programs it contains (same convention as
    display_results_hamlib). Compilers not yet run — e.g. old Phoenix before its
    support benchmark — have no CSV and are silently skipped."""
    pts = {}
    for label, key in COMPILERS:
        path = os.path.join(RESULTS, f"result_hamlib_{key}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        if len(df) < N_PROGRAMS:
            print(f"  (note {label}: {len(df)}/{N_PROGRAMS} programs in CSV)")
        pts[label] = (opt_rate(df, "depth_2q(opt)", "depth_2q"),      # x
                      opt_rate(df, "num_2q_gates(opt)", "num_2q_gates"))  # y
    return pts


# label placement: (ha, dx, dy) in points, hand-tuned to avoid collisions
LABELS = {
    "Qiskit":      ("left",   9,  1),
    "Tetris":      ("right", -9,  4),
    "QuCLEAR":     ("right", -9,  0),
    "TKet":        ("left",   9,  1),
    "Paulihedral": ("left",   9, -1),
    "Phoenix":     ("left",   9,  4),
    "Phoenix++":   ("left",  11, -3),
}


def main():
    pts = load()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11.5, "axes.linewidth": 0.9,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    if HERO not in pts:
        raise SystemExit(
            f"No full {HERO} results (results/result_hamlib_phoenixpp.csv with "
            f"{N_PROGRAMS} programs). Run `make -f Makefile-Hamlib phoenixpp` and "
            "`python summarize_results_hamlib.py -c phoenixpp` first."
        )
    px, py = pts[HERO]
    xlim, ylim = (0.05, 0.58), (0.42, 0.66)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # region Pareto-dominated by Phoenix++: worse (higher) on both axes
    ax.add_patch(Rectangle((px, py), xlim[1] - px, ylim[1] - py,
                           facecolor=SHADE, alpha=0.45, edgecolor="none", zorder=0))
    ax.annotate(f"Pareto-dominated\nby {HERO}", xy=(0.205, 0.588),
                color=MUTED, fontsize=10, style="italic", ha="center",
                va="center", linespacing=1.3, zorder=1)

    # baselines (recessive field) then Phoenix++ (hero) on top
    for name, (x, y) in pts.items():
        hero = name == HERO
        ax.scatter(x, y, s=360 if hero else 96,
                   marker="*" if hero else "o",
                   facecolor=ACCENT if hero else "white",
                   edgecolor=ACCENT if hero else FIELD,
                   linewidth=1.4, zorder=5 if hero else 4)
        ha, dx, dy = LABELS.get(name, ("left", 9, 1))
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=11,
                    color=ACCENT if hero else INK,
                    fontweight="bold" if hero else "normal", zorder=6)

    # chrome
    ax.set_xlabel("2-qubit depth  ·  optimization rate (lower is better)",
                  color=INK, fontsize=11.5)
    ax.set_ylabel("2-qubit gate count  ·  optimization rate (lower is better)",
                  color=INK, fontsize=11.5)
    ax.set_title("Phoenix++ Pareto-dominates SOTA on HamLib",
                 color=INK, fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)

    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=ACCENT,
               markeredgecolor=ACCENT, markersize=16, label="Phoenix++ (ours)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=FIELD, markeredgewidth=1.4, markersize=9,
               label="Baseline compilers"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=10.5, handletextpad=0.4, labelcolor=INK,
              bbox_to_anchor=(0.005, 0.99))

    ax.annotate("n = 100 programs · all-to-all · geomean of per-program ratios",
                xy=(0.5, -0.15), xycoords="axes fraction", ha="center",
                color=MUTED, fontsize=8.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved -> {os.path.relpath(OUT, HERE)}")
    print("points (depth, count):")
    for name, (x, y) in pts.items():
        print(f"  {name:12s} ({x:.3f}, {y:.3f})")


if __name__ == "__main__":
    main()
