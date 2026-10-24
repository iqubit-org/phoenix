#!/usr/bin/env python
"""Sensitivity of Symphony to the adaptive-emission density threshold rho.

One figure, ``experiments/figures/hamlib_rho_scan.pdf``, with two panels
(two-qubit gate count, two-qubit depth).  The x axis is ``rho_threshold``; the
y axis is the geometric-mean ratio of the metric against the default
``rho_threshold=0.35`` (below 1 is better than the default), with one curve per
HamLib category and a heavier black one for the whole suite.  The dashed
vertical line marks the default; the solid hairline at 1 marks parity with it.

Each program's metrics are a step function of the threshold (the emission
predicate only compares it against the active tableau density), so the curves
are sampled step functions: a flat stretch around the default means the
default is not a delicately tuned value.

Theme and grid are shared with the other HamLib figures.

Input:  experiments/analysis/ablation_rho_scan/rho_scan.csv
        (written by ablate_rho_threshold_scan.py)
Run:    python experiments/analysis/plot_rho_threshold.py
"""
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gmean

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# Shared style helpers.
from plot_hamlib_result import EXP_DIR, KEY, OUT_DIR, draw_grid

DATA = Path(__file__).resolve().parent / "ablation_rho_scan" / "rho_scan.csv"
OUT_PATH = OUT_DIR / "hamlib_rho_scan.pdf"

REFERENCE_RHO = 0.35

# (results key, legend label, colour, marker); drawn in this order, the suite-wide
# curve last.  The four hues are the blue / yellow / magenta / green slots of the
# dataviz reference palette (hamlib_pareto.pdf draws from the same one): every
# curve can cross every other, and this is the four-hue subset that clears the
# colour-blind and normal-vision separation checks on all pairs.
CATEGORIES = [
    ("binaryoptimization", "Binary", "#2a78d6", "o"),
    ("discreteoptimization", "Discrete", "#eda100", "s"),
    ("chemistry", "Chem.", "#e87ba4", "^"),
    ("condensedmatter", "Cond.", "#008300", "D"),
]
ALL_LABEL = "All"
ALL_COLOR = "#222222"
REFERENCE_INK = "#898781"

# (column, panel title)
METRICS = [
    ("num_2q", "Two-qubit gate count"),
    ("depth_2q", "Two-qubit depth"),
]

Y_TICKS = [0.8, 1.0, 1.5, 2.0, 3.0, 4.0]


def load_ratios() -> pd.DataFrame:
    """Per-(program, rho) ratio of each metric against the reference threshold."""
    df = pd.read_csv(DATA)
    df = df.rename(columns={"name": "program"})
    df["rho_threshold"] = df["rho_threshold"].round(6)
    reference = df[np.isclose(df["rho_threshold"], REFERENCE_RHO)]
    if reference.empty:
        raise SystemExit(f"{DATA} holds no rho_threshold={REFERENCE_RHO} rows")
    metrics = [metric for metric, _ in METRICS]
    merged = df.merge(reference[[*KEY, *metrics]], on=list(KEY), suffixes=("", "_ref"))
    for metric in metrics:
        valid = (merged[metric] > 0) & (merged[f"{metric}_ref"] > 0)
        merged[f"{metric}_ratio"] = np.where(valid, merged[metric] / merged[f"{metric}_ref"], np.nan)
    return merged


def geomean_curve(ratios: pd.DataFrame, metric: str) -> pd.Series:
    """Geomean ratio per threshold, over the programs available at that threshold."""
    column = f"{metric}_ratio"
    return ratios.dropna(subset=[column]).groupby("rho_threshold")[column].apply(gmean)


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans",
                  rc={"axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11,
                      "xtick.labelsize": 11, "ytick.labelsize": 11,
                      "pdf.fonttype": 42, "ps.fonttype": 42})

    ratios = load_ratios()
    n_rho = ratios["rho_threshold"].nunique()
    n_programs = len(ratios[list(KEY)].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), sharey=True, layout="constrained")
    for ax, (metric, title) in zip(axes, METRICS):
        ax.axhline(1.0, color=REFERENCE_INK, lw=0.9, zorder=2)
        ax.axvline(REFERENCE_RHO, color=REFERENCE_INK, lw=0.9, ls=(0, (4, 3)), zorder=2)

        for key, label, color, marker in CATEGORIES:
            curve = geomean_curve(ratios[ratios["category"] == key], metric)
            ax.plot(curve.index, curve.values, color=color, lw=1.45, marker=marker,
                    markersize=4.5, markeredgewidth=0, zorder=3, label=label)
        curve = geomean_curve(ratios, metric)
        ax.plot(curve.index, curve.values, color=ALL_COLOR, lw=2.0, marker="o",
                markersize=4.5, markeredgewidth=0, zorder=4, label=ALL_LABEL)

        ax.set(yscale="log", xlim=(-0.03, 1.03), title=title)
        ax.set_xlabel(r"Density threshold $\rho$")
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        draw_grid(ax)
        ax.xaxis.grid(False, which="both")  # horizontal reference lines only
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)

    # Per program: metric at rho / metric at the default; geomean per category.
    axes[0].set_ylabel(rf"Relative to $\rho$ = {REFERENCE_RHO}")
    # The default sits at 0.35, so the label goes just right of the line, at the top.
    axes[0].annotate("default", (REFERENCE_RHO, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                     fontsize=10, color="#52514e")
    # One shared key above both panels keeps the plotting area free.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(handles),
               frameon=False, handlelength=1.8, columnspacing=1.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"saved -> {os.path.relpath(OUT_PATH, EXP_DIR)}   ({n_programs} programs x {n_rho} thresholds)")
    for metric, title in METRICS:
        curve = geomean_curve(ratios, metric)
        print(f"    {title:22s} suite geomean vs rho={REFERENCE_RHO}: "
              f"min x{curve.min():.3f} at rho={curve.idxmin():.2f}, "
              f"max x{curve.max():.3f} at rho={curve.idxmax():.2f}")


if __name__ == "__main__":
    main()
