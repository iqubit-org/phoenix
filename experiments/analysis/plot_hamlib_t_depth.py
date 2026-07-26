#!/usr/bin/env python
"""Single scatter: baseline T-depth vs Symphony T-depth over the HamLib suite.

One figure, ``experiments/figures/hamlib_t_depth.pdf``. Each point is one
(baseline, program) pair; marker/colour identify the baseline. Points ABOVE the
y = x diagonal are programs where the baseline needs a deeper T-stage; the shaded
bands mark 1.5x / 2x / 3x / 5x / 10x ratios.

Colours, markers, reference bands and grid are imported from plot_hamlib_result.py
so this figure sits next to hamlib_2q_depth.pdf with an identical visual language
(each baseline keeps the same colour it has in the two-qubit figures).

Input:  experiments/analysis/t_cost_data/{symphony,qiskit,tket,quclear,phoenix}.csv
Run:    python experiments/analysis/plot_hamlib_t_depth.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean

# Imported first: it configures MPLCONFIGDIR and the Agg backend before matplotlib
# is loaded, and gives us the shared style/reference-band helpers.
from plot_hamlib_result import (
    BASELINES,
    KEY,
    OUT_DIR,
    EXP_DIR,
    POINT_ZORDER,
    STYLE,
    axis_limits,
    draw_grid,
    draw_reference,
)

import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path(__file__).resolve().parent / "t_cost_data"
OUT_PATH = OUT_DIR / "hamlib_t_depth.pdf"

METRIC = "t_depth"
HERO_KEY = "symphony"
HERO_LABEL = "Symphony"

# Drawn baselines, in legend order. Labels match plot_hamlib_result.BASELINES so
# STYLE hands back the same colour/marker each one uses in the 2q figures.
T_BASELINES = [("Qiskit", "qiskit"), ("TKET", "tket"), ("QuCLEAR", "quclear"), ("Phoenix", "phoenix")]


def read_metric(key: str) -> pd.DataFrame:
    """Load one compiler's CSV, keyed by (category, program).

    Tolerates a suite that is still running: rows may be in any order, the file
    may cover only part of the 100 programs, and a run killed mid-write can leave
    a truncated final line. Nothing here depends on row order or row count.
    """
    df = pd.read_csv(DATA / f"{key}.csv")
    missing = {*KEY, METRIC} - set(df.columns)
    if missing:
        raise ValueError(f"{key}.csv is missing column(s) {sorted(missing)}")
    df = df[[*KEY, METRIC]].copy()
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[*KEY, METRIC])
    df = df[df[METRIC] > 0]  # log axes; a 0 would silently drop the point anyway
    # A resumed run can in principle append a program twice; keep the newest.
    df = df.drop_duplicates(subset=list(KEY), keep="last")
    return df.rename(columns={METRIC: key})


def load_pairs():
    """Per-baseline (symphony, baseline) point sets, joined on (category, program).

    The join is what makes an incomplete or unsorted CSV harmless: pairing is by
    key, never by position, so QuCLEAR's partial suite simply yields fewer points
    instead of mis-pairing programs.
    """
    hero = read_metric(HERO_KEY)
    data, stats = {}, {}
    for label, key in T_BASELINES:
        path = DATA / f"{key}.csv"
        if not path.exists():
            print(f"  skip {label}: {os.path.relpath(path, EXP_DIR)} not found")
            continue
        merged = hero.merge(read_metric(key), on=list(KEY), how="inner")
        if merged.empty:
            print(f"  skip {label}: no programs in common with {HERO_LABEL}")
            continue
        ratio = float(gmean(merged[key] / merged[HERO_KEY]))
        data[label] = merged.rename(columns={HERO_KEY: "symphony", key: "baseline"})
        stats[label] = {"ratio": ratio, "n": len(merged)}
    return data, stats, len(hero)


def legend_label(label: str, st: dict, n_hero: int) -> str:
    """``Qiskit  (x1.83)``; partial suites additionally carry their program count."""
    # suffix = "" if st["n"] == n_hero else f", n={st['n']}"
    suffix=""
    return f"{label}  (×{st['ratio']:.2f}{suffix})"


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans",
                  rc={"axes.labelsize": 15, "legend.fontsize": 13,
                      "xtick.labelsize": 11, "ytick.labelsize": 11,
                      "pdf.fonttype": 42, "ps.fonttype": 42})

    data, stats, n_hero = load_pairs()
    if not data:
        raise SystemExit("no baseline had any program in common with Symphony")

    x_all = np.concatenate([d["symphony"].to_numpy() for d in data.values()])
    y_all = np.concatenate([d["baseline"].to_numpy() for d in data.values()])
    xlo, xhi = axis_limits(x_all)
    ylo, yhi = axis_limits(y_all)

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    draw_reference(ax, min(xlo, ylo), max(xhi, yhi))

    for label, _ in T_BASELINES:
        if label not in data:
            continue
        points, style = data[label], STYLE[label]
        ax.scatter(
            points["symphony"], points["baseline"],
            marker=style["marker"], s=50, facecolor=style["color"],
            edgecolor="grey", linewidth=0.4, alpha=0.75, zorder=POINT_ZORDER,
            label=legend_label(label, stats[label], n_hero),
        )

    ax.set(xscale="log", yscale="log", xlim=(xlo, xhi), ylim=(ylo, yhi),
           xlabel=f"T-depth ({HERO_LABEL})", ylabel="T-depth (Baseline)")
    draw_grid(ax)
    # Lower right: the only corner with no points (every baseline is at or above
    # the diagonal), so the legend never occludes data.
    ax.legend(title=f"Geomean$\\times$ vs {HERO_LABEL}", title_fontsize=13,
              loc="lower right", frameon=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"saved -> {os.path.relpath(OUT_PATH, EXP_DIR)}")
    for label, _ in T_BASELINES:
        if label in stats:
            st = stats[label]
            above = int((data[label]["baseline"] > data[label]["symphony"]).sum())
            print(f"    {label:9s} geomean baseline/{HERO_LABEL} = x{st['ratio']:.3f}"
                  f"   ({st['n']:3d}/{n_hero} programs, {above} above the diagonal)")


if __name__ == "__main__":
    main()
