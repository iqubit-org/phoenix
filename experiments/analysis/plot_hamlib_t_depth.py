#!/usr/bin/env python
"""Baseline vs Symphony T-depth over the HamLib suite: scatter + violin.

Two figures in ``experiments/figures``:

* ``hamlib_t_depth_scatter.pdf`` — each point is one (baseline, program) pair;
  marker/colour identify the baseline. Points ABOVE the y = x diagonal are
  programs where the baseline needs a deeper T-stage; the shaded bands mark
  1.5x / 2x / 3x / 5x / 10x ratios.
* ``hamlib_t_depth_violin.pdf`` — the same pairs collapsed to per-program
  ratios (baseline / Symphony), one violin per baseline on a log axis. The
  dashed parity line at 1 is the scatter's diagonal, the shaded horizontal
  bands are its 1.5x .. 10x bands, and the white diamond marks the geomean.

Colours, markers, reference bands and grid are imported from plot_hamlib_result.py
so these figures sit next to hamlib_2q_depth.pdf with an identical visual
language (each baseline keeps the same colour it has in the two-qubit figures).

Input:  experiments/analysis/t_cost_data/{symphony,qiskit,tket,paulihedral,tetris,quclear,phoenix}.csv
Run:    python experiments/analysis/plot_hamlib_t_depth.py
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

# Shared style and reference-band helpers.
from plot_hamlib_result import (
    EXP_DIR,
    KEY,
    OUT_DIR,
    POINT_ZORDER,
    REFERENCE_BAND_ZORDER,
    REFERENCE_BANDS,
    REFERENCE_COLOR,
    REFERENCE_DIVIDER_COLOR,
    REFERENCE_LINE_ZORDER,
    STYLE,
    axis_limits,
    draw_grid,
    draw_reference,
)

DATA = Path(__file__).resolve().parent / "t_cost_data"
SCATTER_PATH = OUT_DIR / "hamlib_t_depth_scatter.pdf"
VIOLIN_PATH = OUT_DIR / "hamlib_t_depth_violin.pdf"

METRIC = "t_depth"
HERO_KEY = "symphony"
HERO_LABEL = "Symphony"

# Drawn baselines, in legend order. Labels match plot_hamlib_result.BASELINES so
# STYLE hands back the same colour/marker each one uses in the 2q figures.
T_BASELINES = [
    ("Qiskit", "qiskit"),
    ("TKET", "tket"),
    ("Paulihedral", "paulihedral"),
    ("Tetris", "tetris"),
    ("QuCLEAR", "quclear"),
    ("Phoenix", "phoenix"),
]


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


def plot_scatter(data, stats, n_hero) -> None:
    x_all = np.concatenate([d["symphony"].to_numpy() for d in data.values()])
    y_all = np.concatenate([d["baseline"].to_numpy() for d in data.values()])
    xlo, xhi = axis_limits(x_all)
    ylo, yhi = axis_limits(y_all)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
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

    fig.savefig(SCATTER_PATH, bbox_inches="tight")
    plt.close(fig)


def plot_violin(data, stats) -> None:
    """One violin per baseline over the per-program ratio baseline/Symphony.

    The ratio is the scatter's distance-to-diagonal, so the parity line here IS
    the diagonal there, and the shaded bands are draw_reference's bands laid
    flat (same colour, same stacked alpha, so 1x-1.5x is the darkest). KDE
    runs in log space (log_scale=True); cut=0 keeps it inside the data range,
    which the heavy upper tails (ratios into the hundreds) would otherwise
    smear far past the observed maximum. Each body carries dashed quartile
    segments (q25/q75), a solid median and the geomean diamond.
    """
    order = [label for label, _ in T_BASELINES if label in data]
    long = pd.concat(
        [pd.DataFrame({"Baseline": label,
                       "ratio": data[label]["baseline"] / data[label]["symphony"]})
         for label in order],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(7, 3.85), constrained_layout=True)

    ax.axhline(1.0, color="#222222", lw=1.5, ls="--", alpha=0.75,
               zorder=REFERENCE_LINE_ZORDER)
    for k in REFERENCE_BANDS:                       # ratio in [1, k]  <=>  y in [x, k*x]
        ax.axhspan(1.0, k, color=REFERENCE_COLOR, alpha=0.10, lw=0,
                   zorder=REFERENCE_BAND_ZORDER)
    for k in REFERENCE_BANDS:
        ax.axhline(k, color=REFERENCE_DIVIDER_COLOR, lw=0.55, alpha=0.45,
                   zorder=REFERENCE_LINE_ZORDER)

    # inner="quartile" lays a dashed segment across the body at q25, the median
    # and q75, each clipped to the silhouette's width at that height.
    n_before = len(ax.lines)  # the parity/band axhlines drawn above
    sns.violinplot(
        data=long, x="Baseline", y="ratio", order=order,
        hue="Baseline", palette={label: STYLE[label]["color"] for label in order},
        legend=False, log_scale=True, cut=0, density_norm="width", width=0.75,
        inner="quartile", linewidth=0.9,
        inner_kws={"color": "0.35", "linewidth": 1.3, "linestyle": (0, (4, 2))},
        saturation=0.9, ax=ax, zorder=3,
    )
    # The inner lines do not inherit the zorder handed to the bodies, so they
    # land underneath the (opaque) fills and have to be lifted back out.
    inner = ax.lines[n_before:]
    if len(inner) != 3 * len(order):
        raise RuntimeError(f"expected 3 quartile lines per violin, got {len(inner)}")
    # Seaborn emits them in (q25, median, q75) order. The median goes: the
    # geomean diamond already carries central tendency, and a full-width line
    # through the fattest part of every body only added clutter.
    for i, line in enumerate(inner):
        if i % 3 == 1:
            line.remove()
        else:
            line.set_zorder(4)

    # Geomean: white diamond on the violin, headline number above it. Together
    # they replace the scatter legend's ``(×2.15)`` annotations.
    for i, label in enumerate(order):
        g = stats[label]["ratio"]
        ax.scatter(i, g, marker="D", s=22, facecolor="white",
                   edgecolor="#222222", linewidth=0.8, zorder=5)
        top = long.loc[long["Baseline"] == label, "ratio"].max()
        ax.annotate(f"$\\times${g:.2f}", (i, top), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold", color="#222222")

    lo, hi = long["ratio"].min(), long["ratio"].max()
    ax.set_ylim(lo / 1.35, hi * 2.6)  # extra log-headroom for the ×labels
    ax.set_xlabel("")
    ax.set_ylabel(f"T-depth ratio (Baseline / {HERO_LABEL})", fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.yaxis.grid(True, which="major", lw=0.5, alpha=0.35)
    ax.yaxis.grid(True, which="minor", lw=0.3, alpha=0.15)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    fig.savefig(VIOLIN_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans",
                  rc={"axes.labelsize": 13, "legend.fontsize": 12,
                      "xtick.labelsize": 11, "ytick.labelsize": 11,
                      "pdf.fonttype": 42, "ps.fonttype": 42})

    data, stats, n_hero = load_pairs()
    if not data:
        raise SystemExit("no baseline had any program in common with Symphony")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_scatter(data, stats, n_hero)
    plot_violin(data, stats)

    for path in (SCATTER_PATH, VIOLIN_PATH):
        print(f"saved -> {os.path.relpath(path, EXP_DIR)}")
    for label, _ in T_BASELINES:
        if label in stats:
            st = stats[label]
            above = int((data[label]["baseline"] > data[label]["symphony"]).sum())
            print(f"    {label:9s} geomean baseline/{HERO_LABEL} = x{st['ratio']:.3f}"
                  f"   ({st['n']:3d}/{n_hero} programs, {above} above the diagonal)")


if __name__ == "__main__":
    main()
