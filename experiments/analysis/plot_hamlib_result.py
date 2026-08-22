#!/usr/bin/env python
"""Per-program scatter: each SOTA baseline vs Symphony on the 100 HamLib
programs (all-to-all, post-O3).

Eight figures are produced (figsize 6.2x5.0):
  * hamlib_2q_count.pdf  -- x = Symphony 2-qubit gate count,  y = baseline 2-qubit gate count
  * hamlib_2q_depth.pdf  -- x = Symphony 2-qubit depth,       y = baseline 2-qubit depth
  * hamlib_<baseline>_vs_symphony.pdf -- one per baseline, with both
    two-qubit gate-count and circuit-depth scatter points.

Every point is one (baseline, program) pair, coloured by baseline. Points ABOVE
the y = x diagonal are cases where the baseline needs more two-qubit resources;
the shaded bands mark 1.5x / 2x / 3x ratios.

Style follows experiments/analysis/reference_ratio_plotting/plot_routing_result.py.
Run from anywhere:  python experiments/analysis/plot_hamlib_result.py
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
from matplotlib.lines import Line2D

EXP_DIR = Path(__file__).resolve().parents[1]          # experiments/
RESULTS = EXP_DIR / "results"
OUT_DIR = EXP_DIR / "figures"
KEY = ("category", "program")

HERO_KEY = "phoenixpp"
HERO_LABEL = "Symphony"

# display label -> results CSV key (order = legend/z-order; drawn baselines only)
BASELINES = [
    ("Qiskit", "qiskit"),
    ("TKET", "tket"),
    ("Paulihedral", "paulihedral"),
    ("Tetris", "tetris"),
    ("QuCLEAR", "quclear"),
    ("Phoenix", "phoenix"),
]

METRICS = {
    "num_2q_gates(opt)": {
        "out": "hamlib_2q_count.pdf",
        "x_label": "Two-qubit gate count (Symphony)",
        "y_label": "Two-qubit gate count (Baseline)",
    },
    "depth_2q(opt)": {
        "out": "hamlib_2q_depth.pdf",
        "x_label": "Two-qubit circuit depth (Symphony)",
        "y_label": "Two-qubit circuit depth (Baseline)",
    },
}

COMBINED_METRICS = {
    "num_2q_gates(opt)": "Two-qubit gate count",
    "depth_2q(opt)": "Two-qubit circuit depth",
}

REFERENCE_COLOR = "#8DA0CB"
REFERENCE_DIVIDER_COLOR = "#B9C7DD"
# REFERENCE_BANDS = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
REFERENCE_BANDS = (1.5, 2.0, 3.0, 5.0, 10.0)
AXIS_PADDING = 1.12
GRID_ZORDER = 0
REFERENCE_BAND_ZORDER = 1
REFERENCE_LINE_ZORDER = 2
POINT_ZORDER = 3

_palette = sns.color_palette("Set2", len(BASELINES))
_markers = ["o", "s", "^", "D", "P", "X"]
STYLE = {label: {"color": _palette[i], "marker": _markers[i % len(_markers)]}
         for i, (label, _) in enumerate(BASELINES)}


def load_metric(metric):
    """Long-form df of (Baseline, program, phoenixpp value, baseline value) for a
    metric, joined per-program to Phoenix++, plus per-baseline geomean ratio."""
    hero = pd.read_csv(RESULTS / f"result_hamlib_{HERO_KEY}.csv")[[*KEY, metric]]
    hero = hero.rename(columns={metric: "phoenixpp"})
    rows, ratios = [], {}
    for label, key in BASELINES:
        path = RESULTS / f"result_hamlib_{key}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)[[*KEY, metric]].rename(columns={metric: "baseline"})
        merged = hero.merge(df, on=list(KEY), how="inner")
        merged = merged[(merged["phoenixpp"] > 0) & (merged["baseline"] > 0)]
        ratios[label] = float(gmean(merged["baseline"] / merged["phoenixpp"]))
        merged = merged.assign(Baseline=f"{label}  (×{ratios[label]:.2f})")
        merged["_label"] = label
        rows.append(merged)
    return pd.concat(rows, ignore_index=True), ratios


def load_baseline_metrics(label, key):
    """Metric-specific Symphony/baseline pairs and ratios for one baseline."""
    columns = [*KEY, *COMBINED_METRICS]
    hero = pd.read_csv(RESULTS / f"result_hamlib_{HERO_KEY}.csv")[columns]
    baseline = pd.read_csv(RESULTS / f"result_hamlib_{key}.csv")[columns]
    merged = hero.merge(baseline, on=list(KEY), suffixes=("_symphony", "_baseline"))

    data, ratios = {}, {}
    for metric in COMBINED_METRICS:
        symphony = f"{metric}_symphony"
        baseline_metric = f"{metric}_baseline"
        points = merged[[*KEY, symphony, baseline_metric]].rename(
            columns={symphony: "symphony", baseline_metric: "baseline"}
        )
        points = points[(points["symphony"] > 0) & (points["baseline"] > 0)]
        data[metric] = points
        ratios[metric] = float(gmean(points["baseline"] / points["symphony"]))

    return data, ratios


def axis_limits(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    # On a wide log axis, 5% leaves only a few pixels for edge markers.
    return vals.min() / AXIS_PADDING, vals.max() * AXIS_PADDING


def draw_reference(ax, lo, hi):
    ref = np.geomspace(max(lo, 1e-9), hi, 256)
    ax.plot(
        ref, ref, color="#222222", lw=1.5, ls="--", alpha=0.75,
        zorder=REFERENCE_LINE_ZORDER,
    )
    for k in REFERENCE_BANDS:                       # y = k*x  => baseline needs k x more
        ax.fill_between(
            ref, ref, k * ref, color=REFERENCE_COLOR, alpha=0.10, lw=0,
            zorder=REFERENCE_BAND_ZORDER,
        )
    for k in REFERENCE_BANDS:
        ax.plot(
            ref, k * ref, color=REFERENCE_DIVIDER_COLOR, lw=0.55, alpha=0.45,
            zorder=REFERENCE_LINE_ZORDER,
        )


def draw_grid(ax):
    """Keep both major and minor grid lines below every reference band."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#D7DCE2", lw=0.75, zorder=GRID_ZORDER)
    ax.grid(True, which="minor", color="#EEF1F5", lw=0.45, zorder=GRID_ZORDER)
    for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
        gridline.set_zorder(GRID_ZORDER)


def lighter(color, fraction=0.62):
    """Mix an RGB color with white while retaining the baseline hue."""
    return tuple(channel + (1.0 - channel) * fraction for channel in color)


def plot_metric(metric, out_path):
    df, ratios = load_metric(metric)
    cfg = METRICS[metric]
    palette = {f"{lab}  (×{ratios[lab]:.2f})": STYLE[lab]["color"] for lab in ratios}
    markers = {f"{lab}  (×{ratios[lab]:.2f})": STYLE[lab]["marker"] for lab in ratios}
    order = [f"{lab}  (×{ratios[lab]:.2f})" for lab, _ in BASELINES if lab in ratios]

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    xlo, xhi = axis_limits(df["phoenixpp"])
    ylo, yhi = axis_limits(df["baseline"])
    draw_reference(ax, xlo, xhi)
    sns.scatterplot(
        data=df, x="phoenixpp", y="baseline",
        hue="Baseline", style="Baseline", hue_order=order, style_order=order,
        palette=palette, markers=markers,
        s=50, edgecolor="k", linewidth=0.4, alpha=0.8, zorder=POINT_ZORDER, ax=ax,
    )
    ax.set(xscale="log", yscale="log", xlim=(xlo, xhi), ylim=(ylo, yhi),
           xlabel=cfg["x_label"], ylabel=cfg["y_label"])
    draw_grid(ax)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(f"Geomean$\\times$ vs {HERO_LABEL}",
                      prop={"size": 10})
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {os.path.relpath(out_path, EXP_DIR)}")
    for lab, _ in BASELINES:
        if lab in ratios:
            print(f"    {lab:12s} geomean baseline/{HERO_LABEL} = x{ratios[lab]:.3f}")


def plot_baseline_metrics(label, key, out_path):
    """Plot gate count and circuit depth together for one baseline."""
    data, ratios = load_baseline_metrics(label, key)
    style = STYLE[label]
    x_values = np.concatenate([points["symphony"].to_numpy() for points in data.values()])
    y_values = np.concatenate([points["baseline"].to_numpy() for points in data.values()])
    xlo, xhi = axis_limits(x_values)
    ylo, yhi = axis_limits(y_values)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    draw_reference(ax, xlo, xhi)

    count = data["num_2q_gates(opt)"]
    depth = data["depth_2q(opt)"]
    ax.scatter(
        count["symphony"], count["baseline"], marker=style["marker"], s=56,
        facecolor=style["color"], edgecolor="#202020", linewidth=0.45, alpha=0.8,
        zorder=POINT_ZORDER,
    )
    ax.scatter(
        depth["symphony"], depth["baseline"], marker=style["marker"], s=56,
        facecolor=lighter(style["color"]), edgecolor=style["color"], linewidth=1.05, alpha=0.8,
        zorder=POINT_ZORDER,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(xlo, xhi),
        ylim=(ylo, yhi),
        xlabel=f"Two-qubit metric value ({HERO_LABEL})",
        ylabel=f"Two-qubit metric value ({label})",
        # title=f"{label} vs {HERO_LABEL}",
    )
    draw_grid(ax)

    handles = [
        Line2D(
            [], [], marker=style["marker"], linestyle="none", markersize=7,
            markerfacecolor=style["color"], markeredgecolor="#202020", markeredgewidth=0.6,
            label=f"Gate count  (×{ratios['num_2q_gates(opt)']:.2f})",
        ),
        Line2D(
            [], [], marker=style["marker"], linestyle="none", markersize=7,
            markerfacecolor=lighter(style["color"]), markeredgecolor=style["color"], markeredgewidth=1.1,
            label=f"Circuit depth  (×{ratios['depth_2q(opt)']:.2f})",
        ),
    ]
    ax.legend(
        handles=handles,
        title=f"Geomean {label}/{HERO_LABEL}",
        title_fontsize=12,
        loc="upper left",
        frameon=True,
        fontsize=12
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {os.path.relpath(out_path, EXP_DIR)}")


def main():
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans",
                  rc={"axes.labelsize": 15, "legend.fontsize": 13,
                      "xtick.labelsize": 11, "ytick.labelsize": 11,
                      "pdf.fonttype": 42, "ps.fonttype": 42})
    for metric, cfg in METRICS.items():
        plot_metric(metric, OUT_DIR / cfg["out"])
    for label, key in BASELINES:
        plot_baseline_metrics(label, key, OUT_DIR / f"hamlib_{key}_vs_symphony.pdf")


if __name__ == "__main__":
    main()
