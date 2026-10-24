#!/usr/bin/env python3
"""Plot absolute UCCSD compilation runtime against input two-qubit cost.

The script reads the exact-name files ``runtime_uccsd_<compiler>.csv`` from
this directory.  Platform-tagged archival files such as
``runtime_uccsd_phoenixpp.csv`` are intentionally ignored.

The x position of each UCCSD instance is its common, unoptimised input-circuit
two-qubit gate count (#2Q), on a logarithmic scale.  These costs are loaded
from ``experiments/output_uccsd/naive/all2all``.

Run from any working directory:

    python experiments/analysis/plot_runtime_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
RUNTIME_DATA_DIR = ANALYSIS_DIR / "runtime_data"
EXPERIMENTS_DIR = ANALYSIS_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "uccsd"
NAIVE_QASM_DIR = EXPERIMENTS_DIR / "output_uccsd" / "naive" / "all2all"
DEFAULT_OUTPUT = EXPERIMENTS_DIR / "figures" / "uccsd_runtime_comparison.pdf"

# The baseline order, colours, and markers deliberately match
# plot_hamlib_result.py.  Symphony receives the immediately following Set2
# colour, so it is visually distinct without changing baseline identities.
COMPILERS = (
    ("Qiskit", "qiskit", "o"),
    ("TKET", "tket", "s"),
    ("Paulihedral", "paulihedral", "^"),
    ("Tetris", "tetris", "D"),
    ("QuCLEAR", "quclear", "P"),
    ("PHOENIX", "phoenix", "X"),
    ("Symphony", "phoenixpp", "*"),
)
HERO_KEY = "phoenixpp"
# seaborn's violinplot desaturates its fills; plot_hamlib_t_depth.py passes
# saturation=0.9, so the curves here use the same toned-down Set2 colours.
SATURATION = 0.9


@dataclass(frozen=True)
class UCCSDMetadata:
    """Common input-circuit metadata used to position a UCCSD instance."""

    num_qubits: int
    num_2q_gates: int
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    return parser.parse_args()


def load_runtime_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["program", "runtime"]:
            raise ValueError(f"Unexpected schema in {path}; expected program,runtime")
        rows: dict[str, float] = {}
        for row in reader:
            program = row["program"]
            runtime = float(row["runtime"])
            if not program or not isfinite(runtime) or runtime <= 0:
                raise ValueError(f"Invalid runtime record in {path}: {row}")
            if program in rows:
                raise ValueError(f"Duplicate program {program!r} in {path}")
            rows[program] = runtime
    if not rows:
        raise ValueError(f"No runtime records in {path}")
    return rows


def load_uccsd_metadata(programs: set[str]) -> dict[str, UCCSDMetadata]:
    """Load qubit count plus naive two-qubit count and total circuit depth."""
    from qiskit import QuantumCircuit

    metadata: dict[str, UCCSDMetadata] = {}
    for program in programs:
        benchmark_path = BENCHMARK_DIR / f"{program}.json"
        qasm_path = NAIVE_QASM_DIR / f"{program}.qasm"
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Cannot find benchmark metadata for {program}: {benchmark_path}")
        if not qasm_path.exists():
            raise FileNotFoundError(f"Cannot find naive circuit for {program}: {qasm_path}")
        with benchmark_path.open() as handle:
            num_qubits = int(json.load(handle)["num_qubits"])
        circuit = QuantumCircuit.from_qasm_file(qasm_path)
        metadata[program] = UCCSDMetadata(
            num_qubits=num_qubits,
            num_2q_gates=circuit.num_nonlocal_gates(),
            depth=circuit.depth(),
        )
    return metadata


def compact_integer(value: int) -> str:
    """Format large circuit metrics compactly while retaining one decimal."""
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


def main() -> None:
    args = parse_args()

    loaded: dict[str, tuple[str, str, dict[str, float]]] = {}
    missing: list[Path] = []
    for label, key, marker in COMPILERS:
        path = RUNTIME_DATA_DIR / f"runtime_uccsd_{key}.csv"
        if not path.exists():
            missing.append(path)
            continue
        loaded[key] = (label, marker, load_runtime_csv(path))

    if HERO_KEY not in loaded:
        raise SystemExit(f"Missing required Symphony runtime data: {RUNTIME_DATA_DIR / 'runtime_uccsd_phoenixpp.csv'}")
    if len(loaded) < 2:
        raise SystemExit("At least Symphony and one baseline CSV are required")
    if missing:
        print("warning: missing runtime CSVs (not plotted):")
        for path in missing:
            print(f"  {path.name}")

    all_programs = set().union(*(rows for _, _, rows in loaded.values()))
    metadata = load_uccsd_metadata(all_programs)
    programs = sorted(all_programs, key=lambda program: metadata[program].num_2q_gates)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    # Same grid as the hamlib_*_vs_symphony scatter figures.
    from plot_hamlib_result import draw_grid

    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axis = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    palette = [sns.desaturate(color, SATURATION) for color in sns.color_palette("Set2", len(COMPILERS))]

    legend_handles: list[Line2D] = []
    for index, (label, key, marker) in enumerate(COMPILERS):
        if key not in loaded:
            continue
        _, _, rows = loaded[key]
        observed = [program for program in programs if program in rows]
        hero = key == HERO_KEY
        color = palette[index]
        # linestyle = "--" if key in {"paulihedral", "tetris"} else "-"
        linestyle = "-"
        axis.plot(
            [metadata[program].num_2q_gates for program in observed],
            [rows[program] for program in observed],
            color=color,
            linestyle=linestyle,
            linewidth=1.8 if hero else 1.45,
            marker=marker,
            markersize=11 if hero else 9,
            markeredgecolor="grey" if hero else color,
            markeredgewidth=0.75 if hero else 0.0,
            zorder=4 if hero else 2,
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8 if hero else 1.45,
                markersize=11 if hero else 9,
                markeredgecolor="grey" if hero else color,
                markeredgewidth=0.75 if hero else 0.0,
                label=f"{label}",
            )
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Program size (#2Q in naive synthesis)", fontweight="bold")
    axis.set_ylabel("Compilation latency (s)", fontweight="bold")
    # axis.set_title("Absolute compilation runtime", pad=10)
    draw_grid(axis)

    # One column in the upper-left corner: every curve rises to the right, so
    # that corner is the only empty region tall enough for seven entries.
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        handlelength=2.0,
        labelspacing=0.3,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
