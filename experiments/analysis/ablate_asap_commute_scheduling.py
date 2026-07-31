#!/usr/bin/env python
"""Compare ASAP scheduling with and without exact-commutation relaxation.

For every Hamlib benchmark, this script compiles the four configurations:

    1. SCHEDULE_ASAP=False, SCHEDULE_ASAP_COMMUTE=False
    2. SCHEDULE_ASAP=True,  SCHEDULE_ASAP_COMMUTE=False
    3. SCHEDULE_ASAP=False, SCHEDULE_ASAP_COMMUTE=True
    4. SCHEDULE_ASAP=True,  SCHEDULE_ASAP_COMMUTE=True

The two reported comparisons hold the exact-commute setting fixed and measure
the effect of turning ASAP on.  Their depth ratio is ``asap_on / asap_off``;
therefore a ratio below one, or a negative ``(ratio - 1) * 100`` percentage,
means a shallower two-qubit circuit.

Outputs:
    experiments/analysis/ablation_data/asap_commute_scheduling.json
        Per-benchmark depths and ratios for all four configurations.
    experiments/analysis/ablation_data/asap_commute_scheduling.csv
        Per-category and overall geometric-mean / best-case summary table.
"""

import contextlib
import csv
import io
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import warnings

warnings.filterwarnings("ignore")

import phoenix
import phoenix.primitive.holistic as holistic_mod
from phoenix.hamiltonian import Hamiltonian


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS = os.path.join(REPO, "experiments")
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
JSON_PATH = os.path.join(EXPERIMENTS, "analysis", "ablation_data", "asap_commute_scheduling.json")
CSV_PATH = os.path.join(EXPERIMENTS, "analysis", "ablation_data", "asap_commute_scheduling.csv")

# Keep this order aligned with the requested terminal/CSV table.
CATEGORIES = (
    ("binaryoptimization", "Binary optimization", 15),
    ("discreteoptimization", "Discrete optimization", 15),
    ("chemistry", "Chemistry", 35),
    ("condensedmatter", "Condensed matter", 35),
)

CONFIGURATIONS = (
    ("asap_off_commute_off", False, False),
    ("asap_on_commute_off", True, False),
    ("asap_off_commute_on", False, True),
    ("asap_on_commute_on", True, True),
)

COMPARISONS = {
    "asap_without_commute": {
        "label": "ASAP w/o Commute",
        "off_key": "asap_off_commute_off",
        "on_key": "asap_on_commute_off",
        "description": "SCHEDULE_ASAP on/off with SCHEDULE_ASAP_COMMUTE=False.",
    },
    "asap_with_commute": {
        "label": "ASAP w/ Commute",
        "off_key": "asap_off_commute_on",
        "on_key": "asap_on_commute_on",
        "description": "SCHEDULE_ASAP on/off with SCHEDULE_ASAP_COMMUTE=True.",
    },
}


def two_qubit_depth(qc: Any) -> int:
    return qc.depth(lambda inst: inst.operation.num_qubits == 2)


def compile_depth(ham: Hamiltonian, *, asap: bool, exact_commute: bool) -> int:
    """Compile one arm while restoring global scheduler flags afterwards."""
    old_asap = holistic_mod.SCHEDULE_ASAP
    old_exact_commute = holistic_mod.SCHEDULE_ASAP_COMMUTE
    try:
        holistic_mod.SCHEDULE_ASAP = asap
        holistic_mod.SCHEDULE_ASAP_COMMUTE = exact_commute
        with contextlib.redirect_stdout(io.StringIO()):
            qc = phoenix.compile_hamiltonian_simulation(ham)
        return two_qubit_depth(qc)
    finally:
        holistic_mod.SCHEDULE_ASAP = old_asap
        holistic_mod.SCHEDULE_ASAP_COMMUTE = old_exact_commute


def ratio_and_improvement(on_depth: int, off_depth: int) -> tuple[float | None, float | None]:
    if off_depth == 0:
        return None, None
    ratio = on_depth / off_depth
    return ratio, 100.0 * (ratio - 1.0)


def run_case(category: str, filename: str, data: dict[str, Any]) -> dict[str, Any]:
    ham = Hamiltonian(data["paulis"], data["coeffs"])
    depths: dict[str, int] = {}
    for key, asap, exact_commute in CONFIGURATIONS:
        depths[key] = compile_depth(ham, asap=asap, exact_commute=exact_commute)

    comparisons: dict[str, dict[str, float | int | None]] = {}
    for key, definition in COMPARISONS.items():
        off_depth = depths[definition["off_key"]]
        on_depth = depths[definition["on_key"]]
        ratio, improvement = ratio_and_improvement(on_depth, off_depth)
        comparisons[key] = {
            "off_depth_2q": off_depth,
            "on_depth_2q": on_depth,
            "depth_ratio_on_over_off": ratio,
            "improvement_rate_pct": improvement,
        }

    return {
        "category": category,
        "name": filename[:-5],
        "qubits": ham.num_qubits,
        "paulis": len(ham.paulis),
        "depth_2q": depths,
        "comparisons": comparisons,
    }


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize_cases(cases: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    valid = [
        case["comparisons"][comparison]
        for case in cases
        if case["comparisons"][comparison]["depth_ratio_on_over_off"] is not None
        and case["comparisons"][comparison]["depth_ratio_on_over_off"] > 0.0
    ]
    ratios = [record["depth_ratio_on_over_off"] for record in valid]
    ratio = geometric_mean(ratios)
    best = min(valid, key=lambda record: record["depth_ratio_on_over_off"], default=None)
    return {
        "valid_cases": len(valid),
        "depth_ratio_geomean": ratio,
        "improvement_rate_geomean_pct": None if ratio is None else 100.0 * (ratio - 1.0),
        "best_depth_ratio": None if best is None else best["depth_ratio_on_over_off"],
        "best_improvement_rate_pct": None if best is None else best["improvement_rate_pct"],
    }


def build_summary(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_cases = [case for category, _label, _count in CATEGORIES for case in results[category]]
    rows: list[dict[str, Any]] = []
    for category, label, expected_count in CATEGORIES:
        cases = results[category]
        rows.append(
            {
                "category": category,
                "category_label": label,
                "expected_cases": expected_count,
                "completed_cases": len(cases),
                "asap_without_commute": summarize_cases(cases, "asap_without_commute"),
                "asap_with_commute": summarize_cases(cases, "asap_with_commute"),
            }
        )
    rows.append(
        {
            "category": "all",
            "category_label": "All",
            "expected_cases": sum(count for _category, _label, count in CATEGORIES),
            "completed_cases": len(all_cases),
            "asap_without_commute": summarize_cases(all_cases, "asap_without_commute"),
            "asap_with_commute": summarize_cases(all_cases, "asap_with_commute"),
        }
    )
    return rows


def format_improvement(rate: float | None, ratio: float | None) -> str:
    if rate is None or ratio is None:
        return "n/a"
    return f"{rate:+.1f}% ({ratio:.3f}x)"


def case_count_label(row: dict[str, Any]) -> str:
    expected = row["expected_cases"]
    completed = row["completed_cases"]
    count = str(expected) if completed == expected else f"{completed}/{expected}"
    return f"{row['category_label']} ({count})"


def print_markdown_table(summary: list[dict[str, Any]]) -> None:
    print("| Relative improv. rate | ASAP w/o Commute | ASAP w/o Commute | ASAP w/ Commute | ASAP w/ Commute |")
    print("| --- | --- | --- | --- | --- |")
    print("| Category (#) | Depth improv. (Avg) | Depth improv. (Max) | Depth improv. (Avg) | Depth improv. (Max) |")
    for row in summary:
        without = row["asap_without_commute"]
        with_commute = row["asap_with_commute"]
        print(
            "| {category} | {without_avg} | {without_best} | {with_avg} | {with_best} |".format(
                category=case_count_label(row),
                without_avg=format_improvement(
                    without["improvement_rate_geomean_pct"], without["depth_ratio_geomean"]
                ),
                without_best=format_improvement(
                    without["best_improvement_rate_pct"], without["best_depth_ratio"]
                ),
                with_avg=format_improvement(
                    with_commute["improvement_rate_geomean_pct"], with_commute["depth_ratio_geomean"]
                ),
                with_best=format_improvement(
                    with_commute["best_improvement_rate_pct"], with_commute["best_depth_ratio"]
                ),
            )
        )


def write_csv(summary: list[dict[str, Any]]) -> None:
    fieldnames = [
        "category",
        "category_label",
        "completed_cases",
        "expected_cases",
        "asap_without_commute_avg_ratio",
        "asap_without_commute_avg_improvement_rate_pct",
        "asap_without_commute_max_ratio",
        "asap_without_commute_max_improvement_rate_pct",
        "asap_with_commute_avg_ratio",
        "asap_with_commute_avg_improvement_rate_pct",
        "asap_with_commute_max_ratio",
        "asap_with_commute_max_improvement_rate_pct",
    ]
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            without = row["asap_without_commute"]
            with_commute = row["asap_with_commute"]
            writer.writerow(
                {
                    "category": row["category"],
                    "category_label": row["category_label"],
                    "completed_cases": row["completed_cases"],
                    "expected_cases": row["expected_cases"],
                    "asap_without_commute_avg_ratio": without["depth_ratio_geomean"],
                    "asap_without_commute_avg_improvement_rate_pct": without["improvement_rate_geomean_pct"],
                    "asap_without_commute_max_ratio": without["best_depth_ratio"],
                    "asap_without_commute_max_improvement_rate_pct": without["best_improvement_rate_pct"],
                    "asap_with_commute_avg_ratio": with_commute["depth_ratio_geomean"],
                    "asap_with_commute_avg_improvement_rate_pct": with_commute["improvement_rate_geomean_pct"],
                    "asap_with_commute_max_ratio": with_commute["best_depth_ratio"],
                    "asap_with_commute_max_improvement_rate_pct": with_commute["best_improvement_rate_pct"],
                }
            )


def main() -> None:
    # Fail fast on a renamed/removed scheduler flag. Without this, every case
    # raises AttributeError, is caught per-case below, and the run still
    # overwrites the saved results with an all-"n/a" summary.
    for flag in ("SCHEDULE_ASAP", "SCHEDULE_ASAP_COMMUTE"):
        if not hasattr(holistic_mod, flag):
            raise SystemExit(
                f"phoenix.primitive.holistic has no attribute {flag!r}; the scheduler "
                "flags were renamed or removed. Update this script before rerunning "
                "(the existing result files are left untouched)."
            )

    results = {category: [] for category, _label, _count in CATEGORIES}
    errors: list[dict[str, str]] = []

    for category, label, expected_count in CATEGORIES:
        directory = os.path.join(HAMLIB, category)
        filenames = [name for name in sorted(os.listdir(directory)) if name.endswith(".json")]
        print(f"Running {label}: {len(filenames)} benchmarks", file=sys.stderr, flush=True)
        if len(filenames) != expected_count:
            print(
                f"warning: expected {expected_count} {label} benchmarks, found {len(filenames)}",
                file=sys.stderr,
                flush=True,
            )
        for index, filename in enumerate(filenames, start=1):
            try:
                with open(os.path.join(directory, filename)) as handle:
                    data = json.load(handle)
                results[category].append(run_case(category, filename, data))
            except Exception as exc:  # Keep the remaining long-running cases usable.
                errors.append({"category": category, "name": filename, "error": repr(exc)})
                print(
                    f"skip {label} [{index}/{len(filenames)}] {filename}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    summary = build_summary(results)
    payload = {
        "metadata": {
            "metric": "two-qubit circuit depth after phoenix.compile_hamiltonian_simulation",
            "ratio_definition": "SCHEDULE_ASAP=True depth / SCHEDULE_ASAP=False depth",
            "improvement_rate_definition": "(depth_ratio_on_over_off - 1) * 100; negative is better",
            "comparisons": COMPARISONS,
        },
        "results": results,
        "errors": errors,
        "summary": summary,
    }
    with open(JSON_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(summary)
    print_markdown_table(summary)
    print(f"\nsaved raw data -> {JSON_PATH}")
    print(f"saved summary  -> {CSV_PATH}")
    if errors:
        print(f"warning: {len(errors)} benchmark(s) failed; their errors are stored in the JSON file.")


if __name__ == "__main__":
    main()
