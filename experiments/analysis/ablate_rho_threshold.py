#!/usr/bin/env python
"""Ablate density-gated two-qubit emission in holistic compilation.

Every HAMLib benchmark is compiled through the full default Phoenix pipeline
with three ``rho_threshold`` values.  The arms differ only in the predicate
used by ``peel_forward``'s ``_emit`` closure:

    1. ``rho_threshold=0.00``: fixed weight-1 emission;
    2. ``rho_threshold=0.35``: the production adaptive policy;
    3. ``rho_threshold=1.00``: fixed aggressive weight-2 emission.

Per-case JSON retains all three absolute circuit metrics.  The CSV reports
each pairwise comparison by category and overall.  Ratios are
``candidate / baseline``: a ratio below one (or negative percentage) is
better.

Outputs:
    experiments/analysis/ablation_data/rho_threshold.json
        Per-benchmark metrics, pairwise comparisons, errors, and summaries.
    experiments/analysis/ablation_data/rho_threshold.csv
        Per-category and overall geometric-mean / best-case summary table.

The JSON and CSV are checkpointed after each completed benchmark, so an
interrupted run can continue with ``--resume``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
DATA_DIR = os.path.join(REPO, "experiments", "analysis", "ablation_data")
JSON_PATH = os.path.join(DATA_DIR, "rho_threshold.json")
CSV_PATH = os.path.join(DATA_DIR, "rho_threshold.csv")
SCHEMA_VERSION = 2

CATEGORIES = (
    ("binaryoptimization", "Binary optimization", 15),
    ("discreteoptimization", "Discrete optimization", 15),
    ("chemistry", "Chemistry", 35),
    ("condensedmatter", "Condensed matter", 35),
)

ARMS = (
    ("rho_0_00", 0.00, "Fixed weight-1 emission"),
    ("rho_0_35", 0.35, "Adaptive production policy"),
    ("rho_1_00", 1.00, "Fixed aggressive weight-2 emission"),
)
ARM_THRESHOLDS = {key: rho for key, rho, _label in ARMS}
COMPARISONS = (
    ("rho_0_35_over_0_00", "rho_0_00", "rho_0_35"),
    ("rho_1_00_over_0_00", "rho_0_00", "rho_1_00"),
    ("rho_0_35_over_1_00", "rho_1_00", "rho_0_35"),
)
METRICS = ("num_2q", "depth_2q")


def circuit_metrics(qc: Any) -> dict[str, int]:
    return {
        "num_2q": sum(1 for inst in qc.data if inst.operation.num_qubits == 2),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def compile_arm(ham: Any, rho_threshold: float) -> dict[str, int | float]:
    """Compile one density-threshold arm with every other option at default."""
    import phoenix

    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        qc = phoenix.compile_hamiltonian_simulation(ham, rho_threshold=rho_threshold)
    return {
        **circuit_metrics(qc),
        "compile_seconds": time.perf_counter() - start,
    }


def compare_arms(
    arms: dict[str, dict[str, int | float]], baseline_key: str, candidate_key: str
) -> dict[str, float | None]:
    comparison: dict[str, float | None] = {}
    for metric in METRICS:
        baseline = arms[baseline_key][metric]
        candidate = arms[candidate_key][metric]
        ratio = candidate / baseline if baseline else None
        comparison[f"{metric}_ratio"] = ratio
        comparison[f"{metric}_improvement_rate_pct"] = (
            None if ratio is None else 100.0 * (ratio - 1.0)
        )
    return comparison


def run_case(category: str, path: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    from phoenix.hamiltonian import Hamiltonian

    with open(path) as handle:
        data = json.load(handle)
    ham = Hamiltonian(data["paulis"], data["coeffs"])
    arms = {key: compile_arm(ham, rho) for key, rho, _label in ARMS}
    comparisons = {
        key: compare_arms(arms, baseline_key, candidate_key)
        for key, baseline_key, candidate_key in COMPARISONS
    }
    return {
        "category": category,
        "name": os.path.basename(path)[:-5],
        "qubits": ham.num_qubits,
        "paulis": len(ham.paulis),
        "arms": arms,
        "comparisons": comparisons,
    }


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize_cases(cases: list[dict[str, Any]], comparison_key: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        ratios = [
            case["comparisons"][comparison_key][f"{metric}_ratio"]
            for case in cases
            if case["comparisons"][comparison_key][f"{metric}_ratio"] is not None
            and case["comparisons"][comparison_key][f"{metric}_ratio"] > 0.0
        ]
        ratio_geomean = geometric_mean(ratios)
        ratio_best = min(ratios, default=None)
        ratio_worst = max(ratios, default=None)
        summary.update(
            {
                f"{metric}_valid_cases": len(ratios),
                f"{metric}_ratio_geomean": ratio_geomean,
                f"{metric}_improvement_geomean_pct": (
                    None if ratio_geomean is None else 100.0 * (ratio_geomean - 1.0)
                ),
                f"{metric}_ratio_best": ratio_best,
                f"{metric}_improvement_best_pct": (
                    None if ratio_best is None else 100.0 * (ratio_best - 1.0)
                ),
                f"{metric}_ratio_worst": ratio_worst,
                f"{metric}_improvement_worst_pct": (
                    None if ratio_worst is None else 100.0 * (ratio_worst - 1.0)
                ),
                f"{metric}_improved_cases": sum(ratio < 1.0 - 1e-12 for ratio in ratios),
                f"{metric}_equal_cases": sum(abs(ratio - 1.0) <= 1e-12 for ratio in ratios),
                f"{metric}_regressed_cases": sum(ratio > 1.0 + 1e-12 for ratio in ratios),
            }
        )
    return summary


def build_summary(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    category_rows = list(CATEGORIES) + [
        (
            "all",
            "All",
            sum(count for _category, _label, count in CATEGORIES),
        )
    ]
    all_cases = [case for category, _label, _count in CATEGORIES for case in results[category]]
    for category, label, expected_cases in category_rows:
        cases = all_cases if category == "all" else results[category]
        for comparison_key, baseline_key, candidate_key in COMPARISONS:
            rows.append(
                {
                    "category": category,
                    "category_label": label,
                    "completed_cases": len(cases),
                    "expected_cases": expected_cases,
                    "comparison": comparison_key,
                    "baseline": baseline_key,
                    "candidate": candidate_key,
                    **summarize_cases(cases, comparison_key),
                }
            )
    return rows


def csv_fieldnames() -> list[str]:
    fields = [
        "category",
        "category_label",
        "completed_cases",
        "expected_cases",
        "comparison",
        "baseline",
        "candidate",
    ]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_valid_cases",
                f"{metric}_ratio_geomean",
                f"{metric}_improvement_geomean_pct",
                f"{metric}_ratio_best",
                f"{metric}_improvement_best_pct",
                f"{metric}_ratio_worst",
                f"{metric}_improvement_worst_pct",
                f"{metric}_improved_cases",
                f"{metric}_equal_cases",
                f"{metric}_regressed_cases",
            ]
        )
    return fields


def write_csv(summary: list[dict[str, Any]]) -> None:
    fields = csv_fieldnames()
    temporary_path = f"{CSV_PATH}.tmp"
    with open(temporary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in summary)
    os.replace(temporary_path, CSV_PATH)


def metadata(complete: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "benchmark_suite": "HAMLib (100 programs)",
        "pipeline": (
            "phoenix.compile_hamiltonian_simulation with every default setting "
            "except rho_threshold"
        ),
        "arms": {
            key: {"rho_threshold": rho, "description": description}
            for key, rho, description in ARMS
        },
        "comparisons": {
            key: {"baseline": baseline_key, "candidate": candidate_key}
            for key, baseline_key, candidate_key in COMPARISONS
        },
        "ratio_definition": "candidate / baseline",
        "improvement_rate_definition": "(ratio - 1) * 100; negative is better",
        "metrics": list(METRICS),
    }


def write_outputs(
    results: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, str]],
    *,
    complete: bool,
) -> list[dict[str, Any]]:
    for category in results:
        results[category].sort(key=lambda case: case["name"])
    summary = build_summary(results)
    payload = {
        "metadata": metadata(complete),
        "results": results,
        "errors": errors,
        "summary": summary,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    temporary_path = f"{JSON_PATH}.tmp"
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, JSON_PATH)
    write_csv(summary)
    return summary


def fmt(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{100.0 * (ratio - 1.0):+.1f}% ({ratio:.3f}x)"


def print_markdown_table(summary: list[dict[str, Any]]) -> None:
    print("| Category (#) | Candidate / baseline | 2Q count avg. | 2Q count best | 2Q depth avg. | 2Q depth best |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary:
        count = f"{row['completed_cases']}/{row['expected_cases']}"
        comparison = f"{row['candidate']} / {row['baseline']}"
        print(
            "| {} ({}) | {} | {} | {} | {} | {} |".format(
                row["category_label"],
                count,
                comparison,
                fmt(row["num_2q_ratio_geomean"]),
                fmt(row["num_2q_ratio_best"]),
                fmt(row["depth_2q_ratio_geomean"]),
                fmt(row["depth_2q_ratio_best"]),
            )
        )


def load_previous_results() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    with open(JSON_PATH) as handle:
        payload = json.load(handle)
    if payload.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{JSON_PATH} uses an incompatible result schema; rerun without --resume."
        )
    results = {
        category: list(payload["results"].get(category, []))
        for category, _label, _count in CATEGORIES
    }
    return results, list(payload.get("errors", []))


def parse_args() -> argparse.Namespace:
    default_jobs = min(8, max(1, (os.cpu_count() or 2) - 2))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("ABLATE_JOBS", default_jobs)),
        help=f"worker processes (default: ABLATE_JOBS or {default_jobs})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed cases from a matching output JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    if args.resume and os.path.exists(JSON_PATH):
        results, errors = load_previous_results()
    else:
        results = {category: [] for category, _label, _count in CATEGORIES}
        errors: list[dict[str, str]] = []

    completed = {(category, case["name"]) for category in results for case in results[category]}
    tasks = []
    for category, _label, _expected in CATEGORIES:
        directory = os.path.join(HAMLIB, category)
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".json") and (category, filename[:-5]) not in completed:
                tasks.append((category, os.path.join(directory, filename)))

    total_expected = sum(count for _category, _label, count in CATEGORIES)
    print(
        f"{len(tasks)} pending of {total_expected} HAMLib benchmarks x {len(ARMS)} arms, "
        f"{args.jobs} workers",
        file=sys.stderr,
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_case, category, path): (category, path) for category, path in tasks
        }
        for done, future in enumerate(as_completed(futures), start=1):
            category, path = futures[future]
            filename = os.path.basename(path)
            try:
                case = future.result()
                results[category].append(case)
                adaptive = case["comparisons"]["rho_0_35_over_0_00"]
                aggressive = case["comparisons"]["rho_1_00_over_0_00"]
                print(
                    f"[{done}/{len(tasks)}] {category}/{filename}: "
                    f"adaptive 2q={adaptive['num_2q_ratio']:.3f}x "
                    f"depth={adaptive['depth_2q_ratio']:.3f}x; "
                    f"aggressive 2q={aggressive['num_2q_ratio']:.3f}x "
                    f"depth={aggressive['depth_2q_ratio']:.3f}x",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                errors.append({"category": category, "name": filename, "error": repr(exc)})
                print(
                    f"[{done}/{len(tasks)}] SKIP {category}/{filename}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            write_outputs(results, errors, complete=False)

    completed_count = sum(len(cases) for cases in results.values())
    complete = completed_count == total_expected and not errors
    summary = write_outputs(results, errors, complete=complete)
    print_markdown_table(summary)
    print(f"\nsaved raw data -> {JSON_PATH}")
    print(f"saved summary  -> {CSV_PATH}")
    if errors:
        print(f"warning: {len(errors)} benchmark(s) failed; errors stored in the JSON file.")


if __name__ == "__main__":
    main()
