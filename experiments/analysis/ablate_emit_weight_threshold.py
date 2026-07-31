#!/usr/bin/env python
"""Ablate weight-1 versus weight-2 emission in holistic compilation.

Every HAMLib benchmark is compiled through the full default Phoenix pipeline
twice.  The two arms differ only in the predicate used by ``peel_forward``'s
``_emit`` closure:

    1. ``weight <= 1``: peel every two-qubit Pauli down to one qubit;
    2. ``weight <= 2``: production default, allowing aggressive 2Q blocks.

Ratios are ``weight<=2 / weight<=1``.  Thus, a ratio below one and a negative
improvement percentage mean that weight-2 emission produced a better circuit.

Outputs:
    experiments/analysis/ablation_data/emit_weight_threshold.json
        Per-benchmark metrics for both arms, errors, and aggregate summaries.
    experiments/analysis/ablation_data/emit_weight_threshold.csv
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
JSON_PATH = os.path.join(DATA_DIR, "emit_weight_threshold.json")
CSV_PATH = os.path.join(DATA_DIR, "emit_weight_threshold.csv")

CATEGORIES = (
    ("binaryoptimization", "Binary optimization", 15),
    ("discreteoptimization", "Discrete optimization", 15),
    ("chemistry", "Chemistry", 35),
    ("condensedmatter", "Condensed matter", 35),
)

ARMS = (("emit_w_le_1", 1), ("emit_w_le_2", 2))
BASELINE_KEY = "emit_w_le_1"
CANDIDATE_KEY = "emit_w_le_2"
METRICS = ("num_2q", "depth_2q")


def circuit_metrics(qc: Any) -> dict[str, int]:
    return {
        "num_2q": sum(1 for inst in qc.data if inst.operation.num_qubits == 2),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def compile_arm(ham: Any, emit_max_weight: int) -> dict[str, int | float]:
    """Compile one arm with every option except the emission threshold at default."""
    import phoenix

    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        qc = phoenix.compile_hamiltonian_simulation(
            ham,
            emit_max_weight=emit_max_weight,
        )
    return {
        **circuit_metrics(qc),
        "compile_seconds": time.perf_counter() - start,
    }


def run_case(category: str, path: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    from phoenix.hamiltonian import Hamiltonian

    with open(path) as handle:
        data = json.load(handle)
    ham = Hamiltonian(data["paulis"], data["coeffs"])
    arms = {key: compile_arm(ham, threshold) for key, threshold in ARMS}

    comparison: dict[str, float | None] = {}
    for metric in METRICS:
        baseline = arms[BASELINE_KEY][metric]
        candidate = arms[CANDIDATE_KEY][metric]
        ratio = candidate / baseline if baseline else None
        comparison[f"{metric}_ratio"] = ratio
        comparison[f"{metric}_improvement_rate_pct"] = (
            None if ratio is None else 100.0 * (ratio - 1.0)
        )

    return {
        "category": category,
        "name": os.path.basename(path)[:-5],
        "qubits": ham.num_qubits,
        "paulis": len(ham.paulis),
        "arms": arms,
        "comparison": comparison,
    }


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        ratios = [
            case["comparison"][f"{metric}_ratio"]
            for case in cases
            if case["comparison"][f"{metric}_ratio"] is not None
            and case["comparison"][f"{metric}_ratio"] > 0.0
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
    all_cases = [case for category, _label, _count in CATEGORIES for case in results[category]]
    for category, label, expected_cases in CATEGORIES:
        cases = results[category]
        rows.append(
            {
                "category": category,
                "category_label": label,
                "completed_cases": len(cases),
                "expected_cases": expected_cases,
                **summarize_cases(cases),
            }
        )
    rows.append(
        {
            "category": "all",
            "category_label": "All",
            "completed_cases": len(all_cases),
            "expected_cases": sum(count for _category, _label, count in CATEGORIES),
            **summarize_cases(all_cases),
        }
    )
    return rows


def csv_fieldnames() -> list[str]:
    fields = ["category", "category_label", "completed_cases", "expected_cases"]
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
        "complete": complete,
        "benchmark_suite": "HAMLib (100 programs)",
        "pipeline": "phoenix.compile_hamiltonian_simulation with all defaults except emit_max_weight",
        "arms": {key: {"emit_predicate": f"weight <= {threshold}"} for key, threshold in ARMS},
        "baseline": BASELINE_KEY,
        "candidate": CANDIDATE_KEY,
        "ratio_definition": f"{CANDIDATE_KEY} / {BASELINE_KEY}",
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
    print("| Category (#) | 2Q count avg. | 2Q count max. | 2Q depth avg. | 2Q depth max. |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in summary:
        count = f"{row['completed_cases']}/{row['expected_cases']}"
        print(
            "| {} ({}) | {} | {} | {} | {} |".format(
                row["category_label"],
                count,
                fmt(row["num_2q_ratio_geomean"]),
                fmt(row["num_2q_ratio_best"]),
                fmt(row["depth_2q_ratio_geomean"]),
                fmt(row["depth_2q_ratio_best"]),
            )
        )


def load_previous_results() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    with open(JSON_PATH) as handle:
        payload = json.load(handle)
    results = {category: list(payload["results"].get(category, [])) for category, _l, _n in CATEGORIES}
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
        help="reuse completed cases from an existing output JSON",
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
                count_ratio = case["comparison"]["num_2q_ratio"]
                depth_ratio = case["comparison"]["depth_2q_ratio"]
                print(
                    f"[{done}/{len(tasks)}] {category}/{filename}: "
                    f"2q={count_ratio:.3f}x depth={depth_ratio:.3f}x",
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
