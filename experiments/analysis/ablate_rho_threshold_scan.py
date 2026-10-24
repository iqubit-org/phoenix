#!/usr/bin/env python
"""Scan the density threshold of adaptive two-qubit emission over [0, 1].

``ablate_rho_threshold.py`` compares the production ``rho_threshold=0.35``
against its two degenerate endpoints.  This script answers the follow-up
question -- how sensitive is the result to that default? -- by compiling every
HAMLib benchmark through the full default Phoenix pipeline on a uniform grid of
``rho_threshold`` values (default step 0.05, i.e. 21 points), with every other
option untouched.

The emission predicate only compares ``rho_threshold`` against the active
tableau density at emission time, so each program's metrics are a step function
of the threshold; the grid samples that step function.

Outputs (``experiments/analysis/ablation_rho_scan``):
    rho_scan.json
        Per-benchmark absolute metrics for every grid point, plus errors.
    rho_scan.csv
        The same data in long form: one row per (benchmark, rho_threshold).
    rho_scan_summary.csv
        Per-category and overall geometric-mean ratios against the reference
        ``rho_threshold=0.35`` (``candidate / reference``; below one is better).

Work is split into one task per (benchmark, rho_threshold) and the outputs are
checkpointed after every task, so an interrupted run can continue with
``--resume``.  ``plot_rho_threshold.py`` draws the figure from ``rho_scan.csv``.
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

# One thread per worker process.  Qiskit's Rust passes (rayon) and BLAS each
# default to one thread per core, so N workers would otherwise run N x cores
# threads and stall the machine.  Spawned workers re-import this module, so the
# caps reach them too; an explicit setting in the environment still wins.
for _variable in (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
DATA_DIR = os.path.join(REPO, "experiments", "analysis", "ablation_rho_scan")
SCHEMA_VERSION = 1

CATEGORIES = (
    ("binaryoptimization", "Binary optimization", 15),
    ("discreteoptimization", "Discrete optimization", 15),
    ("chemistry", "Chemistry", 35),
    ("condensedmatter", "Condensed matter", 35),
)

REFERENCE_RHO = 0.35
DEFAULT_STEP = 0.05
METRICS = ("num_2q", "depth_2q")


def rho_grid(step: float) -> list[float]:
    """Uniform grid on [0, 1]; the reference threshold is always included."""
    if not 0.0 < step <= 1.0:
        raise ValueError("--step must lie in (0, 1]")
    count = round(1.0 / step)
    if abs(count * step - 1.0) > 1e-9:
        raise ValueError("--step must divide 1 evenly")
    grid = {round(index * step, 6) for index in range(count + 1)}
    grid.add(REFERENCE_RHO)
    return sorted(grid)


def arm_key(rho: float) -> str:
    """``0.35 -> 'rho_0_35'``, the key convention of ``ablate_rho_threshold.py``."""
    return f"rho_{rho:.2f}".replace(".", "_")


def circuit_metrics(qc: Any) -> dict[str, int]:
    return {
        "num_2q": sum(1 for inst in qc.data if inst.operation.num_qubits == 2),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def run_arm(category: str, path: str, rho_threshold: float) -> dict[str, Any]:
    """Compile one benchmark at one threshold with every other option at default."""
    warnings.filterwarnings("ignore")
    import phoenix
    from phoenix.hamiltonian import Hamiltonian

    with open(path) as handle:
        data = json.load(handle)
    # HAMLib labels are big-endian; Phoenix expects little-endian Pauli strings.
    ham = Hamiltonian([pauli[::-1] for pauli in data["paulis"]], data["coeffs"])
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        qc = phoenix.compile_hamiltonian_simulation(ham, rho_threshold=rho_threshold)
    return {
        "category": category,
        "name": os.path.basename(path)[:-5],
        "qubits": ham.num_qubits,
        "paulis": len(ham.paulis),
        "rho_threshold": rho_threshold,
        **circuit_metrics(qc),
        "compile_seconds": time.perf_counter() - start,
    }


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def build_summary(
    results: dict[str, list[dict[str, Any]]], grid: list[float]
) -> list[dict[str, Any]]:
    """Geomean ratio against the reference threshold, per category and overall.

    Only benchmarks holding both the grid point and the reference contribute,
    so a partially finished scan still summarises consistently.
    """
    reference_key = arm_key(REFERENCE_RHO)
    category_rows = list(CATEGORIES) + [
        ("all", "All", sum(count for _category, _label, count in CATEGORIES))
    ]
    all_cases = [case for category, _label, _count in CATEGORIES for case in results[category]]
    rows: list[dict[str, Any]] = []
    for category, label, expected_cases in category_rows:
        cases = all_cases if category == "all" else results[category]
        for rho in grid:
            key = arm_key(rho)
            paired = [
                (case["arms"][key], case["arms"][reference_key])
                for case in cases
                if key in case["arms"] and reference_key in case["arms"]
            ]
            row: dict[str, Any] = {
                "category": category,
                "category_label": label,
                "rho_threshold": rho,
                "completed_cases": len(paired),
                "expected_cases": expected_cases,
                "identical_cases": sum(
                    all(arm[metric] == reference[metric] for metric in METRICS)
                    for arm, reference in paired
                ),
            }
            for metric in METRICS:
                ratios = [
                    arm[metric] / reference[metric]
                    for arm, reference in paired
                    if reference[metric] and arm[metric]
                ]
                ratio_geomean = geometric_mean(ratios)
                row.update(
                    {
                        f"{metric}_valid_cases": len(ratios),
                        f"{metric}_ratio_geomean": ratio_geomean,
                        f"{metric}_change_geomean_pct": (
                            None if ratio_geomean is None else 100.0 * (ratio_geomean - 1.0)
                        ),
                        f"{metric}_improved_cases": sum(ratio < 1.0 - 1e-12 for ratio in ratios),
                        f"{metric}_equal_cases": sum(abs(ratio - 1.0) <= 1e-12 for ratio in ratios),
                        f"{metric}_regressed_cases": sum(ratio > 1.0 + 1e-12 for ratio in ratios),
                    }
                )
            rows.append(row)
    return rows


def write_csv(path: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    os.replace(temporary_path, path)


def long_rows(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for category, _label, _count in CATEGORIES:
        for case in results[category]:
            for arm in sorted(case["arms"].values(), key=lambda arm: arm["rho_threshold"]):
                rows.append(
                    {
                        "category": category,
                        "name": case["name"],
                        "qubits": case["qubits"],
                        "paulis": case["paulis"],
                        **arm,
                    }
                )
    return rows


def metadata(grid: list[float], complete: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "benchmark_suite": "HAMLib (100 programs)",
        "pipeline": (
            "phoenix.compile_hamiltonian_simulation with every default setting "
            "except rho_threshold"
        ),
        "pauli_input_conversion": "HAMLib big-endian labels reversed for Phoenix",
        "rho_grid": grid,
        "reference_rho": REFERENCE_RHO,
        "arm_key_convention": "rho_<threshold with two decimals, '.' -> '_'>",
        "ratio_definition": "candidate / reference",
        "change_definition": "(ratio - 1) * 100; negative is better",
        "metrics": list(METRICS),
    }


def write_outputs(
    data_dir: str,
    results: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    grid: list[float],
    *,
    complete: bool,
) -> list[dict[str, Any]]:
    for category in results:
        results[category].sort(key=lambda case: case["name"])
    summary = build_summary(results, grid)
    payload = {
        "metadata": metadata(grid, complete),
        "results": results,
        "errors": errors,
        "summary": summary,
    }
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "rho_scan.json")
    temporary_path = f"{json_path}.tmp"
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, json_path)

    write_csv(
        os.path.join(data_dir, "rho_scan.csv"),
        ["category", "name", "qubits", "paulis", "rho_threshold", *METRICS, "compile_seconds"],
        long_rows(results),
    )
    write_csv(os.path.join(data_dir, "rho_scan_summary.csv"), list(summary[0]), summary)
    return summary


def print_markdown_table(summary: list[dict[str, Any]], grid: list[float]) -> None:
    def fmt(ratio: float | None) -> str:
        return "n/a" if ratio is None else f"{100.0 * (ratio - 1.0):+.1f}%"

    by_key = {(row["category"], row["rho_threshold"]): row for row in summary}
    labels = [label for _category, label, _count in CATEGORIES] + ["All"]
    keys = [category for category, _label, _count in CATEGORIES] + ["all"]
    print(f"2Q count / 2Q depth, geomean change vs rho_threshold={REFERENCE_RHO}")
    print("| rho | " + " | ".join(labels) + " |")
    print("| ---: | " + " | ".join("---:" for _ in labels) + " |")
    for rho in grid:
        cells = [
            f"{fmt(by_key[key, rho]['num_2q_ratio_geomean'])} / "
            f"{fmt(by_key[key, rho]['depth_2q_ratio_geomean'])}"
            for key in keys
        ]
        print(f"| {rho:.2f} | " + " | ".join(cells) + " |")


def load_previous_results(
    data_dir: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    json_path = os.path.join(data_dir, "rho_scan.json")
    with open(json_path) as handle:
        payload = json.load(handle)
    if payload.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{json_path} uses an incompatible result schema; rerun without --resume."
        )
    results = {
        category: list(payload["results"].get(category, []))
        for category, _label, _count in CATEGORIES
    }
    # Failed tasks are retried on resume, so stale errors are dropped.
    return results, []


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
        "--step",
        type=float,
        default=DEFAULT_STEP,
        help=f"grid spacing on [0, 1]; must divide 1 (default: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only scan the first N benchmarks of each category (smoke test)",
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help="directory for rho_scan.{json,csv} and rho_scan_summary.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed (benchmark, rho_threshold) pairs from the output JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    grid = rho_grid(args.step)
    data_dir = os.path.abspath(args.output_dir)

    if args.resume and os.path.exists(os.path.join(data_dir, "rho_scan.json")):
        results, errors = load_previous_results(data_dir)
    else:
        results = {category: [] for category, _label, _count in CATEGORIES}
        errors: list[dict[str, Any]] = []

    cases = {
        (category, case["name"]): case for category in results for case in results[category]
    }
    tasks = []
    expected = 0
    for category, _label, _expected in CATEGORIES:
        directory = os.path.join(HAMLIB, category)
        filenames = sorted(name for name in os.listdir(directory) if name.endswith(".json"))
        for filename in filenames[: args.limit]:
            path = os.path.join(directory, filename)
            done_arms = cases.get((category, filename[:-5]), {}).get("arms", {})
            for rho in grid:
                expected += 1
                if arm_key(rho) not in done_arms:
                    tasks.append((os.path.getsize(path), category, path, rho))
    # Largest inputs first, so the long compilations do not form a serial tail.
    tasks.sort(key=lambda task: (-task[0], task[1], task[2], task[3]))

    print(
        f"{len(tasks)} pending of {expected} (benchmark, rho_threshold) pairs: "
        f"{len(grid)} thresholds, {args.jobs} workers",
        file=sys.stderr,
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_arm, category, path, rho): (category, path, rho)
            for _size, category, path, rho in tasks
        }
        for done, future in enumerate(as_completed(futures), start=1):
            category, path, rho = futures[future]
            name = os.path.basename(path)[:-5]
            try:
                arm = future.result()
                case = cases.get((category, name))
                if case is None:
                    case = {
                        "category": category,
                        "name": name,
                        "qubits": arm["qubits"],
                        "paulis": arm["paulis"],
                        "arms": {},
                    }
                    cases[category, name] = case
                    results[category].append(case)
                case["arms"][arm_key(rho)] = {
                    "rho_threshold": rho,
                    **{metric: arm[metric] for metric in METRICS},
                    "compile_seconds": arm["compile_seconds"],
                }
                print(
                    f"[{done}/{len(tasks)}] {category}/{name} rho={rho:.2f}: "
                    f"2q={arm['num_2q']} depth={arm['depth_2q']} "
                    f"({arm['compile_seconds']:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {"category": category, "name": name, "rho_threshold": rho, "error": repr(exc)}
                )
                print(
                    f"[{done}/{len(tasks)}] SKIP {category}/{name} rho={rho:.2f}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            write_outputs(data_dir, results, errors, grid, complete=False)

    completed = sum(
        arm_key(rho) in case["arms"] for case in cases.values() for rho in grid
    )
    full_suite = args.limit is None
    complete = full_suite and completed == expected and not errors
    summary = write_outputs(data_dir, results, errors, grid, complete=complete)
    print_markdown_table(summary, grid)
    print(f"\nsaved raw data -> {os.path.join(data_dir, 'rho_scan.json')}")
    print(f"saved long form -> {os.path.join(data_dir, 'rho_scan.csv')}")
    print(f"saved summary   -> {os.path.join(data_dir, 'rho_scan_summary.csv')}")
    if errors:
        print(f"warning: {len(errors)} task(s) failed; errors stored in the JSON file.")


if __name__ == "__main__":
    main()
