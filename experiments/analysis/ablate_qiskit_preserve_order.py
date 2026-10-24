#!/usr/bin/env python
"""Measure what ``preserve_order`` costs the Qiskit baseline on the HAMLib suite.

``bench_utils.qiskit_pass`` synthesizes the Pauli evolution with Qiskit's Rustiq
plugin.  With ``preserve_order=False`` (the default, used for every published
Qiskit number) Rustiq may reorder the Pauli rotations freely; with
``preserve_order=True`` it has to keep the input order (up to commutation).
This script compiles every HAMLib benchmark both ways, exactly as
``bench_hamlib.py`` calls the pass (all-to-all, common post-optimization
included), and reports how much the constraint changes the circuits.

Ratios are ``preserve_order=True / preserve_order=False``; a ratio above one
means that keeping the input order costs gates / depth.

Outputs (``experiments/analysis/ablation_qiskit_preserve_order``):
    qiskit_preserve_order.json
        Per-benchmark absolute metrics for both settings, ratios, errors and
        the summary.
    qiskit_preserve_order.csv
        One row per benchmark: both settings side by side plus the ratios.
    qiskit_preserve_order_summary.csv
        Per-category and overall geometric-mean ratios, with the number of
        benchmarks that got better / stayed equal / got worse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (REPO, os.path.join(REPO, "experiments", "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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

HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
DATA_DIR = os.path.join(REPO, "experiments", "analysis", "ablation_qiskit_preserve_order")
JSON_PATH = os.path.join(DATA_DIR, "qiskit_preserve_order.json")
CSV_PATH = os.path.join(DATA_DIR, "qiskit_preserve_order.csv")
SUMMARY_CSV_PATH = os.path.join(DATA_DIR, "qiskit_preserve_order_summary.csv")
# Published Qiskit numbers (preserve_order=False); only used for a consistency check.
REFERENCE_CSV_PATH = os.path.join(REPO, "experiments", "results", "result_hamlib_qiskit.csv")

CATEGORIES = (
    ("binaryoptimization", "Binary optimization", 15),
    ("discreteoptimization", "Discrete optimization", 15),
    ("chemistry", "Chemistry", 35),
    ("condensedmatter", "Condensed matter", 35),
)

# (key, preserve_order); the first arm is the reference of every ratio.
ARMS = (
    ("reorder", False),
    ("preserve", True),
)

METRICS = ("num_2q", "depth_2q", "num_gates", "depth")
METRIC_LABELS = {"num_2q": "Num2Q", "depth_2q": "Depth2Q", "num_gates": "NumGates", "depth": "Depth"}


def compile_metrics(paulis: list[str], coeffs: list[float], preserve_order: bool) -> dict[str, float]:
    import bench_utils

    start = time.perf_counter()
    qc = bench_utils.qiskit_pass(paulis, coeffs, preserve_order=preserve_order)
    elapsed = time.perf_counter() - start
    return {
        "num_2q": qc.num_nonlocal_gates(),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits > 1),
        "num_gates": qc.size(),
        "depth": qc.depth(),
        "compile_seconds": elapsed,
    }


def run_case(category: str, path: str) -> dict[str, Any]:
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

    with open(path) as handle:
        data = json.load(handle)
    paulis, coeffs = data["paulis"], data["coeffs"]
    arms = {key: compile_metrics(paulis, coeffs, preserve_order) for key, preserve_order in ARMS}
    reference, candidate = arms[ARMS[0][0]], arms[ARMS[1][0]]
    ratios = {
        metric: (candidate[metric] / reference[metric]) if reference[metric] else None
        for metric in METRICS
    }
    return {
        "category": category,
        "name": os.path.basename(path)[:-5],
        "qubits": len(paulis[0]),
        "paulis": len(paulis),
        "arms": arms,
        "ratios": ratios,
    }


def geometric_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v and v > 0.0 and math.isfinite(v)]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "identical_cases": sum(
            all(c["arms"]["reorder"][m] == c["arms"]["preserve"][m] for m in METRICS) for c in cases
        ),
    }
    for metric in METRICS:
        ratios = [c["ratios"][metric] for c in cases if c["ratios"][metric]]
        gm = geometric_mean(ratios)
        out[f"{metric}_valid_cases"] = len(ratios)
        out[f"{metric}_ratio_geomean"] = gm
        out[f"{metric}_change_geomean_pct"] = None if gm is None else 100.0 * (gm - 1.0)
        out[f"{metric}_improved_cases"] = sum(r < 1.0 for r in ratios)
        out[f"{metric}_equal_cases"] = sum(r == 1.0 for r in ratios)
        out[f"{metric}_regressed_cases"] = sum(r > 1.0 for r in ratios)
        out[f"{metric}_ratio_min"] = min(ratios, default=None)
        out[f"{metric}_ratio_max"] = max(ratios, default=None)
        out[f"{metric}_total_reorder"] = sum(c["arms"]["reorder"][metric] for c in cases)
        out[f"{metric}_total_preserve"] = sum(c["arms"]["preserve"][metric] for c in cases)
    return out


def build_summary(results: dict[str, list[dict[str, Any]]], expected: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    all_cases = [c for category, _label, _n in CATEGORIES for c in results[category]]
    for category, label, _n in CATEGORIES:
        rows.append({
            "category": category,
            "category_label": label,
            "expected_cases": expected[category],
            "completed_cases": len(results[category]),
            **summarize(results[category]),
        })
    rows.append({
        "category": "all",
        "category_label": "All",
        "expected_cases": sum(expected.values()),
        "completed_cases": len(all_cases),
        **summarize(all_cases),
    })
    return rows


def fmt(row: dict[str, Any], metric: str) -> str:
    gm = row[f"{metric}_ratio_geomean"]
    if gm is None:
        return "n/a"
    return "{:+.1f}% ({:.3f}x; {}/{}/{})".format(
        100.0 * (gm - 1.0),
        gm,
        row[f"{metric}_improved_cases"],
        row[f"{metric}_equal_cases"],
        row[f"{metric}_regressed_cases"],
    )


def print_markdown_table(summary: list[dict[str, Any]]) -> None:
    print("preserve_order=True relative to preserve_order=False: geomean change (ratio; better/equal/worse)")
    print("| Category (#) | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |")
    print("| --- |" + " --- |" * len(METRICS))
    for row in summary:
        label = f"{row['category_label']} ({row['completed_cases']}/{row['expected_cases']})"
        print("| {} | {} |".format(label, " | ".join(fmt(row, metric) for metric in METRICS)))


def print_extremes(results: dict[str, list[dict[str, Any]]], count: int = 5) -> None:
    cases = [c for cs in results.values() for c in cs if c["ratios"]["num_2q"]]
    if not cases:
        return
    ordered = sorted(cases, key=lambda c: c["ratios"]["num_2q"])
    for title, chosen in (
        ("largest Num2Q increase with preserve_order=True", ordered[::-1][:count]),
        ("largest Num2Q decrease with preserve_order=True", ordered[:count]),
    ):
        print(f"\n{title}:")
        for c in chosen:
            print("  {:.3f}x  {} -> {}  {}/{}".format(
                c["ratios"]["num_2q"],
                c["arms"]["reorder"]["num_2q"],
                c["arms"]["preserve"]["num_2q"],
                c["category"],
                c["name"],
            ))


def write_case_csv(results: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = ["category", "name", "qubits", "paulis"]
    for key, _preserve_order in ARMS:
        fieldnames += [f"{metric}_{key}" for metric in METRICS] + [f"compile_seconds_{key}"]
    fieldnames += [f"{metric}_ratio" for metric in METRICS]
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for category, _label, _n in CATEGORIES:
            for case in results[category]:
                rec = {k: case[k] for k in fieldnames[:4]}
                for key, _preserve_order in ARMS:
                    for metric in METRICS:
                        rec[f"{metric}_{key}"] = case["arms"][key][metric]
                    rec[f"compile_seconds_{key}"] = round(case["arms"][key]["compile_seconds"], 3)
                for metric in METRICS:
                    rec[f"{metric}_ratio"] = case["ratios"][metric]
                writer.writerow(rec)


def write_summary_csv(summary: list[dict[str, Any]]) -> None:
    fieldnames = ["category", "category_label", "completed_cases", "expected_cases", "identical_cases"]
    for metric in METRICS:
        fieldnames += [
            f"{metric}_valid_cases",
            f"{metric}_ratio_geomean",
            f"{metric}_change_geomean_pct",
            f"{metric}_improved_cases",
            f"{metric}_equal_cases",
            f"{metric}_regressed_cases",
            f"{metric}_ratio_min",
            f"{metric}_ratio_max",
            f"{metric}_total_reorder",
            f"{metric}_total_preserve",
        ]
    with open(SUMMARY_CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in fieldnames})


def check_against_reference(results: dict[str, list[dict[str, Any]]]) -> str | None:
    """Compare the preserve_order=False arm with the stored benchmark table, when that table exists."""
    if not os.path.exists(REFERENCE_CSV_PATH):
        return None
    with open(REFERENCE_CSV_PATH, newline="") as handle:
        stored = {(row["category"], row["program"]): row for row in csv.DictReader(handle)}
    columns = {"num_2q": "num_2q_gates(opt)", "depth_2q": "depth_2q(opt)", "num_gates": "num_gates(opt)", "depth": "depth(opt)"}
    compared = same = 0
    for category, cases in results.items():
        for case in cases:
            row = stored.get((category, case["name"]))
            if row is None:
                continue
            compared += 1
            same += all(int(row[col]) == case["arms"]["reorder"][metric] for metric, col in columns.items())
    return f"preserve_order=False arm equals experiments/results/result_hamlib_qiskit.csv on {same}/{compared} benchmarks"


def parse_args() -> argparse.Namespace:
    default_jobs = max(1, (os.cpu_count() or 2) - 2)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("ABLATE_JOBS", default_jobs)),
        help=f"worker processes (default: ABLATE_JOBS or {default_jobs})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N benchmarks of every category (smoke test; outputs are not written)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = []
    expected: dict[str, int] = {}
    for category, _label, _n in CATEGORIES:
        directory = os.path.join(HAMLIB, category)
        filenames = sorted(name for name in os.listdir(directory) if name.endswith(".json"))
        filenames = filenames[: args.limit]
        expected[category] = len(filenames)
        tasks += [(category, os.path.join(directory, name)) for name in filenames]
    print(f"{len(tasks)} benchmarks x {len(ARMS)} settings, {args.jobs} workers", file=sys.stderr, flush=True)

    results: dict[str, list[dict[str, Any]]] = {c: [] for c, _l, _n in CATEGORIES}
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, category, path): (category, path) for category, path in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            category, path = futures[future]
            name = os.path.basename(path)
            try:
                results[category].append(future.result())
                print(f"[{done}/{len(tasks)}] {category}/{name}", file=sys.stderr, flush=True)
            except Exception as exc:
                errors.append({"category": category, "name": name, "error": repr(exc)})
                print(f"[{done}/{len(tasks)}] SKIP {category}/{name}: {exc!r}", file=sys.stderr, flush=True)

    for category in results:
        results[category].sort(key=lambda c: c["name"])
    summary = build_summary(results, expected)
    print_markdown_table(summary)
    print_extremes(results)
    reference_note = check_against_reference(results)
    if reference_note:
        print(f"\n{reference_note}")

    if args.limit is not None:
        print("\n--limit given: smoke test only, nothing written.")
    else:
        payload = {
            "metadata": {
                "compiler": "bench_utils.qiskit_pass (Rustiq synthesis + common post-optimization, all-to-all)",
                "metrics": "gate count, 2q gate count, depth and 2q depth of the compiled circuit",
                "ratio_definition": "preserve_order=True / preserve_order=False",
                "arms": [list(arm) for arm in ARMS],
                "reference_check": reference_note,
            },
            "results": results,
            "errors": errors,
            "summary": summary,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(JSON_PATH, "w") as handle:
            json.dump(payload, handle, indent=2)
        write_case_csv(results)
        write_summary_csv(summary)
        print(f"\nsaved raw data  -> {JSON_PATH}")
        print(f"saved case table -> {CSV_PATH}")
        print(f"saved summary   -> {SUMMARY_CSV_PATH}")
    if errors:
        print(f"warning: {len(errors)} benchmark(s) failed; see the errors above" + (" and in the JSON file." if args.limit is None else "."))


if __name__ == "__main__":
    main()
