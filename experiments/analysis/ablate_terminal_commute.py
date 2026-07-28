#!/usr/bin/env python
"""Ablate SCHEDULE_TERMINAL_COMMUTE (commutation-aware scheduling of the
terminal Clifford tail) on the Hamlib suite.

For every Hamlib benchmark, four configurations are compiled through the full
default pipeline (``phoenix.compile_hamiltonian_simulation``, post-transpile
included; SCHEDULE_ASAP and SCHEDULE_EXACT_COMMUTE stay at their defaults):

    1. terminal="auto",   SCHEDULE_TERMINAL_COMMUTE=False
    2. terminal="auto",   SCHEDULE_TERMINAL_COMMUTE=True
    3. terminal="replay", SCHEDULE_TERMINAL_COMMUTE=False
    4. terminal="replay", SCHEDULE_TERMINAL_COMMUTE=True

The two reported comparisons hold the terminal mode fixed and measure the
effect of turning the tail scheduling on. ``auto`` is the default pipeline
(the pass is inert when the synth tail wins the 2q-count comparison);
``replay`` isolates the pass by forcing the CNOT-equiv tail everywhere.
Ratios are ``on / off``; a ratio below one means fewer 2q gates / shallower.

Outputs:
    experiments/analysis/ablation_data/terminal_commute_scheduling.json
    experiments/analysis/ablation_data/terminal_commute_scheduling.csv
"""

import contextlib
import csv
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import warnings

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAMLIB = os.path.join(REPO, "benchmarks", "hamlib")
DATA_DIR = os.path.join(REPO, "experiments", "analysis", "ablation_data")
JSON_PATH = os.path.join(DATA_DIR, "terminal_commute_scheduling.json")
CSV_PATH = os.path.join(DATA_DIR, "terminal_commute_scheduling.csv")

CATEGORIES = (
    ("binaryoptimization", "Binary optimization", 15),
    ("discreteoptimization", "Discrete optimization", 15),
    ("chemistry", "Chemistry", 35),
    ("condensedmatter", "Condensed matter", 35),
)

CONFIGURATIONS = (
    ("auto_tc_off", "auto", False),
    ("auto_tc_on", "auto", True),
    ("replay_tc_off", "replay", False),
    ("replay_tc_on", "replay", True),
)

COMPARISONS = {
    "auto": ("auto_tc_off", "auto_tc_on"),
    "replay": ("replay_tc_off", "replay_tc_on"),
}

METRICS = ("num_2q", "depth_2q")


def compile_metrics(ham, *, terminal: str, terminal_commute: bool) -> dict[str, int]:
    import phoenix
    import phoenix.primitive.holistic as holistic_mod

    old = holistic_mod.SCHEDULE_TERMINAL_COMMUTE
    try:
        holistic_mod.SCHEDULE_TERMINAL_COMMUTE = terminal_commute
        with contextlib.redirect_stdout(io.StringIO()):
            qc = phoenix.compile_hamiltonian_simulation(ham, terminal=terminal)
        return {
            "num_2q": sum(1 for inst in qc.data if inst.operation.num_qubits == 2),
            "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits == 2),
        }
    finally:
        holistic_mod.SCHEDULE_TERMINAL_COMMUTE = old


def run_case(category: str, path: str) -> dict[str, Any]:
    import warnings as _warnings

    _warnings.filterwarnings("ignore")
    from phoenix.hamiltonian import Hamiltonian

    with open(path) as handle:
        data = json.load(handle)
    ham = Hamiltonian(data["paulis"], data["coeffs"])
    arms = {
        key: compile_metrics(ham, terminal=terminal, terminal_commute=tc)
        for key, terminal, tc in CONFIGURATIONS
    }
    comparisons: dict[str, dict[str, float | None]] = {}
    for name, (off_key, on_key) in COMPARISONS.items():
        record: dict[str, float | None] = {}
        for metric in METRICS:
            off, on = arms[off_key][metric], arms[on_key][metric]
            record[f"{metric}_ratio"] = (on / off) if off else None
        comparisons[name] = record
    return {
        "category": category,
        "name": os.path.basename(path)[:-5],
        "qubits": ham.num_qubits,
        "paulis": len(ham.paulis),
        "arms": arms,
        "comparisons": comparisons,
    }


def geometric_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v and v > 0.0 and math.isfinite(v)]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def summarize(cases: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    out: dict[str, Any] = {"valid_cases": len(cases)}
    for metric in METRICS:
        ratios = [
            c["comparisons"][comparison][f"{metric}_ratio"]
            for c in cases
            if c["comparisons"][comparison][f"{metric}_ratio"]
        ]
        gm = geometric_mean(ratios)
        out[f"{metric}_ratio_geomean"] = gm
        out[f"{metric}_improvement_geomean_pct"] = None if gm is None else 100.0 * (gm - 1.0)
        out[f"{metric}_ratio_best"] = min(ratios, default=None)
    return out


def build_summary(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    all_cases = [c for category, _l, _n in CATEGORIES for c in results[category]]
    for category, label, expected in CATEGORIES:
        rows.append({
            "category": category,
            "category_label": label,
            "expected_cases": expected,
            "completed_cases": len(results[category]),
            **{name: summarize(results[category], name) for name in COMPARISONS},
        })
    rows.append({
        "category": "all",
        "category_label": "All",
        "expected_cases": sum(n for _c, _l, n in CATEGORIES),
        "completed_cases": len(all_cases),
        **{name: summarize(all_cases, name) for name in COMPARISONS},
    })
    return rows


def fmt(gm: float | None) -> str:
    return "n/a" if gm is None else f"{100.0 * (gm - 1.0):+.1f}% ({gm:.3f}x)"


def print_markdown_table(summary: list[dict[str, Any]]) -> None:
    print("| Category (#) | auto Num2Q | auto Depth2Q | replay Num2Q | replay Depth2Q |")
    print("| --- | --- | --- | --- | --- |")
    for row in summary:
        label = f"{row['category_label']} ({row['completed_cases']}/{row['expected_cases']})"
        print("| {} | {} | {} | {} | {} |".format(
            label,
            fmt(row["auto"]["num_2q_ratio_geomean"]),
            fmt(row["auto"]["depth_2q_ratio_geomean"]),
            fmt(row["replay"]["num_2q_ratio_geomean"]),
            fmt(row["replay"]["depth_2q_ratio_geomean"]),
        ))


def write_csv(summary: list[dict[str, Any]]) -> None:
    fieldnames = ["category", "category_label", "completed_cases", "expected_cases"]
    for name in COMPARISONS:
        for metric in METRICS:
            fieldnames += [
                f"{name}_{metric}_ratio_geomean",
                f"{name}_{metric}_improvement_geomean_pct",
                f"{name}_{metric}_ratio_best",
            ]
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            rec = {k: row[k] for k in fieldnames[:4]}
            for name in COMPARISONS:
                for metric in METRICS:
                    rec[f"{name}_{metric}_ratio_geomean"] = row[name][f"{metric}_ratio_geomean"]
                    rec[f"{name}_{metric}_improvement_geomean_pct"] = row[name][f"{metric}_improvement_geomean_pct"]
                    rec[f"{name}_{metric}_ratio_best"] = row[name][f"{metric}_ratio_best"]
            writer.writerow(rec)


def main() -> None:
    jobs = int(os.environ.get("ABLATE_JOBS", max(1, (os.cpu_count() or 2) - 2)))
    tasks = []
    for category, label, _expected in CATEGORIES:
        directory = os.path.join(HAMLIB, category)
        for name in sorted(os.listdir(directory)):
            if name.endswith(".json"):
                tasks.append((category, os.path.join(directory, name)))
    print(f"{len(tasks)} benchmarks x {len(CONFIGURATIONS)} arms, {jobs} workers",
          file=sys.stderr, flush=True)

    results: dict[str, list[dict[str, Any]]] = {c: [] for c, _l, _n in CATEGORIES}
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_case, category, path): (category, path)
                   for category, path in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            category, path = futures[future]
            name = os.path.basename(path)
            try:
                results[category].append(future.result())
                print(f"[{done}/{len(tasks)}] {category}/{name}", file=sys.stderr, flush=True)
            except Exception as exc:
                errors.append({"category": category, "name": name, "error": repr(exc)})
                print(f"[{done}/{len(tasks)}] SKIP {category}/{name}: {exc!r}",
                      file=sys.stderr, flush=True)

    for category in results:
        results[category].sort(key=lambda c: c["name"])
    summary = build_summary(results)
    payload = {
        "metadata": {
            "metric": "2q gate count and 2q depth after phoenix.compile_hamiltonian_simulation",
            "ratio_definition": "SCHEDULE_TERMINAL_COMMUTE=True / False, terminal mode fixed per comparison",
            "configurations": [list(c) for c in CONFIGURATIONS],
        },
        "results": results,
        "errors": errors,
        "summary": summary,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(summary)
    print_markdown_table(summary)
    print(f"\nsaved raw data -> {JSON_PATH}")
    print(f"saved summary  -> {CSV_PATH}")
    if errors:
        print(f"warning: {len(errors)} benchmark(s) failed; errors stored in the JSON file.")


if __name__ == "__main__":
    main()
