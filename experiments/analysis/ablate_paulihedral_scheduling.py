#!/usr/bin/env python
"""Compare Paulihedral's two block-scheduling passes, GCO and DO, on HAMLib and UCCSD.

Paulihedral ships two schedulers: gate-count-oriented (GCO, lexicographic order of the
blocks) and depth-oriented (DO, blocks sorted by active length, then small blocks packed
next to large ones into parallel layers).  ``bench_utils.paulihedral_pass`` uses DO by
default (as in Paulihedral's own evaluation); this script compiles every benchmark both
ways, exactly as ``bench_hamlib.py``
(HAMLib, all-to-all) and ``bench_uccsd.py`` (UCCSD on all2all / hhex / square, with the
benchmark's own excitation blocks) call the pass, and reports the difference.  DO uses
the artifact's hyperparameters (``length = n // 2``, ``maxiter = 30``).

All-to-all runs go through Paulihedral's FT backend, hhex / square runs through its SC
backend.  Ratios are ``DO / GCO``; a ratio below one means DO is better.

Outputs (``experiments/analysis/ablation_paulihedral_scheduling``):
    paulihedral_scheduling.json
        Per-case absolute metrics for both passes, ratios, errors and the summary.
    paulihedral_scheduling.csv
        One row per (benchmark, device): both passes side by side plus the ratios.
    paulihedral_scheduling_summary.csv
        Geometric-mean ratios per HAMLib category, per UCCSD device and overall, with
        the number of cases that got better / stayed equal / got worse under DO.
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

# One thread per worker process (see ablate_qiskit_preserve_order.py); an explicit
# setting in the environment still wins.
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
UCCSD = os.path.join(REPO, "benchmarks", "uccsd")
DATA_DIR = os.path.join(REPO, "experiments", "analysis", "ablation_paulihedral_scheduling")
JSON_PATH = os.path.join(DATA_DIR, "paulihedral_scheduling.json")
CSV_PATH = os.path.join(DATA_DIR, "paulihedral_scheduling.csv")
SUMMARY_CSV_PATH = os.path.join(DATA_DIR, "paulihedral_scheduling_summary.csv")
# Published Paulihedral numbers (GCO); only used for a consistency check.
REFERENCE_HAMLIB_CSV = os.path.join(REPO, "experiments", "results", "result_hamlib_paulihedral.csv")
REFERENCE_UCCSD_CSV = os.path.join(REPO, "experiments", "results", "result_uccsd_paulihedral.csv")

HAMLIB_CATEGORIES = (
    ("binaryoptimization", "Binary optimization"),
    ("discreteoptimization", "Discrete optimization"),
    ("chemistry", "Chemistry"),
    ("condensedmatter", "Condensed matter"),
)
UCCSD_DEVICES = ("all2all", "hhex", "square")

# The first arm is the reference of every ratio.
ARMS = ("gco", "do")

METRICS = ("num_2q", "depth_2q", "num_gates", "depth")
METRIC_LABELS = {"num_2q": "Num2Q", "depth_2q": "Depth2Q", "num_gates": "NumGates", "depth": "Depth"}


def compile_metrics(suite: str, device: str, path: str, scheduling: str) -> dict[str, float]:
    import bench_utils
    import phoenix

    with open(path) as handle:
        data = json.load(handle)
    paulis, coeffs = data["paulis"], data["coeffs"]
    start = time.perf_counter()
    if suite == "hamlib":
        qc = bench_utils.paulihedral_pass(paulis, coeffs, scheduling=scheduling)
    else:
        coupling_map = getattr(phoenix.utils, f"gene_{device}_coupling_map")(data["num_qubits"])
        qc = bench_utils.paulihedral_pass(
            paulis, coeffs, coupling_map=coupling_map, block_sizes=data.get("block_sizes"), scheduling=scheduling
        )
    elapsed = time.perf_counter() - start
    return {
        "num_2q": qc.num_nonlocal_gates(),
        "depth_2q": qc.depth(lambda inst: inst.operation.num_qubits > 1),
        "num_gates": qc.size(),
        "depth": qc.depth(),
        "compile_seconds": elapsed,
    }


def run_arm(case: dict[str, Any], scheduling: str) -> dict[str, float]:
    import warnings as _warnings

    _warnings.filterwarnings("ignore")
    return compile_metrics(case["suite"], case["device"], case["path"], scheduling)


def geometric_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v and v > 0.0 and math.isfinite(v)]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "identical_cases": sum(all(c["arms"]["gco"][m] == c["arms"]["do"][m] for m in METRICS) for c in cases),
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
        out[f"{metric}_total_gco"] = sum(c["arms"]["gco"][metric] for c in cases)
        out[f"{metric}_total_do"] = sum(c["arms"]["do"][metric] for c in cases)
    return out


def build_summary(cases: list[dict[str, Any]], expected: dict[str, int]) -> list[dict[str, Any]]:
    groups = [(f"hamlib/{key}", f"HAMLib {label}", lambda c, k=key: c["suite"] == "hamlib" and c["group"] == k)
              for key, label in HAMLIB_CATEGORIES]
    groups.append(("hamlib/all", "HAMLib all", lambda c: c["suite"] == "hamlib"))
    groups += [(f"uccsd/{device}", f"UCCSD {device}", lambda c, d=device: c["suite"] == "uccsd" and c["device"] == d)
               for device in UCCSD_DEVICES]
    groups.append(("uccsd/all", "UCCSD all", lambda c: c["suite"] == "uccsd"))
    rows = []
    for key, label, member in groups:
        chosen = [c for c in cases if member(c)]
        rows.append({
            "group": key,
            "group_label": label,
            "expected_cases": expected.get(key, 0),
            "completed_cases": len(chosen),
            "seconds_gco": sum(c["arms"]["gco"]["compile_seconds"] for c in chosen),
            "seconds_do": sum(c["arms"]["do"]["compile_seconds"] for c in chosen),
            **summarize(chosen),
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
    print("DO relative to GCO: geomean change (ratio; better/equal/worse)")
    print("| Group (#) | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " | compile s (GCO / DO) |")
    print("| --- |" + " --- |" * (len(METRICS) + 1))
    for row in summary:
        label = f"{row['group_label']} ({row['completed_cases']}/{row['expected_cases']})"
        print("| {} | {} | {} |".format(label, " | ".join(fmt(row, metric) for metric in METRICS),
                                        f"{row['seconds_gco']:.0f} / {row['seconds_do']:.0f}"))


def print_uccsd_table(cases: list[dict[str, Any]]) -> None:
    rows = sorted((c for c in cases if c["suite"] == "uccsd"), key=lambda c: (UCCSD_DEVICES.index(c["device"]), c["qubits"]))
    if not rows:
        return
    print("\nUCCSD, Num2Q / Depth2Q (GCO -> DO):")
    for c in rows:
        g, d = c["arms"]["gco"], c["arms"]["do"]
        print("  {:8s} {:9s} {:7d} -> {:7d} ({:.3f}x)   {:7d} -> {:7d} ({:.3f}x)".format(
            c["device"], c["name"], g["num_2q"], d["num_2q"], c["ratios"]["num_2q"],
            g["depth_2q"], d["depth_2q"], c["ratios"]["depth_2q"]))


def print_extremes(cases: list[dict[str, Any]], metric: str, count: int = 5) -> None:
    chosen = sorted((c for c in cases if c["suite"] == "hamlib" and c["ratios"][metric]), key=lambda c: c["ratios"][metric])
    if not chosen:
        return
    for title, part in (("largest increase", chosen[::-1][:count]), ("largest decrease", chosen[:count])):
        print(f"\nHAMLib {METRIC_LABELS[metric]}, {title} with DO:")
        for c in part:
            print("  {:.3f}x  {} -> {}  {}/{}".format(
                c["ratios"][metric], c["arms"]["gco"][metric], c["arms"]["do"][metric], c["group"], c["name"]))


def write_case_csv(cases: list[dict[str, Any]]) -> None:
    fieldnames = ["suite", "group", "device", "name", "qubits", "paulis"]
    for arm in ARMS:
        fieldnames += [f"{metric}_{arm}" for metric in METRICS] + [f"compile_seconds_{arm}"]
    fieldnames += [f"{metric}_ratio" for metric in METRICS]
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            rec = {k: case[k] for k in fieldnames[:6]}
            for arm in ARMS:
                for metric in METRICS:
                    rec[f"{metric}_{arm}"] = case["arms"][arm][metric]
                rec[f"compile_seconds_{arm}"] = round(case["arms"][arm]["compile_seconds"], 3)
            for metric in METRICS:
                rec[f"{metric}_ratio"] = case["ratios"][metric]
            writer.writerow(rec)


def write_summary_csv(summary: list[dict[str, Any]]) -> None:
    fieldnames = ["group", "group_label", "completed_cases", "expected_cases", "identical_cases", "seconds_gco", "seconds_do"]
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
            f"{metric}_total_gco",
            f"{metric}_total_do",
        ]
    with open(SUMMARY_CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k) for k in fieldnames})


def check_against_reference(cases: list[dict[str, Any]]) -> list[str]:
    """Compare the GCO arm with the stored benchmark tables, when they exist."""
    notes = []
    if os.path.exists(REFERENCE_HAMLIB_CSV):
        with open(REFERENCE_HAMLIB_CSV, newline="") as handle:
            stored = {(row["category"], row["program"]): row for row in csv.DictReader(handle)}
        columns = {"num_2q": "num_2q_gates(opt)", "depth_2q": "depth_2q(opt)", "num_gates": "num_gates(opt)", "depth": "depth(opt)"}
        compared = same = 0
        for case in cases:
            row = stored.get((case["group"], case["name"])) if case["suite"] == "hamlib" else None
            if row is None:
                continue
            compared += 1
            same += all(int(row[col]) == case["arms"]["gco"][metric] for metric, col in columns.items())
        notes.append(f"HAMLib GCO arm equals experiments/results/result_hamlib_paulihedral.csv on {same}/{compared} benchmarks")
    if os.path.exists(REFERENCE_UCCSD_CSV):
        with open(REFERENCE_UCCSD_CSV, newline="") as handle:
            stored = {row["program"]: row for row in csv.DictReader(handle)}
        compared = same = 0
        for case in cases:
            row = stored.get(case["name"]) if case["suite"] == "uccsd" else None
            if row is None:
                continue
            dev = case["device"]
            columns = {"num_2q": f"num_2q_gates({dev})", "depth_2q": f"depth_2q({dev})", "num_gates": f"num_gates({dev})", "depth": f"depth({dev})"}
            if not all(col in row and row[col] for col in columns.values()):
                continue
            compared += 1
            same += all(int(row[col]) == case["arms"]["gco"][metric] for metric, col in columns.items())
        notes.append(f"UCCSD GCO arm equals experiments/results/result_uccsd_paulihedral.csv on {same}/{compared} cases")
    return notes


def collect_cases(limit: int | None, suites: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cases: list[dict[str, Any]] = []
    expected: dict[str, int] = {}
    if "hamlib" in suites:
        for category, _label in HAMLIB_CATEGORIES:
            directory = os.path.join(HAMLIB, category)
            filenames = sorted(name for name in os.listdir(directory) if name.endswith(".json"))[:limit]
            expected[f"hamlib/{category}"] = len(filenames)
            for name in filenames:
                cases.append({"suite": "hamlib", "group": category, "device": "all2all", "path": os.path.join(directory, name)})
        expected["hamlib/all"] = sum(v for k, v in expected.items() if k.startswith("hamlib/"))
    if "uccsd" in suites:
        filenames = sorted((name for name in os.listdir(UCCSD) if name.endswith(".json")),
                           key=lambda n: int(n.split("_")[-1].split(".")[0]))[:limit]
        for device in UCCSD_DEVICES:
            expected[f"uccsd/{device}"] = len(filenames)
            for name in filenames:
                cases.append({"suite": "uccsd", "group": "uccsd", "device": device, "path": os.path.join(UCCSD, name)})
        expected["uccsd/all"] = len(filenames) * len(UCCSD_DEVICES)
    for case in cases:
        with open(case["path"]) as handle:
            data = json.load(handle)
        case["name"] = os.path.basename(case["path"])[:-5]
        case["qubits"] = len(data["paulis"][0])
        case["paulis"] = len(data["paulis"])
    return cases, expected


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
        "--suite",
        choices=("all", "hamlib", "uccsd"),
        default="all",
        help="benchmark suite(s) to run (default: all); outputs are only written for 'all'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N benchmarks of every HAMLib category / of UCCSD (smoke test; outputs are not written)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = ["hamlib", "uccsd"] if args.suite == "all" else [args.suite]
    cases, expected = collect_cases(args.limit, suites)
    # Largest programs first, so that the slowest compilations do not end up alone at the tail.
    tasks = sorted(((i, arm) for i in range(len(cases)) for arm in ARMS),
                   key=lambda t: -cases[t[0]]["qubits"] * cases[t[0]]["paulis"])
    print(f"{len(cases)} cases x {len(ARMS)} passes, {args.jobs} workers", file=sys.stderr, flush=True)

    arms: dict[int, dict[str, dict[str, float]]] = {i: {} for i in range(len(cases))}
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_arm, cases[i], arm): (i, arm) for i, arm in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            i, arm = futures[future]
            case = cases[i]
            tag = f"{case['suite']}/{case['group']}/{case['name']} [{case['device']}, {arm}]"
            try:
                arms[i][arm] = future.result()
                print(f"[{done}/{len(tasks)}] {tag} {arms[i][arm]['compile_seconds']:.1f}s", file=sys.stderr, flush=True)
            except Exception as exc:
                errors.append({"case": tag, "error": repr(exc)})
                print(f"[{done}/{len(tasks)}] SKIP {tag}: {exc!r}", file=sys.stderr, flush=True)

    completed = []
    for i, case in enumerate(cases):
        if set(arms[i]) != set(ARMS):
            continue
        gco, do = arms[i]["gco"], arms[i]["do"]
        completed.append({
            **{k: v for k, v in case.items() if k != "path"},
            "arms": arms[i],
            "ratios": {metric: (do[metric] / gco[metric]) if gco[metric] else None for metric in METRICS},
        })
    summary = build_summary(completed, expected)

    print_markdown_table(summary)
    print_uccsd_table(completed)
    print_extremes(completed, "depth_2q")
    print_extremes(completed, "num_2q")
    reference_notes = check_against_reference(completed)
    for note in reference_notes:
        print(f"\n{note}")

    if args.limit is not None or args.suite != "all":
        print("\n--limit / --suite given: smoke test only, nothing written.")
    else:
        payload = {
            "metadata": {
                "compiler": "bench_utils.paulihedral_pass (FT backend on all-to-all, SC backend on hhex/square)",
                "metrics": "gate count, 2q gate count, depth and 2q depth of the compiled circuit",
                "ratio_definition": "DO / GCO",
                "do_hyperparameters": "length = num_qubits // 2, maxiter = 30 (as in the Paulihedral artifact)",
                "reference_check": reference_notes,
            },
            "results": completed,
            "errors": errors,
            "summary": summary,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(JSON_PATH, "w") as handle:
            json.dump(payload, handle, indent=2)
        write_case_csv(completed)
        write_summary_csv(summary)
        print(f"\nsaved raw data   -> {JSON_PATH}")
        print(f"saved case table -> {CSV_PATH}")
        print(f"saved summary    -> {SUMMARY_CSV_PATH}")
    if errors:
        print(f"warning: {len(errors)} compilation(s) failed; see the errors above" + (" and in the JSON file." if args.limit is None else "."))


if __name__ == "__main__":
    main()
