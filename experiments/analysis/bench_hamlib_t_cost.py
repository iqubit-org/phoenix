#!/usr/bin/env python
"""Clifford+T cost (T-count / T-depth) of each compiler over the 100-program HamLib suite.

One compiler per invocation; the result lands in ``analysis/t_cost_data/<compiler>.csv``:

    python bench_hamlib_t_cost.py -c symphony            # all 100, all cores
    python bench_hamlib_t_cost.py -c paulihedral         # Paulihedral baseline
    python bench_hamlib_t_cost.py -c tetris              # Tetris baseline
    python bench_hamlib_t_cost.py -c qiskit -j 32        # pin the worker count
    python bench_hamlib_t_cost.py -c quclear --resume    # continue an interrupted run

Every compiler pass runs with ``optimize=False``, so the T-cost is measured on the raw
compiled circuit without Qiskit's post-optimization.

Parallelism has two levels, and they are mutually exclusive by design:
  * ``-j > 1`` -- benchmarks run concurrently, one per worker process, and the
    Clifford+T synthesis inside each worker falls back to serial automatically
    (``phoenix.utils._synth_rz_angles`` refuses to nest process pools). This is
    the efficient mode for the full suite on a many-core machine.
  * ``-j 1``   -- one benchmark at a time, but ``synth_to_clifford_t`` then uses
    every core for the Ross-Selinger synthesis. Useful for a single large program.

Rows are appended and flushed as each benchmark finishes, so an interrupted run
loses nothing and ``--resume`` picks up where it stopped. The file is sorted by
(category, program) at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import phoenix
import phoenix.utils
from bench_utils import (
    paulihedral_pass,
    phoenix_pass,
    qiskit_pass,
    quclear_pass,
    tetris_pass,
    tket_pass,
)

SUITE_CSV = PROJECT_ROOT / "benchmarks" / "description_hamlib.csv"
BENCH_DIR = PROJECT_ROOT / "benchmarks" / "hamlib"
OUT_DIR = Path(__file__).resolve().parent / "t_cost_data"

COMPILERS = ("symphony", "phoenix", "qiskit", "paulihedral", "tetris", "quclear", "tket")
FIELDS = ["category", "program", "num_qubits", "num_paulis", "t_count", "t_depth", "elapsed"]


def compile_circuit(compiler: str, paulis, coeffs):
    """Dispatch to the compiler pass under test. Every pass runs with ``optimize=False``.

    Each pass is called exactly as bench_hamlib.py calls it (defaults untouched, so
    tket keeps greedy=True), so the T-cost numbers line up with results/result_hamlib_*.csv.
    """
    if compiler == "symphony":
        # holistic grouping == the default of compile_hamiltonian_simulation (the hero)
        return phoenix_pass(paulis, coeffs, optimize=False)
    if compiler == "phoenix":
        # same-support grouping == the DAC'25 Phoenix baseline
        return phoenix_pass(paulis, coeffs, grouping="support", optimize=False)
    if compiler == "qiskit":
        return qiskit_pass(paulis, coeffs, optimize=False)
    if compiler == "paulihedral":
        return paulihedral_pass(paulis, coeffs, optimize=False)
    if compiler == "tetris":
        return tetris_pass(paulis, coeffs, optimize=False)
    if compiler == "quclear":
        return quclear_pass(paulis, coeffs, optimize=False)
    if compiler == "tket":
        return tket_pass(paulis, coeffs, optimize=False)
    raise ValueError(f"unknown compiler {compiler!r}")


def get_t_cost(qc) -> tuple[int, int]:
    """T-count and T-depth of a Clifford+T circuit."""
    return qc.count_ops().get("t", 0), qc.depth(lambda instr: instr.operation.name == "t")


def run_one(task: tuple) -> dict:
    """Compile one HamLib program, synthesize to Clifford+T, report its T-cost.

    Runs in a worker process; exceptions are returned rather than raised so one
    bad program cannot take down the whole suite.
    """
    category, program, compiler, epsilon = task
    t0 = time.perf_counter()
    try:
        with open(BENCH_DIR / category / f"{program}.json") as f:
            data = json.load(f)
        qc = compile_circuit(compiler, data["paulis"], data["coeffs"])
        qc_clifford_t = phoenix.utils.synth_to_clifford_t(qc, epsilon)
        t_count, t_depth = get_t_cost(qc_clifford_t)
        return {
            "category": category,
            "program": program,
            # Counted from the Pauli list, not the JSON's ``num_terms`` field, which
            # is stale for 71/100 programs; this matches description_hamlib.csv.
            "num_qubits": len(data["paulis"][0]),
            "num_paulis": len(data["paulis"]),
            "t_count": t_count,
            "t_depth": t_depth,
            "elapsed": round(time.perf_counter() - t0, 3),
        }
    except Exception:
        return {
            "category": category,
            "program": program,
            "error": traceback.format_exc(limit=3).strip().splitlines()[-1],
            "elapsed": round(time.perf_counter() - t0, 3),
        }


def load_suite() -> list[dict]:
    with open(SUITE_CSV) as f:
        return list(csv.DictReader(f))


def load_done(out_csv: Path) -> set[tuple[str, str]]:
    """(category, program) pairs already present in the output CSV."""
    if not out_csv.exists():
        return set()
    with open(out_csv) as f:
        return {(r["category"], r["program"]) for r in csv.DictReader(f)}


def sort_csv(out_csv: Path) -> None:
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["category"], r["program"]))
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T-count / T-depth of one compiler over the HamLib suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--compiler", required=True, choices=COMPILERS, help="Compiler under test")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                        help="Benchmarks to run concurrently (1 = serial outer loop, "
                             "which lets Clifford+T synthesis use all cores instead)")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-10,
                        help="Ross-Selinger approximation error per rotation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip programs already present in the output CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N programs (smoke test)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV (default: t_cost_data/<compiler>.csv)")
    args = parser.parse_args()

    out_csv = args.output or OUT_DIR / f"{args.compiler}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    suite = load_suite()
    if args.limit:
        suite = suite[: args.limit]

    if args.resume:
        done = load_done(out_csv)
    else:
        done = set()
        out_csv.unlink(missing_ok=True)
    todo = [r for r in suite if (r["category"], r["program"]) not in done]

    tasks = [(r["category"], r["program"], args.compiler, args.epsilon) for r in todo]
    jobs = max(1, min(args.jobs, len(tasks))) if tasks else 1

    print(f"compiler={args.compiler}  programs={len(tasks)} (skipped {len(suite) - len(tasks)})  "
          f"jobs={jobs}  epsilon={args.epsilon:g}  ->  {out_csv}", flush=True)
    if not tasks:
        return

    failures: list[dict] = []
    t_start = time.perf_counter()
    # Header only when the file is genuinely empty -- `done` being empty is not the
    # same thing (a resumed run may find a header-only CSV from an aborted start).
    need_header = not out_csv.exists() or out_csv.stat().st_size == 0

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            writer.writeheader()
            f.flush()

        def record(row: dict, i: int) -> None:
            tag = f"[{i}/{len(tasks)}]"
            if "error" in row:
                failures.append(row)
                print(f"{tag} FAIL {row['program']}: {row['error']}", flush=True)
                return
            writer.writerow(row)
            f.flush()  # keep the file usable if the run is interrupted
            print(f"{tag} {row['program']}  T-count={row['t_count']}  "
                  f"T-depth={row['t_depth']}  ({row['elapsed']}s)", flush=True)

        if jobs == 1:
            # Serial outer loop: synth_to_clifford_t is then free to use every core.
            for i, task in enumerate(tasks, 1):
                record(run_one(task), i)
        else:
            # "fork" keeps the already-imported qiskit/phoenix/tket stack in the
            # workers instead of re-importing it once per worker.
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
                futures = {pool.submit(run_one, t): t for t in tasks}
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        row = future.result()
                    except Exception as exc:
                        # Worker died outright (OOM kill, segfault). Record it and keep
                        # going so one bad program does not discard the whole run;
                        # --resume retries only what is missing.
                        category, program = futures[future][0], futures[future][1]
                        row = {"category": category, "program": program,
                               "error": f"{type(exc).__name__}: {exc}", "elapsed": 0.0}
                    record(row, i)

    sort_csv(out_csv)

    elapsed = time.perf_counter() - t_start
    ok = len(tasks) - len(failures)
    print(f"\ndone in {elapsed:.1f}s -- {ok}/{len(tasks)} succeeded "
          f"-> {os.path.relpath(out_csv, PROJECT_ROOT)}", flush=True)
    for row in failures:
        print(f"  failed: {row['category']}/{row['program']}: {row['error']}", flush=True)


if __name__ == "__main__":
    main()
