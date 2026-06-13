#!/usr/bin/env python
"""
Profile per-program Phoenix compilation time for the `make phoenix -f Makefile-Hamlib`
pipeline (compiler=phoenix, with_O3=True), across all Hamlib categories.

Each benchmark is compiled in an isolated subprocess with a wall-clock timeout.
Programs exceeding the timeout are marked as "timeout".

Usage:
    # driver (run from scripts/ dir)
    ./profile_phoenix_compile.py [--timeout 180] [--out ../PHOENIX_COMPILE_TIMING.md]

    # worker (internal)
    ./profile_phoenix_compile.py --worker <json_path>
"""

import sys

sys.path.append("../..")

import os
import json
import time
import argparse
import subprocess

INPUT_JSON_DPATH = "../../benchmarks/hamlib"
CATEGORIES = ["binaryoptimization", "chemistry", "condensedmatter", "discreteoptimization"]


def run_worker(json_path):
    """Compile one benchmark, print `COMPILE_TIME <seconds>` to stdout."""
    import warnings

    warnings.filterwarnings("ignore")
    import bench_utils

    with open(json_path, "r") as f:
        data = json.load(f)

    t0 = time.perf_counter()
    bench_utils.phoenix_pass(data["paulis"], data["coeffs"], with_O3=True)
    dt = time.perf_counter() - t0
    print("COMPILE_TIME {:.4f}".format(dt))


def program_props(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    paulis = data["paulis"]
    n_pauli = len(paulis)
    n_qubit = len(paulis[0]) if paulis else 0
    return n_qubit, n_pauli


def profile_one(json_path, timeout):
    """Spawn a worker subprocess; return (status, compile_time_or_None, wall_time)."""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", os.path.abspath(json_path)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout", None, time.perf_counter() - t0
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return "error", None, wall
    ct = None
    for line in proc.stdout.splitlines():
        if line.startswith("COMPILE_TIME"):
            ct = float(line.split()[1])
    if ct is None:
        return "error", None, wall
    return "ok", ct, wall


def fmt_time(status, ct):
    if status == "timeout":
        return "timeout (>{}s)"
    if status == "error":
        return "error"
    return "{:.2f}".format(ct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=180, help="per-program timeout in seconds (default 180 = 3 min)")
    parser.add_argument("--out", type=str, default="../PHOENIX_COMPILE_TIMING.md")
    parser.add_argument("--progress", type=str, default="../phoenix_compile_timing.json")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker)
        return

    results = {}  # category -> list of dicts
    for cat in CATEGORIES:
        cat_dir = os.path.join(INPUT_JSON_DPATH, cat)
        fnames = sorted(os.listdir(cat_dir))
        results[cat] = []
        for i, fname in enumerate(fnames):
            jp = os.path.join(cat_dir, fname)
            n_qubit, n_pauli = program_props(jp)
            status, ct, wall = profile_one(jp, args.timeout)
            rec = {
                "name": fname.replace(".json", ""),
                "n_qubit": n_qubit,
                "n_pauli": n_pauli,
                "status": status,
                "compile_time": ct,
            }
            results[cat].append(rec)
            disp = "timeout" if status == "timeout" else ("error" if status == "error" else "{:.2f}s".format(ct))
            print(
                "[{}] {}/{} {}  q={} P={}  -> {}".format(cat, i + 1, len(fnames), rec["name"], n_qubit, n_pauli, disp),
                flush=True,
            )
            # incremental dump
            with open(args.progress, "w") as f:
                json.dump(results, f, indent=2)

    write_markdown(results, args.out, args.timeout)
    print("\nWrote", args.out)


def write_markdown(results, out_path, timeout):
    lines = []
    lines.append("# Phoenix Compilation Timing on Hamlib Benchmarks\n")
    lines.append(
        "Per-program compilation time for the `make phoenix -f Makefile-Hamlib` pipeline "
        "(`compiler=phoenix`, `with_O3=True`), measured by isolating each benchmark in its own "
        "subprocess and timing the `phoenix_pass` call (Hamiltonian construction + "
        "`compile_hamiltonian_simulation` + O3 all-to-all optimization).\n"
    )
    lines.append("- **Timeout threshold:** {} s (3 min). Programs exceeding it are marked `timeout`.".format(timeout))
    lines.append("- **Program property:** `Qubits` = Pauli-string length, `Paulis` = number of Pauli terms.")
    lines.append("- Measurements are sequential (one program at a time) for fair per-program timing.\n")

    # Summary
    total = sum(len(v) for v in results.values())
    n_timeout = sum(1 for v in results.values() for r in v if r["status"] == "timeout")
    n_error = sum(1 for v in results.values() for r in v if r["status"] == "error")
    n_ok = total - n_timeout - n_error
    lines.append("## Summary\n")
    lines.append("| Category | Programs | Completed | Timeout | Error |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for cat, recs in results.items():
        c_to = sum(1 for r in recs if r["status"] == "timeout")
        c_er = sum(1 for r in recs if r["status"] == "error")
        c_ok = len(recs) - c_to - c_er
        lines.append("| {} | {} | {} | {} | {} |".format(cat, len(recs), c_ok, c_to, c_er))
    lines.append("| **Total** | **{}** | **{}** | **{}** | **{}** |".format(total, n_ok, n_timeout, n_error))
    lines.append("")

    # Per-category detail
    for cat, recs in results.items():
        lines.append("## {}\n".format(cat))
        lines.append("| Program | Qubits | Paulis | Compile time (s) |")
        lines.append("| --- | ---: | ---: | ---: |")
        # sort by n_qubit then n_pauli for readability
        for r in sorted(recs, key=lambda r: (r["n_qubit"], r["n_pauli"])):
            if r["status"] == "timeout":
                t = "**timeout**"
            elif r["status"] == "error":
                t = "error"
            else:
                t = "{:.2f}".format(r["compile_time"])
            lines.append("| {} | {} | {} | {} |".format(r["name"], r["n_qubit"], r["n_pauli"], t))
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
