#!/usr/bin/env python
"""Ablate support-mode group ordering on UCCSD benchmarks.

Compares the current support baseline

    hamiltonian.group_same_weights()[::-1]

against the natural order returned by ``group_same_weights()``.  The rest of
the support pipeline is intentionally kept identical to ``compiler.py``:
same-group simplification, TSP ordering, and optional ``post_transpile``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from natsort import natsorted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import phoenix
from phoenix.compiler import _simplify_groups
from phoenix.hamiltonian import Hamiltonian
from phoenix.primitive.ordering import order_circuits
from phoenix.utils import post_transpile

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "benchmarks" / "uccsd"
DEFAULT_JSON = ROOT / "experiments" / "support_group_reverse_ablation.json"
DEFAULT_LOG = ROOT / "experiments" / "support_group_reverse_ablation.log"


def metrics(qc):
    return {
        "num_2q": int(sum(1 for inst in qc.data if inst.operation.num_qubits == 2)),
        "depth_2q": int(qc.depth(lambda inst: inst.operation.num_qubits == 2)),
        "num_gates": int(qc.size()),
        "depth": int(qc.depth()),
    }


def compile_support(
    ham: Hamiltonian,
    *,
    reverse_groups: bool,
    optimize: bool,
    order_method: str,
    backend: str,
    parallel_search: bool,
    search_patience: int | None,
):
    hams = ham.group_same_weights()
    if reverse_groups:
        hams = hams[::-1]
    circuits = _simplify_groups(hams, backend, parallel=parallel_search, patience=search_patience)
    qc = order_circuits(circuits, method=order_method)
    if optimize:
        qc = post_transpile(qc)
    return qc, len(hams)


def geomean(xs):
    xs = [float(x) for x in xs if x and np.isfinite(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--order-method", default="tsp", choices=["trivial", "greedy", "tsp"])
    parser.add_argument("--backend", default="sequential", choices=["sequential", "joblib", "concurrent.futures"])
    parser.add_argument("--parallel-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-patience", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N benchmarks.")
    args = parser.parse_args()

    paths = [args.input / name for name in natsorted(os.listdir(args.input)) if name.endswith(".json")]
    if args.limit is not None:
        paths = paths[: args.limit]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_log.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with args.out_log.open("w") as log:
        def emit(msg: str):
            print(msg, flush=True)
            print(msg, file=log, flush=True)

        emit(
            "support group-order ablation: "
            f"input={args.input} optimize={args.optimize} order={args.order_method} "
            f"backend={args.backend} parallel_search={args.parallel_search}"
        )

        for idx, path in enumerate(paths, start=1):
            with path.open() as f:
                data = json.load(f)
            # Phoenix uses little-endian Pauli strings in the benchmark harness.
            paulis = [p[::-1] for p in data["paulis"]]
            ham = Hamiltonian(paulis, data["coeffs"])
            row = {
                "name": path.stem,
                "num_qubits": int(data["num_qubits"]),
                "num_paulis": len(data["paulis"]),
            }
            emit(f"[{idx}/{len(paths)}] {path.stem}: {row['num_paulis']}P {row['num_qubits']}q")

            for label, reverse in [("original", False), ("reversed", True)]:
                t0 = time.perf_counter()
                qc, ngroups = compile_support(
                    ham,
                    reverse_groups=reverse,
                    optimize=args.optimize,
                    order_method=args.order_method,
                    backend=args.backend,
                    parallel_search=args.parallel_search,
                    search_patience=args.search_patience,
                )
                dt = time.perf_counter() - t0
                row[label] = {
                    **metrics(qc),
                    "time": round(dt, 3),
                    "num_groups": int(ngroups),
                }
                emit(
                    f"  {label:8s} 2q={row[label]['num_2q']:8d} "
                    f"depth2q={row[label]['depth_2q']:8d} t={dt:8.2f}s"
                )

            row["ratio_reversed_over_original"] = {
                "num_2q": row["reversed"]["num_2q"] / row["original"]["num_2q"]
                if row["original"]["num_2q"]
                else None,
                "depth_2q": row["reversed"]["depth_2q"] / row["original"]["depth_2q"]
                if row["original"]["depth_2q"]
                else None,
                "time": row["reversed"]["time"] / row["original"]["time"]
                if row["original"]["time"]
                else None,
            }
            emit(
                "  ratio reversed/original "
                f"2q={row['ratio_reversed_over_original']['num_2q']:.4f} "
                f"depth2q={row['ratio_reversed_over_original']['depth_2q']:.4f} "
                f"time={row['ratio_reversed_over_original']['time']:.4f}"
            )
            results.append(row)

            with args.out_json.open("w") as f:
                json.dump({"results": results}, f, indent=2)

        ratios_2q = [r["ratio_reversed_over_original"]["num_2q"] for r in results]
        ratios_d2 = [r["ratio_reversed_over_original"]["depth_2q"] for r in results]
        ratios_t = [r["ratio_reversed_over_original"]["time"] for r in results]
        summary = {
            "num_cases": len(results),
            "geomean_reversed_over_original": {
                "num_2q": geomean(ratios_2q),
                "depth_2q": geomean(ratios_d2),
                "time": geomean(ratios_t),
            },
            "wins_reversed": {
                "num_2q": int(sum(r < 1 for r in ratios_2q)),
                "depth_2q": int(sum(r < 1 for r in ratios_d2)),
            },
            "losses_reversed": {
                "num_2q": int(sum(r > 1 for r in ratios_2q)),
                "depth_2q": int(sum(r > 1 for r in ratios_d2)),
            },
        }
        payload = {
            "settings": {
                "input": str(args.input),
                "optimize": args.optimize,
                "order_method": args.order_method,
                "backend": args.backend,
                "parallel_search": args.parallel_search,
                "search_patience": args.search_patience,
            },
            "summary": summary,
            "results": results,
        }
        with args.out_json.open("w") as f:
            json.dump(payload, f, indent=2)
        emit("\nsummary:")
        emit(json.dumps(summary, indent=2))
        emit(f"saved json -> {args.out_json}")
        emit(f"saved log  -> {args.out_log}")


if __name__ == "__main__":
    main()
