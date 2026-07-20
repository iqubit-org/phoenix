#!/usr/bin/env python3
"""Serial, resumable per-instance compilation-time measurements.

Examples (from this directory or any other working directory)::

    python runtime_testing.py -c phoenixpp -b uccsd
    python runtime_testing.py -c phoenixpp -b hamlib --limit 1

The timer covers Hamiltonian construction and compilation, but excludes JSON
loading, CSV I/O, process start-up, and progress reporting.  Each completed
instance is appended immediately to ``runtime_<suite>_<compiler>.csv`` so an
interrupted run can be resumed safely.  Existing program names are skipped
unless ``--overwrite`` is specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Callable

from tqdm import tqdm

ANALYSIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ANALYSIS_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"

# Keep the runtime harness executable from any working directory.
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


SUPPORTED_COMPILERS = (
    "naive",
    "phoenix",
    "phoenixpp",
    "qiskit",
    "tket",
    "paulihedral",
    "tetris",
    "quclear",
    "pauliopt",
)


def repetitions_for(num_paulis: int) -> int:
    """Choose 2--5 repetitions from the number of input Pauli strings.

    Short instances need more repetitions to smooth wall-clock noise, whereas
    large instances are expensive enough that two repetitions suffice.
    """
    if num_paulis <= 500:
        return 5
    if num_paulis <= 2_000:
        return 4
    if num_paulis <= 8_000:
        return 3
    return 2


def instances_for(suite: str) -> list[tuple[str, Path]]:
    """Return stable, human-readable program identifiers and JSON paths."""
    if suite == "uccsd":
        root = BENCHMARK_ROOT / "uccsd"
        files = sorted(root.glob("*.json"))
        return [(path.stem, path) for path in files]

    root = BENCHMARK_ROOT / "hamlib"
    files = sorted(root.rglob("*.json"))
    return [(str(path.relative_to(root).with_suffix("")), path) for path in files]


def read_instance(path: Path) -> tuple[list[str], list[float]]:
    with path.open() as handle:
        data = json.load(handle)
    paulis = data["paulis"]
    coeffs = data["coeffs"]
    if not paulis:
        raise ValueError(f"{path} contains no Pauli strings")
    if len(paulis) != len(coeffs):
        raise ValueError(f"{path} has mismatched Pauli-string and coefficient counts")
    return paulis, coeffs


def phoenix_runner(grouping: str) -> Callable[[list[str], list[float]], object]:
    """Return a native PHOENIX runner without importing external baselines."""
    import phoenix

    def run(paulis: list[str], coeffs: list[float]) -> object:
        # Benchmark JSON labels are big-endian; PHOENIX follows Qiskit's
        # little-endian convention, matching experiments/scripts/bench_utils.py.
        hamiltonian = phoenix.Hamiltonian([p[::-1] for p in paulis], coeffs)
        return phoenix.compile_hamiltonian_simulation(hamiltonian, grouping=grouping)

    return run


def compiler_runner(compiler: str) -> Callable[[list[str], list[float]], object]:
    """Build a no-argument-per-instance compiler callable.

    The two PHOENIX variants are intentionally imported directly, allowing
    their runtime to be measured without requiring optional Tetris/QuCLEAR
    packages.  Other compilers reuse the common benchmark wrappers so their
    settings stay aligned with the quality experiments.
    """
    if compiler == "phoenixpp":
        return phoenix_runner("holistic")
    if compiler == "phoenix":
        return phoenix_runner("support")

    import bench_utils

    runners: dict[str, Callable[[list[str], list[float]], object]] = {
        "naive": bench_utils.naive_pass,
        "qiskit": bench_utils.qiskit_pass,
        "tket": bench_utils.tket_pass,
        "paulihedral": bench_utils.paulihedral_pass,
        "tetris": bench_utils.tetris_pass,
        "quclear": bench_utils.quclear_pass,
        "pauliopt": bench_utils.pauliopt_pass,
    }
    return runners[compiler]


def recorded_programs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["program", "runtime"]:
            raise ValueError(f"Unexpected CSV schema in {path}; expected program,runtime")
        return {row["program"] for row in reader}


def append_result(path: Path, program: str, runtime: float) -> None:
    """Append and flush one durable result record."""
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["program", "runtime"])
        writer.writerow([program, f"{runtime:.9f}"])
        handle.flush()
        os.fsync(handle.fileno())


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--compiler", choices=SUPPORTED_COMPILERS, required=True)
    parser.add_argument("-b", "--benchmark-suite", choices=("hamlib", "uccsd"), required=True)
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Measure at most this many pending instances (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this compiler/suite's previous CSV before measuring.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = ANALYSIS_DIR / f"runtime_{args.benchmark_suite}_{args.compiler}(MacOS).csv"
    if args.overwrite and output.exists():
        output.unlink()

    completed = recorded_programs(output)
    all_instances = instances_for(args.benchmark_suite)
    if not all_instances:
        raise FileNotFoundError(f"No {args.benchmark_suite} instances found under {BENCHMARK_ROOT}")
    pending = [(name, path) for name, path in all_instances if name not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print(f"No pending {args.benchmark_suite}/{args.compiler} instances; results are in {output}")
        return 0

    run_compiler = compiler_runner(args.compiler)
    failures: list[tuple[str, str]] = []
    progress = tqdm(pending, desc=f"{args.benchmark_suite}/{args.compiler}", unit="case")
    for program, path in progress:
        try:
            paulis, coeffs = read_instance(path)
            nreps = repetitions_for(len(paulis))
            samples: list[float] = []
            for _ in range(nreps):
                start = time.perf_counter()
                run_compiler(paulis, coeffs)
                samples.append(time.perf_counter() - start)
            runtime = fmean(samples)
            append_result(output, program, runtime)
            progress.set_postfix(reps=nreps, seconds=f"{runtime:.3f}", refresh=False)
        except Exception as exc:  # Continue so a single case does not lose completed measurements.
            failures.append((program, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"FAILED {program}: {failures[-1][1]}")

    if failures:
        print(f"Completed with {len(failures)} failed instance(s); successful rows were saved to {output}.")
        for program, reason in failures:
            print(f"  {program}: {reason}")
        return 1

    print(f"Saved {len(pending)} runtime measurement(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
