#!/usr/bin/env python
"""
Compile the UCCSD benchmarks with every compiler pass that exposes an ``optimize``
switch, with ``optimize=False`` (i.e., without the common post-optimization) and
all other pass parameters at their defaults. All-to-all topology only.

Outputs go to ../output_uccsd/noopt/<compiler>/<benchmark>.qasm
"""

import sys

sys.path.append("../..")

import os
import json
import time
import argparse
import warnings
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import qiskit.qasm2
from natsort import natsorted
import bench_utils

warnings.filterwarnings("ignore")

from rich.console import Console

console = Console()

INPUT_JSON_DPATH = "../../benchmarks/uccsd"
OUTPUT_DPATH = "../output_uccsd/noopt"

# Same names as bench_uccsd.py; naive_pass and pauliopt_pass have no ``optimize`` switch.
COMPILER_PASSES = {
    "tket": bench_utils.tket_pass,
    "qiskit": bench_utils.qiskit_pass,
    "paulihedral": bench_utils.paulihedral_pass,
    "tetris": bench_utils.tetris_pass,
    "quclear": bench_utils.quclear_pass,
    "phoenix": partial(bench_utils.phoenix_pass, grouping="support"),
    "phoenixpp": partial(bench_utils.phoenix_pass, grouping="holistic"),
}


def process_one(fname, compiler):
    """Compile a single UCCSD benchmark without post-optimization and dump the result."""
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

    output_dpath = os.path.join(OUTPUT_DPATH, compiler)
    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace(".json", ".qasm"))
    with open(fname, "r") as f:
        data = json.load(f)

    t0 = time.perf_counter()
    kwargs = {}
    if compiler in ("paulihedral", "tetris"):  # these two consume the benchmark's own blocks
        kwargs["block_sizes"] = data.get("block_sizes")
    circ = COMPILER_PASSES[compiler](data["paulis"], data["coeffs"], optimize=False, **kwargs)
    elapsed = time.perf_counter() - t0

    qiskit.qasm2.dump(circ, output_fname)
    # Return plain stats only: pickling these large circuits back to the parent is slow.
    return {
        "output": output_fname,
        "ops": dict(circ.count_ops()),
        "num_2q": circ.num_nonlocal_gates(),
        "depth_2q": circ.depth(lambda instr: instr.operation.num_qubits > 1),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="UCCSD benchmarks with optimize=False (all2all)")
    parser.add_argument(
        "-c",
        "--compiler",
        nargs="+",
        default=list(COMPILER_PASSES),
        choices=list(COMPILER_PASSES),
        help="Compilers to run (default: all)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()

    # Largest programs first so the long-running tasks do not queue behind small ones.
    json_fnames = [os.path.join(INPUT_JSON_DPATH, fname) for fname in natsorted(os.listdir(INPUT_JSON_DPATH))][::-1]
    for compiler in args.compiler:
        os.makedirs(os.path.join(OUTPUT_DPATH, compiler), exist_ok=True)

    console.print("compilers: {}".format(", ".join(args.compiler)))
    console.print("output directory: {}".format(OUTPUT_DPATH))
    console.print("parallel jobs: {}".format(args.jobs))

    num_failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_one, fname, compiler): (compiler, fname)
            for fname in json_fnames
            for compiler in args.compiler
        }
        for future in as_completed(futures):
            compiler, fname = futures[future]
            try:
                stats = future.result()
                console.print(
                    "Done {:12s} {} | #2Q={} Depth-2Q={} | {:.1f}s | {}".format(
                        compiler,
                        os.path.basename(fname),
                        stats["num_2q"],
                        stats["depth_2q"],
                        stats["elapsed"],
                        stats["ops"],
                    )
                )
            except Exception as e:
                num_failed += 1
                console.print("[red]Failed[/red] {} {}: {!r}".format(compiler, fname, e))

    sys.exit(1 if num_failed else 0)


if __name__ == "__main__":
    main()
