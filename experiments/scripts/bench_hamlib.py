#!/usr/bin/env python
"""
Benchmarking on Hamlib benchmarks with given "compiler" and given "category" in Hamlib programs.
"""

import sys

sys.path.append("../..")

import os
import json
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import qiskit
import qiskit.qasm2
from natsort import natsorted
import phoenix
import bench_utils
from functools import partial

from rich.console import Console

console = Console()

INPUT_QASM_DPATH = "../../benchmarks/hamlib_qasm"
INPUT_JSON_DPATH = "../../benchmarks/hamlib_json"
OUTPUT_DPATH = "../output_hamlib"

CATEGORIES = ["binaryoptimization", "chemistry", "condensedmatter", "discreteoptimization"]

COMPILER_PASSES = {
    "phoenix": bench_utils.phoenix_pass,
    "phoenix+": partial(bench_utils.phoenix_pass, grouping=False),
    "paulihedral": bench_utils.paulihedral_pass,
    "tetris": bench_utils.tetris_pass,
    "pauliopt": bench_utils.pauliopt_pass,
    "quclear": bench_utils.quclear_pass,
    "tket": bench_utils.tket_pass,
    "qiskit": bench_utils.qiskit_pass,
}


def process_one(fname, compiler, output_dpath):
    """Compile a single Hamlib benchmark and dump the result. Returns (fname, status, circ_or_msg)."""
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace(".json", ".qasm"))

    # if os.path.exists(output_fname):
    #     return fname, 'cached', output_fname

    with open(fname, "r") as f:
        data = json.load(f)

    compiler_pass = COMPILER_PASSES[compiler]
    circ = compiler_pass(data["paulis"], data["coeffs"], with_O3=True)

    qiskit.qasm2.dump(circ, output_fname)
    return fname, "ok", circ


def main():
    parser = argparse.ArgumentParser(description="Benchmarking on hamlib100 with Phoenix compiler")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type of benchmarks (binaryoptimization, chemistry, condensedmatter, discreteoptimization)",
    )
    parser.add_argument("-c", "--compiler", default="phoenix", type=str, help="Compiler (default: phoenix)")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()

    if args.compiler not in COMPILER_PASSES:
        raise ValueError("Unsupported compiler: {}".format(args.compiler))
    if args.type not in CATEGORIES:
        raise ValueError("Unsupported category: {}".format(args.type))

    json_fnames = [
        os.path.join(INPUT_JSON_DPATH, args.type, fname)
        for fname in natsorted(os.listdir(os.path.join(INPUT_JSON_DPATH, args.type)), reverse=True)
    ]

    output_dpath = os.path.join(OUTPUT_DPATH, args.compiler, args.type)

    if not os.path.exists(output_dpath):
        os.makedirs(output_dpath)

    console.print("program type: {}".format(args.type))
    console.print("compiler: {}".format(args.compiler))
    console.print("output directory: {}".format(output_dpath))
    console.print("parallel jobs: {}".format(args.jobs))

    warnings.filterwarnings("ignore")

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_one, fname, args.compiler, output_dpath): fname for fname in json_fnames}
        for future in as_completed(futures):
            fname = futures[future]
            try:
                _, status, payload = future.result()
                if status == "ok":
                    console.print("Done", fname)
                    phoenix.utils.print_circ_info(payload)
                elif status == "skipped":
                    console.print("[yellow]Skipped[/yellow] {} ({})".format(fname, payload))
                elif status == "cached":
                    console.print("[cyan]Cached[/cyan] {} -> {}".format(fname, payload))
            except Exception as e:
                console.print("[red]Failed[/red] {}: {}".format(fname, e))


if __name__ == "__main__":
    main()
