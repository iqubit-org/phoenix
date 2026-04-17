#!/usr/bin/env python
"""
Optimize logical circuits by Qiskit O3 for UCCSD benchmarks
"""

import sys

sys.path.append("../..")

import os
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import phoenix
import qiskit
import qiskit.qasm2
from natsort import natsorted

warnings.filterwarnings("ignore")

from rich.console import Console

console = Console()


def process_one(all2all_fname, all2all_opt_fname):
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

    circ = qiskit.QuantumCircuit.from_qasm_file(all2all_fname)
    circ = phoenix.utils.qiskit_O3_all2all(circ)
    qiskit.qasm2.dump(circ, all2all_opt_fname)
    return all2all_fname, all2all_opt_fname, circ


def main():
    parser = argparse.ArgumentParser(description="Optimize logical circuits by Qiskit O3 for UCCSD benchmarks")
    parser.add_argument("-c", "--compiler", default="phoenix", type=str, help="For which compiler (default: phoenix)")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()

    output_dpath = "../output_uccsd/"
    all2all_dpath = os.path.join(output_dpath, args.compiler, "all2all")
    all2all_opt_dpath = os.path.join(output_dpath, args.compiler, "all2all_opt")

    if not os.path.exists(all2all_dpath):
        raise FileNotFoundError("Directory not found: {}".format(all2all_dpath))
    if not os.path.exists(all2all_opt_dpath):
        os.makedirs(all2all_opt_dpath)

    console.rule("Applying Qiskit O3 for {}".format(args.compiler))
    console.print("parallel jobs: {}".format(args.jobs))

    tasks = []
    for fname in natsorted(os.listdir(all2all_dpath)):
        tasks.append((os.path.join(all2all_dpath, fname), os.path.join(all2all_opt_dpath, fname)))

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_one, src, dst): (src, dst) for src, dst in tasks}
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                _, _, circ = future.result()
                console.print("Converted {} -> {}".format(src, dst))
                phoenix.utils.print_circ_info(circ)
            except Exception as e:
                console.print("[red]Failed[/red] {}: {}".format(src, e))


if __name__ == "__main__":
    main()
