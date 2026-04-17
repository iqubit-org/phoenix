#!/usr/bin/env python
"""
Compile Hamlib JSON benchmarks that are still missing under QuCLEAR output.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import qiskit.qasm2
from tqdm import tqdm
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import bench_utils  # noqa: E402

console = Console()


def find_missing_jsons(input_root: Path, output_root: Path) -> list[Path]:
    compiled_relpaths = {
        qasm_path.relative_to(output_root).with_suffix("")
        for qasm_path in output_root.rglob("*.qasm")
    }
    return [
        json_path
        for json_path in sorted(input_root.rglob("*.json"))
        if json_path.relative_to(input_root).with_suffix("") not in compiled_relpaths
    ]


def compile_one(json_path: Path, input_root: Path, output_root: Path) -> tuple[Path, Path, dict[str, int]]:
    with json_path.open("r") as f:
        data = json.load(f)

    circ = bench_utils.quclear_pass(data["paulis"], data["coeffs"], with_O3=True)

    output_path = output_root / json_path.relative_to(input_root).with_suffix(".qasm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qiskit.qasm2.dump(circ, output_path)

    metrics = {
        "num_qubits": circ.num_qubits,
        "num_gates": circ.size(),
        "num_2q_gates": circ.num_nonlocal_gates(),
        "depth": circ.depth(),
        "depth_2q": circ.depth(lambda instr: instr.operation.num_qubits > 1),
    }
    return json_path, output_path, metrics


def main() -> None:
    warnings.filterwarnings("ignore")

    input_root = (REPO_ROOT / "benchmarks" / "hamlib_json").resolve()
    output_root = (REPO_ROOT / "experiments" / "output_hamlib" / "quclear").resolve()

    missing_jsons = find_missing_jsons(input_root, output_root)

    console.print(f"input root: {input_root}")
    console.print(f"output root: {output_root}")
    console.print(f"missing json files: {len(missing_jsons)}")

    if not missing_jsons:
        console.print("[green]No missing JSON files found.[/green]")
        return

    for json_path in missing_jsons:
        console.print(json_path.relative_to(input_root))

    for json_path in tqdm(missing_jsons, desc="Compiling with QuCLEAR"):
        rel_json = json_path.relative_to(input_root)
        try:
            _, output_path, metrics = compile_one(json_path, input_root, output_root)
            console.print(f"[green]Compiled[/green] {rel_json} -> {output_path.relative_to(output_root)}")
            console.print(metrics)
        except Exception as exc:
            console.print(f"[red]Failed[/red] {rel_json}: {exc}")


if __name__ == "__main__":
    main()
