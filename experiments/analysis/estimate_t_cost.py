#!/usr/bin/env python3
"""Estimate Clifford+T resources for an OpenQASM circuit with Qiskit GridSynth.

The input is first rebased to ``{rz, sx, x, cx}``, where every gate except
``rz(theta)`` is Clifford.  Each resulting RZ rotation is then synthesized by
GridSynth, up to a global phase, using the requested approximation tolerance.
The reported T-depth respects the circuit dependencies: two-qubit gates
synchronize the T-layer frontiers of their operands, while every synthesized
RZ contributes its local T-count sequentially on its qubit.  Synthesis uses
Qiskit's Rust ``rsgridsynth`` binding (``qiskit.synthesis.gridsynth_rz``),
which implements the Ross--Selinger algorithm.

Usage:

    python estimate_t_cost.py <input_qasm_filename>

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from qiskit import QuantumCircuit, transpile
from qiskit.synthesis import gridsynth_rz

T_GATE_NAMES = {"t", "tdg"}
NON_CLIFFORD_GATE_NAMES = {
    "p",
    "rx",
    "rxx",
    "rxy",
    "ry",
    "ryy",
    "rzx",
    "u",
    "u1",
    "u2",
    "u3",
}
CLIFFORD_T_BASIS = ("rz", "sx", "x", "cx")


@dataclass(frozen=True)
class TResources:
    """RZ-only Clifford+T estimate for one input circuit."""

    num_rz: int
    num_existing_t: int
    num_unique_rz_angles: int
    t_count: int
    t_depth: int
    unsupported_gate_counts: dict[str, int]


class QiskitGridSynth:
    """Memoized Ross--Selinger synthesis via Qiskit's ``rsgridsynth`` binding."""

    def __init__(self, epsilon: float) -> None:
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("GridSynth epsilon must be finite and positive")
        self.epsilon = epsilon
        self._cache: dict[str, int] = {}

    @property
    def num_unique_angles(self) -> int:
        return len(self._cache)

    def t_count(self, angle: float) -> int:
        if not math.isfinite(angle):
            raise ValueError(f"RZ angle must be finite, got {angle!r}")

        # RZ(0) has no T cost and does not require entering the synthesizer.
        if angle == 0.0:
            return 0

        angle_text = format(angle, ".17g")
        if angle_text in self._cache:
            return self._cache[angle_text]

        synthesized = gridsynth_rz(angle, epsilon=self.epsilon)
        count = sum(instruction.operation.name.lower() in T_GATE_NAMES for instruction in synthesized.data)
        self._cache[angle_text] = count
        return count


def estimate_t_resources(
    circuit: QuantumCircuit, rz_t_count: Callable[[float], int], unique_angles: Callable[[], int]
) -> TResources:
    """Estimate T-count and dependency-respecting T-depth for ``circuit``.

    The estimator synthesizes only explicit ``rz`` instructions.  Other
    arbitrary rotations are reported separately, rather than silently being
    counted as Clifford gates.  Their presence makes the result an RZ-only
    lower bound for the full circuit's T cost.
    """
    t_layers = [0] * circuit.num_qubits
    total_t_count = 0
    num_rz = 0
    num_existing_t = 0
    unsupported: dict[str, int] = {}

    def qubit_indices(instruction) -> list[int]:
        return [circuit.find_bit(qubit).index for qubit in instruction.qubits]

    def synchronize(indices: list[int]) -> None:
        if len(indices) < 2:
            return
        frontier = max(t_layers[index] for index in indices)
        for index in indices:
            t_layers[index] = frontier

    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name.lower()
        indices = qubit_indices(instruction)

        if name == "rz":
            if len(indices) != 1 or len(operation.params) != 1:
                raise ValueError(f"Unexpected RZ instruction: {operation!r}")
            try:
                angle = float(operation.params[0])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Cannot synthesize symbolic RZ angle {operation.params[0]!r}") from error
            local_t_count = rz_t_count(angle)
            total_t_count += local_t_count
            t_layers[indices[0]] += local_t_count
            num_rz += 1
            continue

        if name in T_GATE_NAMES:
            if len(indices) != 1:
                raise ValueError(f"Unexpected {name!r} instruction: {operation!r}")
            total_t_count += 1
            t_layers[indices[0]] += 1
            num_existing_t += 1
            continue

        if name in NON_CLIFFORD_GATE_NAMES:
            unsupported[name] = unsupported.get(name, 0) + 1

        # Any multi-qubit operation transfers the accumulated T-layer
        # dependency between its wires, even if the operation itself is
        # Clifford and therefore adds no T layer.
        synchronize(indices)

    return TResources(
        num_rz=num_rz,
        num_existing_t=num_existing_t,
        num_unique_rz_angles=unique_angles(),
        t_count=total_t_count,
        t_depth=max(t_layers, default=0),
        unsupported_gate_counts=unsupported,
    )


def load_qasm(path: Path) -> QuantumCircuit:
    """Load OpenQASM 2, with an OpenQASM 3 fallback when available."""
    try:
        return QuantumCircuit.from_qasm_file(path)
    except Exception as qasm2_error:
        try:
            from qiskit import qasm3

            return qasm3.load(path)
        except Exception as qasm3_error:
            raise ValueError(
                f"Could not parse {path} as OpenQASM 2 or OpenQASM 3:\n"
                f"OpenQASM 2: {qasm2_error}\nOpenQASM 3: {qasm3_error}"
            ) from qasm3_error


def rebase_to_clifford_rz(circuit: QuantumCircuit) -> QuantumCircuit:
    """Express Qiskit ``u`` gates as RZ rotations and Clifford gates.

    ``sx``, ``x``, and ``cx`` are Clifford, so their exact fault-tolerant cost
    is zero.  This makes the T estimate meaningful for the existing benchmark
    QASM files, which are commonly saved in the ``{u, cx}`` basis.
    """
    try:
        return transpile(circuit, basis_gates=list(CLIFFORD_T_BASIS), optimization_level=0)
    except Exception as error:
        raise ValueError(f"Could not rebase circuit to {CLIFFORD_T_BASIS}: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_qasm_filename", type=Path, help="Input OpenQASM file")
    parser.add_argument(
        "--epsilon", type=float, default=1e-10, help="Ross--Selinger approximation epsilon (default: 1e-10)"
    )
    parser.add_argument(
        "--no-rebase",
        action="store_true",
        help="Analyze only the RZ gates already explicit in the input QASM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_qasm_filename.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input QASM file not found: {input_path}")
    circuit = load_qasm(input_path)
    if not args.no_rebase:
        circuit = rebase_to_clifford_rz(circuit)
    synthesizer = QiskitGridSynth(epsilon=args.epsilon)
    resources = estimate_t_resources(circuit, synthesizer.t_count, lambda: synthesizer.num_unique_angles)

    print(f"Input: {input_path}")
    print(f"Qubits: {circuit.num_qubits}")
    if not args.no_rebase:
        print("Basis rebase: {rz, sx, x, cx} (optimization level 0)")
    print(f"RZ gates synthesized: {resources.num_rz} ({resources.num_unique_rz_angles} unique nonzero angles)")
    print(f"Existing T/Tdg gates: {resources.num_existing_t}")
    print(f"Qiskit Ross--Selinger GridSynth epsilon: {args.epsilon} (global phase ignored)")
    print(f"T-count: {resources.t_count}")
    print(f"T-depth: {resources.t_depth}")
    if resources.unsupported_gate_counts:
        unsupported = ", ".join(f"{name}={count}" for name, count in sorted(resources.unsupported_gate_counts.items()))
        print(f"WARNING: RZ-only estimate excludes other non-Clifford gates: {unsupported}")
    if resources.num_rz == 0:
        print("WARNING: no explicit RZ gates were found; the estimate is zero unless the input already contains T/Tdg.")


if __name__ == "__main__":
    main()
