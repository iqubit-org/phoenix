"""Unit tests for the RZ-only GridSynth resource estimator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "experiments" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from estimate_t_cost import QiskitGridSynth, estimate_t_resources, rebase_to_clifford_rz  # noqa: E402, I001


def test_t_depth_respects_two_qubit_dependencies() -> None:
    circuit = QuantumCircuit(2)
    circuit.rz(0.1, 0)
    circuit.cx(0, 1)
    circuit.rz(0.2, 1)
    circuit.rz(0.3, 0)

    t_counts = {round(0.1, 1): 2, round(0.2, 1): 3, round(0.3, 1): 1}
    resources = estimate_t_resources(
        circuit,
        lambda angle: t_counts[round(angle, 1)],
        unique_angles=lambda: len(t_counts),
    )

    assert resources.num_rz == 3
    assert resources.t_count == 6
    assert resources.t_depth == 5
    assert resources.unsupported_gate_counts == {}


def test_non_rz_nonclifford_gates_are_reported() -> None:
    circuit = QuantumCircuit(1)
    circuit.rz(0.1, 0)
    circuit.rx(0.2, 0)
    circuit.t(0)

    resources = estimate_t_resources(circuit, lambda _angle: 2, unique_angles=lambda: 1)

    assert resources.t_count == 3
    assert resources.t_depth == 3
    assert resources.unsupported_gate_counts == {"rx": 1}


def test_rebase_converts_u_gate_to_rz_plus_cliffords() -> None:
    circuit = QuantumCircuit(1)
    circuit.u(0.1, 0.2, 0.3, 0)

    rebased = rebase_to_clifford_rz(circuit)
    names = {instruction.operation.name for instruction in rebased.data}

    assert names <= {"rz", "sx", "x"}
    assert "rz" in names


def test_qiskit_gridsynth_synthesizes_pi_over_four_as_one_t_gate() -> None:
    synthesizer = QiskitGridSynth(epsilon=1e-10)
    assert synthesizer.t_count(math.pi / 4) == 1
    assert synthesizer.t_count(-math.pi / 2) == 0
