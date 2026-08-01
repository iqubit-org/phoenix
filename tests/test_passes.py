"""Tests for dependency-aware Clifford boundary peeling."""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from phoenix.basics import CNOTEquivCliffordGate
from phoenix.passes import peel_front_cliffords, peel_tail_cliffords


def _operation_names(circuit: QuantumCircuit) -> list[str]:
    return [instruction.operation.name for instruction in circuit.data]


def test_peel_front_cliffords_recursively_across_independent_wires():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.t(1)
    circuit.cx(0, 2)
    circuit.s(2)
    circuit.x(0)

    front, remainder = peel_front_cliffords(circuit)

    assert _operation_names(front)[:2] == ["h", "cx"]
    assert set(_operation_names(front)[2:]) == {"s", "x"}
    assert _operation_names(remainder) == ["t"]
    assert Operator(front.compose(remainder)).equiv(Operator(circuit))
    assert _operation_names(circuit) == ["h", "t", "cx", "s", "x"]


def test_peel_tail_cliffords_recursively_across_independent_wires():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.t(1)
    circuit.cx(0, 2)
    circuit.s(2)
    circuit.x(0)

    remainder, tail = peel_tail_cliffords(circuit)

    assert _operation_names(remainder) == ["t"]
    assert _operation_names(tail) == ["h", "cx", "s", "x"]
    assert Operator(remainder.compose(tail)).equiv(Operator(circuit))
    assert _operation_names(circuit) == ["h", "t", "cx", "s", "x"]


def test_peel_tail_cliffords_does_not_cross_a_non_clifford_on_the_same_wire():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(0)
    circuit.s(1)

    remainder, tail = peel_tail_cliffords(circuit)

    # The terminal S on q1 is peeled, but the T on q0 prevents CX and H from
    # entering the trailing Clifford subcircuit.
    assert _operation_names(remainder) == ["h", "cx", "t"]
    assert _operation_names(tail) == ["s"]
    assert Operator(remainder.compose(tail)).equiv(Operator(circuit))


def test_peel_passes_recognize_phoenix_custom_clifford_gates():
    front_circuit = QuantumCircuit(2)
    front_circuit.append(CNOTEquivCliffordGate("X", "Y"), [0, 1])
    front_circuit.t(0)

    tail_circuit = QuantumCircuit(2)
    tail_circuit.t(0)
    tail_circuit.append(CNOTEquivCliffordGate("X", "Y"), [0, 1])

    front, front_remainder = peel_front_cliffords(front_circuit)
    tail_remainder, tail = peel_tail_cliffords(tail_circuit)

    assert _operation_names(front) == ["cxy"]
    assert _operation_names(front_remainder) == ["t"]
    assert _operation_names(tail_remainder) == ["t"]
    assert _operation_names(tail) == ["cxy"]


def test_measurement_is_a_clifford_peeling_boundary():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)

    front, front_remainder = peel_front_cliffords(circuit)
    tail_remainder, tail = peel_tail_cliffords(circuit)

    assert _operation_names(front) == ["h"]
    assert _operation_names(front_remainder) == ["measure"]
    assert _operation_names(tail_remainder) == ["h", "measure"]
    assert _operation_names(tail) == []


def test_barrier_is_a_clifford_peeling_boundary():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.barrier(0)
    circuit.s(0)

    front, front_remainder = peel_front_cliffords(circuit)
    tail_remainder, tail = peel_tail_cliffords(circuit)

    assert _operation_names(front) == ["h"]
    assert _operation_names(front_remainder) == ["barrier", "s"]
    assert _operation_names(tail_remainder) == ["h", "barrier"]
    assert _operation_names(tail) == ["s"]
