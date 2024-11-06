import numpy as np
from functools import reduce
from operator import add

from functools import reduce, partial
from operator import add
from typing import List, Tuple

from phoenix.basic.circuits import Circuit
from phoenix.basic.gates import Clifford2QGate
from phoenix.models.paulis import BSF
from phoenix.models.cliffords import Clifford2Q


def config_to_circuit(config):
    circ = Circuit()
    for item in config:
        if isinstance(item, Clifford2Q):
            # circ += item.as_cnot_circuit()
            circ.append(item.as_gate())
        if isinstance(item, BSF):
            circ += item.as_cnot_circuit()
    return circ


def optimize_clifford_circuit_by_qiskit(circ: Circuit, optimization_level=1) -> Circuit:
    import qiskit
    from itertools import product
    basis_gates = ['h', 's', 'sdg', 'cx', 'rz', 'u3']
    for p0, p1 in product(['x', 'y', 'z'], repeat=2):
        basis_gates.append(f'c{p0}{p1}')
    return Circuit.from_qiskit(qiskit.transpile(circ.to_qiskit(), optimization_level=optimization_level,
                                                basis_gates=basis_gates))


def left_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
    if num_qubits is None:
        num_qubits = circ.num_qubits_with_dummy
    circ = Circuit([g for g in circ if g.num_qregs > 1])
    left_end = np.full(num_qubits, -1)
    for num_layer, layer in enumerate(circ.layer()):
        for q in reduce(add, [g.qregs for g in layer]):
            if left_end[q] < 0:
                left_end[q] = num_layer
        if np.all(left_end >= 0):
            break
    return left_end


# def right_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
#     if num_qubits is None:
#         num_qubits = circ.num_qubits_with_dummy
#     circ = Circuit([g for g in circ if g.num_qregs > 1])
#     right_end = np.full(num_qubits, -1)
#     for num_layer, layer in enumerate(circ.inverse().layer()):
#         for q in reduce(add, [g.qregs for g in layer]):
#             if right_end[q] < 0:
#                 right_end[q] = num_layer
#         if np.all(right_end >= 0):
#             break
#     return right_end


def right_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
    if num_qubits is None:
        num_qubits = circ.num_qubits_with_dummy
    circ = Circuit([g for g in circ if g.num_qregs > 1])
    right_end = np.full(num_qubits, -1)
    for num_layer, layer in enumerate(reversed(circ.layer())):
        for q in reduce(add, [g.qregs for g in layer]):
            if right_end[q] < 0:
                right_end[q] = num_layer
        if np.all(right_end >= 0):
            break
    return right_end


# def end_empty_layers(circ: Circuit):
#     circ = Circuit([g for g in circ if g.num_qregs > 1])
#     left_end = {i: -1 for i in circ.qubits}
#     right_end = {i: -1 for i in circ.qubits}
#     for num_layer, layer in enumerate(circ.layer()):
#         for q in reduce(add, [g.qregs for g in layer]):
#             if left_end[q] < 0:
#                 left_end[q] = num_layer
#         if np.all(np.array(list(left_end.values())) > 0):
#             break
#     for num_layer, layer in enumerate(circ.inverse().layer()):
#         for q in reduce(add, [g.qregs for g in layer]):
#             if right_end[q] < 0:
#                 right_end[q] = num_layer
#         if np.all(np.array(list(right_end.values())) > 0):
#             break
#
#     return left_end, right_end


# def end_empty_layers(circ: Circuit, num_qubits: int = None) -> Tuple[np.ndarray, np.ndarray]:
#     if num_qubits is None:
#         num_qubits = circ.num_qubits_with_dummy
#     circ = Circuit([g for g in circ if g.num_qregs > 1])
#     left_end = np.full(num_qubits, -1)
#     right_end = np.full(num_qubits, -1)
#     for num_layer, layer in enumerate(circ.layer()):
#         for q in reduce(add, [g.qregs for g in layer]):
#             if left_end[q] < 0:
#                 left_end[q] = num_layer
#         if np.all(left_end >= 0):
#             break
#     for num_layer, layer in enumerate(circ.inverse().layer()):
#         for q in reduce(add, [g.qregs for g in layer]):
#             if right_end[q] < 0:
#                 right_end[q] = num_layer
#         if np.all(right_end >= 0):
#             break
#
#     return left_end, right_end


def front_layer_cliffords(circ: Circuit) -> List[Clifford2QGate]:
    raise NotImplementedError


def last_layer_cliffords(circ: Circuit) -> List[Clifford2QGate]:
    raise front_layer_cliffords(Circuit(list(reversed(circ))))


def clifford_equal(lhs: Clifford2QGate, rhs: Clifford2QGate):
    return (lhs.pauli_0 == rhs.pauli_1 and lhs.pauli_1 == rhs.pauli_0 and
            lhs.tqs == rhs.tqs and lhs.cqs == rhs.cqs)


def common_cliffords(lhs: List[Clifford2QGate], rhs: List[Clifford2QGate]) -> List[Clifford2QGate]:
    return [cliff for cliff in lhs if any(clifford_equal(cliff, c) for c in rhs)]


class CircuitTetris:
    def __init__(self, circ: Circuit, num_qubits: int = None):
        if num_qubits is None:
            self.num_qubits = circ.num_qubits_with_dummy
        self.circuit = circ.clone()
        self.left_end = left_end_empty_layers(self.circuit, self.num_qubits)
        self.right_end = right_end_empty_layers(self.circuit, self.num_qubits)
        self.front_cliffs = front_layer_cliffords(self.circuit)
        self.last_cliffs = last_layer_cliffords(self.circuit)

    def update(self):
        """Update the left_end, right_end, front_cliffs, and last_cliffs attributes"""
        self.left_end = left_end_empty_layers(self.circuit, self.num_qubits)
        self.right_end = right_end_empty_layers(self.circuit, self.num_qubits)
        self.front_cliffs = front_layer_cliffords(self.circuit)
        self.last_cliffs = last_layer_cliffords(self.circuit)

    def append(self, circ: Circuit):
        # while intersection := self.last_cliffs.intersection(front_layer_cliffords(circ)):
        while commons := common_cliffords(self.last_cliffs, front_layer_cliffords(circ)):
            # remove self-inverse cliffords within intersection from self.last_cliffs and front_layer_cliffords(circ)
            # self.circuit.reverse()  # reverse the circuit because remove() method removes the first occurrence
            for cliff in commons:
                self.circuit.remove(cliff)
                circ.remove(cliff)
            # self.circuit.reverse()  # reverse the circuit back
            self.last_cliffs = last_layer_cliffords(self.circuit)  # update last_cliffs

        self.circuit += circ
        self.update()

    def prepend(self, circ: Circuit):
        # while intersection := self.front_cliffs.intersection(last_layer_cliffords(circ)):
        while commons := common_cliffords(self.front_cliffs, last_layer_cliffords(circ)):
            # remove self-inverse cliffords within intersection from self.front_cliffs and last_layer_cliffords(circ)
            # circ.reverse()  # reverse the circuit because remove() method removes the first occurrence
            for cliff in commons:
                self.circuit.remove(cliff)
                circ.remove(cliff)
            # circ.reverse()  # reverse the circuit back
            self.front_cliffs = front_layer_cliffords(self.circuit)  # update front_cliffs

        self.circuit = circ + self.circuit
        self.update()

    # def reduce_left_end(self):
    #     self.left_end -= self.left_end.min()


def assembling_score(tetris: CircuitTetris, circ: Circuit, approach: str):
    """Prepend/Append-assembling score"""
    if approach == 'append':
        ...
    elif approach == 'prepend':
        ...
    else:
        raise ValueError('Invalid approach (must be either "append" or "prepend")')


def depth_overhead(lhs: np.ndarray, rhs: np.ndarray):
    effective = (lhs >= 0) & (rhs >= 0)
    # idx = np.where((lhs >= 0) & (rhs >= 0))[0]
    xor = np.logical_xor(lhs, rhs)
    print(xor)
    if np.all(xor[np.where(effective)[0]]):
        print('Yes')
        return 0
    return (lhs[np.where(xor & effective)] + rhs[np.where(xor & effective)]).sum()


def order_blocks(blocks: List[Circuit]) -> Circuit:
    def wire_width(circ):
        return sum([g for g in circ.gates if g.num_qregs > 1])

    # blocks is already sorted .... by the least_overlap rule
    blocks = list(
        map(partial(optimize_clifford_circuit_by_qiskit, optimization_level=2), blocks))  # TODO: remove this line

    num_qubits = max(reduce(add, [block.qubits for block in blocks])) + 1
    tetris = CircuitTetris(blocks.pop(0), num_qubits=num_qubits)

    while blocks:
        # look ahead for 30 blocks
        append_scores = {i: assembling_score(tetris, block, 'append') for i, block in enumerate(blocks[:30])}
        prepend_scores = {i: assembling_score(tetris, block, 'prepend') for i, block in enumerate(blocks[:30])}
        if max(append_scores.values()) >= max(prepend_scores.values()):
            i = sorted(append_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
            block = blocks.pop(i)
            tetris.append(block)
        else:
            i = sorted(prepend_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
            block = blocks.pop(i)
            tetris.prepend(block)

    return tetris.circuit
