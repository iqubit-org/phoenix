import numpy as np
from functools import reduce
from copy import copy
from operator import add

from phoenix.basic.circuits import Circuit
from phoenix.basic.gates import Clifford2QGate
from phoenix.utils.passes import front_full_width_circuit, obtain_front_layer, obtain_last_layer
from typing import List, Tuple, Union

from rich.console import Console

console = Console()


class CircuitTetris:
    def __init__(self, circ: Circuit, num_qubits: int = None):
        self.num_qubits = num_qubits if num_qubits is not None else circ.num_qubits_with_dummy
        self.circuit = circ.clone()
        self.left_end = left_end_empty_layers(self.circuit, self.num_qubits)
        self.right_end = right_end_empty_layers(self.circuit, self.num_qubits)

        self.front_cliffs = obtain_front_layer(self.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))
        self.last_cliffs = obtain_last_layer(self.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))

    def copy(self):
        """Return a copy of the CircuitTetris instance (Copy all fields by reference copying)"""
        tetris = copy(self)
        tetris.circuit = self.circuit.clone()
        return tetris

    def update(self):
        """Update the left_end, right_end, front_cliffs, and last_cliffs attributes"""
        self.left_end = left_end_empty_layers(self.circuit, self.num_qubits)
        self.right_end = right_end_empty_layers(self.circuit, self.num_qubits)
        self.front_cliffs = obtain_front_layer(self.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))
        self.last_cliffs = obtain_last_layer(self.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))

    def left_assemble(self, tetris: 'CircuitTetris'):
        """Assemble the left end of the CircuitTetris with another CircuitTetris instance"""
        ...

    def right_assemble(self, tetris: 'CircuitTetris'):
        """Assemble the right end of the CircuitTetris with another CircuitTetris instance"""
        while commons := common_cliffords(self.last_cliffs, tetris.front_cliffs):
            for lhs_cliff, rhs_cliff in zip(*commons):
                self.circuit.remove(lhs_cliff)
                tetris.circuit.remove(rhs_cliff)
            self.last_cliffs = obtain_last_layer(self.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))
            tetris.front_cliffs = obtain_front_layer(tetris.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))

        self.circuit += tetris.circuit
        self.update()


def clifford_equal(lhs: Clifford2QGate, rhs: Clifford2QGate):
    """Define the value-equality of Clifford2QGate instances rather than the original reference-equality criterion"""
    return lhs.pauli_0 == rhs.pauli_0 and lhs.pauli_1 == rhs.pauli_1 and lhs.tqs == rhs.tqs and lhs.cqs == rhs.cqs


def common_cliffords(lhs: List[Clifford2QGate], rhs: List[Clifford2QGate]) -> Union[
    Tuple[List[Clifford2QGate], List[Clifford2QGate]], None]:
    """Get the common 2Q Clifford gates from two lists of Clifford2QGate instances"""
    lhs_common = [cliff for cliff in lhs if any(clifford_equal(cliff, c) for c in rhs)]
    if lhs_common:
        return lhs_common, [cliff for cliff in rhs if any(clifford_equal(cliff, c) for c in lhs)]


def left_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
    """
    E.g.,
                                                             ┌──────┐
        q_0: ────────────────────────────────────────────────┤0     ├────────
             ┌──────┐                                        │      │
        q_1: ┤0     ├────────────────────────────────────────┤      ├────────
             │      │                        ┌──────┐┌──────┐│      │┌──────┐
        q_2: ┤  Cxx ├────────────────────────┤0     ├┤0     ├┤  Cyy ├┤0     ├
             │      │┌──────┐┌──────┐┌──────┐│  Czz ││      ││      ││      │
        q_3: ┤1     ├┤0     ├┤0     ├┤0     ├┤1     ├┤  Cxx ├┤      ├┤  Czy ├
             └──────┘│      ││      ││      │└──────┘│      ││      ││      │  ==> left_end = [6, 0, 4, 0, 5, 1, 2, 3]
        q_4: ────────┤  Cxx ├┤      ├┤      ├────────┤1     ├┤1     ├┤1     ├
                     │      ││  Cxx ││      │        └──────┘└──────┘└──────┘
        q_5: ────────┤1     ├┤      ├┤  Cxz ├────────────────────────────────
                     └──────┘│      ││      │
        q_6: ────────────────┤1     ├┤      ├────────────────────────────────
                             └──────┘│      │
        q_7: ────────────────────────┤1     ├────────────────────────────────
                                     └──────┘
    """
    if num_qubits is None:
        num_qubits = circ.num_qubits_with_dummy
    left_end = np.full(num_qubits, -1)
    # circ_part = front_layer_circuit(circ, lambda g: g.num_qregs > 1)  # to efficiently compute left_end
    circ_part = front_full_width_circuit(circ, lambda g: g.num_qregs > 1)
    for num_layer, layer in enumerate(circ_part.layer()):
        for q in reduce(add, [g.qregs for g in layer]):
            if left_end[q] < 0:
                left_end[q] = num_layer
        if np.all(left_end >= 0):
            break
    left_end[left_end == -1] = left_end.max() + 1
    return left_end


def right_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
    """
    E.g.,
                                                             ┌──────┐
        q_0: ────────────────────────────────────────────────┤0     ├────────
             ┌──────┐                                        │      │
        q_1: ┤0     ├────────────────────────────────────────┤      ├────────
             │      │                        ┌──────┐┌──────┐│      │┌──────┐
        q_2: ┤  Cxx ├────────────────────────┤0     ├┤0     ├┤  Cyy ├┤0     ├
             │      │┌──────┐┌──────┐┌──────┐│  Czz ││      ││      ││      │
        q_3: ┤1     ├┤0     ├┤0     ├┤0     ├┤1     ├┤  Cxx ├┤      ├┤  Czy ├
             └──────┘│      ││      ││      │└──────┘│      ││      ││      │  ==> right_end = [1, 7, 0, 3, 0, 6, 5, 4]
        q_4: ────────┤  Cxx ├┤      ├┤      ├────────┤1     ├┤1     ├┤1     ├
                     │      ││  Cxx ││      │        └──────┘└──────┘└──────┘
        q_5: ────────┤1     ├┤      ├┤  Cxz ├────────────────────────────────
                     └──────┘│      ││      │
        q_6: ────────────────┤1     ├┤      ├────────────────────────────────
                             └──────┘│      │
        q_7: ────────────────────────┤1     ├────────────────────────────────
                                     └──────┘
    """
    if num_qubits is None:
        num_qubits = circ.num_qubits_with_dummy
    right_end = np.full(num_qubits, -1)
    for num_layer, layer in enumerate(reversed(circ.nonlocal_structure.layer())):
        for q in reduce(add, [g.qregs for g in layer]):
            if right_end[q] < 0:
                right_end[q] = num_layer
        if np.all(right_end >= 0):
            break
    right_end[right_end == -1] = right_end.max() + 1
    return right_end


def assembling_overhead(lhs: CircuitTetris, rhs: CircuitTetris, efficient: bool = False):
    """
    Assembling overhead considering
        - depth overhead
        - gate cancellation opportunity
        - .... (TODO: maybe others when exploring co-optimization opportunities)
    """
    lhs, rhs = lhs.copy(), rhs.copy()

    cost = depth_overhead(lhs.right_end, rhs.left_end)

    if efficient:
        return cost

    while commons := common_cliffords(lhs.last_cliffs, rhs.front_cliffs):
        indices = np.unique([cliff.qregs for cliff in commons[0]])
        cost -= indices.size  # ! a pair of 2Q Cliffords are canceled, but we do not use *2 factor

        mask = np.ones_like(rhs.left_end, dtype=bool)
        mask[indices] = False
        if np.all(rhs.left_end[mask]):
            cost -= mask.size
        if np.all(lhs.right_end[mask]):
            cost -= mask.size

        for lhs_cliff, rhs_cliff in zip(*commons):
            lhs.circuit.remove(lhs_cliff)
            rhs.circuit.remove(rhs_cliff)

        lhs.last_cliffs = obtain_last_layer(lhs.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))
        rhs.front_cliffs = obtain_front_layer(rhs.circuit, predicate=lambda g: isinstance(g, Clifford2QGate))

    return cost


def depth_overhead(lhs: np.ndarray, rhs: np.ndarray):
    """Calculate circuit depth(2q) overhead once lhs and rhs are concatenated"""
    xor = np.logical_xor(lhs, rhs)
    if np.all(xor[(lhs == 0) | (rhs == 0)]):
        cost = (lhs + rhs - 1).sum()
    else:
        cost = (lhs + rhs).sum()
    return cost


def order_blocks(blocks: List[Circuit], efficient: bool = False) -> Circuit:
    """Order unarranged simplified subcircuits (blocks) in a tetris-heuristic strategy"""
    num_qubits = max(reduce(add, [block.qubits for block in blocks])) + 1
    tetris = CircuitTetris(blocks.pop(0), num_qubits=num_qubits)
    tetris_list = [CircuitTetris(blk, num_qubits=num_qubits) for blk in blocks]

    # LOOKAHEAD = 40
    LOOKAHEAD = 25
    # TODO: what is the suitable number of blocks to look ahead?
    # ! after field test, 40 is a good lookahead length    LOOKAHEAD = 40
    while tetris_list:
        costs = {i: assembling_overhead(tetris, tts, efficient=efficient) for i, tts in
                 enumerate(tetris_list[:LOOKAHEAD])}
        i = sorted(costs.items(), key=lambda x: x[1])[0][0]
        tetris.right_assemble(tetris_list.pop(i))

    return tetris.circuit
