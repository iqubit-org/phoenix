import numpy as np
from functools import reduce
from operator import add

from phoenix.basic.circuits import Circuit
from phoenix.basic.gates import Clifford2QGate
from typing import List, Tuple, Union

from rich.console import Console

console = Console()


class CircuitTetris:
    def __init__(self, circ: Circuit, num_qubits: int = None):
        self.num_qubits = num_qubits if num_qubits is not None else circ.num_qubits_with_dummy
        self.circuit = circ.clone()
        self.left_end = left_end_empty_layers(self.circuit, self.num_qubits)
        self.right_end = right_end_empty_layers(self.circuit, self.num_qubits)
        self.front_cliffs = front_layer_cliffords(self.circuit)
        self.last_cliffs = last_layer_cliffords(self.circuit)

    def copy(self):
        """Return a copy of the CircuitTetris instance"""
        from copy import copy
        # copy all field by reference copy
        tetris = copy(self)
        tetris.circuit = self.circuit.clone()
        return tetris

    def update(self):
        """Update the left_end, right_end, front_cliffs, and last_cliffs attributes"""
        self.left_end = left_end_empty_layers(self.circuit, self.num_qubits)
        self.right_end = right_end_empty_layers(self.circuit, self.num_qubits)
        self.front_cliffs = front_layer_cliffords(self.circuit)
        self.last_cliffs = last_layer_cliffords(self.circuit)

    def left_assemble(self, tetris: 'CircuitTetris'):
        """Assemble the left end of the CircuitTetris with another CircuitTetris instance"""
        ...

    def right_assemble(self, tetris: 'CircuitTetris'):
        """Assemble the right end of the CircuitTetris with another CircuitTetris instance"""
        while commons := common_cliffords(self.last_cliffs, tetris.front_cliffs):
            # lhs_common, rhs_common = commons
            # for cliff in lhs_common:
            #     self.circuit.remove(cliff)
            # for cliff in rhs_common:
            #     tetris.circuit.remove(cliff)
            # lhs_common, rhs_common = commons
            for lhs_cliff, rhs_cliff in zip(*commons):
                self.circuit.remove(lhs_cliff)
                tetris.circuit.remove(rhs_cliff)
            self.last_cliffs = last_layer_cliffords(self.circuit)  # update last_cliffs of self
            tetris.front_cliffs = front_layer_cliffords(tetris.circuit)  # update front_cliffs of tetris

        self.circuit += tetris.circuit
        self.update()

    # def append(self, circ: Circuit):
    #     # while intersection := self.last_cliffs.intersection(front_layer_cliffords(circ)):
    #     while commons := common_cliffords(self.last_cliffs, front_layer_cliffords(circ)):
    #         lhs_common, rhs_common = commons
    #         # remove self-inverse cliffords within intersection from self.last_cliffs and front_layer_cliffords(circ)
    #         # self.circuit.reverse()  # reverse the circuit because remove() method removes the first occurrence
    #         for cliff in lhs_common:
    #             self.circuit.remove(cliff)
    #         for cliff in rhs_common:
    #             circ.remove(cliff)
    #         # for cliff in commons:
    #         #     self.circuit.remove(cliff)
    #         #     circ.remove(cliff)
    #         # self.circuit.reverse()  # reverse the circuit back
    #         self.last_cliffs = last_layer_cliffords(self.circuit)  # update last_cliffs
    #
    #     self.circuit += circ
    #     self.update()

    # def prepend(self, circ: Circuit):
    #     # while intersection := self.front_cliffs.intersection(last_layer_cliffords(circ)):
    #     while commons := common_cliffords(self.front_cliffs, last_layer_cliffords(circ)):
    #         # remove self-inverse cliffords within intersection from self.front_cliffs and last_layer_cliffords(circ)
    #         # circ.reverse()  # reverse the circuit because remove() method removes the first occurrence
    #         for cliff in commons:
    #             self.circuit.remove(cliff)
    #             circ.remove(cliff)
    #         # circ.reverse()  # reverse the circuit back
    #         self.front_cliffs = front_layer_cliffords(self.circuit)  # update front_cliffs
    #
    #     self.circuit = circ + self.circuit
    #     self.update()

    # def reduce_left_end(self):
    #     self.left_end -= self.left_end.min()


def front_layer_cliffords(circ: Circuit) -> List[Clifford2QGate]:
    """Obtain 2Q Clifford gates within the front layer of a circuit"""
    # for running efficiency, only part of 2Q Clifford gates are sufficient to obtain the front layer
    from phoenix.utils.passes import obtain_front_layer

    circ_part = Circuit()
    for g in circ.nonlocal_structure:
        circ_part.append(g)
        if circ_part.num_qubits == circ.num_qubits:
            break
    return [g for g in obtain_front_layer(circ_part) if isinstance(g, Clifford2QGate)]


def last_layer_cliffords(circ: Circuit) -> List[Clifford2QGate]:
    return front_layer_cliffords(Circuit(list(reversed(circ))))


def clifford_equal(lhs: Clifford2QGate, rhs: Clifford2QGate):
    return (lhs.pauli_0 == rhs.pauli_1 and lhs.pauli_1 == rhs.pauli_0 and
            lhs.tqs == rhs.tqs and lhs.cqs == rhs.cqs)


def common_cliffords(lhs: List[Clifford2QGate], rhs: List[Clifford2QGate]) -> Union[
    Tuple[List[Clifford2QGate], List[Clifford2QGate]], None]:
    """Get the common 2Q Clifford gates from two lists of Clifford2QGate instances"""
    lhs_common = [cliff for cliff in lhs if any(clifford_equal(cliff, c) for c in rhs)]
    if lhs_common:
        return lhs_common, [cliff for cliff in rhs if any(clifford_equal(cliff, c) for c in lhs)]
    return None


def left_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
    if num_qubits is None:
        num_qubits = circ.num_qubits_with_dummy
    circ = circ.nonlocal_structure
    left_end = np.full(num_qubits, -1)
    for num_layer, layer in enumerate(circ.layer()):
        for q in reduce(add, [g.qregs for g in layer]):
            if left_end[q] < 0:
                left_end[q] = num_layer
        if np.all(left_end >= 0):
            break
    left_end[left_end == -1] = left_end.max() + 1
    return left_end


def right_end_empty_layers(circ: Circuit, num_qubits: int = None) -> np.ndarray:
    if num_qubits is None:
        num_qubits = circ.num_qubits_with_dummy
    circ = circ.nonlocal_structure
    right_end = np.full(num_qubits, -1)
    for num_layer, layer in enumerate(reversed(circ.layer())):
        for q in reduce(add, [g.qregs for g in layer]):
            if right_end[q] < 0:
                right_end[q] = num_layer
        if np.all(right_end >= 0):
            break

    right_end[right_end == -1] = right_end.max() + 1

    return right_end


def assembling_overhead(lhs: CircuitTetris, rhs: CircuitTetris, approach: str = None):
    """Prepend/Append-assembling score"""
    # if approach == 'append':
    #     ...
    # elif approach == 'prepend':
    #     ...
    # else:
    #     raise ValueError('Invalid approach (must be either "append" or "prepend")')
    ...
    lhs, rhs = lhs.copy(), rhs.copy()

    cost = depth_overhead(lhs.right_end, rhs.left_end)

    while commons := common_cliffords(lhs.last_cliffs, rhs.front_cliffs):
        # lhs_common, rhs_common = commons
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

        lhs.update()
        rhs.update()

    return cost

    # return - depth_overhead(lhs.right_end, rhs.left_end)


# def assembling_score(tetris: CircuitTetris, circ: Circuit, approach: str = None):
#     """Prepend/Append-assembling score"""
#     # if approach == 'append':
#     #     ...
#     # elif approach == 'prepend':
#     #     ...
#     # else:
#     #     raise ValueError('Invalid approach (must be either "append" or "prepend")')
#     ...
#
#
#     while commons := common_cliffords(tetris.last_cliffs, front_layer_cliffords(circ)):
#         ...
#
#     return - depth_overhead(tetris.right_end,
#                             left_end_empty_layers(circ, tetris.num_qubits))


def depth_overhead(lhs: np.ndarray, rhs: np.ndarray):  # , lhs_num_layers:int):
    """Calculate circuit depth(2q) overhead once lhs and rhs are concatenated"""
    # effective = (lhs >= 0) & (rhs >= 0)
    # idx = np.where((lhs >= 0) & (rhs >= 0))[0]
    # xor = np.logical_xor(lhs, rhs)
    # # print(xor)
    # if np.all(xor[np.where(effective)[0]]):
    #     return 0
    # xor = np.logical_xor(lhs, rhs)
    # if np.all(xor[(lhs == 0) | (rhs == 0)]):
    #     return 0
    # return (lhs[np.where(xor & effective)] + rhs[np.where(xor & effective)]).sum()

    xor = np.logical_xor(lhs, rhs)
    if np.all(xor[(lhs == 0) | (rhs == 0)]):
        # cost = ((lhs - 1) + rhs)[rhs != -1].sum()
        cost = (lhs + rhs - 1).sum()
        # cost = 0
    else:
        # lhs[lhs == -1] = lhs_num_layers
        # cost = (lhs + rhs)[(rhs != -1)].sum()
        cost = (lhs + rhs).sum()
    return cost


def order_blocks(blocks: List[Circuit]) -> Circuit:
    num_qubits = max(reduce(add, [block.qubits for block in blocks])) + 1
    tetris = CircuitTetris(blocks.pop(0), num_qubits=num_qubits)
    tetris_list = [CircuitTetris(blk, num_qubits=num_qubits) for blk in blocks]

    # while blocks:
    #     # look ahead for 30 blocks
    #     append_scores = {i: assembling_score(tetris, block, 'append') for i, block in enumerate(blocks[:30])}
    #     prepend_scores = {i: assembling_score(tetris, block, 'prepend') for i, block in enumerate(blocks[:30])}
    #     if max(append_scores.values()) >= max(prepend_scores.values()):
    #         i = sorted(append_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
    #         block = blocks.pop(i)
    #         tetris.append(block)
    #     else:
    #         i = sorted(prepend_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
    #         block = blocks.pop(i)
    #         tetris.prepend(block)

    # while blocks:
    #     # look ahead finite blocks
    #     scores = {i: assembling_score(tetris, block) for i, block in enumerate(blocks[:20])}
    #     i = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]
    #     tetris.append(blocks.pop(i))

    while tetris_list:
        # TODO: what is the suitable number of blocks to look ahead?
        costs = {i: assembling_overhead(tetris, tts) for i, tts in enumerate(tetris_list[:50])}
        i = sorted(costs.items(), key=lambda x: x[1])[0][0]
        tetris.right_assemble(tetris_list.pop(i))

        # tetris.right(tetris_list.pop(i))

    return tetris.circuit
