import numpy as np
from functools import reduce
from operator import add
from phoenix.basic.circuits import Circuit


def order_blocks(blocks: Circuit) -> Circuit:
    def wire_width(circ):
        return sum([g for g in circ.gates if g.num_qregs > 1])

    def end_empty_layers(circ: Circuit):
        circ = Circuit([g for g in circ if g.num_qregs > 1])
        left_ends = {i: -1 for i in circ.qubits}
        right_end = {i: -1 for i in circ.qubits}
        for num_layer, layer in enumerate(circ.layer()):
            for q in reduce(add, [g.qregs for g in layer]):
                if left_ends[q] < 0:
                    left_ends[q] = num_layer
            if np.all(np.array(list(left_ends.values())) > 0):
                break
        for num_layer, layer in enumerate(circ.inverse().layer()):
            for q in reduce(add, [g.qregs for g in layer]):
                if right_end[q] < 0:
                    right_end[q] = num_layer
            if np.all(np.array(list(right_end.values())) > 0):
                break

        return left_ends, right_end
