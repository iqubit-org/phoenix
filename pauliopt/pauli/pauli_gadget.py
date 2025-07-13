from collections import deque
from typing import List

import networkx as nx
import numpy as np

from pauliopt.circuits import Circuit
from pauliopt.gates import H, V, Rz, CX, Vdg
from pauliopt.pauli_strings import Pauli, X, Y, Z, I
from pauliopt.topologies import Topology
from pauliopt.utils import AngleExpr


def decompose_cnot_ladder_z(ctrl: int, trg: int, arch: Topology):
    cnot_ladder = []
    shortest_path = arch.shortest_path(ctrl, trg)

    prev = ctrl
    for current in shortest_path[1:-1]:
        cnot_ladder.append((current, prev))
        cnot_ladder.append((prev, current))
        prev = current
    cnot_ladder.append((shortest_path[-2], trg))
    return reversed(cnot_ladder)


def find_minimal_cx_assignment(column: np.array, arch: Topology, q0=None):
    if not np.all(np.isin(column, [0, 1])):
        raise Exception(f"Expected binary array as column, got: {column}")

    G = nx.Graph()
    for i in range(len(column)):
        G.add_node(i)

    for i in range(len(column)):
        for j in range(len(column)):
            if column[i] != 0 and column[j] != 0 and i != j:
                G.add_edge(i, j, weight=4 * arch.dist(i, j) - 2)

    mst_branches = list(nx.minimum_spanning_edges(G, data=False, algorithm="prim"))
    incident = {q: set() for q in range(len(column))}
    for fst, snd in mst_branches:
        incident[fst].add((fst, snd))
        incident[snd].add((snd, fst))
    if q0 is None:
        q0 = np.argmax(
            column
        )  # Assume that 0 is always the first qubit aka first non zero
    visited = set()
    queue = deque([q0])
    cnot_ladder = []
    while queue:
        q = queue.popleft()
        visited.add(q)
        for tail, head in incident[q]:
            if head not in visited:
                cnot_ladder += decompose_cnot_ladder_z(head, tail, arch)
                queue.append(head)
    return cnot_ladder, q0


class PPhase:
    _angle: AngleExpr

    def __init__(self, angle: AngleExpr):
        self._angle = angle

    def __matmul__(self, paulis: List[Pauli]):
        return PauliGadget(self._angle, paulis)

    def __str__(self):
        return f"PPhase({self._angle})"


class PauliGadget:
    def __init__(self, angle: AngleExpr, paulis: List[Pauli]):
        self.angle = angle
        self.paulis = paulis

    def __len__(self):
        return len(self.paulis)

    def __repr__(self):
        return self.to_string()

    def __getitem__(self, item):
        return self.paulis[item]

    @property
    def num_qubits(self):
        return len(self.paulis)

    def to_string(self, pad_length=0):
        pad = " " * (pad_length - len(str(self.angle)))
        return f"PPhase({self.angle}) @ {pad} [ {', '.join([pauli.value for pauli in self.paulis])} ]"

    def num_legs(self):
        return sum([1 for pauli in self.paulis if pauli != Pauli.I])

    def copy(self):
        return PauliGadget(self.angle, self.paulis.copy())

    def decompose(self, q0):
        cliffords = []
        column = np.asarray(self.paulis)
        for pauli_idx in range(len(column)):
            if column[pauli_idx] == I:
                pass
            elif column[pauli_idx] == X:
                cliffords.append(H(pauli_idx))
            elif column[pauli_idx] == Y:
                cliffords.append(V(pauli_idx))
            elif column[pauli_idx] == Z:  # Z
                pass
            else:
                raise Exception(f"unknown column type: {column[pauli_idx]}")
        for q in range(self.num_qubits):
            if q == q0:
                continue
            if self[q] != I:
                cliffords.append(CX(q, q0))
        return cliffords, q0

    def swap_rows(self, row1, row2):
        self.paulis[row1], self.paulis[row2] = self.paulis[row2], self.paulis[row1]

    def two_qubit_count(self, topology, leg_cache=None):
        if leg_cache is None:
            leg_cache = {}

        col_binary = [1 if self[q] != I else 0 for q in range(self.num_qubits)]
        col_id = "".join([str(int(el)) for el in col_binary])
        if col_id in leg_cache.keys():
            return leg_cache[col_id]
        else:
            cnot_amount = 2 * len(
                find_minimal_cx_assignment(np.asarray(col_binary), topology)[0]
            )
            leg_cache[col_id] = cnot_amount
        return cnot_amount

    def mutual_legs(self, other: "PauliGadget"):
        if len(self.paulis) != len(other.paulis):
            raise Exception(
                f"Paulis must be of equal length to have mutual legs. But are {len(self.paulis)}, "
                f"{len(other.paulis)}"
            )

        match_count = 0
        for p_1, p_2 in zip(self.paulis, other.paulis):
            leg_present_1 = p_1 != Pauli.I
            leg_present_2 = p_2 != Pauli.I
            if leg_present_2 and leg_present_1:
                match_count += 1
            elif leg_present_1 != leg_present_2:
                match_count -= 1
            else:
                match_count -= 1
        return match_count

    def commutes(self, other: "PauliGadget"):
        if len(self.paulis) != len(other.paulis):
            raise Exception(
                f"Paulis must be of equal length to commute. But are {len(self.paulis)}, "
                f"{len(other.paulis)}"
            )

        mismatchcount = 0
        for p_1, p_2 in zip(self.paulis, other.paulis):
            if p_1 != p_2 and p_1 != Pauli.I and p_2 != Pauli.I:
                mismatchcount += 1
        return mismatchcount % 2 == 0

    def to_circuit(self, topology=None, time=1):
        num_qubits = self.num_qubits
        if topology is None:
            topology = Topology.complete(num_qubits)

        circ = Circuit(num_qubits)

        column = np.asarray(self.paulis)
        column_binary = np.where(column == I, 0, 1)
        if np.all(column_binary == 0):
            circ.global_phase += self.angle
            return circ
        cnot_ladder, q0 = find_minimal_cx_assignment(column_binary, topology)
        for pauli_idx in range(len(column)):
            if column[pauli_idx] == I:
                pass
            elif column[pauli_idx] == X:
                circ.add_gate(H(pauli_idx))  # Had
            elif column[pauli_idx] == Y:
                circ.add_gate(V(pauli_idx))
            elif column[pauli_idx] == Z:  # Z
                pass
            else:
                print(column)
                raise Exception(f"unknown column type: {column[pauli_idx]}")

        if len(cnot_ladder) > 0:
            for pauli_idx, target in reversed(cnot_ladder):
                circ.add_gate(CX(pauli_idx, target))

            circ.add_gate(Rz(self.angle * time, q0))

            for pauli_idx, target in cnot_ladder:
                circ.add_gate(CX(pauli_idx, target))
        else:
            target = np.argmax(column_binary)
            circ.add_gate(Rz(self.angle * time, target))

        for pauli_idx in range(len(column)):
            if column[pauli_idx] == I:
                pass
            elif column[pauli_idx] == X:
                circ.add_gate(H(pauli_idx))  # Had
            elif column[pauli_idx] == Y:
                circ.add_gate(Vdg(pauli_idx))
            elif column[pauli_idx] == Z:
                pass
            else:
                raise Exception(f"unknown column type: {column[pauli_idx]}")
        return circ

    def to_qiskit(self, time=1, topology=None):
        return self.to_circuit(topology, time).to_qiskit()

    def permute(self, permutation: dict):
        for k, v in permutation.items():
            self.paulis[k], self.paulis[v] = self.paulis[v], self.paulis[k]

    def assign_time(self, time):
        if isinstance(self.angle, float):
            self.angle = self.angle * time
        else:
            self.angle = self.angle.to_qiskit * time

    def set_angle(self, angle):
        self.angle = float(angle)
