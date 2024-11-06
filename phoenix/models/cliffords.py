"""Clifford operators."""

import numpy as np
import pandas as pd
import qiskit.quantum_info as qi
from copy import deepcopy
from typing import List, Tuple
from itertools import product
from phoenix.basic import gates
from phoenix.basic.circuits import Circuit
from phoenix import decompose
from phoenix.utils.operations import is_equiv_unitary

X = qi.Pauli('X')
Y = qi.Pauli('Y')
Z = qi.Pauli('Z')
I = qi.Pauli('I')

PAULIS_1Q = {'I': I.to_matrix(), 'X': X.to_matrix(), 'Y': Y.to_matrix(), 'Z': Z.to_matrix()}
PAULIS_2Q = {}
for name1, P1 in PAULIS_1Q.items():
    for name2, P2 in PAULIS_1Q.items():
        PAULIS_2Q[name1 + name2] = np.kron(P1, P2)


class Clifford1Q:
    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self.targ = None

    def __repr__(self) -> str:
        return self.name

    def on(self, targ: int) -> 'Clifford1Q':
        cliff = deepcopy(self)
        cliff.targ = targ
        return cliff

    def transform(self, pauli: str) -> str:
        """Transformation effect on a Pauli operator."""
        P = PAULIS_1Q[pauli]
        Q = self.data @ P @ self.data.conj().T
        for pauli_, P_ in PAULIS_1Q.items():
            if is_equiv_unitary(Q, P_):
                return pauli_


class Clifford2Q:
    def __init__(self, pauli_0: str, pauli_1: str):
        assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
        P0, P1 = qi.Pauli(pauli_0), qi.Pauli(pauli_1)
        self.name = 'C({}, {})'.format(pauli_0, pauli_1)
        self.pauli_0, self.pauli_1 = pauli_0, pauli_1
        self.data = qi.SparsePauliOp([I ^ I, P0 ^ I, I ^ P1, P0 ^ P1], [1 / 2, 1 / 2, 1 / 2, -1 / 2]).to_matrix()
        self.ctrl, self.targ = None, None

    def __repr__(self) -> str:
        if self.ctrl is not None and self.targ is not None:
            return f'C({self.pauli_0}, {self.pauli_1}) @ ({self.ctrl}, {self.targ})'
        return self.name

    def on(self, ctrl: int, targ: int) -> 'Clifford2Q':
        cliff = deepcopy(self)
        cliff.ctrl, cliff.targ = ctrl, targ
        return cliff

    def transform(self, pauli: str) -> Tuple[str, int]:
        """Transformation effect on a pair of Pauli operators."""
        P = PAULIS_2Q[pauli]
        Q = self.data @ P @ self.data.conj().T
        for pauli_, P_ in PAULIS_2Q.items():
            if np.allclose(Q, P_):
                return pauli_, 0
            if np.allclose(Q, -P_):
                return pauli_, 1

    def wrt_cz(self) -> Tuple[List[str], List[str]]:
        """
        Return local Clifford operators, with respect to which it is equivalent to a CZ gate.

        Returns:
            wrt0, wrt1: List[str]
                Local Clifford operators ("H" or "S") for the first and second qubits.

        Examples:
            >>> cliff = Clifford2Q('X', 'Z')
            >>> cliff.wrt_cz()
            (['H'], [])
            >>> cliff = Clifford2Q('Y', 'Z')
            >>> cliff.wrt_cz()
            (['H', 'S'], [])
        """
        wrt0 = []
        if self.pauli_0 == 'X':
            wrt0.append('H')  # X = H Z H
        elif self.pauli_0 == 'Y':
            wrt0.append('H')
            wrt0.append('S')  # Y = S X S† = S H Z H S†
        wrt1 = []
        if self.pauli_1 == 'X':
            wrt1.append('H')
        elif self.pauli_1 == 'Y':
            wrt1.append('H')
            wrt1.append('S')
        return wrt0, wrt1

    def wrt_cx(self) -> Tuple[List[str], List[str]]:
        """
        Return local Clifford operators, with respect to which it is equivalent to a CX gate.

        Returns:
            wrt0, wrt1: List[str]
                Local Clifford operators ("H" or "S") for the first and second qubits.

        Examples:
            >>> cliff = Clifford2Q('X', 'Z')
            >>> cliff.wrt_cx()
            (['H'], [])
            >>> cliff = Clifford2Q('Y', 'Z')
            >>> cliff.wrt_cx()
            (['H', 'S'], ['H'])
        """
        wrt0 = []
        if self.pauli_0 == 'X':
            wrt0.append('H')  # X = H Z H
        elif self.pauli_0 == 'Y':
            wrt0.append('H')
            wrt0.append('S')  # Y = S X S† = S H Z H S†
        wrt1 = []
        if self.pauli_1 == 'Z':
            wrt1.append('H')
        elif self.pauli_1 == 'Y':
            wrt1.append('S')  # Y = S X S†
        return wrt0, wrt1

    def as_cnot_circuit(self) -> Circuit:
        """
        Return the circuit representation of the Clifford operator as a CNOT-basis circuit.

        Examples:
            >>> cliff = Clifford2Q('Y', 'Y').on(ctrl=0, targ=1)
            >>> cliff.as_cnot_circuit().to_cirq()
            q_0: ───S^-1───H───@───H───S───
                               │
            q_1: ───S^-1───────X───S───────
        """
        assert self.ctrl is not None and self.targ is not None
        ctrl, targ = self.ctrl, self.targ
        wrt0, wrt1 = self.wrt_cx()
        circ = Circuit()
        for opr in reversed(wrt0):
            if opr == 'H':
                circ.append(gates.H.on(ctrl))
            elif opr == 'S':
                circ.append(gates.SDG.on(ctrl))
            else:
                raise ValueError('Invalid operator')
        for opr in reversed(wrt1):
            if opr == 'H':
                circ.append(gates.H.on(targ))
            elif opr == 'S':
                circ.append(gates.SDG.on(targ))
            else:
                raise ValueError('Invalid operator')

        circ.append(gates.X.on(targ, ctrl))

        for opr in wrt0:
            if opr == 'H':
                circ.append(gates.H.on(ctrl))
            elif opr == 'S':
                circ.append(gates.S.on(ctrl))
            else:
                raise ValueError('Invalid operator')
        for opr in wrt1:
            if opr == 'H':
                circ.append(gates.H.on(targ))
            elif opr == 'S':
                circ.append(gates.S.on(targ))
            else:
                raise ValueError('Invalid operator')

        return circ

    def as_su4_circuit(self) -> Circuit:
        """
        Return the circuit representation of the Clifford operator as a SU(4)-basis circuit.

        Examples:
            >>> cliff = Clifford2Q('Y', 'Y').on(ctrl=0, targ=1)
            >>> cliff.as_su4_circuit().to_qiskit().draw()
                 ┌─────────────────┐┌───────────────┐┌───────────────┐
            q_0: ┤ U3(0,-π/4,-π/4) ├┤0              ├┤ U3(π/2,0,π/2) ├─
                 └┬───────────────┬┘│  Can(π/2,0,0) │├───────────────┴┐
            q_1: ─┤ U3(0,π/4,π/4) ├─┤1              ├┤ U3(π/2,0,-π/2) ├
                  └───────────────┘ └───────────────┘└────────────────┘
        """
        return decompose.can_decompose(gates.UnivGate(self.data).on([self.ctrl, self.targ]))

    def as_gate(self) -> gates.Clifford2QGate:
        return gates.Clifford2QGate(self.pauli_0, self.pauli_1).on([self.ctrl, self.targ])

def assemble_paulistr_with_sign(pauli, sign):
    if sign == 0:
        return pauli
    if sign == 1:
        return '-' + pauli


_TRANSFORM_TABLE_2Q = {}
_PAULIS_2Q = [''.join(pair) for pair in product(['I', 'X', 'Y', 'Z'], ['I', 'X', 'Y', 'Z'])]
for pauli_0, pauli_1 in product(['X', 'Y', 'Z'], ['X', 'Y', 'Z']):
    cg = Clifford2Q(pauli_0, pauli_1)  # controlled gate
    _TRANSFORM_TABLE_2Q[cg.name] = [assemble_paulistr_with_sign(*cg.transform(pauli)) for pauli in _PAULIS_2Q]
TRANSFORM_TABLE_2Q = pd.DataFrame(_TRANSFORM_TABLE_2Q, index=_PAULIS_2Q)
