"""Binary Symplectic Form (BSF) class and related functions."""

import numpy as np
from copy import deepcopy
from functools import reduce
from typing import List, Union

from phoenix.basic import gates
from phoenix.basic.circuits import Circuit
from phoenix.models.cliffords import Clifford1Q, Clifford2Q
from phoenix.utils.ops import params_u3

from rich.console import Console

console = Console()


class BSF:
    def __init__(self, paulis: Union[List[str], str],
                 coeffs: Union[List[float], float] = None,
                 signs: Union[List[int], int] = None):
        self.mat = np.vstack([pauli_to_bsf_vec(p) for p in np.atleast_1d(paulis)])
        self.coeffs = np.atleast_1d(coeffs) if coeffs is not None else np.atleast_1d(np.repeat(None, len(paulis)))
        self.signs = np.atleast_1d(signs) if signs is not None else np.zeros_like(self.coeffs).astype(int)

    def __repr__(self) -> str:
        return f'BSF(size=[{self.num_paulis}, {self.num_qubits}], num_nonlocals={self.num_nonlocal_paulis}, total_weight={self.total_weight})'

    @property
    def num_qubits(self) -> int:
        return self.mat.shape[1] // 2

    @property
    def num_paulis(self) -> int:
        return self.mat.shape[0]

    @property
    def paulis(self) -> List[str]:
        paulis = []
        for i in range(self.mat.shape[0]):
            ops = []
            for x, z in zip(self.x[i], self.z[i]):
                if x and z:
                    ops.append('Y')
                elif x:
                    ops.append('X')
                elif z:
                    ops.append('Z')
                else:
                    ops.append('I')
            paulis.append(''.join(ops))
        return paulis

    @property
    def x(self) -> np.ndarray:
        return self.mat[:, :self.num_qubits]

    @property
    def z(self) -> np.ndarray:
        return self.mat[:, self.num_qubits:]

    @property
    def with_ops(self) -> np.ndarray:
        return self.x | self.z

    @property
    def nontrivial_qubits_acted(self):
        """Pauli-wise qubit indices referring to nontrivial actions"""
        # qubits = {i: [] for i in range(self.num_paulis)}
        qubits = {}
        for i in range(self.num_paulis):
            qubits[i] = np.where(self.with_ops[i])[0]
        return qubits

    @property
    def qubits_with_ops(self):
        """Which qubits involve non-identity operations."""
        # return np.where(reduce(np.bitwise_or, self.with_ops))[0]
        return np.where(self.with_ops.sum(axis=0) > 0)[0]

    @property
    def qubits_with_nonlocal_ops(self):
        """Which qubits involve non-local operations."""
        # nonlocal_paulis = self.which_nonlocal_paulis
        qubits = set()
        for pauli in self.paulis:
            if self.num_qubits - pauli.count('I') > 1:
                qubits = qubits.union(set(np.where(np.array(list(pauli)) != 'I')[0]))

        return np.array(list(qubits))

    @property
    def weights(self) -> np.ndarray:
        return np.hstack((np.sum(self.x, axis=1).reshape(-1, 1), np.sum(self.z, axis=1).reshape(-1, 1)))

    @property
    def total_weight(self) -> int:
        if not self.num_paulis:  # it is an empty tableau
            return 0
        if ''.join(self.paulis).count('I') == self.num_paulis * self.num_qubits:  # all are I
            return 0
        if not self.num_nonlocal_paulis:  # only 1Q rotations
            return 1
        return reduce(np.bitwise_or, self.with_ops[self.which_nonlocal_paulis]).sum()

    @property
    def which_nonlocal_paulis(self) -> np.ndarray:
        return np.where(self.with_ops.sum(axis=1) > 1)[0]

    @property
    def which_local_paulis(self) -> np.ndarray:
        return np.where(self.with_ops.sum(axis=1) <= 1)[0]

    @property
    def num_nonlocal_paulis(self) -> int:
        return np.count_nonzero(self.with_ops.sum(axis=1) > 1)

    @property
    def num_local_paulis(self) -> int:
        return np.count_nonzero(self.with_ops.sum(axis=1) <= 1)

    def reverse(self) -> 'BSF':
        """Reverse the order of Pauli exponentiations."""
        # bsf = deepcopy(self)
        bsf = deepcopy(self)
        bsf.mat = bsf.mat[::-1]
        bsf.coeffs = bsf.coeffs[::-1]
        bsf.signs = bsf.signs[::-1]
        return bsf

    def pop_local_paulis(self) -> 'BSF':
        """Pop local Paulis."""

        """
        popping ...
        [1 3 4]
        [0 2 5]
        initial: [0.01 0.02 0.03 0.04 0.05 0.06]
        [0.02 0.04 0.05] [0.01 0.02 0.03]
        popped local bsf: ['XIIIIIII', 'IIIIZIII', 'ZIIIIIII'], [0.02 0.04 0.05]
        current bsf: ['YIZZIIII', 'XIYIYIII', 'ZIYIYIII'], [0.01 0.02 0.03]
        [] [0.01 0.02 0.03]
        [0.01 0.02 0.03] ['IIZZIIII', 'XIIIIIII', 'ZIIIIIII'] [0 0 0]
        """
        popped = deepcopy(self)
        popped.mat, popped.coeffs, popped.signs = popped.mat[self.which_local_paulis], popped.coeffs[
            self.which_local_paulis], popped.signs[self.which_local_paulis]
        self.mat, self.coeffs, self.signs = self.mat[self.which_nonlocal_paulis], self.coeffs[
            self.which_nonlocal_paulis], self.signs[self.which_nonlocal_paulis]
        return popped

    def apply_h(self, idx: int) -> 'BSF':
        """Apply H gate to qubit idx."""
        bsf = deepcopy(self)
        bsf.mat[:, [idx, idx + self.num_qubits]] = self.mat[:, [idx + self.num_qubits, idx]]
        bsf.signs ^= self.x[:, idx] & self.z[:, idx]
        return bsf

    def apply_s(self, idx: int) -> 'BSF':
        """Apply S gate to qubit idx."""
        bsf = deepcopy(self)
        bsf.mat[:, idx + self.num_qubits] = self.mat[:, idx + self.num_qubits] ^ self.mat[:, idx]
        bsf.signs ^= self.x[:, idx] & self.z[:, idx]
        return bsf

    def apply_sdg(self, idx: int) -> 'BSF':
        bsf = deepcopy(self)
        bsf.mat[:, idx + self.num_qubits] = self.mat[:, idx + self.num_qubits] ^ self.mat[:, idx]
        bsf.signs ^= (self.x[:, idx] == 1) & (self.z[:, idx] == 0)
        return bsf

    def apply_cx(self, ctrl: int, targ: int) -> 'BSF':
        """Apply CX gate."""
        bsf = deepcopy(self)
        bsf.mat[:, targ] = self.x[:, ctrl] ^ self.x[:, targ]
        bsf.mat[:, ctrl + self.num_qubits] = self.z[:, ctrl] ^ self.z[:, targ]
        bsf.signs ^= ((self.x[:, ctrl] == 1) & (self.x[:, targ] == 0) & (self.z[:, ctrl] == 0) &
                      (self.z[:, targ] == 1)) | ((self.x[:, ctrl] == 1) & (self.x[:, targ] == 1) &
                                                 (self.z[:, ctrl] == 1) & (self.z[:, targ] == 1))
        return bsf

    def apply_cz(self, ctrl: int, targ: int) -> 'BSF':
        """Apply CZ gate."""
        bsf = deepcopy(self)
        bsf.mat[:, ctrl + self.num_qubits] = self.x[:, targ] ^ self.z[:, ctrl]
        bsf.mat[:, targ + self.num_qubits] = self.x[:, ctrl] ^ self.z[:, targ]
        bsf.signs ^= ((self.x[:, ctrl] == 1) & (self.x[:, targ] == 1) & (self.z[:, ctrl] == 0) &
                      (self.z[:, targ] == 1)) | ((self.x[:, ctrl] == 1) & (self.x[:, targ] == 1) &
                                                 (self.z[:, ctrl] == 1) & (self.z[:, targ] == 0))
        return bsf

    def apply_clifford_1q(self, clifford: Clifford1Q, idx: int) -> 'BSF':
        """Apply 1-qubit Clifford operator."""
        raise NotImplementedError

    def apply_clifford_2q(self, clifford: Clifford2Q, ctrl: int, targ: int) -> 'BSF':
        """Apply 2-qubit Clifford operator."""
        wrt0_cz, wrt1_cz = clifford.wrt_cz()
        # print(wrt0_cz, wrt1_cz)
        bsf = deepcopy(self)

        for opr in reversed(wrt0_cz):
            if opr == 'H':
                bsf = bsf.apply_h(ctrl)
            elif opr == 'S':
                bsf = bsf.apply_sdg(ctrl)
            else:
                raise ValueError('Invalid operator')

        for opr in reversed(wrt1_cz):
            if opr == 'H':
                bsf = bsf.apply_h(targ)
            elif opr == 'S':
                bsf = bsf.apply_sdg(targ)
            else:
                raise ValueError('Invalid operator')

        bsf = bsf.apply_cz(ctrl, targ)

        for opr in wrt0_cz:
            if opr == 'H':
                bsf = bsf.apply_h(ctrl)
            elif opr == 'S':
                bsf = bsf.apply_s(ctrl)
            else:
                raise ValueError('Invalid operator')

        for opr in wrt1_cz:
            if opr == 'H':
                bsf = bsf.apply_h(targ)
            elif opr == 'S':
                bsf = bsf.apply_s(targ)
            else:
                raise ValueError('Invalid operator')

        return bsf

    def as_cnot_circuit(self, high_level=False) -> Circuit:
        circ = Circuit()
        for paulistr, coeff, sign in zip(self.paulis, self.coeffs, self.signs):
            assert coeff.imag == 0, "Imaginary coefficients are not supported"
            coeff = coeff.real * (-1) ** sign
            theta = 2 * coeff
            indices = np.where(np.array(list(paulistr)) != 'I')[0].tolist()
            if len(indices) == 0:
                continue
            elif len(indices) == 1:
                if paulistr[indices[0]] == 'X':
                    if high_level:
                        circ.append(gates.RX(theta).on(indices))
                    else:
                        circ.append(
                            *[gates.H.on(i) for i in indices],
                            gates.RZ(theta).on(indices),
                            *[gates.H.on(i) for i in indices]
                        )
                elif paulistr[indices[0]] == 'Y':
                    if high_level:
                        circ.append(gates.RY(theta).on(indices))
                    else:
                        circ.append(
                            *[gates.SDG.on(i) for i in indices],
                            *[gates.H.on(i) for i in indices],
                            gates.RZ(theta).on(indices),
                            *[gates.H.on(i) for i in indices],
                            *[gates.S.on(i) for i in indices]
                        )
                elif paulistr[indices[0]] == 'Z':
                    circ.append(gates.RZ(theta).on(indices))
                else:
                    raise ValueError(f"Invalid Pauli string {paulistr}")
            else:
                if high_level:
                    assert len(indices) == 2, "Only support 2-qubit Pauli strings"
                    for idx in indices:
                        if paulistr[idx] == 'X':
                            circ.append(gates.H.on(idx))
                        elif paulistr[idx] == 'Y':
                            circ.append(gates.SDG.on(idx), gates.H.on(idx))
                    circ.append(gates.RZZ(theta).on(indices))
                    for idx in indices:
                        if paulistr[idx] == 'X':
                            circ.append(gates.H.on(idx))
                        elif paulistr[idx] == 'Y':
                            circ.append(gates.H.on(idx), gates.S.on(idx))
                else:
                    for idx in indices:
                        if paulistr[idx] == 'X':
                            circ.append(gates.H.on(idx))
                        elif paulistr[idx] == 'Y':
                            circ.append(gates.SDG.on(idx), gates.H.on(idx))
                    circ.append(*[
                        gates.X.on(indices[i + 1], indices[i]) for i in range(len(indices) - 1)
                    ])
                    circ.append(gates.RZ(theta).on(indices[-1]))
                    circ.append(*[
                        gates.X.on(indices[i], indices[i - 1]) for i in range(len(indices) - 1, 0, -1)
                    ])
                    for idx in indices:
                        if paulistr[idx] == 'X':
                            circ.append(gates.H.on(idx))
                        elif paulistr[idx] == 'Y':
                            circ.append(gates.H.on(idx), gates.S.on(idx))

        return circ

    def as_su4_circuit(self) -> Circuit:
        from phoenix.models.hamiltonians import HamiltonianModel
        from phoenix.synthesis.utils import can_decompose

        circ = Circuit()
        if self.num_nonlocal_paulis == 0:
            # all are single-qubit rotation gates
            for paulistr, coeff, sign in zip(self.paulis, self.coeffs, self.signs):
                if paulistr.count('I') == self.num_qubits:
                    continue
                assert coeff.imag == 0, "Imaginary coefficients are not supported"
                circ.append(local_pauli_to_u3(paulistr, coeff.real, sign))
        else:
            assert self.total_weight == 2, "Not implemented for total_weight > 2"

            # console.rule('debugging')
            # console.print(self.paulilist)
            # console.print(self.with_ops)
            # console.print(self.qubits_with_nonlocal_ops, self.qubits_with_ops)

            # qubits_with_nonlocal_ops: [2 3]
            # qubits_with_ops: [0 2 3]

            # 1) synthesize operations on qubits involving non-local operations (might include some local operations)
            paulis_part = []
            coeffs_part = []
            signs_part = []
            paulis_remain = []
            coeffs_remain = []
            signs_remain = []
            # console.print('qubits with nonlocal ops', self.qubits_with_nonlocal_ops)
            for i in range(self.num_paulis):
                if set(self.nontrivial_qubits_acted[i]).issubset(set(self.qubits_with_nonlocal_ops)):
                    pauli = self.paulis[i]
                    # console.print('{} {}'.format(pauli, [pauli[q] for q in self.qubits_with_nonlocal_ops]))
                    paulis_part.append(''.join([pauli[q] for q in self.qubits_with_nonlocal_ops]))
                    # console.print(paulis_part)
                    coeffs_part.append(self.coeffs[i])
                    signs_part.append(self.signs[i])
                else:
                    paulis_remain.append(self.paulis[i])
                    coeffs_remain.append(self.coeffs[i])
                    signs_remain.append(self.signs[i])
            coeffs_part, signs_part = np.array(coeffs_part), np.array(signs_part)
            circ += can_decompose(gates.UnivGate(
                HamiltonianModel(paulis_part, coeffs_part * (-1) ** signs_part).unitary_evolution()
            ).on(self.qubits_with_nonlocal_ops.tolist()))

            # 2) synthesize remaining local operations
            for i in range(self.num_paulis):
                if not set(self.nontrivial_qubits_acted[i]).issubset(set(self.qubits_with_nonlocal_ops)):
                    circ.append(local_pauli_to_u3(self.paulis[i], self.coeffs[i], self.signs[i]))

        return circ


def pauli_to_bsf_vec(pauli: str) -> np.ndarray:
    ops = np.array(list(pauli))
    ops_x = (ops == 'X') + (ops == 'Y')
    ops_z = (ops == 'Z') + (ops == 'Y')
    vec = np.hstack((ops_x, ops_z)).astype(int)
    return vec


def local_pauli_to_u3(paulistr: str, coeff: float, sign: float = 1):
    theta = 2 * coeff * (-1) ** sign
    idx = np.where(np.array(list(paulistr)) != 'I')[0].item()

    if paulistr.count('I') == len(paulistr):
        return gates.I.on(0)  # ! return identity gate

    if paulistr[idx] == 'X':
        return gates.U3(*params_u3(gates.RX(theta).data)).on(idx)
    elif paulistr[idx] == 'Y':
        return gates.U3(*params_u3(gates.RY(theta).data)).on(idx)
    elif paulistr[idx] == 'Z':
        return gates.U3(*params_u3(gates.RZ(theta).data)).on(idx)
    else:
        raise ValueError(f"Invalid Pauli string {paulistr}")
