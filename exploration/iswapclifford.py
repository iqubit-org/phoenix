import sys

sys.path.append('../')

from phoenix import gates
import numpy as np
from copy import deepcopy
from phoenix.models.cliffords import PAULIS_2Q
from typing import Tuple, List, Union
import qiskit.quantum_info as qi
from phoenix.models import BSF, Clifford2Q
from phoenix.models.cliffords import CLIFFORD_2Q_SET
from phoenix.synthesis.simplification import heuristic_bsf_cost
from rich.console import Console
from itertools import combinations, product

console = Console()

Hz = gates.H.data
Hy = gates.S.data @ gates.H.data


class Clifford2QiSWAP:
    def __init__(self, pauli_0: str, pauli_1: str):
        assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
        I = qi.Pauli('I')
        P0, P1 = qi.Pauli(pauli_0), qi.Pauli(pauli_1)
        self.name = 'iSWAP({}, {})'.format(pauli_0, pauli_1)
        self.pauli_0, self.pauli_1 = pauli_0, pauli_1
        self.data = gates.ISWAP.data
        if self.pauli_0 == 'X':
            self.data = np.kron(Hz, I) @ self.data @ np.kron(Hz, I).conj().T
        elif self.pauli_0 == 'Y':
            self.data = np.kron(Hy, I) @ self.data @ np.kron(Hy, I).conj().T
        if self.pauli_1 == 'X':
            self.data = np.kron(I, Hz) @ self.data @ np.kron(I, Hz).conj().T
        elif self.pauli_1 == 'Y':
            self.data = np.kron(I, Hy) @ self.data @ np.kron(I, Hy).conj().T
        self.ctrl, self.targ = None, None

    def __repr__(self) -> str:
        if self.ctrl is not None and self.targ is not None:
            return f'iSWAP({self.pauli_0}, {self.pauli_1}) @ ({self.ctrl}, {self.targ})'
        return self.name

    def on(self, ctrl: int, targ: int) -> 'Clifford2QiSWAP':
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

    def wrt_iswap(self) -> Tuple[List[str], List[str]]:
        """
        Return local Clifford operators, with respect to which it is equivalent to a CZ gate.

        Returns:
            wrt0, wrt1: List[str]
                Local Clifford operators ("H" or "S") for the first and second qubits.

        Examples:
            >>> cliff = Clifford2QiSWAP('X', 'Z')
            >>> cliff.wrt_iswap()
            (['H'], [])
            >>> cliff = Clifford2QiSWAP('Y', 'Z')
            >>> cliff.wrt_iswap()
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


def _apply_iswap_on_bsf(bsf: BSF, ctrl, targ) -> BSF:
    bsf = bsf.apply_s(ctrl)
    bsf = bsf.apply_h(ctrl)
    bsf = bsf.apply_s(targ)
    bsf = bsf.apply_cx(ctrl, targ)
    bsf = bsf.apply_cx(targ, ctrl)
    bsf = bsf.apply_h(targ)
    return bsf


def apply_clifford_iswap_on_bsf(bsf, clifford: Clifford2QiSWAP, ctrl: int, targ: int) -> 'BSF':
    """Apply 2-qubit Clifford-iSWAP operator."""
    wrt0_iswap, wrt1_iswap = clifford.wrt_iswap()
    # print(wrt0_iswap, wrt1_iswap)
    bsf = deepcopy(bsf)

    for opr in reversed(wrt0_iswap):
        if opr == 'H':
            bsf = bsf.apply_h(ctrl)
        elif opr == 'S':
            bsf = bsf.apply_sdg(ctrl)
        else:
            raise ValueError('Invalid operator')

    for opr in reversed(wrt1_iswap):
        if opr == 'H':
            bsf = bsf.apply_h(targ)
        elif opr == 'S':
            bsf = bsf.apply_sdg(targ)
        else:
            raise ValueError('Invalid operator')

    bsf = _apply_iswap_on_bsf(bsf, ctrl, targ)  # ! here in is the modified functionality

    for opr in wrt0_iswap:
        if opr == 'H':
            bsf = bsf.apply_h(ctrl)
        elif opr == 'S':
            bsf = bsf.apply_s(ctrl)
        else:
            raise ValueError('Invalid operator')

    for opr in wrt1_iswap:
        if opr == 'H':
            bsf = bsf.apply_h(targ)
        elif opr == 'S':
            bsf = bsf.apply_s(targ)
        else:
            raise ValueError('Invalid operator')

    return bsf


def search_cliffords_by_iswap(bsf: BSF, avoid: Tuple[int, int] = None) -> Tuple[BSF, float, Clifford2QiSWAP]:
    avoid = set(avoid) if avoid is not None else set()
    clifford_candidates = np.array([cg.on(qubits[0], qubits[1]) for cg in CLIFFORD_2Q_ISWAP_SET for qubits in
                                    np.array(list(combinations(bsf.qubits_with_ops, 2))) if set(qubits) != set(avoid)])

    # use numpy vectorization to accelerate computation
    def trans_bsf(cliff: Clifford2QiSWAP) -> BSF:
        # return bsf.apply_clifford_2q(cliff, cliff.ctrl, cliff.targ)
        return apply_clifford_iswap_on_bsf(bsf, cliff, cliff.ctrl, cliff.targ)

    trans_bsf = np.vectorize(trans_bsf)

    bsfs = trans_bsf(clifford_candidates)
    costs = heuristic_bsf_cost(bsfs)

    # select the candidates with the minimum cost
    argmin = np.argmin(costs)
    return bsfs[argmin], costs[argmin], clifford_candidates[argmin]


def simplify_bsf_by_iswap(bsf: BSF) -> Tuple[BSF, List[Tuple[Clifford2QiSWAP, BSF]]]:
    """Simplify a Pauli Tableau, until its weights are simultaneously 2."""
    bsf = deepcopy(bsf)
    cliffords_with_locals = []
    avoid = None
    while bsf.total_weight > 2:
        local_bsf = bsf.pop_local_paulis()
        # if local_bsf.total_weight > 0:
        #     console.print(local_bsf)
        #     console.print(local_bsf.paulilist)
        # local_paulis, local_coeffs = local_bsf.paulilist, local_bsf.coeffs
        t, c, cliff = search_cliffords_by_iswap(bsf, avoid)
        avoid = cliff.ctrl, cliff.targ
        # cliffords_with_locals.append((cliff, list(zip(local_paulis, local_coeffs))))
        cliffords_with_locals.append((cliff, local_bsf))
        console.print(
            'applied {} --> {} cost: {} (is_simplified: {})'.format(cliff, t.paulis, c, t.total_weight <= 2))
        bsf = t
    # console.print('Now BSF: {}, cliff_with_locals: {}'.format(bsf, cliffords_with_locals))
    return bsf, cliffords_with_locals


CLIFFORD_2Q_ISWAP_SET = [Clifford2QiSWAP(pauli_0, pauli_1) for pauli_0, pauli_1 in
                         product(['X', 'Y', 'Z'], ['X', 'Y', 'Z'])]


def search_cliffords_by_mixed(bsf: BSF, avoid: Tuple[int, int] = None) -> Tuple[BSF, float, Clifford2QiSWAP]:
    avoid = set(avoid) if avoid is not None else set()

    cnot_clifford_candidates = np.array([cg.on(qubits[0], qubits[1]) for cg in CLIFFORD_2Q_SET for qubits in
                                         np.array(list(combinations(bsf.qubits_with_ops, 2))) if
                                         set(qubits) != set(avoid)])

    iswap_clifford_candidates = np.array([cg.on(qubits[0], qubits[1]) for cg in CLIFFORD_2Q_ISWAP_SET for qubits in
                                          np.array(list(combinations(bsf.qubits_with_ops, 2))) if
                                          set(qubits) != set(avoid)])
    clifford_candidates = np.concatenate((iswap_clifford_candidates, cnot_clifford_candidates))

    # shuffle clifford_candidates
    np.random.shuffle(clifford_candidates)

    # use numpy vectorization to accelerate computation
    def trans_bsf(cliff: Union[Clifford2QiSWAP, Clifford2Q]) -> BSF:
        if isinstance(cliff, Clifford2QiSWAP):
            return apply_clifford_iswap_on_bsf(bsf, cliff, cliff.ctrl, cliff.targ)
        if isinstance(cliff, Clifford2Q):
            return bsf.apply_clifford_2q(cliff, cliff.ctrl, cliff.targ)

    trans_bsf = np.vectorize(trans_bsf)

    bsfs = trans_bsf(clifford_candidates)
    costs = heuristic_bsf_cost(bsfs)

    # select the candidates with the minimum cost
    argmin = np.argmin(costs)
    return bsfs[argmin], costs[argmin], clifford_candidates[argmin]


def simplify_bsf_by_mixed(bsf: BSF) -> Tuple[BSF, List[Tuple[Clifford2QiSWAP, BSF]]]:
    """Simplify a Pauli Tableau, until its weights are simultaneously 2."""
    bsf = deepcopy(bsf)
    cliffords_with_locals = []
    avoid = None
    while bsf.total_weight > 2:
        local_bsf = bsf.pop_local_paulis()
        # if local_bsf.total_weight > 0:
        #     console.print(local_bsf)
        #     console.print(local_bsf.paulilist)
        # local_paulis, local_coeffs = local_bsf.paulilist, local_bsf.coeffs
        t, c, cliff = search_cliffords_by_mixed(bsf, avoid)
        avoid = cliff.ctrl, cliff.targ
        # cliffords_with_locals.append((cliff, list(zip(local_paulis, local_coeffs))))
        cliffords_with_locals.append((cliff, local_bsf))
        console.print(
            'applied {} --> {} cost: {} (is_simplified: {})'.format(cliff, t.paulis, c, t.total_weight <= 2))
        bsf = t
    # console.print('Now BSF: {}, cliff_with_locals: {}'.format(bsf, cliffords_with_locals))
    return bsf, cliffords_with_locals
