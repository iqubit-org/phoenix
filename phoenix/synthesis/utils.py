import qiskit
from itertools import product
from phoenix.basic.circuits import Circuit
from phoenix.basic import gates
from phoenix.models.cliffords import Clifford2Q
from phoenix.models.paulis import BSF
from phoenix import decompose
from typing import List, Tuple
import numpy as np


def unroll_u3(circ: Circuit, by: str = 'zyz') -> Circuit:
    """
    Unroll U1, U2 and U3 gate by Euler decomposition
    """
    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 1 and isinstance(g, (gates.U2, gates.U3)):
            circ_unrolled += decompose.euler_decompose(g, basis=by, with_phase=False)
        elif g.num_qregs == 1 and isinstance(g, gates.U1):
            circ_unrolled.append(gates.RZ(g.angle).on(g.tq))
        else:
            circ_unrolled.append(g)

    return circ_unrolled


def unique_paulis_and_coeffs(paulis: List[str], coeffs: List[float]) -> Tuple[List[str], List[float]]:
    """Remove duplicates in Pauli strings and sum up their coefficients."""
    if len(np.unique(paulis)) == len(paulis):
        return paulis, coeffs
    unique_paulis = []
    unique_coeffs = []
    for pauli, coeff in zip(paulis, coeffs):
        if len(pauli) == pauli.count('I'):
            continue
        if pauli in unique_paulis:
            idx = unique_paulis.index(pauli)
            unique_coeffs[idx] += coeff
        else:
            unique_paulis.append(pauli)
            unique_coeffs.append(coeff)
    return unique_paulis, unique_coeffs


def optimize_clifford_circuit_by_qiskit(circ: Circuit, optimization_level=1) -> Circuit:
    """Topology-preserved optimization by Qiskit"""
    basis_gates = ['h', 's', 'sdg', 'rz', 'u3', 'cx', 'can']
    for p0, p1 in product(['x', 'y', 'z'], repeat=2):
        basis_gates.append(f'c{p0}{p1}')
    return Circuit.from_qiskit(qiskit.transpile(circ.to_qiskit(), optimization_level=optimization_level,
                                                basis_gates=basis_gates))


def config_to_circuit(config, by: str = 'cnot', optimize=True):
    circ = Circuit()
    for item in config:
        if isinstance(item, Clifford2Q):
            circ.append(item.as_gate())
        if isinstance(item, BSF):
            if by == 'cnot':
                circ += item.as_cnot_circuit()
            elif by == 'su4':
                circ += item.as_su4_circuit()

    # by default, we use Qiskit O2 to consolidate redundant 1Q and CNOT gates
    if by == 'cnot' and optimize:
        return optimize_clifford_circuit_by_qiskit(circ, 2)
    return circ


def unroll_su4(circ: Circuit, by: str = 'can') -> Circuit:
    """
    Unroll two-qubit gates, i.e., SU(4) gates, by canonical decomposition or other methods
    """
    if by not in ['can', 'cnot']:
        raise ValueError("Only support canonical (by='can') and CNOT unrolling (by='cnot').")

    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 2 and not g.cqs:  # arbitrary two-qubit gate
            if by == 'can':
                circ_unrolled += decompose.can_decompose(g)
            if by == 'cnot':
                circ_unrolled += decompose.kak_decompose(g)
        else:
            circ_unrolled.append(g)

    return circ_unrolled
