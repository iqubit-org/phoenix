from phoenix.basic.circuits import Circuit
from phoenix.basic import gates
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
