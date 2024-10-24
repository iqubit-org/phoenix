"""One-qubit gate decomposition"""

from phoenix.basic import Gate, Circuit, gates
from phoenix.utils.operations import params_zyz, params_u3

OPTIONAL_BASIS = ['zyz', 'u3']


def euler_decompose(g: Gate, basis: str = 'zyz', with_phase: bool = True) -> Circuit:
    """
    One-qubit Euler decomposition.

    !NOTE: Currently only support 'ZYZ' and 'U3' decomposition.

    Args:
        g (Gate): single-qubit quantum gate
        basis (str): decomposition basis
        with_phase (bool): whether return global phase in form of a `GlobalPhase` gate

    Returns:
        Circuit, quantum circuit after Euler decomposition.
    """
    if len(g.tqs) != 1 or g.cqs:
        raise ValueError(f'{g} is not a single-qubit gate with designated qubit for Euler decomposition')
    basis = basis.lower()
    tq = g.tq
    circ = Circuit()
    if basis == 'zyz':
        alpha, (theta, phi, lamda) = params_zyz(g.data)
        circ.append(gates.RZ(lamda).on(tq))
        circ.append(gates.RY(theta).on(tq))
        circ.append(gates.RZ(phi).on(tq))
        if with_phase:
            circ.append(gates.GlobalPhase(alpha).on(tq))
    elif basis == 'u3':
        phase, (theta, phi, lamda) = params_u3(g.data, return_phase=True)
        circ.append(gates.U3(theta, phi, lamda).on(tq))
        if with_phase:
            circ.append(gates.GlobalPhase(phase).on(tq))
    else:
        raise ValueError(f'{basis} is not a supported decomposition method of {OPTIONAL_BASIS}')
    return circ
