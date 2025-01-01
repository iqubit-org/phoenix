import cirq
import qiskit
from math import sqrt, pi
from scipy import linalg
from itertools import product
from phoenix.basic.circuits import Circuit
from phoenix.basic import gates, Gate
from phoenix.models.cliffords import Clifford2Q
from phoenix.models.paulis import BSF
from phoenix.utils.ops import params_zyz, params_u3, is_tensor_prod, kron_decomp, replace_close_to_zero_with_zero
from typing import List, Tuple
import numpy as np


def tensor_product_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    """
    Tensor product decomposition of a 2-qubit gate.

    Args:
        g (Gate): 2-qubit gate composed by tensor product
        return_u3 (bool): return gates in form of `U3` or `UnivMathGate`

    Returns:
        Circuit, including two single-qubit gates.
    """
    if len(g.tqs) != 2 or g.cqs:
        raise ValueError(f'{g} is not a 2-qubit gate with designated qubits')
    if not is_tensor_prod(g.data):
        raise ValueError(f'{g} is not a tensor-product unitary gate.')

    u0, u1 = kron_decomp(g.data)
    circ = Circuit()
    if return_u3:
        circ.append(gates.U3(*params_u3(u0)).on(g.tqs[0]))
        circ.append(gates.U3(*params_u3(u1)).on(g.tqs[1]))
    else:
        circ.append(gates.UnivGate(u0, 'U0').on(g.tqs[0]))
        circ.append(gates.UnivGate(u1, 'U1').on(g.tqs[1]))
    # return optimize_circuit(circ)
    return circ


def kak_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    r"""
    KAK decomposition (CNOT basis) of an arbitrary two-qubit gate.

    Step 1: decompose it into

             ┌──────────┐
        ──B0─┤          ├─A0──
             │ exp(-iH) │
        ──B1─┤          ├─A1──
             └──────────┘
    .. math::

        \left( A_0 \otimes A_1 \right) e^{-iH}\left( B_0 \otimes B_1 \right)

    Step 2: synthesize parameterized gates exp(-iH) using three CNOT gates

        ──B0────●────U0────●────V0────●────W─────A0──
                │          │          │
        ──B1────X────U1────X────V1────X────W†────A1──

    Args:
        g (QuantumGate): 2-qubit quantum gate
        return_u3 (bool): return gates in form of `U3` or `UnivMathGate`

    Returns:
        Circuit, including at most 3 CNOT gates and 6 single-qubit gates.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """
    # ! Since the KAK decomposition implemented bw our own is not robust, we use the Cirq's implementation by default
    if len(g.tqs) != 2 or g.cqs:
        raise ValueError(f'{g} is not an arbitrary 2-qubit gate with designated qubits')

    if is_tensor_prod(g.data):
        return tensor_product_decompose(g, return_u3)
    pauli_i = gates.I.data
    pauli_x = gates.X.data
    pauli_z = gates.Z.data

    res = cirq.kak_decomposition(g.data)
    b0 = res.single_qubit_operations_before[0]
    b1 = res.single_qubit_operations_before[1]
    a0 = res.single_qubit_operations_after[0]
    a1 = res.single_qubit_operations_after[1]
    h1, h2, h3 = [-k for k in res.interaction_coefficients]
    u0 = 1j / sqrt(2) * (pauli_x + pauli_z) @ linalg.expm(-1j * (h1 - pi / 4) * pauli_x)
    v0 = -1j / sqrt(2) * (pauli_x + pauli_z)
    u1 = linalg.expm(-1j * h3 * pauli_z)
    v1 = linalg.expm(1j * h2 * pauli_z)
    w = (pauli_i - 1j * pauli_x) / sqrt(2)

    # list of operators
    rots1 = [b0, u0, v0, a0 @ w]  # rotation gate on idx1
    rots2 = [b1, u1, v1, a1 @ w.conj().T]

    idx1, idx2 = g.tqs
    circ = Circuit()
    if return_u3:
        circ.append(
            gates.U3(*params_u3(rots1[0])).on(idx1),
            gates.U3(*params_u3(rots2[0])).on(idx2),
            gates.X.on(idx2, idx1),
            gates.U3(*params_u3(rots1[1])).on(idx1),
            gates.U3(*params_u3(rots2[1])).on(idx2),
            gates.X.on(idx2, idx1),
            gates.U3(*params_u3(rots1[2])).on(idx1),
            gates.U3(*params_u3(rots2[2])).on(idx2),
            gates.X.on(idx2, idx1),
            gates.U3(*params_u3(rots1[3])).on(idx1),
            gates.U3(*params_u3(rots2[3])).on(idx2)
        )
    else:
        circ.append(
            gates.UnivGate(rots1[0], 'B0').on(idx1),
            gates.UnivGate(rots2[0], 'B1').on(idx2),
            gates.X.on(idx2, idx1),
            gates.UnivGate(rots1[1], 'U0').on(idx1),
            gates.UnivGate(rots2[1], 'U1').on(idx2),
            gates.X.on(idx2, idx1),
            gates.UnivGate(rots1[2], 'V0').on(idx1),
            gates.UnivGate(rots2[2], 'V1').on(idx2),
            gates.X.on(idx2, idx1),
            gates.UnivGate(rots1[3], 'W0').on(idx1),
            gates.UnivGate(rots2[3], 'W1').on(idx2)
        )

    return remove_identities(circ)


def can_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    """
    Similar to KAK decomposition, but returning Canonical with single-qubit gates.
    The interleaved Canonical gate is defined as
        \mathrm{Can}(\theta_1, \theta_2, \theta_3) = e^{- i \frac{1}{2}(\theta_1 XX + \theta_2 YY + \theta_3 ZZ)}
        in this package
    """
    # ! Since the KAK decomposition implemented bw our own is not robust, we use the Cirq's implementation by default
    if len(g.tqs) != 2 or g.cqs:
        raise ValueError(f'{g} is not an arbitrary 2-qubit gate with designated qubits')

    if is_tensor_prod(g.data):
        return tensor_product_decompose(g, return_u3)

    res = cirq.kak_decomposition(g.data)
    kak_coeffs = np.array(res.interaction_coefficients)
    angles = -2 * replace_close_to_zero_with_zero(kak_coeffs)
    angles = -angles[0], -angles[1], angles[2]
    b0 = res.single_qubit_operations_before[0]
    b1 = gates.Z.data @ res.single_qubit_operations_before[1]
    a0 = res.single_qubit_operations_after[0]
    a1 = res.single_qubit_operations_after[1] @ gates.Z.data

    circ = Circuit()
    if return_u3:
        circ.append(
            gates.U3(*params_u3(b0)).on(g.tqs[0]),
            gates.U3(*params_u3(b1)).on(g.tqs[1]),
            # gate.Can(t1, t2, t3).on(g.tqs),
            gates.Can(*angles).on(g.tqs),
            gates.U3(*params_u3(a0)).on(g.tqs[0]),
            gates.U3(*params_u3(a1)).on(g.tqs[1]),
        )
    else:
        circ.append(
            gates.UnivGate(b0, 'B0').on(g.tqs[0]),
            gates.UnivGate(b1, 'B1').on(g.tqs[1]),
            # gate.Can(t1, t2, t3).on(g.tqs),
            gates.Can(*angles).on(g.tqs),
            gates.UnivGate(a0, 'A0').on(g.tqs[0]),
            gates.UnivGate(a1, 'A1').on(g.tqs[1]),
        )

    return remove_identities(circ)


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
    OPTIONAL_BASIS = ['zyz', 'u3']

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


def remove_identities(circuit: Circuit) -> Circuit:
    """
    Optimize the quantum circuit, i.e., removing identity operators.
    Naive strategy: remove all identity operators.

    Args:
        circuit (Circuit): original input circuit.

    Returns:
        Circuit, the optimized quantum circuit.
    """
    from phoenix.utils.ops import is_equiv_unitary

    circuit_opt = Circuit()
    for g in circuit:
        if not (g.num_qregs == 1 and is_equiv_unitary(g.data, gates.I.data)):
            circuit_opt.append(g)
    return circuit_opt


def unroll_u3(circ: Circuit, by: str = 'zyz') -> Circuit:
    """
    Unroll U1, U2 and U3 gate by Euler decomposition
    """
    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 1 and isinstance(g, (gates.U2, gates.U3)):
            circ_unrolled += euler_decompose(g, basis=by, with_phase=False)
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
                circ_unrolled += can_decompose(g)
            if by == 'cnot':
                circ_unrolled += kak_decompose(g)
        else:
            circ_unrolled.append(g)

    return circ_unrolled
