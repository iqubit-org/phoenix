"""Quantum Shannon Decomposition"""

from typing import List
import numpy as np
from scipy import linalg
from phoenix.basic.gates import Gate
from phoenix.basic import gates
from phoenix.decompose.fixed.pauli_related import ccx_decompose
from phoenix.decompose.universal.one_qubit_decompose import euler_decompose
from phoenix.decompose.universal.two_qubit_decompose import abc_decompose
from phoenix.utils.operations import controlled_unitary_matrix
from phoenix.utils.functions import is_power_of_two
from phoenix.basic.circuits import Circuit, optimize_circuit


def cu_decompose(g: Gate) -> Circuit:
    """
    Decomposition for arbitrary-dimension controlled-U gate decomposition, with m control qubits and n target qubits

    When recursively calling the function itself, `m` decreases, while `n` holds constant.

    Decomposition rules:
        1. when m == 0: use `qs_decompose()`
        2. when m == 1
            - when n == 1: use `abc_decompose()`
            - when n > 1: use `qs_decompose()`
        3. when m > 1:
            - when it is Toffoli gate: use `ccx_decompose()`
            - otherwise: V is matrix-square-root of U
                 ─/──●───        ─/───────●──────────●────●──/─
                     │                    │          │    │
                 ────●───   ==   ────●────X────●─────X────┼────
                     │               │         │          │
                 ────U───        ────V─────────V†─────────V────

    Args:
        g (QuantumGate): instance of quantum gate

    Returns:
        Circuit, composed of 1-qubit gates and CNOT gates.
    """
    m = len(g.cqs)
    n = len(g.tqs)

    if m == 0:
        return qs_decompose(g)

    if m == 1:
        if n == 1:
            # normal 2-qubit controlled-U gate
            if isinstance(g, gates.XGate):
                return Circuit([g])
            return abc_decompose(g)
        if n > 1:
            # 1 control qubits, 2+ target qubits
            cu = controlled_unitary_matrix(g.data)
            return qs_decompose(gates.UnivGate(cu, 'CU').on(g.cqs + g.tqs))

    if m == 2 and isinstance(g, gates.XGate) and n == 1:
        # Toffoli gate
        return ccx_decompose(g)

    v = linalg.sqrtm(g.data)
    vh = np.conj(np.transpose(v))
    cqs_1st, cq_2nd = g.cqs[:-1], g.cqs[-1]
    circ = Circuit()
    circ += cu_decompose(gates.UnivGate(v, 'V').on(g.tqs, cq_2nd))
    circ += cu_decompose(gates.X.on(cq_2nd, cqs_1st))
    circ += cu_decompose(gates.UnivGate(vh, 'Vh').on(g.tqs, cq_2nd))
    circ += cu_decompose(gates.X.on(cq_2nd, cqs_1st))
    circ += cu_decompose(gates.UnivGate(v, 'V').on(g.tqs, cqs_1st))
    return circ


def qs_decompose(g: Gate) -> Circuit:
    r"""
    Quantum Shannon decomposition for arbitrary-dimension unitary gate.

       ┌───┐               ┌───┐     ┌───┐     ┌───┐
      ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
       │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
     /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
       └───┘          └───┘     └───┘     └───┘     └───┘

    The number of CNOT gates in the decomposed circuit is

    .. math::

        O(4^n)

    Args:
        g (Gate): instance of quantum gate

    Returns:
        Circuit, composed of 1-qubit gates and CNOT gates.

    References:
        'Synthesis of Quantum Logic Circuits'
        https://arxiv.org/abs/quant-ph/0406176
    """
    if g.cqs:
        raise ValueError(f'{g} is a controlled gate. Use cu_decompose() instead.')
    n = g.n_qubits

    if n == 1:
        return euler_decompose(g, basis='u3', with_phase=False)

    (u1, u2), rads, (v1h, v2h) = linalg.cossin(g.data, separate=True, p=2 ** (n - 1), q=2 ** (n - 1))
    rads *= 2
    circ_left = demultiplex_pair(v1h, v2h, g.tqs[1:], g.tqs[0])
    circ_middle = demultiplex_pauli('Y', g.tqs[0], g.tqs[1:], *rads)
    circ_right = demultiplex_pair(u1, u2, g.tqs[1:], g.tqs[0])
    return optimize_circuit(circ_left + circ_middle + circ_right)


def demultiplex_pair(u1: np.ndarray, u2: np.ndarray, tqs: List[int], cq: int) -> Circuit:
    """
    Decompose a multiplexor defined by a pair of unitary matrices operating on the same subspace.

    That is, decompose

        cq   ────□────
              ┌──┴──┐
        tqs /─┤     ├─
              └─────┘

    represented by the block diagonal matrix

            ┏         ┓
            ┃ U1      ┃
            ┃      U2 ┃
            ┗         ┛

    to
                  ┌───┐
       cq  ───────┤ Rz├──────
             ┌───┐└─┬─┘┌───┐
       tqs /─┤ W ├──□──┤ V ├─
             └───┘     └───┘

    by means of simultaneous unitary diagonalization.

    Args:
        u1 (ndarray): applied if the control qubit is |0>
        u2 (ndarray): applied if the control qubit is |1>
        tqs (List[int]): target qubit indices
        cq (int): control qubit index

    Returns:
        Circuit, composed of 1-qubit gates and CNOT gates.
    """
    if u1.shape != u2.shape:
        raise ValueError(f'Input matrices have different dimensions: {u1.shape}, {u2.shape}.')
    tqs = tqs.copy()
    u1u2h = u1 @ u2.conj().T
    if np.allclose(u1u2h, u1u2h.conj().T):  # is hermitian
        eigvals, v = linalg.eigh(u1u2h)
    else:
        evals, v = linalg.schur(u1u2h, output='complex')
        eigvals = np.diag(evals)
    dvals = np.sqrt(eigvals)
    rads = 2 * np.angle(dvals.conj())
    w = np.diag(dvals) @ v.conj().T @ u2
    circ_left = qs_decompose(gates.UnivGate(w, 'W').on(tqs))
    circ_middle = demultiplex_pauli('Z', cq, tqs, *rads)
    circ_right = qs_decompose(gates.UnivGate(v, 'V').on(tqs))
    return circ_left + circ_middle + circ_right


def demultiplex_pauli(sigma: str, tq: int, cqs: List[int], *args, permute_cnot: bool = False) -> Circuit:
    """
    Decompose a Pauli-rotation (RY or RZ) multiplexor defined by 2^(n-1) rotation angles.

         ────□───        ─────────●─────────●────
             │                    │         │
         ─/──□───   ==   ─/──□────┼────□────┼──/─
             │               │    │    │    │
         ────R───        ────R────X────R────X────

    Args:
        sigma (str): Axis of rotation Pauli matrix, 'Y' or 'Z'.
        tq (int): target qubit index
        cqs (List[int]): control qubit indices
        *args: 2^(n-1) rotation angles in which n-1 is the length of `cqs`
        permute_cnot (bool): whether permute positions of CNOT gates

    Returns:
        Circuit, composed of 1-qubit Pauli-rotation gates and CNOT gates.
    """
    if not is_power_of_two(len(args)) or len(args) < 2:
        raise ValueError('Number of angle parameters is illegal (should be power of 2 and no less than 2).')
    if len(args) != 2 ** len(cqs):
        raise ValueError(f'Number of angle parameters ({len(args)}) does not coincide with control qubits ({len(cqs)})')
    n = int(np.log2(len(args))) + 1
    cqs = cqs.copy()
    circ = Circuit()

    if n == 2:
        circ.append(
            getattr(gates, f'R{sigma.upper()}')((args[0] + args[1]) / 2).on(tq),
            gates.X.on(tq, cqs[0]),
            getattr(gates, f'R{sigma.upper()}')((args[0] - args[1]) / 2).on(tq),
            gates.X.on(tq, cqs[0])
        )
        if permute_cnot:
            circ.append(circ.pop(0))
    elif n == 3:
        (s0, s1), (t0, t1) = _cal_demultiplex_rads(args)
        cq_1st = cqs.pop(0)
        cq_2nd = cqs.pop(0)
        circ.append(
            getattr(gates, f'R{sigma.upper()}')(s0.item()).on(tq),
            gates.X.on(tq, cq_2nd),
            getattr(gates, f'R{sigma.upper()}')(s1.item()).on(tq),
            gates.X.on(tq, cq_1st),
            getattr(gates, f'R{sigma.upper()}')(t1.item()).on(tq),
            gates.X.on(tq, cq_2nd),
            getattr(gates, f'R{sigma.upper()}')(t0.item()).on(tq),
            gates.X.on(tq, cq_1st)
        )
    else:
        (s0, s1), (t0, t1) = _cal_demultiplex_rads(args)
        cq_1st = cqs.pop(0)
        cq_2nd = cqs.pop(0)
        circ += demultiplex_pauli(sigma, tq, cqs, *s0)
        circ.append(gates.X.on(tq, cq_2nd))
        circ += demultiplex_pauli(sigma, tq, cqs, *s1)
        circ.append(gates.X.on(tq, cq_1st))
        circ += demultiplex_pauli(sigma, tq, cqs, *t1)
        circ.append(gates.X.on(tq, cq_2nd))
        circ += demultiplex_pauli(sigma, tq, cqs, *t0)
        circ.append(gates.X.on(tq, cq_1st))
    return circ


def _cal_demultiplex_rads(rads):
    r"""
    Calculation rotation angles for two-level decomposing of a Pauli-rotation multiplexor.

    Reshape `rads` into a blocked matrix in presentation of

        ┏                           ┓
        ┃ θ_{00}                    ┃
        ┃                           ┃
        ┃       θ_{01}              ┃
        ┃                           ┃
        ┃             θ_{10}        ┃
        ┃                           ┃
        ┃                   θ_{11}  ┃
        ┗                           ┛

    Then calculate `\phi`

        ┏           ┓         ┏              ┓         ┏              ┓
        ┃ φ_0       ┃         ┃ θ_{00}       ┃         ┃ θ_{10}       ┃
        ┃           ┃ = 1/2 * ┃              ┃ + 1/2 * ┃              ┃
        ┃       φ_1 ┃         ┃       θ_{01} ┃         ┃       θ_{11} ┃
        ┗           ┛         ┗              ┛         ┗              ┛

    and `\lambda`

        ┏           ┓         ┏              ┓         ┏              ┓
        ┃ λ_0       ┃         ┃ θ_{00}       ┃         ┃ θ_{10}       ┃
        ┃           ┃ = 1/2 * ┃              ┃ - 1/2 * ┃              ┃
        ┃       λ_1 ┃         ┃       θ_{01} ┃         ┃       θ_{11} ┃
        ┗           ┛         ┗              ┛         ┗              ┛

    Finally, decompose multiplexors in presentation of `\phi` and `\lambda`, respectively.

    Args:
        rads: rotation angles representing the original Pauli-rotation multiplexor

    Returns:
        rotation angles after two-level decomposition
    """
    dim = len(rads)
    rads = np.reshape(rads, [2, 2, int(dim / 2 / 2)])
    p0 = (rads[0, 0, :] + rads[1, 0, :]) / 2
    p1 = (rads[0, 1, :] + rads[1, 1, :]) / 2
    l0 = (rads[0, 0, :] - rads[1, 0, :]) / 2
    l1 = (rads[0, 1, :] - rads[1, 1, :]) / 2
    return ((p0 + p1) / 2, (p0 - p1) / 2), ((l0 + l1) / 2, (l0 - l1) / 2)
