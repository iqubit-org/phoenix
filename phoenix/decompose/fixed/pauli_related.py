import numpy as np
from phoenix.basic import gates
from phoenix.basic import Gate, Circuit
from typing import List


def mcx_decompose(MCX: Gate, avail_qubits: List[int]) -> Circuit:
    """
    Decompose multi-controlled X gate into CCX gates according to rules in Barenco and Adriano (1995).

    n: number of available qubits
    m: number of MCX control qubits

    There must be m ≤ n-2 !!!

    If 3 ≤ m ≤ ⌈n/2⌉, apply the decomposition rule like below:

        q_0: ───@───        q_0: ───────────@───────────────@───────
                │                           │               │
        q_1: ───@───        q_1: ───────────@───────────────@───────
                │                           │               │
        q_2: ───@───        q_2: ───────@───┼───@───────@───┼───@───
                │                       │   │   │       │   │   │
        q_3: ───@───   =>   q_3: ───@───┼───┼───┼───@───┼───┼───┼───
                │                   │   │   │   │   │   │   │   │
        q_4: ───┼───        q_4: ───┼───@───X───@───┼───@───X───@───
                │                   │   │       │   │   │       │
        q_5: ───│───        q_5: ───@───X───────X───@───X───────X───
                │                   │               │
        q_6: ───X───        q_6: ───X───────────────X───────────────

    If ⌈n/2⌉ < m ≤ n-2, apply the decomposition rule like below:

        q_0: ───@───        q_0: ───@───────@───────
                │                   │       │
        q_1: ───@───        q_1: ───@───────@───────
                │                   │       │
        q_2: ───@───        q_2: ───@───────@───────
                │                   │       │
        q_3: ───@───   =>   q_3: ───@───────@───────
                │                   │       │
        q_4: ───@───        q_4: ───┼───@───┼───@───
                │                   │   │   │   │
        q_5: ───@───        q_5: ───┼───@───┼───@───
                │                   │   │   │   │
        q_6: ───┼───        q_6: ───X───@───X───@───
                │                       │       │
        q_7: ───X───        q_7: ───────X───────X───

    References:
        Barenco, Adriano, et al. "Elementary gates for quantum computation." Physical review A 52.5 (1995): 3457.

    """
    n = len(avail_qubits)
    m = len(MCX.cqs)
    # print('n={}, m={}'.format(n, m))
    assert m >= 3, "MCX must have at least 3 qubits"
    circ = Circuit()
    if 3 <= m <= np.ceil(n / 2):
        # borrow m-2 ancilla qubits
        ancilla_qubits = [q for q in avail_qubits if q not in MCX.qregs][:(m - 2)]
        circ.append(gates.X.on(MCX.tq, [MCX.cqs[-1], ancilla_qubits[-1]]))
        for i in range(m - 3):
            circ.append(gates.X.on(ancilla_qubits[-1 - i], [MCX.cqs[-2 - i], ancilla_qubits[-2 - i]]))
        circ.append(gates.X.on(ancilla_qubits[0], MCX.cqs[:2]))
        for i in range(m - 3):
            circ.append(gates.X.on(ancilla_qubits[1 + i], [MCX.cqs[2 + i], ancilla_qubits[i]]))
        circ.append(gates.X.on(MCX.tq, [MCX.cqs[-1], ancilla_qubits[-1]]))
        for i in range(m - 3):
            circ.append(gates.X.on(ancilla_qubits[-1 - i], [MCX.cqs[-2 - i], ancilla_qubits[-2 - i]]))
        circ.append(gates.X.on(ancilla_qubits[0], MCX.cqs[:2]))
        for i in range(m - 3):
            circ.append(gates.X.on(ancilla_qubits[1 + i], [MCX.cqs[2 + i], ancilla_qubits[i]]))
    elif np.ceil(n / 2) < m <= n - 2:
        # borrow 1 ancilla qubit
        ancilla_qubit = [q for q in avail_qubits if q not in MCX.qregs][0]
        mcx1 = gates.X.on(ancilla_qubit, MCX.cqs[:-2])
        mcx2 = gates.X.on(MCX.tq, MCX.cqs[-2:] + [ancilla_qubit])
        if len(mcx1.cqs) >= 3:
            mcx1_circ = mcx_decompose(mcx1, avail_qubits)
        else:
            mcx1_circ = Circuit([mcx1])
        mcx2_circ = mcx_decompose(mcx2, avail_qubits)
        circ += mcx1_circ + mcx2_circ + mcx1_circ + mcx2_circ
    else:
        raise ValueError("Available qubits are not enough for MCX decomposition")
    return circ


def ccx_decompose(CCX: Gate) -> Circuit:
    """
    Decompose Toffoli gate.

        ──●──       ───────●────────────────────●────T──────────X────T†────X──
          │                │                    │               │          │
        ──●──  ==>  ───────┼──────────●─────────┼──────────●────●────T─────●──
          │                │          │         │          │
        ──X──       ──H────X────T†────X────T────X────T†────X────T────H────────

    Args:
        CCX: Toffoli gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CCX, gates.XGate) and len(CCX.cqs) == 2 and len(CCX.tqs) == 1):
        raise ValueError("CCX must be a two control one target XGate")
    cq1, cq2 = CCX.cqs
    tq = CCX.tq
    return Circuit([
        gates.H.on(tq),
        gates.X.on(tq, cq1),
        gates.T.on(tq).hermitian(),
        gates.X.on(tq, cq2),
        gates.T.on(tq),
        gates.X.on(tq, cq1),
        gates.T.on(tq).hermitian(),
        gates.X.on(tq, cq2),
        gates.T.on(tq),
        gates.T.on(cq1),
        gates.X.on(cq1, cq2),
        gates.H.on(tq),
        gates.T.on(cq2),
        gates.T.on(cq1).hermitian(),
        gates.X.on(cq1, cq2),
    ])


def ccx_decompose_adjacent(CCX: Gate) -> Circuit:
    """
    Decompose Toffoli gate with adjacent qubit coupling patterns (cq1 <--> tq <--> cq2).
    """
    if not (isinstance(CCX, gates.XGate) and len(CCX.cqs) == 2 and len(CCX.tqs) == 1):
        raise ValueError("CCX must be a two control one target XGate")
    cq1, cq2 = CCX.cqs
    tq = CCX.tq
    return Circuit([
        gates.H.on(tq),
        gates.X.on(cq2, tq),
        gates.X.on(tq, cq1),
        gates.T.on(cq2),
        gates.X.on(cq2, tq),
        gates.T.on(cq2),
        gates.T.on(tq),
        gates.X.on(tq, cq1),
        gates.X.on(cq2, tq),
        gates.X.on(tq, cq1),
        gates.TDG.on(cq2),
        gates.X.on(cq2, tq),
        gates.X.on(tq, cq1),
        gates.TDG.on(cq1),
        gates.TDG.on(cq2),
        gates.TDG.on(tq),
        gates.H.on(tq)
    ])


def cy_decompose(CY: Gate) -> Circuit:
    """
    Decompose CY gate.

        ──●──       ────────●───────
          │    ==>          │
        ──Y──       ──S†────X────S──

    Args:
        CY: Controlled-Y gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CY, gates.YGate) and len(CY.cqs) == 1 and len(CY.tqs) == 1):
        raise ValueError("CY must be a one control one target YGate")
    cq = CY.cq
    tq = CY.tq
    return Circuit([
        gates.S.on(tq).hermitian(),
        gates.X.on(tq, cq),
        gates.S.on(tq),
    ])


def cz_decompose(CZ: Gate) -> Circuit:
    """
    Decompose CY gate.

        ──●──       ───────●───────
          │    ==>         │
        ──Z──       ──H────X────H──

    Args:
        CZ: Controlled-Z gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CZ, gates.ZGate) and len(CZ.cqs) == 1 and len(CZ.tqs) == 1):
        raise ValueError("CZ must be a one control one target ZGate")
    cq = CZ.cq
    tq = CZ.tq
    return Circuit([
        gates.H.on(tq),
        gates.X.on(tq, cq),
        gates.H.on(tq),
    ])


def crx_decompose(CRX: Gate) -> Circuit:
    """
    Decompose CRX gate.

    ─────●─────       ───────●────────────────●───────────────────
         │       ==>         │                │
    ───RX(1)───       ──S────X────RY(-1/2)────X────RY(1/2)────S†──

    Args:
        CRX: Controlled-RX gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CRX, gates.RX) and len(CRX.cqs) == 1 and len(CRX.tqs) == 1):
        raise ValueError("CRX must be a one control one target RXGate")
    cq = CRX.cq
    tq = CRX.tq
    return Circuit([
        gates.S.on(tq),
        gates.X.on(tq, cq),
        gates.RY(- CRX.angle / 2).on(tq),
        gates.X.on(tq, cq),
        gates.RY(CRX.angle / 2).on(tq),
        gates.S.on(tq).hermitian(),
    ])


def cry_decompose(CRY: Gate) -> Circuit:
    """
    Decompose CRY gate.

    ─────●─────       ─────────────●────────────────●──
         │       ==>               │                │
    ───RY(1)───       ──RY(1/2)────X────RY(-1/2)────X──

    Args:
        CRY: Controlled-RY gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CRY, gates.RY) and len(CRY.cqs) == 1 and len(CRY.tqs) == 1):
        raise ValueError("CRY must be a one control one target RYGate")
    cq = CRY.cq
    tq = CRY.tq
    return Circuit([
        gates.RY(CRY.angle / 2).on(tq),
        gates.X.on(tq, cq),
        gates.RY(-CRY.angle / 2).on(tq),
        gates.X.on(tq, cq),
    ])


def crz_decompose(CRZ: Gate) -> Circuit:
    """
    Decompose CRZ gate.

    ─────●─────       ─────────────●────────────────●──
         │       ==>               │                │
    ───RZ(1)───       ──RZ(1/2)────X────RZ(-1/2)────X──

    Args:
        CRZ: Controlled-RZ gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CRZ, gates.RZ) and len(CRZ.cqs) == 1 and len(CRZ.tqs) == 1):
        raise ValueError("CRZ must be a one control one target RZGate")
    cq = CRZ.cq
    tq = CRZ.tq
    return Circuit([
        gates.RZ(CRZ.angle / 2).on(tq),
        gates.X.on(tq, cq),
        gates.RZ(-CRZ.angle / 2).on(tq),
        gates.X.on(tq, cq),
    ])
