"""
Decomposition rules for fixed gates composed of H, S and T.
"""
from math import pi
from phoenix.basic import gates
from phoenix.basic import Gate, Circuit


def ch_decompose(CH: Gate) -> Circuit:
    """
    Decompose controlled-H gate.

    ──●──       ─────────────────●───────────────────
      │    ==>                   │
    ──H──       ──S────H────T────X────T†────H────S†──

    Args:
        CH: Controlled-H gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CH, gate.HGate) and len(CH.cqs) == 1 and len(CH.tqs) == 1):
        raise ValueError("CH must be a one control one target HGate")
    cq = CH.cq
    tq = CH.tq
    return Circuit([
        gates.S.on(tq),
        gates.H.on(tq),
        gates.T.on(tq),
        gates.X.on(tq, cq),
        gates.T.on(tq).hermitian(),
        gates.H.on(tq),
        gates.S.on(tq).hermitian(),
    ])


def cs_decompose(CS: Gate) -> Circuit:
    """
    Decompose controlled-S gate.

    ──●──  ==>  ──T────●──────────●──
      │    ==>         │          │
    ──S──  ==>  ──T────X────T†────X──

    Args:
        CS: Controlled-S gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CS, gates.SGate) and len(CS.cqs) == 1 and len(CS.tqs) == 1):
        raise ValueError("CS must be a one control one target SGate")
    cq = CS.cq
    tq = CS.tq
    return Circuit([
        gates.T.on(tq),
        gates.T.on(cq),
        gates.X.on(tq, cq),
        gates.T.on(tq).hermitian(),
        gates.X.on(tq, cq),
    ])


def ct_decompose(CT: Gate) -> Circuit:
    """
    Decompose controlled-T gate.

    ──●──       ──PS(-π/8)───●────────────────●──
      │    ==>               │                │
    ──T──       ──RZ(π/8)────X────RZ(-π/8)────X──

    Args:
        CT: Controlled-T gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CT, gates.TGate) and len(CT.cqs) == 1 and len(CT.tqs) == 1):
        raise ValueError("CT must be a one control one target TGate")
    cq = CT.cq
    tq = CT.tq
    return Circuit([
        gates.PhaseShift(pi / 8).on(cq),
        gates.RZ(pi / 8).on(tq),
        gates.X.on(tq, cq),
        gates.RZ(- pi / 8).on(tq),
        gates.X.on(tq, cq),
    ])
