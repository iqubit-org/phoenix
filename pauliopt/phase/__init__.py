"""
    This module contains code to create and simplify circuits of mixed ZX phase gadgets,
    by conjugation with topologically-aware random circuits of CX gates.
"""

from pauliopt.phase.phase_circuits import (
    Z,
    X,
    PhaseGadget,
    PhaseCircuit,
    PhaseCircuitView,
)
from pauliopt.phase.cx_circuits import (
    CXCircuit,
    CXCircuitLayer,
    CXCircuitView,
    CXCircuitLayerView,
)
from pauliopt.phase.optimized_circuits import (
    OptimizedPhaseCircuit,
    iter_anneal,
    reverse_traversal_anneal,
    reverse_traversal,
)
from pauliopt.utils import pi, π
