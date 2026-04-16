from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import Parameter
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .basics import CNOTEquivCliffordGate
from .hamiltonian import Hamiltonian
from .primitive.ordering import order_circuits
from .primitive.simplification import simplify_hamiltonian
from .primitive.utils import constr_circuit_from_simp_steps


_UNROLL_BASIS_GATES = ["cx", "h", "s", "sdg", "rzx", "rxx", "ryy", "rzz"]
# _SYNTHESIS_BASIS_GATES = ["cx", "rz", "sx", "x"]
_SYNTHESIS_BASIS_GATES = ["cx", "u"]


def _process_same_weight_hamiltonian(ham: Hamiltonian) -> QuantumCircuit:
    """Helper function to process a single Hamiltonian group (used for parallel execution)."""
    ham_, simp_steps = simplify_hamiltonian(ham)
    qc = constr_circuit_from_simp_steps(ham_, simp_steps, optimize=True)
    return qc


def compile_hamiltonian_simulation(
    hamiltonian: Hamiltonian,
    time: float | Parameter = 1.0,
    order: int = 1,
    trotter_steps: int = 1,
    grouping: bool = True,
    optimize: bool = True,
    order_method: str = "trivial",
    backend: str = "sequential",
) -> QuantumCircuit:
    """Compile a Hamiltonian simulation circuit using the Phoenix framework.

    Args:
        hamiltonian: The Hamiltonian to simulate.
        time: Evolution time.
        order: Trotter-Suzuki order (1 or 2).
        trotter_steps: Number of Trotter steps.
        optimize: Whether to apply Qiskit post-optimization.
        order_method: Ordering method for subcircuits.
        backend: Parallelization backend ("joblib", "concurrent.futures", or "sequential").

    Returns:
        The compiled quantum circuit.
    """
    if grouping:
        hams = hamiltonian.group_same_weights()
    else:
        hams = [hamiltonian]

    if len(hams) <= 1:
        circuits = [_process_same_weight_hamiltonian(ham) for ham in hams]
    elif backend == "concurrent.futures":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            circuits = list(executor.map(_process_same_weight_hamiltonian, hams))
    elif backend == "joblib":
        from joblib import Parallel, delayed
        circuits = Parallel(n_jobs=-1)(delayed(_process_same_weight_hamiltonian)(ham) for ham in hams)
    elif backend == "sequential":
        circuits = [_process_same_weight_hamiltonian(ham) for ham in hams]
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'joblib', 'concurrent.futures', or 'sequential'.")

    qc = order_circuits(circuits, method=order_method)

    if optimize:
        qc = optimize_phoenix_circuit_by_qiskit(qc)

    return qc


def optimize_phoenix_circuit_by_qiskit(qc: QuantumCircuit) -> QuantumCircuit:
    """Topology-preserved post-optimization for phoenix-compiled circuits.

    This pass pipeline has two goals:

    1. cancel Phoenix-native Clifford scaffolding as much as possible while the custom gate
       structure is still visible;
    2. unroll the remaining custom Clifford gates to their definitions and let Qiskit resynthesize
       consecutive 2-qubit blocks, so patterns such as ``Rzx -> Ryy -> Rzz`` can be compressed.
    """
    from qiskit.transpiler import passes, PassManager
    from itertools import product

    inverse_list = [
        CNOTEquivCliffordGate(p0, p1)
        for p0, p1 in product(["x", "y", "z"], repeat=2)
    ]
    
    pm = PassManager()
    pm.append(passes.InverseCancellation(inverse_list))
    pm.append(passes.CommutativeInverseCancellation(matrix_based=True))
    pm.append(passes.Optimize1qGatesDecomposition())
    pm.append(passes.CommutativeCancellation())
    pm.append(
        passes.UnrollCustomDefinitions(
            SessionEquivalenceLibrary,
            basis_gates=_UNROLL_BASIS_GATES,
        )
    )
    pm.append(passes.Collect2qBlocks())
    pm.append(passes.ConsolidateBlocks(basis_gates=["cx"]))
    pm.append(passes.UnitarySynthesis(basis_gates=_SYNTHESIS_BASIS_GATES))
    pm.append(passes.Optimize1qGatesDecomposition(basis=_SYNTHESIS_BASIS_GATES[1:]))
    pm.append(passes.CommutativeCancellation())
        
    qc = pm.run(qc)

    return qc


