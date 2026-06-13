from __future__ import annotations

from functools import partial

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

from .basics import CNOTEquivCliffordGate
from .hamiltonian import Hamiltonian
from .primitive.ordering import order_circuits
from .primitive.peel import peel_compile
from .primitive.simplification import simplify_hamiltonian
from .primitive.utils import constr_circuit_from_simp_steps

_UNROLL_BASIS_GATES = ["cx", "h", "s", "sdg", "rzx", "rxx", "ryy", "rzz"]
# _SYNTHESIS_BASIS_GATES = ["cx", "rz", "sx", "x"]
_SYNTHESIS_BASIS_GATES = ["cx", "u"]

# Max width of a Clifford sub-block we hand to Qiskit's optimal (Bravyi-Maslov)
_OPTIMAL_CLIFFORD_MAX_BLOCK_WIDTH = 3


def _process_same_weight_hamiltonian(
    ham: Hamiltonian, parallel: bool = False, patience: int | None = None
) -> QuantumCircuit:
    """Helper function to process a single Hamiltonian group (used for parallel execution)."""
    ham_, simp_steps = simplify_hamiltonian(ham, parallel=parallel, patience=patience)
    qc = constr_circuit_from_simp_steps(ham_, simp_steps)
    return qc


def _simplify_groups(
    hams: list[Hamiltonian], backend: str, parallel: bool, patience: int | None
) -> list[QuantumCircuit]:
    """Simplify each Hamiltonian group into a subcircuit, parallelizing across
    groups via the chosen backend (a single group always runs in-process)."""
    simp = partial(_process_same_weight_hamiltonian, parallel=parallel, patience=patience)
    if len(hams) <= 1 or backend == "sequential":
        return [simp(ham) for ham in hams]
    if backend == "concurrent.futures":
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor() as executor:
            return list(executor.map(simp, hams))
    if backend == "joblib":
        from joblib import Parallel, delayed

        return Parallel(n_jobs=-1)(delayed(simp)(ham) for ham in hams)
    raise ValueError(f"Unknown backend: {backend}. Use 'joblib', 'concurrent.futures', or 'sequential'.")


def compile_hamiltonian_simulation(
    hamiltonian: Hamiltonian,
    time: float | Parameter = 1.0,
    order: int = 1,
    trotter_steps: int = 1,
    grouping: str | None = None,
    parallel_search: bool = True,
    optimize: bool = True,
    terminal='auto',
    order_method: str | None = None,
    backend: str = "sequential",
    search_patience: int | None = None,
) -> QuantumCircuit:
    """Compile a Hamiltonian simulation circuit using the Phoenix framework.

    Args:
        hamiltonian: The Hamiltonian to simulate.
        time: Evolution time.
        order: Trotter-Suzuki order (1 or 2).
        trotter_steps: Number of Trotter steps.
        grouping: Compilation strategy for Pauli terms:
            - ``None`` (default) or ``"peel"``: the peel-forward engine —
              sequential extraction in a forward-only Clifford frame, grouping
              fully emergent, guaranteed termination, zero numeric
              hyperparameters (see ``primitive.peel`` /
              docs/peel_forward_design.md);
            - ``"support"``: exact same-support grouping + legacy BSF greedy
              search (DAC'25 behavior, kept as the ablation baseline).
        optimize: Whether to apply Qiskit post-optimization.
        order_method: Ordering method for subcircuits in support mode
            (None defaults to 'tsp'). Ignored by peel.
        backend: Parallelization backend for support mode ("joblib",
            "concurrent.futures", or "sequential"). Ignored by peel.
        search_patience: Stall patience of the legacy BSF search safety net
            (support mode only). Default: max(16, 2·#active qubits).

    Returns:
        The compiled quantum circuit.
    """
    if grouping is None or grouping == "peel":
        qc = peel_compile(hamiltonian, terminal=terminal)
    elif grouping == "support":
        hams = hamiltonian.group_same_weights()[::-1]
        circuits = _simplify_groups(hams, backend, parallel=parallel_search, patience=search_patience)
        qc = order_circuits(circuits, method=order_method or "tsp")
    else:
        raise ValueError(
            f"Unknown grouping mode: {grouping!r}; options: 'peel' (default), 'support'"
        )

    if optimize:
        qc = optimize_phoenix_circuit_by_qiskit(qc)

    return qc


def optimize_phoenix_circuit_by_qiskit(qc: QuantumCircuit) -> QuantumCircuit:
    """Topology-preserved post-optimization for phoenix-compiled circuits.

    This pass pipeline has three goals:

    1. cancel Phoenix-native Clifford scaffolding as much as possible while the custom gate
       structure is still visible;
    2. collect short runs of Clifford gates and resynthesize each block with Qiskit's
       optimal (Bravyi-Maslov) Clifford synthesis while the custom gate structure
       is still visible;
    3. unroll the remaining custom Clifford gates to their definitions and let Qiskit resynthesize
       consecutive 2-qubit blocks, so patterns such as ``Rzx -> Ryy -> Rzz`` can be compressed.
    """
    from itertools import product

    from qiskit.transpiler import PassManager, passes

    inverse_list = [CNOTEquivCliffordGate(p0, p1) for p0, p1 in product(["x", "y", "z"], repeat=2)]

    pm = PassManager()
    pm.append(passes.InverseCancellation(inverse_list))
    pm.append(passes.CommutativeInverseCancellation(matrix_based=True))
    pm.append(passes.Optimize1qGatesDecomposition())
    pm.append(passes.CommutativeCancellation())
    _append_optimal_clifford_resynthesis_passes(pm, passes)
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


def _append_optimal_clifford_resynthesis_passes(pm, passes_module) -> None:
    """Collect <=3-qubit Clifford runs and resynthesize them optimally.

    Uses ``CollectCliffords(matrix_based=True)`` so Phoenix's custom Clifford
    gates (``CNOTEquivCliffordGate`` and friends) are matched by their unitary
    directly, without needing to unroll first. Each collected block is then
    lowered via :func:`qiskit.synthesis.synth_clifford_bm` (the Bravyi-Maslov
    method), which produces a CX-count-optimal circuit for 1-, 2-, and 3-qubit
    Cliffords. The width cap keeps every block in BM's optimal regime, so this
    pass cannot regress CX count.
    """
    required = ("CollectCliffords", "HighLevelSynthesis")
    if not all(hasattr(passes_module, name) for name in required):
        return

    try:
        from qiskit.transpiler.passes.synthesis import HLSConfig
    except ImportError:
        return

    pm.append(
        passes_module.CollectCliffords(
            matrix_based=True,
            min_block_size=2,
            max_block_width=_OPTIMAL_CLIFFORD_MAX_BLOCK_WIDTH,
        )
    )
    pm.append(
        passes_module.HighLevelSynthesis(
            hls_config=HLSConfig(use_default_on_unspecified=False, clifford=["bm"])
        )
    )
