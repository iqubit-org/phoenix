from __future__ import annotations

from functools import partial

from qiskit import QuantumCircuit
from .hamiltonian import Hamiltonian
from .primitive.ordering import order_circuits
from .primitive.holistic import holistic_compile
from .primitive.simplification import simplify_hamiltonian
from .primitive.utils import constr_circuit_from_simp_steps
from .utils import post_transpile


def _process_same_weight_hamiltonian(
    ham: Hamiltonian, parallel: bool = False, patience: int | None = None, optimize : bool = True
) -> QuantumCircuit:
    """Helper function to process a single Hamiltonian group (used for parallel execution)."""
    ham_, simp_steps = simplify_hamiltonian(ham, parallel=parallel, patience=patience)
    qc = constr_circuit_from_simp_steps(ham_, simp_steps, optimize=optimize)
    return qc


def _simplify_groups(
    hams: list[Hamiltonian], backend: str, parallel: bool, patience: int | None, optimize : bool = True
) -> list[QuantumCircuit]:
    """Simplify each Hamiltonian group into a subcircuit, parallelizing across
    groups via the chosen backend (a single group always runs in-process)."""
    simp = partial(_process_same_weight_hamiltonian, parallel=parallel, patience=patience, optimize=optimize)
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
    grouping: str | None = None,
    parallel_search: bool = True,
    terminal='auto',
    order_method: str | None = None,
    backend: str = "sequential",
    search_patience: int | None = None,
    optimize: bool = True,
) -> QuantumCircuit:
    """Compile a Hamiltonian simulation circuit using the Phoenix framework.

    Args:
        hamiltonian: The Hamiltonian to simulate.
        grouping: Compilation strategy for Pauli terms:
            - ``None`` (default) or ``"holistic"``: the holistic engine —
              forward-frame two-qubit peeling over the whole table, grouping
              fully emergent, guaranteed termination, zero numeric
              hyperparameters (see ``primitive.holistic`` /
              docs/peel_forward_design.md);
            - ``"support"``: exact same-support grouping + legacy BSF greedy
              search (DAC'25 behavior, kept as the ablation baseline).
        optimize: Whether to apply Qiskit post-optimization.
        order_method: Ordering method for subcircuits in support mode
            (None defaults to 'tsp'). Ignored by the holistic engine.
        backend: Parallelization backend for support mode ("joblib",
            "concurrent.futures", or "sequential"). Ignored by the holistic engine.
        search_patience: Stall patience of the legacy BSF search safety net
            (support mode only). Default: max(16, 2·#active qubits).

    Returns:
        The compiled quantum circuit.
    """
    if grouping is None or grouping == "holistic":
        qc = holistic_compile(hamiltonian, terminal=terminal)
    elif grouping == "support":
        hams = hamiltonian.group_same_weights()
        circuits = _simplify_groups(hams, backend, parallel=parallel_search, patience=search_patience, optimize=optimize)
        qc = order_circuits(circuits, method=order_method or "tsp")
    else:
        raise ValueError(
            f"Unknown grouping mode: {grouping!r}; options: None/'holistic' (default), 'support'"
        )
    if optimize:
        qc = post_transpile(qc)

    return qc
