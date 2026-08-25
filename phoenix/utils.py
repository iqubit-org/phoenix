from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

import cirq
import numpy as np
import qiskit
import rustworkx as rx
from cirq.contrib.svg import SVGCircuit
from prettytable import PrettyTable
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Operator
from qiskit.transpiler import CouplingMap, PassManager, passes
from scipy import linalg

warnings.filterwarnings("ignore")

# Below this many distinct Rz angles, spawning worker processes costs more than it saves.
_PARALLEL_GRIDSYNTH_MIN_ANGLES = 256


def remove_1q_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """Remove all single-qubit gates from a QuantumCircuit instance."""
    qc_new = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_new.name = qc.name
    qc_new.global_phase = qc.global_phase

    for instr in qc.data:
        if instr.operation.num_qubits != 1:
            qc_new.append(instr.operation, instr.qubits, instr.clbits)

    return qc_new


def remove_1q_fixed_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """Remove all single-qubit gates from a QuantumCircuit instance."""
    qc_new = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_new.name = qc.name
    qc_new.global_phase = qc.global_phase

    for instr in qc.data:
        if instr.operation.num_qubits != 1 or instr.operation.params:
            qc_new.append(instr.operation, instr.qubits, instr.clbits)

    return qc_new


def infidelity(u: np.ndarray, v: np.ndarray) -> float:
    """Infidelity between two matrices"""
    if u.shape[0] > 2**10:
        raise ValueError(
            "Infidelity calculation might be expensive for large matrices. Consider using a more efficient method or reducing the size."
        )

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")
    d = u.shape[0]
    return 1 - np.abs(np.trace(u.conj().T @ v)) / d


def pauli_exp_synth_cost(paulis: list[str], coeffs: list[float] = None) -> int:
    """Return the rank-based CNOT cost of a two-qubit Pauli exponential.

    The input represents ``exp(-i * sum_j coeffs[j] * paulis[j])``.  Every
    Pauli label must contain exactly two non-identity axes from ``X``, ``Y``,
    and ``Z``.  The labels are assembled into the 3-by-3 interaction matrix
    ``J`` whose row and column order is ``X, Y, Z``.

    If ``coeffs`` is omitted, the coefficients are treated as independent
    generic variables and the structural rank of ``J`` is obtained from a
    maximum matching of its support graph.  If coefficients are supplied,
    repeated Pauli terms are summed and the numerical rank is used instead.

    The returned generic synthesis costs are 0 CNOTs for a zero generator,
    2 CNOTs for ranks one or two, and 3 CNOTs for full rank.  This structural
    criterion does not detect isolated special angles at which a nonzero
    interaction may require fewer CNOTs.
    """
    axis_to_index = {"X": 0, "Y": 1, "Z": 2}
    normalized_paulis = []
    for pauli in paulis:
        if not isinstance(pauli, str):
            raise ValueError(f"Invalid 2Q Pauli label: {pauli!r}")
        label = pauli.upper()
        if len(label) != 2 or any(axis not in axis_to_index for axis in label):
            raise ValueError(f"Invalid 2Q Pauli label: {pauli!r}")
        normalized_paulis.append(label)

    if coeffs is None:
        support = {
            (axis_to_index[label[0]], axis_to_index[label[1]]) for label in normalized_paulis
        }
        matched_left = [-1, -1, -1]

        def augment(left: int, visited_right: set[int]) -> bool:
            for right in range(3):
                if (left, right) not in support or right in visited_right:
                    continue
                visited_right.add(right)
                if matched_left[right] == -1 or augment(matched_left[right], visited_right):
                    matched_left[right] = left
                    return True
            return False

        rank = sum(augment(left, set()) for left in range(3))
    else:
        if len(coeffs) != len(normalized_paulis):
            raise ValueError("The number of coefficients must match the number of Pauli labels.")
        try:
            coefficient_array = np.asarray(coeffs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Coefficients must be finite real numbers.") from exc
        if coefficient_array.ndim != 1 or not np.all(np.isfinite(coefficient_array)):
            raise ValueError("Coefficients must be finite real numbers.")

        interaction_matrix = np.zeros((3, 3), dtype=float)
        for label, coefficient in zip(normalized_paulis, coefficient_array):
            interaction_matrix[axis_to_index[label[0]], axis_to_index[label[1]]] += coefficient
        rank = int(np.linalg.matrix_rank(interaction_matrix))

    if rank == 0:
        return 0
    if rank <= 2:
        return 2
    return 3


def print_circ_info(circ: QuantumCircuit, title=None):
    """Get information of a quantum circuit from its qasm file."""
    num_qubits = circ.num_qubits
    num_gates = circ.size()
    num_nonlocal_gates = circ.num_nonlocal_gates()
    depth = circ.depth()
    depth_nonlocal = circ.depth(lambda instr: instr.operation.num_qubits > 1)

    # use prettytable
    table = PrettyTable()
    if title:
        table.title = title
    table.field_names = ["num_qubits", "num_gates", "num_2q_gates", "depth", "depth_2q"]
    table.add_row([str(num_qubits), str(num_gates), str(num_nonlocal_gates), str(depth), str(depth_nonlocal)])
    print(table)


def is_all2all_coupling_map(coupling_map: CouplingMap) -> bool:
    # ! coupling_map.graph is a directed coupling map
    if coupling_map.size() * (coupling_map.size() - 1) == len(coupling_map.get_edges()):
        return True
    return False


# Basis used to unroll custom Clifford gate definitions before 2q-block resynthesis.
_UNROLL_BASIS_GATES = ["cx", "h", "s", "sdg", "rzx", "rxx", "ryy", "rzz"]
# Final synthesis basis (cx + universal 1q) so every compiled circuit is counted alike.
_SYNTHESIS_BASIS_GATES = ["cx", "u"]
# Max width of a Clifford sub-block handed to Qiskit's optimal (Bravyi-Maslov) synthesis.
_OPTIMAL_CLIFFORD_MAX_BLOCK_WIDTH = 3


def post_transpile(qc: QuantumCircuit) -> QuantumCircuit:
    """Common circuit-level post-optimization.

    Applied uniformly to every compiler's output in the benchmark harness so
    comparisons are apples-to-apples. The pipeline has three goals:

    1. cancel Phoenix-native Clifford scaffolding as much as possible while the
       custom gate structure is still visible (no-op on non-Phoenix circuits);
    2. collect short runs of Clifford gates and resynthesize each block with
       Qiskit's optimal (Bravyi-Maslov) Clifford synthesis;
    3. unroll the remaining custom gates and let Qiskit resynthesize consecutive
       2-qubit blocks (KAK), so patterns such as ``Rzx -> Ryy -> Rzz`` compress.

    On baseline circuits (no Phoenix custom gates) the Phoenix-specific steps
    are inert; the generic steps (2q-block KAK resynthesis, commutative and
    optimal-Clifford resynthesis) still apply, giving every compiler the same
    O3-grade circuit-level optimization.
    """
    from itertools import product

    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

    from .basics import CNOTEquivCliffordGate

    inverse_list = [CNOTEquivCliffordGate(p0, p1) for p0, p1 in product(["x", "y", "z"], repeat=2)]

    pm = PassManager()
    pm.append(passes.InverseCancellation(inverse_list))
    pm.append(passes.CommutativeInverseCancellation(matrix_based=True))
    pm.append(passes.Optimize1qGatesDecomposition())
    pm.append(passes.CommutativeCancellation())
    
    # Collect Clifford blocks up to width 3 and resynthesize them optimally with Bravyi-Maslov synthesis.
    pm.append(passes.CollectCliffords(matrix_based=True, min_block_size=2, max_block_width=_OPTIMAL_CLIFFORD_MAX_BLOCK_WIDTH))
    pm.append(passes.HighLevelSynthesis(hls_config=passes.HLSConfig(use_default_on_unspecified=False, clifford=["bm"])))

    # unroll CNOTEquivCliffordGate if existing
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

    return pm.run(qc)


def compile_by_qiskit(
    paulis: list[str], coeffs: list[float], coupling_map: CouplingMap = None,
    little_endian: bool = True, optimize: bool = True
) -> QuantumCircuit:
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    if not little_endian:
        # ! Qiskit uses little-endian convention for Pauli strings by default
        paulis = [p[::-1] for p in paulis]

    n = len(paulis[0])  # number of qubits
    op = SparsePauliOp(paulis, coeffs)
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(op), range(n))

    hls_config = passes.HLSConfig(
        PauliEvolution=[
            (
                "rustiq",
                {
                    "optimize_count": True,  # Optimize the two-qubit gate count
                    "preserve_order": False,  # Allow reordering of Pauli terms
                    "upto_phase": True,  # Allow a global phase difference
                    "upto_clifford": False,  # Resynthesize the final Clifford operator
                    "resynth_clifford_method": 1,  # Use Qiskit's greedy Clifford resynthesis
                },
            )
        ]
    )
    hls_pass = passes.HighLevelSynthesis(hls_config=hls_config)
    qc = hls_pass(qc)
    if optimize:
        qc = post_transpile(qc)

    if not(coupling_map is None or is_all2all_coupling_map(coupling_map)):
        qc = optimize_with_mapping(qc, coupling_map)

    return qc


def optimize_with_mapping(circ: QuantumCircuit, coupling_map: CouplingMap) -> QuantumCircuit:
    """By default, we use Qiskit's O3 compiler to performa hardware-aware tranpilation and optimization"""
    circ = qiskit.transpile(
        circ,
        optimization_level=3,
        basis_gates=["u", "cx"],
        coupling_map=coupling_map,
        seed_transpiler=1997
    )

    return circ


def compile_by_tket(
    paulis: list[str], coeffs: list[float], greedy: bool = True, little_endian: bool = True, optimize: bool = True
) -> QuantumCircuit:
    import pytket.passes

    if little_endian:
        # ! TKet uses large-endian convention for Pauli strings by default
        paulis = [p[::-1] for p in paulis]

    circ = constr_tket_circuit(paulis, coeffs)
    if greedy:
        pytket.passes.GreedyPauliSimp().apply(circ)
    else:
        pytket.passes.PauliSimp().apply(circ)
    if circ.has_implicit_wireswaps:
        circ.replace_implicit_wire_swaps()

    qc = tket_to_qiskit(circ)
    if optimize:
        # The extra {u, cx} transpile pass is necessary to avoid destructing TKET->Qiskit semantics in post_transpile
        qc = qiskit.transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        qc = post_transpile(qc)
    return qc


def constr_tket_circuit(paulis: list[str], coeffs: list[float]):
    import pytket
    from pytket.circuit import PauliExpBox
    from pytket.pauli import Pauli

    pauli_str_map = {
        "X": Pauli.X,
        "Y": Pauli.Y,
        "Z": Pauli.Z,
    }

    def get_qubits_acted(pauli: str) -> list[int]:
        """Get the qubits acted by a Pauli string."""
        return [i for i, p in enumerate(pauli) if p != "I"]

    circ = pytket.Circuit(len(paulis[0]))

    for pauli, coeff in zip(paulis, coeffs):
        qubits = get_qubits_acted(pauli)
        if len(qubits) == 0:
            continue
        pauli_box = PauliExpBox([pauli_str_map[pauli[q]] for q in qubits], t=coeff * 2 / np.pi)
        circ.add_gate(pauli_box, qubits)

    return circ


def assert_equiv_circuits(qc1: QuantumCircuit, qc2: QuantumCircuit):
    assert Operator(qc1).equiv(Operator(qc2)), "Circuits are not equivalent!"


def tket_to_qiskit(circ) -> QuantumCircuit:
    import pytket.qasm

    return QuantumCircuit.from_qasm_str(pytket.qasm.circuit_to_qasm_str(circ))


def qiskit_to_tket(circ: QuantumCircuit):
    import pytket.qasm

    return pytket.qasm.circuit_from_qasm_str(qiskit.qasm2.dumps(circ))


def gene_all2all_coupling_map(size) -> CouplingMap:
    return CouplingMap.from_full(size)


def gene_chain_coupling_map(size) -> CouplingMap:
    return CouplingMap.from_line(size)


def gene_square_coupling_map(size) -> CouplingMap:
    n = int(np.sqrt(size))
    m = int(np.ceil(size / n))
    g = rx.generators.grid_graph(n, m).subgraph(range(size)).to_directed()
    return CouplingMap(g.edge_list())


def gene_hhex_coupling_map(size) -> CouplingMap:
    if size <= 0:
        return CouplingMap([])

    distance = max(1, int(np.ceil((1 + np.sqrt(10 * size + 6)) / 5)))
    if distance % 2 == 0:
        distance += 1

    coupling_map = CouplingMap.from_heavy_hex(distance)
    if coupling_map.size() == size:
        return coupling_map

    nodes = []
    seen = {0}
    queue = [0]
    queue_index = 0
    while queue_index < len(queue) and len(nodes) < size:
        node = queue[queue_index]
        queue_index += 1
        nodes.append(node)
        for neighbor in coupling_map.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    subgraph = coupling_map.graph.subgraph(nodes)
    coupling_map = CouplingMap(subgraph.edge_list())
    for qubit in range(coupling_map.size(), size):
        coupling_map.add_physical_qubit(qubit)
    return coupling_map


def plot_pauli_strings(paulis, *, little_endian=False, figsize=(5, 10), hide_axis=False,
                       title="Pauli Strings", output_filename=None):
    """
    Visualize a set of equal-length Pauli strings.

    X-axis: qubit index
    Y-axis: Pauli string index
    R/G/B/Gray color: X/Y/Z/I
    """
    import matplotlib.pyplot as plt

    if not paulis:
        raise ValueError("paulis list cannot be empty")

    n_rows = len(paulis)
    n_cols = len(paulis[0])

    if any(len(p) != n_cols for p in paulis):
        raise ValueError("All Pauli strings must have the same length")

    if little_endian:
        paulis = [p[::-1] for p in paulis]

    color_map = {
        "X": np.array([0.95, 0.70, 0.70]),  # soft red
        "Y": np.array([0.70, 0.88, 0.70]),  # soft green
        "Z": np.array([0.68, 0.80, 0.95]),  # soft blue
        "I": np.array([0.93, 0.93, 0.93]),  # soft gray
    }

    img = np.zeros((n_rows, n_cols, 3), dtype=float)

    for i, p in enumerate(paulis):
        for j, ch in enumerate(p):
            if ch not in color_map:
                raise ValueError(f"Unsupported Pauli character: {ch}")
            img[i, j] = color_map[ch]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="upper")

    if not hide_axis:
        ax.set_xlabel("Qubit index")
        ax.set_ylabel("Pauli string index")
    ax.set_title(title)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"q{i}" for i in range(n_cols)])

    # if hide_axis:
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    step = max(1, n_rows // 12)
    yticks = np.arange(0, n_rows, step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$P_{{{i + 1}}}$" for i in yticks])

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=350)
    plt.show()


def plot_pauli_exponential_circuit(paulis, coeffs=None) -> SVGCircuit:

    class MultiPauliRotation(cirq.Gate):
        """Use universal controlled gates to represent generic 2Q Clifford gates"""

        def __init__(self, pauli: str, time: float):
            super().__init__()
            self.pauli = pauli
            self.time = time

        def _num_qubits_(self):
            return len(self.pauli) - self.pauli.count("I")

        def _unitary_(self):
            import qiskit.quantum_info as qi

            return linalg.expm(-1j * qi.SparsePauliOp([self.pauli], [self.time]).to_matrix())

        def _circuit_diagram_info_(self, args):
            if self.time is None:
                if self._num_qubits_() == 1:
                    return [f"R{p.lower()}" for p in self.pauli if p != "I"]
                return [p for p in self.pauli if p != "I"]
            angle_str = f"{self.time * 2:.1f}"
            return [p + f"({angle_str})" for p in self.pauli if p != "I"]

    if coeffs is None:
        coeffs = [None] * len(paulis)
    num_qubits = len(paulis[0])
    circ = cirq.Circuit()
    qubits = cirq.LineQubit.range(num_qubits)
    for pauli, coeff in zip(paulis, coeffs):
        qubits_acted = [qubits[i] for i in range(num_qubits) if pauli[i] != "I"]
        circ.append(MultiPauliRotation(pauli, coeff).on(*qubits_acted))

    return SVGCircuit(circ)



def peel_front_cliffords(circuit: QuantumCircuit) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Split off the maximal Clifford subcircuit reachable from ``circuit``'s inputs.

    At every step, all operations with no predecessor operation are examined.
    Clifford operations are removed together and the process repeats, allowing
    another Clifford to become peelable.  A non-Clifford operation, measurement,
    directive, or operation with classical bits remains in the remainder and
    blocks only the operations that depend on it.

    Returns:
        ``(front_cliffords, remainder)``.  The two circuits use the original
        registers and satisfy ``front_cliffords.compose(remainder) == circuit``
        up to Qiskit's usual circuit-equivalence convention.  The input circuit
        is never modified.
    """

    dag = circuit_to_dag(circuit)
    front_layers: list[list[DAGOpNode]] = []
    while True:
        front_cliffords = [node for node in dag.front_layer() if _is_clifford_node(node)]
        if not front_cliffords:
            break
        front_layers.append(front_cliffords)
        for node in front_cliffords:
            dag.remove_op_node(node)

    return _circuit_from_layers(circuit, front_layers), dag_to_circuit(dag)


def peel_tail_cliffords(circuit: QuantumCircuit) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Split off the maximal Clifford subcircuit reachable from ``circuit``'s outputs.

    This is the output-side counterpart of :func:`peel_front_cliffords`.  At
    each step it removes every terminal Clifford operation, then continues with
    operations that become terminal.  It therefore peels a maximal trailing
    Clifford subcircuit even when the circuit's instruction list interleaves
    independent qubits.

    Returns:
        ``(remainder, tail_cliffords)``.  The two circuits use the original
        registers and satisfy ``remainder.compose(tail_cliffords) == circuit``
        up to Qiskit's usual circuit-equivalence convention.  The input circuit
        is never modified.
    """

    dag = circuit_to_dag(circuit)
    tail_layers: list[list[DAGOpNode]] = []
    while True:
        tail_cliffords = [node for node in _terminal_op_nodes(dag) if _is_clifford_node(node)]
        if not tail_cliffords:
            break
        tail_layers.append(tail_cliffords)
        for node in tail_cliffords:
            dag.remove_op_node(node)

    return dag_to_circuit(dag), _circuit_from_layers(circuit, reversed(tail_layers))


def _terminal_op_nodes(dag: DAGCircuit) -> list[DAGOpNode]:
    """Return operations with no successor operation in the circuit DAG."""

    return [
        node
        for node in dag.op_nodes()
        if not any(isinstance(successor, DAGOpNode) for successor in dag.successors(node))
    ]


def _circuit_from_layers(circuit: QuantumCircuit, layers: Iterable[Iterable[DAGOpNode]]) -> QuantumCircuit:
    """Build a zero-phase subcircuit from dependency-ordered DAG node layers."""

    subcircuit = circuit.copy_empty_like()
    subcircuit.global_phase = 0
    for layer in layers:
        for node in layer:
            subcircuit.append(node.op, node.qargs, node.cargs)
    return subcircuit


def _is_clifford_node(node: DAGOpNode) -> bool:
    """Whether ``node`` is a unitary Clifford operation with no classical I/O."""

    operation = node.op
    has_classical_io = bool(operation.num_clbits or node.cargs)
    if not isinstance(operation, Gate) or operation.num_qubits == 0 or has_classical_io:
        return False

    try:
        Clifford(operation)
    except (QiskitError, TypeError, ValueError):
        return False
    return True


def _gridsynth_rz_batch(args: tuple[list[float], float]) -> list[tuple[list[str], float]]:
    """Worker entry point: Ross-Selinger synthesis for a batch of Rz angles.

    Returns a compact ``(gate names, global phase)`` representation instead of
    ``QuantumCircuit`` objects, which is ~20x cheaper to pickle back to the parent.
    """
    from qiskit.synthesis import gridsynth_rz

    angles, epsilon = args
    results = []
    for theta in angles:
        approx = gridsynth_rz(theta, epsilon)
        results.append(([inst.operation.name for inst in approx.data], float(approx.global_phase)))
    return results


def _rebuild_1q_circuit(names: list[str], global_phase: float) -> QuantumCircuit:
    """Rebuild a 1-qubit circuit from the compact representation of ``_gridsynth_rz_batch``."""
    circ = QuantumCircuit(1, global_phase=global_phase)
    for name in names:
        getattr(circ, name)(0)
    return circ


def _synth_rz_angles(angles: list[float], epsilon: float, num_workers: int | None) -> dict[str, QuantumCircuit]:
    """Synthesize every *distinct* angle in ``angles``, in parallel when it pays off.

    ``gridsynth_rz`` is a Rust extension that holds the GIL, so threads give no
    speedup -- processes are required.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from qiskit.synthesis import gridsynth_rz

    # Deduplicate first: UCCSD circuits repeat many angles.
    uniq: dict[str, float] = {}
    for theta in angles:
        uniq.setdefault(theta.hex(), theta)
    keys = list(uniq)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 16)
    # Process startup (spawn re-imports qiskit) only amortizes over enough angles,
    # and nested pools are not allowed inside a worker process.
    if num_workers <= 1 or len(keys) < _PARALLEL_GRIDSYNTH_MIN_ANGLES or mp.parent_process() is not None:
        return {k: gridsynth_rz(uniq[k], epsilon) for k in keys}

    # Several chunks per worker so that uneven per-angle cost still balances out.
    chunk = max(1, len(keys) // (num_workers * 4))
    batches = [[uniq[k] for k in keys[i:i + chunk]] for i in range(0, len(keys), chunk)]

    # "fork" rather than the macOS/Windows default "spawn": spawn (and forkserver)
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # Windows
        ctx = mp.get_context("spawn")

    try:
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            chunked = list(pool.map(_gridsynth_rz_batch, [(b, epsilon) for b in batches]))
    except Exception as exc:  # pool could not start (frozen app, restricted sandbox, ...)
        warnings.warn(f"parallel gridsynth unavailable ({exc!r}); falling back to serial synthesis", stacklevel=2)
        return {k: gridsynth_rz(uniq[k], epsilon) for k in keys}

    flat = [item for batch in chunked for item in batch]
    return {k: _rebuild_1q_circuit(names, phase) for k, (names, phase) in zip(keys, flat)}


def synth_to_clifford_t(qc: QuantumCircuit, epsilon=1e-10, num_workers: int | None = None) -> QuantumCircuit:
    """Clifford+T synthesis with a controllable approximation error ``epsilon``.

    Args:
        qc: circuit to synthesize.
        epsilon: allowed approximation error per rotation.
        num_workers: processes used for the Ross-Selinger synthesis of the Rz angles,
            which dominates the runtime. ``None`` picks a sensible default from the CPU
            count, ``1`` forces the serial path.

    Note:
        ``gridsynth_rz`` results depend on per-process internal state, so the exact
        circuit is not reproducible across runs -- this is true of the serial path as
        well. T-count varies by ~0.003% with ``num_workers``; pin it if a benchmark
        table needs byte-identical numbers.
    """
    base = qiskit.transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)

    rz_angles = [float(inst.operation.params[0]) for inst in base.data if inst.operation.name == 'rz']
    rz_cache = _synth_rz_angles(rz_angles, float(epsilon), num_workers)

    out = QuantumCircuit(*base.qregs)
    for inst in base.data:
        op, qubits = inst.operation, inst.qubits
        idx = [base.find_bit(q).index for q in qubits]
        if op.name == "cx":
            out.cx(*idx)
        elif op.name == 'x':
            out.x(*idx)
        elif op.name == 'sx':
            out.sx(*idx)
        elif op.name == 'rz':
            out.compose(rz_cache[float(op.params[0]).hex()], [idx[0]], inplace=True)
        else:
            raise ValueError(f"unexpected {op.num_qubits}-qubit gate {op.name!r}")
    return out
