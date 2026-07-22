from __future__ import annotations

import warnings
import numpy as np
import qiskit
import rustworkx as rx
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager, passes

warnings.filterwarnings("ignore")


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


#############################################################################
# Utilities for benchmarking convenience
#############################################################################


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
    if not paulis:
        raise ValueError("paulis list cannot be empty")

    n_rows = len(paulis)
    n_cols = len(paulis[0])

    if any(len(p) != n_cols for p in paulis):
        raise ValueError("All Pauli strings must have the same length")

    if little_endian:
        paulis = [p[::-1] for p in paulis]

    color_map = {
        "X": np.array([0.89, 0.58, 0.58]),  # soft red
        "Y": np.array([0.55, 0.75, 0.55]),  # soft green
        "Z": np.array([0.55, 0.67, 0.87]),  # soft blue
        "I": np.array([0.78, 0.78, 0.78]),  # soft gray
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

    if hide_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    step = max(1, n_rows // 12)
    ax.set_yticks(np.arange(0, n_rows, step))
    ax.set_yticklabels(np.arange(0, n_rows, step))

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=350)
    plt.show()


def synth_to_clifford_t(qc: QuantumCircuit, epsilon=1e-10) -> QuantumCircuit:
    """Clifford+T synthesis with a controllable approximation error ``epsilon``."""
    from qiskit.synthesis import gridsynth_rz, gridsynth_unitary
    # base = qiskit.transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
    base = qiskit.transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
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
            approx = gridsynth_rz(op.params[0], epsilon)
            out.compose(approx, [idx[0]], inplace=True)
        elif op.name == 'u' or op.name == 'u3':
            approx = gridsynth_unitary(Operator(op).data, epsilon)
            out.compose(approx, [idx[0]], inplace=True)
        else:
            raise ValueError(f"unexpected {op.num_qubits}-qubit gate {op.name!r}")
    return out

