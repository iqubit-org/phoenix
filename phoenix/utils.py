from __future__ import annotations

import numpy as np
import qiskit
import rustworkx as rx
from prettytable import PrettyTable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap


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


def qiskit_O3_all2all(circ: QuantumCircuit) -> QuantumCircuit:
    from itertools import combinations

    for q0, q1 in combinations(range(circ.num_qubits), 2):
        circ.cx(q0, q1)
        circ.cx(q0, q1)
    circ = qiskit.transpile(circ, optimization_level=3, basis_gates=["u1", "u2", "u3", "cx"])
    return circ


def qiskit_pass(
    paulis: list[str], coeffs: list[float], coupling_map: CouplingMap = None, with_O3: bool = False
) -> QuantumCircuit:
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.passes import HighLevelSynthesis, HLSConfig

    n = len(paulis[0])  # number of qubits
    op = SparsePauliOp(paulis, coeffs)
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(op), range(n))

    hls_config = HLSConfig(
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
    hls_pass = HighLevelSynthesis(hls_config=hls_config)
    qc = hls_pass(qc)

    if coupling_map is None or is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = qiskit_O3_all2all(qc)
    else:
        qc = optimize_with_mapping(qc, coupling_map)

    return qc


def optimize_with_mapping(circ: QuantumCircuit, coupling_map: CouplingMap, tket_opt: bool = False) -> QuantumCircuit:
    """By default, we use Qiskit's O3 compiler to performa hardware-aware tranpilation and optimization"""
    circ = qiskit.transpile(
        circ,
        optimization_level=3,
        basis_gates=["u1", "u2", "u3", "cx"],
        coupling_map=coupling_map,
        layout_method="sabre",
    )

    # whether to further perform a TKet's topology-preserved optimization pass
    if tket_opt:
        import pytket.passes

        circ = qiskit_to_tket(circ)
        pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
        circ = tket_to_qiskit(circ)

    return circ


def tket_pass(
    paulis: list[str], coeffs: list[float], greedy: bool = True, little_endian: bool = False
) -> QuantumCircuit:
    import pytket.passes

    if little_endian:
        paulis = [p[::-1] for p in paulis]
    circ = constr_tket_circuit(paulis, coeffs)
    if greedy:
        pytket.passes.GreedyPauliSimp().apply(circ)
    else:
        pytket.passes.PauliSimp().apply(circ)
    # GreedyPauliSimp introduces implicit qubit permutations (no explicit SWAP gates).
    # These are lost when converting to QASM/Qiskit, causing incorrect unitaries.
    # Materialize the permutation as explicit SWAP gates before further optimization.
    if circ.has_implicit_wireswaps:
        circ.replace_implicit_wire_swaps()

    # pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)

    qc = tket_to_qiskit(circ)
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


def plot_pauli_strings(paulis, *, little_endian=False, figsize=(5, 10), title="Pauli Strings", output_filename=None):
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

    ax.set_xlabel("Qubit index")
    ax.set_ylabel("Pauli string index")
    ax.set_title(title)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"q{i}" for i in range(n_cols)])

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
