"""
Benchmarking utilities
"""

import qiskit
import pytket
import pytket.qasm
import pytket.passes
import numpy as np
import rustworkx as rx
from typing import Tuple, List
from qiskit.transpiler import CouplingMap, PassManager

import sys

sys.path.append('../..')

from phoenix import Circuit, Gate
from phoenix.utils import arch
from phoenix.models.hamiltonians import HamiltonianModel, console

from tetris.benchmark.mypauli import pauliString
from tetris.utils.hardware import pGraph
from phoenix.synthesis.grouping import group_paulis_and_coeffs
from phoenix.utils.display import print_circ_info

from rich.console import Console

console = Console()

Manhattan = CouplingMap(arch.read_device_topology('../manhattan.graphml').to_directed().edge_list())
Sycamore = CouplingMap(arch.read_device_topology('../sycamore.graphml').to_directed().edge_list())
All2all = CouplingMap(rx.generators.complete_graph(20).to_directed().edge_list())


def qiskit_O3_all2all(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    from itertools import combinations
    for q0, q1 in combinations(range(circ.num_qubits), 2):
        circ.cx(q0, q1)
        circ.cx(q0, q1)
    circ = qiskit.transpile(circ, optimization_level=3, basis_gates=['u1', 'u2', 'u3', 'cx'])
    return circ


def phoenix_pass(paulis: List[str], coeffs: List[float],
                 pre_gates: List[Gate] = None, post_gates: List[Gate] = None) -> qiskit.QuantumCircuit:
    # Phoenix's high-level optimization
    ham = HamiltonianModel(paulis, coeffs)
    # circ = ham.reconfigure_and_generate_circuit() # this is the old version
    circ = ham.phoenix_circuit()

    print_circ_info(circ)

    if pre_gates is not None:
        circ.prepend(*pre_gates)
    if post_gates is not None:
        circ.append(*post_gates)

    # logical optimization by Qiskit
    return qiskit_O3_all2all(circ.to_qiskit())


def paulihedral_pass(paulis: List[str], coeffs: List[float],
                     pre_gates: List[Gate] = None, post_gates: List[Gate] = None,
                     coupling_map: CouplingMap = All2all) -> qiskit.QuantumCircuit:
    from tetris.utils.parallel_bl import gate_count_oriented_scheduling
    from tetris.synthesis_SC import block_opt_SC

    a2 = gate_count_oriented_scheduling(constr_mypauli_blocks(paulis, coeffs))

    qc, total_swaps, total_cx = block_opt_SC(a2, graph=coupling_map_to_pGraph(coupling_map))

    circ = qiskit.QuantumCircuit(qc.num_qubits)
    pre_circ = Circuit(pre_gates).to_qiskit()
    post_circ = Circuit(post_gates).to_qiskit()
    circ.compose(pre_circ, inplace=True)
    circ.compose(qc, inplace=True)
    circ.compose(post_circ, inplace=True)

    if is_all2all_coupling_map(coupling_map):
        circ = qiskit_O3_all2all(circ)
    else:
        # circ = qiskit.transpile(circ, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)
        circ = qiskit.transpile(circ,
                                basis_gates=['u1', 'u2', 'u3', 'cx'],
                                coupling_map=coupling_map,
                                initial_layout=list(range(circ.num_qubits)),
                                layout_method='sabre',
                                optimization_level=3)

    console.print({
        'PH_swap_count': total_swaps,
        'PH_cx_count': total_cx,
        'CNOT': circ.num_nonlocal_gates(),
        'Single': circ.size() - circ.num_nonlocal_gates(),
        'Total': circ.size(),
        'Depth': circ.depth()})

    return circ


def tetris_pass(paulis: List[str], coeffs: List[float],
                pre_gates: List[Gate] = None, post_gates: List[Gate] = None,
                coupling_map: CouplingMap = All2all) -> qiskit.QuantumCircuit:
    from tetris.utils.synthesis_lookahead import synthesis_lookahead

    qc, metrics = synthesis_lookahead(constr_mypauli_blocks(paulis, coeffs),
                                      graph=coupling_map_to_pGraph(coupling_map),
                                      use_bridge=False,
                                      swap_coefficient=3, k=10)

    circ = qiskit.QuantumCircuit(qc.num_qubits)
    pre_circ = Circuit(pre_gates).to_qiskit()
    post_circ = Circuit(post_gates).to_qiskit()
    circ.compose(pre_circ, inplace=True)
    circ.compose(qc, inplace=True)
    circ.compose(post_circ, inplace=True)

    if is_all2all_coupling_map(coupling_map):
        circ = qiskit_O3_all2all(circ)
    else:
        circ = qiskit.transpile(circ,
                                basis_gates=['u1', 'u2', 'u3', 'cx'],
                                coupling_map=coupling_map,
                                initial_layout=list(range(circ.num_qubits)),
                                layout_method='sabre',
                                optimization_level=3)
        # circ = optimize_with_mapping(circ, coupling_map, tket_opt=False)

    metrics.update({'CNOT': circ.num_nonlocal_gates(),
                    'Single': circ.size() - circ.num_nonlocal_gates(),
                    'Total': circ.size(),
                    'Depth': circ.depth()})
    console.print(metrics)

    return circ


def pauliopt_pass(paulis: List[str], coeffs: List[float],
                  pre_gates: List[Gate] = None, post_gates: List[Gate] = None,
                  coupling_map: CouplingMap = All2all) -> qiskit.QuantumCircuit:
    ...


def tket_pass(circ: pytket.Circuit) -> pytket.Circuit:
    from phoenix.synthesis.utils import unroll_u3

    # unroll U3
    circ = unroll_u3(Circuit.from_tket(circ)).to_tket()

    # adaptive PauliSimp
    circ_tmp = circ.copy()
    best_depth_2q = circ.depth_2q()
    best_num_2q_gates = circ.n_2qb_gates()
    while True:
        pytket.passes.PauliSimp().apply(circ_tmp)
        if best_depth_2q > circ_tmp.depth_2q() and best_num_2q_gates > circ_tmp.n_2qb_gates():
            best_depth_2q = circ_tmp.depth_2q()
            best_num_2q_gates = circ_tmp.n_2qb_gates()
            circ = circ_tmp.copy()
        else:
            break

    # full optimization
    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)

    return circ


def qiskit_to_unitary(circ: qiskit.QuantumCircuit) -> np.ndarray:
    from qiskit.quantum_info import Operator
    return Operator(circ.reverse_bits()).to_matrix()


def tket_to_qiskit(circ: pytket.Circuit) -> qiskit.QuantumCircuit:
    return qiskit.QuantumCircuit.from_qasm_str(pytket.qasm.circuit_to_qasm_str(circ))


def qiskit_to_tket(circ: qiskit.QuantumCircuit) -> pytket.Circuit:
    return pytket.qasm.circuit_from_qasm_str(qiskit.qasm2.dumps(circ))


def sabre_map(circ: qiskit.QuantumCircuit, coupling_map: CouplingMap) -> Tuple[
    qiskit.QuantumCircuit, List[int], List[int]]:
    """
    Mapping logical circuits on physical qubits by means of SabreLayout pass in Qiskit.

    Args:
        circ: Input logical quantum circuit
        coupling_map: Physical qubit connectivity graph

    Returns:
        Mapped circuit, initial mapping (physical qubit indices), final mapping (physical qubit indices)
    """
    from qiskit.transpiler import passes

    pm = PassManager([passes.SabreLayout(coupling_map)])
    circ = pm.run(circ)
    # init_mapping_inv = {i: j for i, j in zip(circ.layout.initial_index_layout(), range(circ.num_qubits))}
    # final_mapping_inv = {i: j for i, j in zip(circ.layout.final_index_layout(), range(circ.num_qubits))}
    # init_mapping = {j: i for i, j in init_mapping_inv.items()}
    # final_mapping = {j: i for i, j in final_mapping_inv.items()}
    # circ = Circuit.from_qiskit(circ).rewire(init_mapping_inv)
    # return circ, init_mapping, final_mapping
    return circ, circ.layout.initial_index_layout(), circ.layout.final_index_layout()


def pre_mapping_optimize(circ: pytket.Circuit) -> pytket.Circuit:
    """Pre-mapping optimization on logical circuits by means of TKet's pass"""
    circ = circ.copy()
    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
    return circ


def post_mapping_optimize(circ: pytket.Circuit) -> pytket.Circuit:
    """Post-mapping optimization on physical circuits by means of TKet's pass"""
    circ = circ.copy()
    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
    return circ


def optimize_with_mapping(circ: qiskit.QuantumCircuit, coupling_map: CouplingMap,
                          tket_opt: bool = False) -> qiskit.QuantumCircuit:
    """By default, we use Qiskit's O3 compiler appended by a TKet's topology-preserved optimization pass"""
    circ = qiskit.transpile(circ, optimization_level=3,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=coupling_map, layout_method='sabre')

    if tket_opt:
        circ = qiskit_to_tket(circ)
        pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
        circ = tket_to_qiskit(circ)

    return circ


def coupling_map_to_pGraph(coupling_map: CouplingMap) -> pGraph:
    G = rx.adjacency_matrix(coupling_map.graph)
    C = rx.floyd_warshall_numpy(coupling_map.graph)
    return pGraph(G, C)


def constr_mypauli_blocks(paulis, coeffs) -> List[List[pauliString]]:
    groups = group_paulis_and_coeffs(paulis, coeffs)
    mypauli_blocks = []
    for paulis, coeffs in groups.values():
        mypauli_blocks.append([])
        for p, c in zip(paulis, coeffs):
            mypauli_blocks[-1].append(pauliString(p, c))
    return mypauli_blocks


def is_all2all_coupling_map(coupling_map: CouplingMap) -> bool:
    # directed coupling map
    if coupling_map.size() * (coupling_map.size() - 1) == len(coupling_map.get_edges()):
        return True
    return False

# TODO: verify utils
