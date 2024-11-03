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
from phoenix.transforms.pauli_pass import group_paulis_and_coeffs

from rich.console import Console

console = Console()

Manhattan = CouplingMap(arch.read_device_topology('../manhattan.graphml').to_directed().edge_list())
Sycamore = CouplingMap(arch.read_device_topology('../sycamore.graphml').to_directed().edge_list())
All2all = CouplingMap(rx.generators.complete_graph(50).to_directed().edge_list())

"""
def phoenix_pass(ham: HamiltonianModel, device: rx.rustworkx = None) -> Circuit:
    # topology-agnostic synthesis
    circ = ham.reconfigure_and_generate_circuit()
    if device is None:
        return circ

    # hardware mapping
    circ, _, _ = transforms.circuit_pass.sabre_by_qiskit(circ, device)
    circ = transforms.circuit_pass.phys_circ_opt_by_qiskit(circ)
    return circ
"""


def qiskit_O3_all2all(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    from itertools import combinations
    for q0, q1 in combinations(range(circ.num_qubits), 2):
        circ.cx(q0, q1)
        circ.cx(q0, q1)
    circ = qiskit.transpile(circ, optimization_level=3, basis_gates=['u1', 'u2', 'u3', 'cx'])
    return circ


def phoenix_pass(paulis: List[str], coeffs: List[float],
                 pre_gates: List[Gate] = None, post_gates: List[Gate] = None) -> pytket.Circuit:
    # Phoenix's high-level optimization
    ham = HamiltonianModel(paulis, coeffs)
    circ = ham.reconfigure_and_generate_circuit()
    if pre_gates is not None:
        circ.prepend(*pre_gates)
    if post_gates is not None:
        circ.append(*post_gates)

    # logical optimization by TKet
    circ = circ.to_tket()
    pytket.passes.FullPeepholeOptimise().apply(circ)  # TODO: it might lead to Fault results when dumping to OpenQASM
    return circ


def paulihedral_pass(paulis: List[str], coeffs: List[float],
                     pre_gates: List[Gate] = None, post_gates: List[Gate] = None,
                     coupling_map: CouplingMap = All2all) -> qiskit.QuantumCircuit:
    from tetris.utils.parallel_bl import gate_count_oriented_scheduling
    from tetris.synthesis_SC import block_opt_SC

    oplist = constr_mypauli_blocks(paulis, coeffs)
    n = len(oplist[0][0])
    if coupling_map.size() * (coupling_map.size() - 1) == len(coupling_map.get_edges()) and coupling_map.size() < n:
        # coupling_map is All2all but its size needs to be expanded
        console.print('extending All2all topology')
        coupling_map = CouplingMap(rx.generators.complete_graph(n).to_directed().edge_list())

    a2 = gate_count_oriented_scheduling(oplist)

    console.print(a2)

    qc, total_swaps, total_cx = block_opt_SC(a2, graph=coupling_map_to_pGraph(coupling_map))

    circ = qiskit.QuantumCircuit(qc.num_qubits)
    pre_circ = Circuit(pre_gates).to_qiskit()
    post_circ = Circuit(post_gates).to_qiskit()
    circ.compose(pre_circ, inplace=True)
    circ.compose(qc, inplace=True)
    circ.compose(post_circ, inplace=True)

    circ = qiskit.transpile(circ,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=coupling_map,
                            initial_layout=list(range(circ.num_qubits)),
                            layout_method='sabre',
                            optimization_level=3)
    
    console.print({
        'n_qubits': n,
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

    oplist = constr_mypauli_blocks(paulis, coeffs)
    n = len(oplist[0][0])
    if coupling_map.size() * (coupling_map.size() - 1) == len(coupling_map.get_edges()) and coupling_map.size() < n:
        # coupling_map is All2all but its size needs to be expanded
        console.print('extending All2all topology')
        coupling_map = CouplingMap(rx.generators.complete_graph(n).to_directed().edge_list())

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

    circ = qiskit.transpile(circ,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=coupling_map,
                            initial_layout=list(range(circ.num_qubits)),
                            layout_method='sabre',
                            optimization_level=3)

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
    from phoenix.transforms.circuit_pass import unroll_u3

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
    pytket.passes.FullPeepholeOptimise().apply(circ)

    return circ


def tket_to_qiskit(circ: pytket.Circuit) -> qiskit.QuantumCircuit:
    return qiskit.QuantumCircuit.from_qasm_str(pytket.qasm.circuit_to_qasm_str(circ))


def qiskit_to_tket(circ: qiskit.QuantumCircuit) -> pytket.Circuit:
    return pytket.qasm.circuit_from_qasm_str(circ.qasm())


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


def optimize_with_mapping(circ: qiskit.QuantumCircuit, coupling_map: CouplingMap) -> qiskit.QuantumCircuit:
    circ = qiskit.transpile(circ, optimization_level=3,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=coupling_map, layout_method='sabre')

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

# TODO: verify utils
