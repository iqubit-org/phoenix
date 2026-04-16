"""
Benchmarking utilities
"""
import sys

sys.path.append('../..')

import qiskit
import pytket
import pytket.qasm
import pytket.passes
import numpy as np
import rustworkx as rx
from typing import Tuple, List
from qiskit.transpiler import CouplingMap, PassManager


import phoenix

from tetris.utils.hardware import pGraph
from tetris.benchmark.mypauli import pauliString

from rich.console import Console

console = Console()

# Chain = CouplingMap(rx.generators.path_graph(35).to_directed().edge_list())
# Manhattan = CouplingMap(arch.read_device_topology('../manhattan.graphml').to_directed().edge_list())
# Sycamore = CouplingMap(arch.read_device_topology('../sycamore.graphml').to_directed().edge_list())
# All2all = CouplingMap(rx.generators.complete_graph(35).to_directed().edge_list())


def qiskit_O3_all2all(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    from itertools import combinations
    for q0, q1 in combinations(range(circ.num_qubits), 2):
        circ.cx(q0, q1)
        circ.cx(q0, q1)
    circ = qiskit.transpile(circ, optimization_level=3, basis_gates=['u1', 'u2', 'u3', 'cx'])
    return circ


def phoenix_pass(paulis: List[str], coeffs: List[float],
                 grouping: bool = True,
                 coupling_map: CouplingMap = None,
                 with_O3: bool = False) -> qiskit.QuantumCircuit:
    """Phoenix's high-level optimization"""
    paulis = [p[::-1] for p in paulis]  # ! PHOENIX uses little-endian convention for Pauli strings, reverse the input strings here

    ham = phoenix.Hamiltonian(paulis, coeffs)
    qc = phoenix.compile_hamiltonian_simulation(ham, grouping=grouping)
    # circ = ham.phoenix_circuit(order_blocks=order_blocks, efficient=efficient)

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = phoenix.utils.qiskit_O3_all2all(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def paulihedral_pass(paulis: List[str], coeffs: List[float],
                     coupling_map: CouplingMap = None,
                     with_O3: bool = False) -> qiskit.QuantumCircuit:
    from tetris.utils.parallel_bl import gate_count_oriented_scheduling
    from tetris.synthesis_SC import block_opt_SC

    if coupling_map is None:
        coupling_map = phoenix.utils.gene_all2all_coupling_map(len(paulis[0]))

    a2 = gate_count_oriented_scheduling(constr_mypauli_blocks(paulis, coeffs))

    qc, total_swaps, total_cx = block_opt_SC(a2, graph=coupling_map_to_pGraph(coupling_map))

    qc = qiskit.transpile(qc, optimization_level=2, basis_gates=['u1', 'u2', 'u3', 'cx'])

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = qiskit_O3_all2all(qc)
    else:
        qc = qiskit.transpile(qc,
                              basis_gates=['u1', 'u2', 'u3', 'cx'],
                              coupling_map=coupling_map,
                              initial_layout=list(range(qc.num_qubits)),
                              layout_method='sabre',
                              optimization_level=3)

    # console.print({
    #     'PH_swap_count': total_swaps,
    #     'PH_cx_count': total_cx,
    #     'CNOT': circ.num_nonlocal_gates(),
    #     'Single': circ.size() - circ.num_nonlocal_gates(),
    #     'Total': circ.size(),
    #     'Depth': circ.depth()})

    return qc


def tetris_pass(paulis: List[str], coeffs: List[float],
                coupling_map: CouplingMap = None,
                with_O3: bool = False) -> qiskit.QuantumCircuit:
    from tetris.utils.synthesis_lookahead import synthesis_lookahead

    if coupling_map is None:
        coupling_map = phoenix.utils.gene_all2all_coupling_map(len(paulis[0]))

    qc, metrics = synthesis_lookahead(constr_mypauli_blocks(paulis, coeffs),
                                      graph=coupling_map_to_pGraph(coupling_map),
                                      use_bridge=False,
                                      swap_coefficient=3, k=10)

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = qiskit_O3_all2all(qc)
    else:
        qc = qiskit.transpile(qc,
                              basis_gates=['u1', 'u2', 'u3', 'cx'],
                              coupling_map=coupling_map,
                              initial_layout=list(range(qc.num_qubits)),
                              layout_method='sabre',
                              optimization_level=3)

    metrics.update({'CNOT': qc.num_nonlocal_gates(),
                    'Single': qc.size() - qc.num_nonlocal_gates(),
                    'Total': qc.size(),
                    'Depth': qc.depth()})
    # console.print(metrics)

    return qc


def quclear_pass(paulis: List[str], coeffs: List[float],
                 coupling_map: CouplingMap = None, with_O3: bool = False) -> qiskit.QuantumCircuit:
    from quclear.CE_module import construct_qcc_circuit, CE_recur_tree

    qc, append_clifford, sorted_entanglers = CE_recur_tree(entanglers=paulis, params=coeffs, barrier=False)
    append_clifford = append_clifford.decompose('swap')
    qc.compose(append_clifford, inplace=True)

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = phoenix.utils.qiskit_O3_all2all(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def pauliopt_pass(paulis: List[str], coeffs: List[float], method='steiner_gray',
                  coupling_map: CouplingMap = None, with_O3: bool = False) -> qiskit.QuantumCircuit:
    """Optional method: annealing, steiner_gray, divide_and_conquer (The default steiner_gray which performs the best in fidel tests)"""
    from pauliopt.pauli.pauli_polynomial import PauliPolynomial
    from pauliopt.pauli.pauli_gadget import PPhase
    from pauliopt.pauli_strings import I, X, Y, Z
    from pauliopt.topologies import Topology
    from pauliopt.pauli.synthesis.annealing import annealing_synthesis
    from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
    from pauliopt.pauli.synthesis.synthesis_divide_and_conquer import synthesis_divide_and_conquer

    def apply_permutation(qc: qiskit.QuantumCircuit, permutation: list) -> qiskit.QuantumCircuit:
        """Apply a permutation to a qiskit quantum circuit."""
        register = qc.qregs[0]
        qc_out = qiskit.QuantumCircuit(register)
        for instruction in qc:
            op_qubits = [
                register[permutation[register.index(q)]] for q in instruction.qubits
            ]
            qc_out.append(instruction.operation, op_qubits)
        return qc_out

    pauli_str_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    topology = Topology.complete(len(paulis[0]))
    pp = PauliPolynomial(num_qubits=len(paulis[0]))
    for pauli_str, coeff in zip(paulis, coeffs):
        pp >>= (PPhase(coeff * 2) @ [pauli_str_map[p] for p in pauli_str])

    # TODO: check the correctness of the generated circuit
    if method == 'annealing':
        qc = annealing_synthesis(pp.copy(), topology).to_qiskit()
    elif method == 'steiner_gray':
        qc, gadget_perm, perm = pauli_polynomial_steiner_gray_clifford(pp.copy(), topology)
        qc = apply_permutation(qc.to_qiskit(), perm)
    elif method == 'divide_and_conquer':
        qc, perm = synthesis_divide_and_conquer(pp.copy(), topology)
        qc = apply_permutation(qc.to_qiskit(), perm)
    else:
        raise ValueError(f"Unknown method: {method}")

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = phoenix.utils.qiskit_O3_all2all(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def tket_pass(paulis: List[str], coeffs: List[float], 
              greedy: bool = True, coupling_map: CouplingMap = None, with_O3: bool = False) -> qiskit.QuantumCircuit:
    qc = phoenix.utils.tket_pass(paulis, coeffs, greedy=greedy)

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = phoenix.utils.qiskit_O3_all2all(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def qiskit_pass(paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None, with_O3: bool = False) -> qiskit.QuantumCircuit:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.passes import HighLevelSynthesis, HLSConfig  

    paulis = [p[::-1] for p in paulis]  # ! Qiskit uses little-endian convention for Pauli strings, reverse the input strings here

    n = len(paulis[0])  # number of qubits
    op = SparsePauliOp(paulis, coeffs)
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(op), reversed(range(n)))  

    hls_config = HLSConfig(PauliEvolution=[  
        ("rustiq", {  
            "optimize_count": True,      # 优化双量子比特门数量  
            "preserve_order": False,     # 不保持 Pauli 项顺序  
            "upto_phase": True,         # 允许全局相位差异  
            "upto_clifford": False,     # 合成最终 Clifford 算子 (If True, 类似把尾端Clifford吸收进最终measurement) 
            "resynth_clifford_method": 1  # 使用 Qiskit 贪心合成 （If 2，类似把尾端Clifford吸收进最终measurement）
        })  
    ])
    hls_pass = HighLevelSynthesis(hls_config=hls_config)
    qc = hls_pass(qc)

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = phoenix.utils.qiskit_O3_all2all(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def paulirl_pass(paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None,
                 with_O3: bool = False) -> qiskit.QuantumCircuit:
    """https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes"""
    # TODO: perform PauliRL synthesis



    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = phoenix.utils.qiskit_O3_all2all(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc



# def sabre_map(circ: qiskit.QuantumCircuit, coupling_map: CouplingMap) -> Tuple[
#     qiskit.QuantumCircuit, List[int], List[int]]:
#     """
#     Mapping logical circuits on physical qubits by means of SabreLayout pass in Qiskit.

#     Args:
#         circ: Input logical quantum circuit
#         coupling_map: Physical qubit connectivity graph

#     Returns:
#         Mapped circuit, initial mapping (physical qubit indices), final mapping (physical qubit indices)
#     """
#     from qiskit.transpiler import passes

#     pm = PassManager([passes.SabreLayout(coupling_map)])
#     circ = pm.run(circ)
#     # init_mapping_inv = {i: j for i, j in zip(circ.layout.initial_index_layout(), range(circ.num_qubits))}
#     # final_mapping_inv = {i: j for i, j in zip(circ.layout.final_index_layout(), range(circ.num_qubits))}
#     # init_mapping = {j: i for i, j in init_mapping_inv.items()}
#     # final_mapping = {j: i for i, j in final_mapping_inv.items()}
#     # circ = Circuit.from_qiskit(circ).rewire(init_mapping_inv)
#     # return circ, init_mapping, final_mapping
#     return circ, circ.layout.initial_index_layout(), circ.layout.final_index_layout()


# def pre_mapping_optimize(circ: pytket.Circuit) -> pytket.Circuit:
#     """Pre-mapping optimization on logical circuits by means of TKet's pass"""
#     circ = circ.copy()
#     pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
#     return circ
#
#
# def post_mapping_optimize(circ: pytket.Circuit) -> pytket.Circuit:
#     """Post-mapping optimization on physical circuits by means of TKet's pass"""
#     circ = circ.copy()
#     pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
#     return circ



def coupling_map_to_pGraph(coupling_map: CouplingMap) -> pGraph:
    """Used in Paulihedral/Tetris primitive"""
    G = rx.adjacency_matrix(coupling_map.graph)
    C = rx.floyd_warshall_numpy(coupling_map.graph)
    return pGraph(G, C)


def constr_mypauli_blocks(paulis, coeffs) -> List[List[pauliString]]:
    """Used in Paulihedral/Tetris primitive"""
    groups = phoenix.primitive.group_paulis_and_coeffs(paulis, coeffs)
    
    mypauli_blocks = []
    for paulis, coeffs in groups.values():
        mypauli_blocks.append([])
        for p, c in zip(paulis, coeffs):
            mypauli_blocks[-1].append(pauliString(p, c))
    return mypauli_blocks


def su4_circ_stats(qc: qiskit.QuantumCircuit) -> Tuple[int, int]:
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    from canopus import rebase_to_canonical

    qc = rebase_to_canonical(qc)
    num_su4 = qc.count_ops().get('can', 0)
    depth_su4 = qc.depth(lambda instr: instr.operation.name == 'can')
    return num_su4, depth_su4
