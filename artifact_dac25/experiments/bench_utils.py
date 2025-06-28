"""
Benchmarking utilities
"""
import sys

sys.path.append('../..')

import pytket.circuit
import qiskit
import pytket
import pytket.qasm
import pytket.passes
import numpy as np
import rustworkx as rx
from typing import Tuple, List
from qiskit.transpiler import CouplingMap, PassManager

from phoenix import Circuit, gates
from phoenix.utils import arch
from phoenix.utils.ops import is_tensor_prod
from phoenix.models.hamiltonians import HamiltonianModel

from tetris.utils.hardware import pGraph
from tetris.benchmark.mypauli import pauliString
from phoenix.primitive.grouping import group_paulis_and_coeffs
from rich.console import Console

console = Console()

Chain = CouplingMap(rx.generators.path_graph(35).to_directed().edge_list())
Manhattan = CouplingMap(arch.read_device_topology('./manhattan.graphml').to_directed().edge_list())
Sycamore = CouplingMap(arch.read_device_topology('./sycamore.graphml').to_directed().edge_list())
All2all = CouplingMap(rx.generators.complete_graph(35).to_directed().edge_list())


def qiskit_O3_all2all(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    from itertools import combinations
    for q0, q1 in combinations(range(circ.num_qubits), 2):
        circ.cx(q0, q1)
        circ.cx(q0, q1)
    circ = qiskit.transpile(circ, optimization_level=3, basis_gates=['u1', 'u2', 'u3', 'cx'])
    return circ


def phoenix_pass(paulis: List[str], coeffs: List[float],
                 order_blocks: bool = True,
                 efficient: bool = False,
                 coupling_map: CouplingMap = All2all,
                 with_O3: bool = False) -> qiskit.QuantumCircuit:
    """Phoenix's high-level optimization"""
    ham = HamiltonianModel(paulis, coeffs)
    circ = ham.phoenix_circuit(order_blocks=order_blocks, efficient=efficient)

    # print(circ.to_qiskit())

    # circ = passes.remoev_front_cliffords(circ)
    # circ = passes.remoev_last_cliffords(circ)

    qc = circ.to_qiskit()
    if is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = qiskit_O3_all2all(qc)
    else:
        qc = optimize_with_mapping(qc, coupling_map)

    return qc


def paulihedral_pass(paulis: List[str], coeffs: List[float],
                     coupling_map: CouplingMap = All2all,
                     with_O3: bool = False) -> qiskit.QuantumCircuit:
    from tetris.utils.parallel_bl import gate_count_oriented_scheduling
    from tetris.synthesis_SC import block_opt_SC

    a2 = gate_count_oriented_scheduling(constr_mypauli_blocks(paulis, coeffs))

    qc, total_swaps, total_cx = block_opt_SC(a2, graph=coupling_map_to_pGraph(coupling_map))

    qc = qiskit.transpile(qc, optimization_level=2, basis_gates=['u1', 'u2', 'u3', 'cx'])

    if is_all2all_coupling_map(coupling_map):
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
                coupling_map: CouplingMap = All2all,
                with_O3: bool = False) -> qiskit.QuantumCircuit:
    from tetris.utils.synthesis_lookahead import synthesis_lookahead

    qc, metrics = synthesis_lookahead(constr_mypauli_blocks(paulis, coeffs),
                                      graph=coupling_map_to_pGraph(coupling_map),
                                      use_bridge=False,
                                      swap_coefficient=3, k=10)

    if is_all2all_coupling_map(coupling_map):
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
                 coupling_map: CouplingMap = All2all, with_O3: bool = False) -> qiskit.QuantumCircuit:
    from quclear.CE_module import construct_qcc_circuit, CE_recur_tree

    qc, append_clifford, sorted_entanglers = CE_recur_tree(entanglers=paulis, params=coeffs, barrier=False)
    append_clifford = append_clifford.decompose('swap')
    qc.compose(append_clifford, inplace=True)

    if is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = qiskit_O3_all2all(qc)
    else:
        qc = optimize_with_mapping(qc, coupling_map)

    return qc


def pauliopt_pass(paulis: List[str], coeffs: List[float], method='steiner_gray',
                  coupling_map: CouplingMap = All2all, with_O3: bool = False) -> qiskit.QuantumCircuit:
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

    if is_all2all_coupling_map(coupling_map):
        if with_O3:
            qc = qiskit_O3_all2all(qc)
    else:
        qc = optimize_with_mapping(qc, coupling_map)

    return qc


def tket_pass(paulis: List[str], coeffs: List[float], greedy: bool = True) -> pytket.Circuit:
    circ = constr_tket_circuit(paulis, coeffs)
    if greedy:
        pytket.passes.GreedyPauliSimp().apply(circ)
        pytket.passes.FullPeepholeOptimise().apply(circ)
    else:
        pytket.passes.PauliSquash().apply(circ)
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


def optimize_with_mapping(circ: qiskit.QuantumCircuit, coupling_map: CouplingMap,
                          tket_opt: bool = False) -> qiskit.QuantumCircuit:
    """By default, we use Qiskit's O3 compiler to performa hardware-aware tranpilation and optimization"""
    circ = qiskit_O3_all2all(circ)  # since input is logical circuit, we can first ally an all2all Qiskit O3
    circ = qiskit.transpile(circ, optimization_level=3,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=coupling_map, layout_method='sabre')

    # whether to further perform a TKet's topology-preserved optimization pass
    if tket_opt:
        circ = qiskit_to_tket(circ)
        pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
        circ = tket_to_qiskit(circ)

    return circ


def coupling_map_to_pGraph(coupling_map: CouplingMap) -> pGraph:
    """Used in Paulihedral/Tetris primitive"""
    G = rx.adjacency_matrix(coupling_map.graph)
    C = rx.floyd_warshall_numpy(coupling_map.graph)
    return pGraph(G, C)


def constr_mypauli_blocks(paulis, coeffs) -> List[List[pauliString]]:
    """Used in Paulihedral/Tetris primitive"""
    groups = group_paulis_and_coeffs(paulis, coeffs)
    mypauli_blocks = []
    for paulis, coeffs in groups.values():
        mypauli_blocks.append([])
        for p, c in zip(paulis, coeffs):
            mypauli_blocks[-1].append(pauliString(p, c))
    return mypauli_blocks


def constr_tket_circuit(paulis: List[str], coeffs: List[float]) -> pytket.Circuit:
    from pytket.circuit import PauliExpBox
    from pytket.pauli import Pauli

    pauli_str_map = {
        'X': Pauli.X,
        'Y': Pauli.Y,
        'Z': Pauli.Z,
    }

    def get_qubits_acted(pauli: str) -> List[int]:
        """Get the qubits acted by a Pauli string."""
        return [i for i, p in enumerate(pauli) if p != 'I']

    circ = pytket.Circuit(len(paulis[0]))

    for pauli, coeff in zip(paulis, coeffs):
        qubits = get_qubits_acted(pauli)
        if len(qubits) == 0:
            continue
        pauli_box = PauliExpBox([pauli_str_map[pauli[q]] for q in qubits], t=coeff * 2 / np.pi)
        circ.add_gate(pauli_box, qubits)

    return circ


def is_all2all_coupling_map(coupling_map: CouplingMap) -> bool:
    # ! coupling_map.graph is a directed coupling map
    if coupling_map.size() * (coupling_map.size() - 1) == len(coupling_map.get_edges()):
        return True
    return False


def su4_circ_stats(circ):
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    from bqskit.compiler import Compiler
    from bqskit.passes import QuickPartitioner

    with Compiler() as compiler:
        blocks = list(compiler.compile(circ, QuickPartitioner(2)))

    fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])
    circ_su4 = Circuit([g for g in fused_2q if not is_tensor_prod(g.data)])
    return circ_su4.num_gates, len(circ_su4.layer())
