"""
Transformations on circuits.
"""
import numpy as np
import rustworkx as rx
from operator import add
from typing import List, Union, Set, Tuple, Dict
from functools import reduce
from bqskit.compiler import Compiler
from phoenix.basic import gates
from phoenix.basic.gates import Gate
from phoenix.basic.circuits import Circuit
from phoenix import decompose
from phoenix.utils.operations import params_u3, tensor_1_slot, is_tensor_prod
from phoenix.utils.graphs import filter_nodes, find_predecessors_by_node, find_successors_by_node, node_index
from phoenix.utils.passes import dag_to_circuit, obtain_front_layer, peel_first_1q_gates


################################################################
# Partitioning
################################################################


def sequential_partition(circ: Circuit, grain: int = 2) -> List[Circuit]:
    """
    Partition a list of circuits into groups of grain-qubit blocks (subcircuits) by one-round forward pass.
    ---
    Complexity: O(m*n), m is the number of 2Q gates, n is the number of qubits
    """

    if grain <= 1:
        raise ValueError("grain must be greater than 1.")
    if grain < circ.max_gate_weight:
        raise ValueError("grain must be no less than the maximum gate weight of the circuit.")

    # peel all 1Q gates from the first layer
    circ_nl, first_1q_gates = peel_first_1q_gates(circ)

    blocks = []

    dag = circ_nl.to_dag()
    while circ_nl:  # for each epoch, select a block with the most nonlocal gates
        front_layer = obtain_front_layer(dag)
        block_candidates = [Circuit([g]) for g in front_layer]
        for i, (g, block) in enumerate(zip(front_layer, block_candidates)):
            dag_peeled = dag.copy()
            dag_peeled.remove_node(node_index(dag, g))
            block_candidates[i] = _extend_block_over_dag(block, dag_peeled, grain)

        scores = [_block_score(block) for block in block_candidates]
        block = block_candidates[np.argmax(scores)]  # the selected extended block
        # console.print('selected block: {} with qubits {}'.format(block, block.qubits), style='bold green')
        for g in block:
            circ_nl.remove(g)
            dag.remove_node(node_index(dag, g))
        blocks.append(block)
        # console.print('current num of blocks: {}'.format(len(blocks)))

    # add 1Q gates from first_1q_gates back to corresponding blocks
    dag = circ.to_dag()
    isolated_1q_gates = []
    for g in reversed(first_1q_gates):
        for blk in blocks:
            if dag.successors(node_index(dag, g)) and list(dag.successors(node_index(dag, g)))[0] in blk:
                blk.prepend(g)
                break
        else:
            isolated_1q_gates.append(g)

    # add isolated 1Q gates as blocks
    isolated_1q_gates = list(reversed(isolated_1q_gates))
    isolated_1q_blocks = [Circuit([g]) for g in isolated_1q_gates]
    blocks = isolated_1q_blocks + blocks

    assert sum([blk.num_gates for blk in blocks]) == circ.num_gates, "num_gates mismatch"
    # console.print('num_gates_all_blocks: {}'.format(sum([blk.num_gates for blk in blocks])))
    # console.print('num_gates_circ: {}'.format(len(circ)))
    # console.print('num_gates: {}'.format(circ.num_gates))

    # NOTE: in this algorithm we do not need to unify the blocks since they are already sorted

    return blocks


def _extend_block_over_dag(block: Circuit, dag: rx.PyDiGraph, max_weight: int) -> Circuit:
    """Search applicable gates from the neighbors of block among this DAG to add them to block"""
    block = block.clone()
    # console.print('extending {} on {}'.format(block, block.qubits), style='bold green')
    if not dag.num_nodes():
        return block

    while front_layer := obtain_front_layer(dag):
        optional_gates = _sort_gates_on_ref_qubits(front_layer, block.qubits)
        # console.print('optional gates: {}'.format(optional_gates), style='bold green')
        # console.print([g.qregs for g in optional_gates], style='bold green')
        if len(set(block.qubits + optional_gates[0].qregs)) > max_weight:
            break
        for g in optional_gates:
            if len(set(block.qubits + g.qregs)) <= max_weight:
                block.append(g)
                dag.remove_node(node_index(dag, g))
            else:
                break

    return block


def _sort_gates_on_ref_qubits(gates: List[Gate], ref_qubits: Union[List[int], Set[int]]) -> List[Gate]:
    """
    Sort the gates according to the number of qubits overlapped and additional overhead with the given set of qubits
    Sort by: 1) additional overhead (ascending); 2) number of qubits overlapped (descending)
    """
    return sorted(gates, key=lambda g: (len(set(g.qregs) - set(ref_qubits)),
                                        - len(set(g.qregs) & set(ref_qubits))))


def _block_score(block: Union[List[Gate], Circuit, List[Circuit]], init_score: float = 0) -> float:
    if isinstance(block, Circuit):
        circ = block
    elif isinstance(block, list) and isinstance(block[0], Gate):
        circ = Circuit(block)
    elif isinstance(block, list) and isinstance(block[0], Circuit):
        circ = reduce(add, block)
    else:
        raise ValueError('Invalid type of block.')
    # score_local = 0.1 * (circ.num_gates - circ.num_nonlocal_gates)
    # score_nl = circ.num_nonlocal_gates  # the number of nonlocal gates contributes the most
    score_3q = 10 * len([g for g in circ if g.num_qregs == 3])
    score_2q = 2 * len([g for g in circ if g.num_qregs == 2])
    score_1q = 0.2 * len([g for g in circ if g.num_qregs == 1])

    return score_3q + score_2q + score_1q


################################################################
# Unrolling
################################################################

def unroll_u3(circ: Circuit, by: str = 'zyz') -> Circuit:
    """
    Unroll U1, U2 and U3 gate by Euler decomposition
    """
    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 1 and isinstance(g, (gates.U2, gates.U3)):
            circ_unrolled += decompose.euler_decompose(g, basis=by, with_phase=False)
        elif g.num_qregs == 1 and isinstance(g, gates.U1):
            circ_unrolled.append(gates.RZ(g.angle).on(g.tq))
        else:
            circ_unrolled.append(g)

    return circ_unrolled


def unroll_su4(circ: Circuit, by: str = 'can') -> Circuit:
    """
    Unroll two-qubit gates, i.e., SU(4) gates, by canonical decomposition or other methods
    """
    if by not in ['can', 'cnot']:
        raise ValueError("Only support canonical (by='can') and CNOT unrolling (by='cnot').")

    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 2 and not g.cqs:  # arbitrary two-qubit gate
            if by == 'can':
                circ_unrolled += decompose.can_decompose(g)
            if by == 'cnot':
                circ_unrolled += decompose.kak_decompose(g)
        else:
            circ_unrolled.append(g)

    return circ_unrolled


def unroll_tensor_product(circ: Circuit) -> Circuit:
    """
    Unroll fake two-qubit gate (tensor-product) into two singel-qubit gates
    """
    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == len(g.tqs) == 2 and is_tensor_prod(g.data):
            circ_unrolled += decompose.tensor_product_decompose(g)
        else:
            circ_unrolled.append(g)

    return circ_unrolled


################################################################
# Approximation
################################################################


def approx_to_cnot(circ: Circuit,
                   compiler: Compiler = None,
                   max_synthesis_size: int = 3,
                   optimization_level: int = 1,
                   seed: int = None) -> Circuit:
    """
    Approximate synthesis using CNOT + U3 gates

    Args:
        circ: The input circuit
        compiler: Pre-set compiler for the synthesis. (Default: None)
        max_synthesis_size: The maximum size of a unitary to synthesize or instantiate. Larger circuits will be partitioned.
            Increasing this will most likely lead to better results with an exponential time trade-off. (Default: 3)
        optimization_level: The optimization level to use. (Default: 1)
        seed: The seed for the random number generator. (Default: None)
    """
    import bqskit
    from bqskit.ir.gates import CNOTGate, U3Gate

    circ_bqs = circ.to_bqskit()
    model = bqskit.MachineModel(circ.num_qubits_with_dummy, gate_set={CNOTGate(), U3Gate()})
    circ_bqs_opt = bqskit.compile(
        circ_bqs, model, compiler=compiler,
        max_synthesis_size=max_synthesis_size,
        optimization_level=optimization_level, seed=seed)
    circ_opt = Circuit.from_bqskit(circ_bqs_opt)
    return circ_opt


def approx_to_su4(circ: Circuit,
                  compiler: Compiler = None,
                  max_synthesis_size: int = 3,
                  optimization_level: int = 1,
                  seed: int = None) -> Circuit:
    """
    Approximate synthesis using SU(4) gates, i.e., Canonical + U3

    Args:
        circ: The input circuit
        compiler: Pre-set compiler for the synthesis. (Default: None)
        max_synthesis_size: The maximum size of a unitary to synthesize or instantiate. Larger circuits will be partitioned.
            Increasing this will most likely lead to better results with an exponential time trade-off. (Default: 3)
        optimization_level: The optimization level to use. (Default: 1)
        seed: The seed for the random number generator. (Default: None)
    """
    import bqskit
    from bqskit.ir.gates import CanonicalGate, U3Gate

    circ_bqs = circ.to_bqskit()
    model = bqskit.MachineModel(circ.num_qubits_with_dummy, gate_set={CanonicalGate(), U3Gate()})
    circ_bqs_opt = bqskit.compile(
        circ_bqs, model, compiler=compiler,
        max_synthesis_size=max_synthesis_size,
        optimization_level=optimization_level, seed=seed)
    circ_opt = Circuit.from_bqskit(circ_bqs_opt)
    return circ_opt


################################################################
# Fusing
################################################################

def fuse_blocks(blocks: List[Circuit], name: str = 'U') -> Circuit:
    """
    Fuse each block in the list into a single multi-qubit gate.
    """
    return Circuit([gates.UnivGate(blk.unitary(), name=name).on(blk.qubits) for blk in blocks])


# def fuse_u3_to_su4(circ: Circuit) -> Circuit:
#     """Contract all single-qubit gates into neighboring SU(4)"""
#     dag = circ.to_dag()
#     nodes_2q_gate = filter_nodes(dag, lambda node: isinstance(node, Gate) and node.num_qregs == 2)
#     for g in nodes_2q_gate:
#         print(g)
#         while True:
#             predecessors_1q = find_predecessors_by_node(dag, node_index(dag, g),
#                                                         lambda node: isinstance(node, Gate) and node.num_qregs == 1)
#             successors_1q = find_successors_by_node(dag, node_index(dag, g),
#                                                     lambda node: isinstance(node, Gate) and node.num_qregs == 1)
#             if not predecessors_1q and not successors_1q:  # there is no 1Q gate in the neighborhood
#                 break
#             for g_pred in predecessors_1q:
#                 if g_pred.tq == g.tqs[0]:
#                     g.data = g.matrix() @ tensor_1_slot(g_pred.data, 2, 0)
#                 else:
#                     g.data = g.matrix() @ tensor_1_slot(g_pred.data, 2, 1)
#                 dag.contract_nodes([node_index(dag, g), node_index(dag, g_pred)], g)
#             for g_succ in successors_1q:
#                 if g_succ.tq == g.tqs[0]:
#                     g.data = tensor_1_slot(g_succ.data, 2, 0) @ g.data
#                 else:
#                     g.data = tensor_1_slot(g_succ.data, 2, 1) @ g.data
#                 dag.contract_nodes([node_index(dag, g), node_index(dag, g_succ)], g)
#
#     return dag_to_circuit(dag)


def fuse_neighbor_u3(circ: Circuit) -> Circuit:
    """Fuse neighboring single-qubit gates into one U3 gate"""
    dag = circ.to_dag()
    # nodes_1q_gate = filter_nodes(dag, lambda node: isinstance(node, Gate) and node.num_qregs == 1)
    while nodes_1q_gates_with_1q_successors := filter_nodes(
            dag,
            lambda node: isinstance(node, Gate) and node.num_qregs == 1 and [succ for succ in
                                                                             dag.successors(node_index(dag, node)) if
                                                                             succ.num_qregs == 1]):
        for g in nodes_1q_gates_with_1q_successors:
            # if g has been fused, or g has no successor, skip
            if g not in dag.nodes() or not dag.successors(node_index(dag, g)):
                continue
            succ = dag.successors(node_index(dag, g))[0]
            if succ.num_qregs != 1:  # skip 2Q neighbor gates
                continue
            # fuse them into a new U3 gate
            u3 = gates.U3(*params_u3(succ.data @ g.data)).on(g.tq)
            dag.contract_nodes([node_index(dag, g), node_index(dag, succ)], u3)

    circ = dag_to_circuit(dag)
    for i, g in enumerate(circ):
        if g.num_qregs == 1 and not isinstance(g, gates.U3):
            u3 = gates.U3(*params_u3(g.data)).on(g.tq)
            circ[i] = u3

    return circ


################################################################
# Rebasing
################################################################
def rebase_to_canonical(circ: Circuit, normalize_coordinates: bool = False) -> Circuit:
    """
    Rebase the circuit to Canonical-basis circuit, through only block fusing and gate decomposition,
     without approximate synthesis.

    :param circ: input circuit
    :param normalize_coordinates: whether to normalize canonical coordinates into the range {1/2 ≥ x ≥ y ≥ |z| ≥ 0}
    """
    blocks_2q = sequential_partition(circ, 2)
    fused_2q = fuse_blocks(blocks_2q)
    circ_su4 = unroll_su4(fused_2q, by='can')
    if normalize_coordinates:
        circ_su4 = _normalize_canonical_coordinates(circ_su4)
    circ_su4 = fuse_neighbor_u3(circ_su4)
    return circ_su4


def _normalize_canonical_coordinates(circ: Circuit) -> Circuit:
    """
    Normalize the canonical coordinates of SU(4) gates into the range {1/2 ≥ x ≥ y ≥ |z| ≥ 0}
    """
    circ_normalized = Circuit()
    for g in circ:
        if isinstance(g, gates.Can):
            if g.angles[0] <= 0 and g.angles[1] <= 0:
                circ_normalized.append(
                    gates.Z.on(g.tqs[0]),
                    gates.Can(-g.angles[0], -g.angles[1], g.angles[2]).on(g.tqs, g.cqs),
                    gates.Z.on(g.tqs[0])
                )
            else:
                circ_normalized.append(g)  # TODO ...
        else:
            circ_normalized.append(g)
    return circ_normalized


################################################################
# Mapping
################################################################
def sabre_by_qiskit(circ: Circuit, device: rx.PyGraph,
                    seed: int = None) -> Tuple[Circuit, Dict[int, int], Dict[int, int]]:
    """The running efficiency of qiskit.transpiler.passes.SabreLayout is better than what we implemented"""
    from qiskit.transpiler import passes, PassManager, CouplingMap

    circ = circ.to_qiskit()
    pm = PassManager([passes.SabreLayout(CouplingMap(device.edge_list()), seed=seed)])
    circ = pm.run(circ)
    init_mapping_inv = {i: j for i, j in zip(circ.layout.initial_index_layout(), range(circ.num_qubits))}
    final_mapping_inv = {i: j for i, j in zip(circ.layout.final_index_layout(), range(circ.num_qubits))}
    init_mapping = {j: i for i, j in init_mapping_inv.items()}
    final_mapping = {j: i for i, j in final_mapping_inv.items()}
    circ = Circuit.from_qiskit(circ).rewire(init_mapping_inv)
    return circ, init_mapping, final_mapping


################################################################
# Topology-preserved optimization
################################################################
def phys_circ_opt_by_qiskit(circ: Circuit) -> Circuit:
    import qiskit

    circ_qiskit_opt = qiskit.transpile(circ.to_qiskit(), optimization_level=3, basis_gates=['u1', 'u2', 'u3', 'cx'])
    circ_opt = Circuit.from_qiskit(circ_qiskit_opt)
    assert sorted(circ.qubit_dependency().edges()) == sorted(circ_opt.qubit_dependency().edges()), "Topology mismatch"
    return circ_opt


def phys_circ_opt_by_tket(circ: Circuit) -> Circuit:
    import pytket.passes

    circ_tket = circ.to_tket()
    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ_tket)
    pytket.passes.RemoveRedundancies().apply(circ_tket)
    circ_opt = Circuit.from_tket(circ_tket)
    assert sorted(circ.qubit_dependency().edges()) == sorted(circ_opt.qubit_dependency().edges()), "Topology mismatch"
    return circ_opt
