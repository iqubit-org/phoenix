import qiskit
import rustworkx as rx
from itertools import product
from functools import reduce
from operator import add
from phoenix.basic.circuits import Circuit
from phoenix.basic import gates
from phoenix.basic.gates import Gate
from phoenix.models.cliffords import Clifford2Q
from phoenix.models.paulis import BSF
from phoenix import decompose
from phoenix.utils.passes import peel_first_1q_gates, obtain_front_layer
from phoenix.utils.graphs import filter_nodes, node_index
from phoenix.utils.operations import params_u3
from phoenix.utils.passes import dag_to_circuit
from typing import List, Tuple, Set, Union
import numpy as np


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


def unique_paulis_and_coeffs(paulis: List[str], coeffs: List[float]) -> Tuple[List[str], List[float]]:
    """Remove duplicates in Pauli strings and sum up their coefficients."""
    if len(np.unique(paulis)) == len(paulis):
        return paulis, coeffs
    unique_paulis = []
    unique_coeffs = []
    for pauli, coeff in zip(paulis, coeffs):
        if len(pauli) == pauli.count('I'):
            continue
        if pauli in unique_paulis:
            idx = unique_paulis.index(pauli)
            unique_coeffs[idx] += coeff
        else:
            unique_paulis.append(pauli)
            unique_coeffs.append(coeff)
    return unique_paulis, unique_coeffs


def optimize_clifford_circuit_by_qiskit(circ: Circuit, optimization_level=1) -> Circuit:
    """Topology-preserved optimization by Qiskit"""
    basis_gates = ['h', 's', 'sdg', 'rz', 'u3', 'cx']
    for p0, p1 in product(['x', 'y', 'z'], repeat=2):
        basis_gates.append(f'c{p0}{p1}')
    return Circuit.from_qiskit(qiskit.transpile(circ.to_qiskit(), optimization_level=optimization_level,
                                                basis_gates=basis_gates))


def config_to_circuit(config, optimize=True):
    circ = Circuit()
    for item in config:
        if isinstance(item, Clifford2Q):
            circ.append(item.as_gate())
        if isinstance(item, BSF):
            circ += item.as_cnot_circuit()

    # by default, we use Qiskit O2 to consolidate redundant 1Q and CNOT gates
    if optimize:
        return optimize_clifford_circuit_by_qiskit(circ, 2)
    return circ


# def rebase_to_canonical(circ: Circuit, normalize_coordinates: bool = True) -> Circuit:
#     """
#     Rebase the circuit to Canonical-basis circuit, through only block fusing and gate decomposition,
#      without approximate synthesis.
#
#     :param circ: input circuit
#     :param normalize_coordinates: whether to normalize canonical coordinates into the range {1/2 ≥ x ≥ y ≥ |z| ≥ 0}
#     """
#     blocks_2q = sequential_partition(circ, 2)
#     fused_2q = fuse_blocks(blocks_2q)
#
#     circ_su4 = unroll_su4(fused_2q, by='can')
#     if normalize_coordinates:
#         circ_su4 = _normalize_canonical_coordinates(circ_su4)
#     circ_su4 = fuse_neighbor_u3(circ_su4)
#     return circ_su4


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


def fuse_blocks(blocks: List[Circuit], name: str = 'U') -> Circuit:
    """
    Fuse each block in the list into a single multi-qubit gate.
    """
    return Circuit([gates.UnivGate(blk.unitary(), name=name).on(blk.qubits) for blk in blocks])


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

#
#
# def sequential_partition(circ: Circuit, grain: int = 2) -> List[Circuit]:
#     """
#     Partition a list of circuits into groups of grain-qubit blocks (subcircuits) by one-round forward pass.
#     ---
#     Complexity: O(m*n), m is the number of 2Q gates, n is the number of qubits
#     """
#     if grain <= 1:
#         raise ValueError("grain must be greater than 1.")
#     if grain < circ.max_gate_weight:
#         raise ValueError("grain must be no less than the maximum gate weight of the circuit.")
#
#     # peel all 1Q gates from the first layer
#     circ_nl, first_1q_gates = peel_first_1q_gates(circ)
#
#     blocks = []
#
#     dag = circ_nl.to_dag()
#     while circ_nl:  # for each epoch, select a block with the most nonlocal gates
#         front_layer = obtain_front_layer(dag)
#         block_candidates = [Circuit([g]) for g in front_layer]
#         for i, (g, block) in enumerate(zip(front_layer, block_candidates)):
#             dag_peeled = dag.copy()
#             dag_peeled.remove_node(node_index(dag, g))
#             block_candidates[i] = _extend_block_over_dag(block, dag_peeled, grain)
#
#         scores = [block_score(block) for block in block_candidates]
#         block = block_candidates[np.argmax(scores)]  # the selected extended block
#         # console.print('selected block: {} with qubits {}'.format(block, block.qubits), style='bold green')
#         for g in block:
#             circ_nl.remove(g)
#             dag.remove_node(node_index(dag, g))
#         blocks.append(block)
#         # console.print('current num of blocks: {}'.format(len(blocks)))
#
#     # add 1Q gates from first_1q_gates back to corresponding blocks
#     dag = circ.to_dag()
#     isolated_1q_gates = []
#     for g in reversed(first_1q_gates):
#         for blk in blocks:
#             if dag.successors(node_index(dag, g)) and list(dag.successors(node_index(dag, g)))[0] in blk:
#                 blk.prepend(g)
#                 break
#         else:
#             isolated_1q_gates.append(g)
#
#     # add isolated 1Q gates as blocks
#     isolated_1q_gates = list(reversed(isolated_1q_gates))
#     isolated_1q_blocks = [Circuit([g]) for g in isolated_1q_gates]
#     blocks = isolated_1q_blocks + blocks
#
#     assert sum([blk.num_gates for blk in blocks]) == circ.num_gates, "num_gates mismatch"
#     # console.print('num_gates_all_blocks: {}'.format(sum([blk.num_gates for blk in blocks])))
#     # console.print('num_gates_circ: {}'.format(len(circ)))
#     # console.print('num_gates: {}'.format(circ.num_gates))
#
#     # NOTE: in this algorithm we do not need to unify the blocks since they are already sorted
#
#     return blocks


# def _extend_block_over_dag(block: Circuit, dag: rx.PyDiGraph, max_weight: int) -> Circuit:
#     """Search applicable gates from the neighbors of block among this DAG to add them to block"""
#     block = block.clone()
#     # console.print('extending {} on {}'.format(block, block.qubits), style='bold green')
#     if not dag.num_nodes():
#         return block
#
#     while front_layer := obtain_front_layer(dag):
#         optional_gates = _sort_gates_on_ref_qubits(front_layer, block.qubits)
#         # console.print('optional gates: {}'.format(optional_gates), style='bold green')
#         # console.print([g.qregs for g in optional_gates], style='bold green')
#         if len(set(block.qubits + optional_gates[0].qregs)) > max_weight:
#             break
#         for g in optional_gates:
#             if len(set(block.qubits + g.qregs)) <= max_weight:
#                 block.append(g)
#                 dag.remove_node(node_index(dag, g))
#             else:
#                 break
#
#     return block
#
#
# def _sort_gates_on_ref_qubits(gates: List[Gate], ref_qubits: Union[List[int], Set[int]]) -> List[Gate]:
#     """
#     Sort the gates according to the number of qubits overlapped and additional overhead with the given set of qubits
#     Sort by: 1) additional overhead (ascending); 2) number of qubits overlapped (descending)
#     """
#     return sorted(gates, key=lambda g: (len(set(g.qregs) - set(ref_qubits)),
#                                         - len(set(g.qregs) & set(ref_qubits))))
#


# def block_score(block: Union[List[Gate], Circuit, List[Circuit]]) -> float:
#     if isinstance(block, Circuit):
#         circ = block
#     elif isinstance(block, list) and isinstance(block[0], Gate):
#         circ = Circuit(block)
#     elif isinstance(block, list) and isinstance(block[0], Circuit):
#         circ = reduce(add, block)
#     else:
#         raise ValueError('Invalid type of block.')
#
#     score_3q = 10 * len([g for g in circ if g.num_qregs == 3])
#     score_2q = 2 * len([g for g in circ if g.num_qregs == 2])
#     score_1q = 0.2 * len([g for g in circ if g.num_qregs == 1])
#
#     return score_3q + score_2q + score_1q
