import networkx as nx
import rustworkx as rx
from typing import List, Union, Tuple
from functools import reduce
from operator import add
from phoenix.basic.gates import Gate, UnivGate
from phoenix.basic.circuits import Circuit
from phoenix.utils.graphs import find_predecessors_by_node, find_successors_by_node, node_index
from rich.console import Console

console = Console()


def dag_to_layers(dag: Union[nx.DiGraph, rx.PyDiGraph]) -> List[List]:
    """Convert a DAG to a list of layers in topological order"""
    if not isinstance(dag, (nx.DiGraph, rx.PyDiGraph)):
        raise TypeError("Input must be a networkx.DiGraph or rx.PyDiGraph instance, not {}".format(type(dag)))

    dag = dag.copy()
    layers = []
    while dag.nodes():
        if isinstance(dag, rx.PyDiGraph):
            front_layer, front_layer_indices = obtain_front_layer(dag, return_indices=True)
            dag.remove_nodes_from(front_layer_indices)
        else:
            front_layer = obtain_front_layer(dag)
            dag.remove_nodes_from(front_layer)
        layers.append(front_layer)
    return layers


def dag_to_circuit(dag: Union[nx.DiGraph, rx.PyDiGraph]) -> Circuit:
    """Convert a DAG to a Circuit"""
    node_is_block = isinstance(next(iter(dag.nodes())), Circuit)
    if isinstance(dag, rx.PyDiGraph):
        if node_is_block:
            return Circuit(reduce(add, [dag[idx].gates for idx in rx.topological_sort(dag)]))
        return Circuit([dag[idx] for idx in rx.topological_sort(dag)])
    else:
        if node_is_block:
            return Circuit(reduce(add, list(nx.topological_sort(dag))))
        return Circuit(list(nx.topological_sort(dag)))


def obtain_front_layer(dag_or_circ: Union[Circuit, nx.DiGraph, rx.PyDiGraph],
                       return_indices: bool = False) -> Union[
    List[Union[Gate, Circuit]], Tuple[List[Union[Gate, Circuit]], List[int]]]:
    """
    Obtain front layer (with in_degree == 0) of the DAG.
    Since the node of DAG might be Gate instance or Circuit instance, result is a list of Gate or Circuit.
    """
    if isinstance(dag_or_circ, Circuit):
        dag = dag_or_circ.to_dag()
    else:
        dag = dag_or_circ
    front_layer = []
    front_layer_indices = []
    if isinstance(dag, nx.DiGraph):
        for node in dag.nodes():
            if dag.in_degree(node) == 0:
                front_layer.append(node)
    elif isinstance(dag, rx.PyDiGraph):
        for node_idx in dag.node_indices():
            if dag.in_degree(node_idx) == 0:
                front_layer.append(dag[node_idx])
                front_layer_indices.append(node_idx)
    else:
        raise TypeError("Input must be a Circuit, networkx.DiGraph or rustworkx.PyDiGraph instance, not {}".format(
            type(dag_or_circ)))

    if isinstance(dag, rx.PyDiGraph) and return_indices:
        return front_layer, front_layer_indices
    return front_layer


def sort_blocks_on_qregs(blocks: List[Circuit], descend=False) -> List[Circuit]:
    if descend:
        return sorted(blocks, key=lambda b: max([max(blk.qubits) for blk in blocks]))
    else:
        return sorted(blocks, key=lambda b: min([min(blk.qubits) for blk in blocks]))


def unify_blocks(blocks: List[Circuit], circ: Circuit) -> List[Circuit]:
    """
    Unify the blocks (reorder them) according to original gates orders for the given circuit.
    """
    # construct the mapping from node indices to blocks
    # print()
    # console.print('before unified:')
    # console.print(blocks)

    # for blk in blocks:
    #     console.print(blk)
    #     print(blk.to_cirq())

    # contract the nodes in the same block, replace the contracted node by the block in DAG
    dag = circ.to_dag()
    for block in blocks:
        # dag = nx.relabel_nodes(dag, {block[0]: block})
        dag[node_index(dag, block[0])] = block

        # print('DAG nodes after relabeling', dag.nodes)
        for g in block[1:]:
            # dag = nx.contracted_nodes(dag, block, g, self_loops=False)
            dag.contract_nodes([node_index(dag, block), node_index(dag, g)], block)

    # blocks_layers = list(map(sort_blocks_on_qregs, dag_to_layers(dag)))
    # blocks = reduce(add, blocks_layers)
    # return blocks
    return [dag[idx] for idx in rx.topological_sort(dag)]


def contract_1q_gates_on_dag(dag: rx.PyDiGraph) -> rx.PyDiGraph:
    """
    Aggregate all 1Q gates into neighboring 2Q/3Q gates
    After this pass, each node in DAG is a 2Q/3Q block (Circuit instance), including only one 2Q/3Q gate
    """
    dag = dag.copy()
    nodes_nonlocal_gates = [node for node in dag.nodes() if isinstance(node, Gate) and node.num_qregs > 1]

    # console.print('nodes_2q_gate:', nodes_2q_gate)
    # console.print('nodes_2q_gate (indices):', [node_index(dag, node) for node in nodes_2q_gate])

    for g in nodes_nonlocal_gates:
        # idx = node_index(dag, g)
        # console.print('node_index({}) = {}'.format(g, node_index(dag, g)))
        # console.print({idx: dag[idx] for idx in dag.node_indices()})
        # console.print('contracting {} with node index {}'.format(g, node_index(dag, g)), style='bold red')
        block = Circuit([g])
        # dag = nx.relabel_nodes(dag, {g: block})
        dag[node_index(dag, g)] = block
        while True:
            # predecessors_1q_gate = [g_nb for g_nb in list(dag.predecessors(block)) if
            #                         isinstance(g_nb, Gate) and g_nb.num_qregs == 1]
            # successors_1q_gate = [g_nb for g_nb in list(dag.successors(block)) if
            #                       isinstance(g_nb, Gate) and g_nb.num_qregs == 1]
            predecessors_1q = find_predecessors_by_node(dag, node_index(dag, block),
                                                        lambda node: isinstance(node, Gate) and node.num_qregs == 1)
            successors_1q = find_successors_by_node(dag, node_index(dag, block),
                                                    lambda node: isinstance(node, Gate) and node.num_qregs == 1)
            if not predecessors_1q and not successors_1q:  # there is no 1Q gate in the neighborhood
                break
            # console.print('predecessors_1q: {}'.format(predecessors_1q), style='blue')
            # console.print('successors_1q: {}'.format(successors_1q), style='blue')
            for g_pred in predecessors_1q:
                block.insert(0, g_pred)
                dag.contract_nodes([node_index(dag, block), node_index(dag, g_pred)], block)
                # dag[idx] =
            for g_succ in successors_1q:
                block.append(g_succ)
                dag.contract_nodes([node_index(dag, block), node_index(dag, g_succ)], block)
            # print(block.to_cirq())
            # console.print('Now node_index(block)={}'.format(node_index(dag, block)), style='bold red')

    # relabel node indices of dag according to their topological orders
    dag_contracted = rx.PyDiGraph(multigraph=False)
    # indices_topo = (rx.topological_sort(dag))
    indices_to_new = {idx: i for i, idx in enumerate(rx.topological_sort(dag))}
    # console.print(rx.topological_sort(dag))
    dag_contracted.add_nodes_from([dag[idx] for idx in indices_to_new.keys()])
    for u, v in dag.edge_list():
        dag_contracted.add_edge(indices_to_new[u], indices_to_new[v], {})
        # node_u = dag[u]
        # node_v = dag[v]
        # dag_contracted.add_edge(node_index(dag_contracted, node_u), node_index(dag_contracted, node_v), {})

    return dag_contracted


# def contract_1q_gates_in_circuit(circ: Circuit) -> Circuit:
#     """
#     Agtgreate all 1Q gates into neighboring 2Q gates
#     """
#     dag = circ.to_dag() # each node is a Gate instance
#     dag = contract_1q_gates_on_dag(dag) # each node is a 2-qubit Circuit instance
#     blocks = list(dag.nodes)
#     gates_2q = [UnivGate(blk.unitary()).on(blk.qubits) for blk in blocks]
#     dag = nx.relabel_nodes(dag, {blk: gate for blk, gate in zip(blocks, gates_2q)})
#     return Circuit(list(nx.topological_sort(dag)))


def blocks_to_dag(blocks: List[Circuit], nested: bool = False) -> rx.PyDiGraph:
    """
    Convert blocks (a list of Circuit instances) to a DAG in rustworkx.

    Args:
        blocks: a list of Circuit instances
        nested: default False; if True, each node is also a DAG converted from a Circuit
    """
    blocks = blocks.copy()
    dag = rx.PyDiGraph(multigraph=False)

    block_dag_map = {blk: blk.to_dag() for blk in blocks}
    if nested:
        # dag.add_nodes_from([blk.to_dag() for blk in blocks])
        dag.add_nodes_from(list(block_dag_map.values()))
    else:
        dag.add_nodes_from(blocks)

    while blocks:
        blk = blocks.pop(0)
        qubits = set(blk.qubits)
        for blk_opt in blocks:  # traverse subsequent optional blocks
            qubits_opt = set(blk_opt.qubits)
            if dependent_qubits := qubits_opt & qubits:
                if nested:
                    dag.add_edge(
                        node_index(dag, block_dag_map[blk]),
                        node_index(dag, block_dag_map[blk_opt]),
                        {'qubits': list(dependent_qubits)})
                else:
                    dag.add_edge(
                        node_index(dag, blk),
                        node_index(dag, blk_opt),
                        {'qubits': list(dependent_qubits)})
                qubits -= qubits_opt
            if not qubits:
                break

    return dag




def compact_dag(dag: rx.PyDiGraph, num_2q_threshold: int = 4) -> rx.PyDiGraph:
    """
    Compact a nested DAG (each of its node is also a DAG converted from a Circuit).
    ---
    Opportunity 1: difference between greedy_partition and quick_partition
    Opportunity 2: numerically equivalence when exchanging order of two SU(4) gates
    """
    return compact_dag_2(compact_dag_1(dag, num_2q_threshold), num_2q_threshold)


def compact_dag_old(dag: rx.PyDiGraph, num_2q_threshold: int = 4) -> rx.PyDiGraph:
    """
    Compact a nested DAG (each of its node is also a DAG converted from a Circuit).
    ---
    Opportunity 1: difference between greedy_partition and quick_partition
    Opportunity 2: numerically equivalence when exchanging order of two SU(4) gates
    """
    # Typically, each node is a 3-qubit block, including only 2-qubit gates
    dag = dag.copy()
    circ = Circuit(reduce(add, [blk.gates for blk in dag.nodes()]))
    gate_dependency = circ.to_dag()

    ##########################################################################
    # Opportunity 1
    # nx.set_node_attributes(dag, False, 'resolved')  # this attribute will be True for all nodes in the end
    # nx.set_node_attributes(dag,
    #                        {node: True for node in dag.nodes if node.num_nonlocal_gates < num_2q_threshold},
    #                        'resolved')
    # dag.attrs = {idx: 'resolved' for idx in dag.node_indices()}
    dag.attrs = {
        'resolved': {node: (True if node.num_nonlocal_gates < num_2q_threshold else False) for node in dag.nodes()}}

    # while unresolved_nodes := [node for node in dag.nodes if not dag.nodes[node]['resolved']]:
    while unresolved_nodes := [node for node in dag.nodes() if not dag.attrs['resolved'][node]]:
        node = sorted(unresolved_nodes, key=lambda node: node.num_nonlocal_gates, reverse=True)[0]
        node_idx = node_index(dag, node)
        if pred_nodes := dag.predecessors(node_idx):
            for pred_node in pred_nodes:
                if set(pred_node[-1].qregs).issubset(set(node.qubits)) and all(
                        [(g in node) for g in gate_dependency.successors(node_index(gate_dependency, pred_node[-1]))]):
                    # although the first condition is not necessary, it can help improve running performance
                    node.prepend(pred_node.pop())
                    if pred_node.num_nonlocal_gates < num_2q_threshold:
                        # dag.nodes[pred_node]['resolved'] = True
                        dag.attrs['resolved'][pred_node] = True
        if succ_nodes := dag.successors(node_idx):
            for succ_node in succ_nodes:
                if set(succ_node[0].qregs).issubset(set(node.qubits)) and all(
                        [(pred in node) for pred in
                         gate_dependency.predecessors(node_index(gate_dependency, succ_node[0]))]):
                    # although the first condition is not necessary, it can help improve running performance
                    node.append(succ_node.pop(0))
                    if succ_node.num_nonlocal_gates < num_2q_threshold:
                        # dag.nodes[succ_nodes]['resolved'] = True
                        dag.attrs['resolved'][succ_node] = True
        # dag.nodes[node]['resolved'] = True
        assert node_idx == node_index(dag, node), 'node_idx: {}, node_index: {}'.format(node_idx, node_index(dag, node))
        dag.attrs['resolved'][node] = True
        # nx.set_node_attributes(dag,
        #                        {node: True for node in dag.nodes if num_2q_of_dag(node) < num_2q_threshold},
        #                        'resolved')
    ##########################################################################
    # Opportunity 2
    # ! in fact, the opportunity-2 pass can be iteratively applied
    dag.attrs = {
        'resolved': {node: (True if node.num_nonlocal_gates < num_2q_threshold else False) for node in dag.nodes()}}
    while unresolved_nodes := [node for node in dag.nodes() if not dag.attrs['resolved'][node]]:
        node = sorted(unresolved_nodes, key=lambda node: node.num_nonlocal_gates, reverse=True)[0]
        node_idx = node_index(dag, node)
        if pred_nodes := [node for node in dag.predecessors(node_idx) if
                          node.num_nonlocal_gates <= num_2q_threshold + 1]:
            for pred_node in pred_nodes:
                if pred_node.num_gates > 1 and set(pred_node[-2].qregs).issubset(set(node.qubits)):
                    # although the "subset" condition is not necessary, it can help improve running performance
                    circ_tmp = circ.clone()
                    circ_tmp.remove(pred_node[-1])
                    gate_dependency = circ_tmp.to_dag()
                    if all([(g in node) for g in
                            gate_dependency.successors(node_index(gate_dependency, pred_node[-2]))]):
                        if res := approximately_commutative(pred_node[-2], pred_node[-1]):
                            # u_g = Circuit(pred_node[-2:]).unitary()
                            # v_g = Circuit(res).unitary()
                            # u = (pred_node + node).unitary()
                            # console.print('res:', res)
                            circ.insert(circ.index(pred_node[-2]), res[0])
                            circ.remove(pred_node[-2])
                            circ.insert(circ.index(pred_node[-1]), res[1])
                            circ.remove(pred_node[-1])
                            pred_node.pop()
                            pred_node.pop()
                            pred_node.append(res[0])
                            node.prepend(res[1])
                            if pred_node.num_gates > 1 and set(pred_node[-2].tqs) == set(pred_node[-1].tqs):
                                # fuse two SU(4)s cause they act on the same qubits
                                qubits = sorted(pred_node[-1].tqs)
                                su4 = UnivGate(Circuit([pred_node[-2], pred_node[-1]]).unitary()).on(qubits)
                                circ.insert(circ.index(pred_node[-2]), su4)
                                circ.remove(pred_node[-2])
                                circ.remove(pred_node[-1])
                                pred_node.pop()
                                pred_node.pop()
                                pred_node.append(su4)
                            if pred_node.num_nonlocal_gates < num_2q_threshold:
                                # dag.nodes[pred_node]['resolved'] = True
                                dag.attrs['resolved'][pred_node] = True
        if succ_nodes := [node for node in dag.successors(node_idx) if node.num_nonlocal_gates <= num_2q_threshold + 1]:
            for succ_node in succ_nodes:
                if succ_node.num_gates > 1 and set(succ_node[1].qregs).issubset(set(node.qubits)):
                    # although the "subset" condition is not necessary, it can help improve running performance
                    circ_tmp = circ.clone()
                    circ_tmp.remove(succ_node[0])
                    gate_dependency = circ_tmp.to_dag()
                    if all([(pred in node) for pred in
                            gate_dependency.predecessors(node_index(gate_dependency, succ_node[1]))]):
                        if res := approximately_commutative(succ_node[0], succ_node[1]):
                            circ.insert(circ.index(succ_node[0]), res[0])
                            circ.remove(succ_node[0])
                            circ.insert(circ.index(succ_node[1]), res[1])
                            circ.remove(succ_node[1])
                            succ_node.pop(0)
                            succ_node.pop(0)
                            succ_node.prepend(res[1])
                            node.append(res[0])
                            if succ_node.num_gates > 1 and set(succ_node[0].tqs) == set(succ_node[1].tqs):
                                # fuse two SU(4)s cause they act on the same qubits
                                qubits = sorted(succ_node[0].tqs)
                                su4 = UnivGate(Circuit([succ_node[0], succ_node[1]]).unitary()).on(qubits)
                                circ.insert(circ.index(succ_node[0]), su4)
                                circ.remove(succ_node[0])
                                circ.remove(succ_node[1])
                                succ_node.pop(0)
                                succ_node.pop(0)
                                succ_node.prepend(su4)
                            if succ_node.num_nonlocal_gates < num_2q_threshold:
                                dag.attrs['resolved'][succ_node] = True
        dag.attrs['resolved'][node] = True

    return dag


def peel_first_1q_gates(circ: Circuit) -> Tuple[Circuit, List[Gate]]:
    circ_nl = circ.clone()
    first_1q_gates = []
    while first_layer_1q := [g for g in obtain_front_layer(circ_nl) if g.num_qregs == 1]:
        first_1q_gates.extend(first_layer_1q)
        for g_1q in first_layer_1q:
            circ_nl.remove(g_1q)
    return circ_nl, first_1q_gates


def peel_last_1q_gates(circ: Circuit) -> Tuple[Circuit, List[Gate]]:
    circ_nl_rev = Circuit(circ.gates[::-1])
    last_1q_gates = []
    while last_layer_1q := [g for g in obtain_front_layer(circ_nl_rev) if g.num_qregs == 1]:
        last_1q_gates.extend(last_layer_1q)
        for g_1q in last_layer_1q:
            circ_nl_rev.remove(g_1q)
    last_1q_gates.reverse()
    return Circuit(circ_nl_rev.gates[::-1]), last_1q_gates


def peel_first_and_last_1q_gates(circ: Circuit) -> Tuple[Circuit, List[Gate], List[Gate]]:
    circ_nl, first_1q_gates = peel_first_1q_gates(circ)
    circ_nl, last_1q_gates = peel_last_1q_gates(circ_nl)
    return circ_nl, first_1q_gates, last_1q_gates


def compact_dag_1(dag: rx.PyDiGraph, num_2q_threshold: int = 4) -> rx.PyDiGraph:
    """
    Compact a nested DAG (each of its node is also a DAG converted from a Circuit).
    ---
    Opportunity 1: difference between greedy_partition and quick_partition
    """
    # Typically, each node is a 3-qubit block, including only 2-qubit gates
    dag = dag.copy()
    circ = Circuit(reduce(add, [blk.gates for blk in dag.nodes()]))
    gate_dependency = circ.to_dag()

    ##########################################################################
    # Opportunity 1
    # nx.set_node_attributes(dag, False, 'resolved')  # this attribute will be True for all nodes in the end
    # nx.set_node_attributes(dag,
    #                        {node: True for node in dag.nodes if node.num_nonlocal_gates < num_2q_threshold},
    #                        'resolved')
    # dag.attrs = {idx: 'resolved' for idx in dag.node_indices()}
    dag.attrs = {
        'resolved': {node: (True if node.num_nonlocal_gates < num_2q_threshold else False) for node in dag.nodes()}}

    # while unresolved_nodes := [node for node in dag.nodes if not dag.nodes[node]['resolved']]:
    while unresolved_nodes := [node for node in dag.nodes() if not dag.attrs['resolved'][node]]:
        node = sorted(unresolved_nodes, key=lambda node: node.num_nonlocal_gates, reverse=True)[0]
        node_idx = node_index(dag, node)
        if pred_nodes := dag.predecessors(node_idx):
            for pred_node in pred_nodes:
                if set(pred_node[-1].qregs).issubset(set(node.qubits)) and all(
                        [(g in node) for g in gate_dependency.successors(node_index(gate_dependency, pred_node[-1]))]):
                    # although the first condition is not necessary, it can help improve running performance
                    node.prepend(pred_node.pop())
                    if pred_node.num_nonlocal_gates < num_2q_threshold:
                        # dag.nodes[pred_node]['resolved'] = True
                        dag.attrs['resolved'][pred_node] = True
        if succ_nodes := dag.successors(node_idx):
            for succ_node in succ_nodes:
                if set(succ_node[0].qregs).issubset(set(node.qubits)) and all(
                        [(pred in node) for pred in
                         gate_dependency.predecessors(node_index(gate_dependency, succ_node[0]))]):
                    # although the first condition is not necessary, it can help improve running performance
                    node.append(succ_node.pop(0))
                    if succ_node.num_nonlocal_gates < num_2q_threshold:
                        # dag.nodes[succ_nodes]['resolved'] = True
                        dag.attrs['resolved'][succ_node] = True
        # dag.nodes[node]['resolved'] = True
        assert node_idx == node_index(dag, node), 'node_idx: {}, node_index: {}'.format(node_idx, node_index(dag, node))
        dag.attrs['resolved'][node] = True

    return dag


def compact_dag_2(dag: rx.PyDiGraph, num_2q_threshold: int = 4) -> rx.PyDiGraph:
    """
    Compact a nested DAG (each of its node is also a DAG converted from a Circuit).
    ---
    Opportunity 2: numerically equivalence when exchanging order of two SU(4) gates
    """
    # Typically, each node is a 3-qubit block, including only 2-qubit gates
    dag = dag.copy()
    circ = Circuit(reduce(add, [blk.gates for blk in dag.nodes()]))
    gate_dependency = circ.to_dag()

    ##########################################################################
    # Opportunity 2
    # ! in fact, the opportunity-2 pass can be iteratively applied
    dag.attrs = {
        'resolved': {node: (True if node.num_nonlocal_gates < num_2q_threshold else False) for node in dag.nodes()}}
    while unresolved_nodes := [node for node in dag.nodes() if not dag.attrs['resolved'][node]]:
        node = sorted(unresolved_nodes, key=lambda node: node.num_nonlocal_gates, reverse=True)[0]
        node_idx = node_index(dag, node)
        if pred_nodes := [node for node in dag.predecessors(node_idx) if
                          node.num_nonlocal_gates <= num_2q_threshold + 1]:
            for pred_node in pred_nodes:
                if pred_node.num_gates > 1 and set(pred_node[-2].qregs).issubset(set(node.qubits)):
                    # although the "subset" condition is not necessary, it can help improve running performance
                    circ_tmp = circ.clone()
                    circ_tmp.remove(pred_node[-1])
                    gate_dependency = circ_tmp.to_dag()
                    if all([(g in node) for g in
                            gate_dependency.successors(node_index(gate_dependency, pred_node[-2]))]):
                        if res := approximately_commutative(pred_node[-2], pred_node[-1]):
                            # u_g = Circuit(pred_node[-2:]).unitary()
                            # v_g = Circuit(res).unitary()
                            # u = (pred_node + node).unitary()
                            # console.print('res:', res)
                            circ.insert(circ.index(pred_node[-2]), res[0])
                            circ.remove(pred_node[-2])
                            circ.insert(circ.index(pred_node[-1]), res[1])
                            circ.remove(pred_node[-1])
                            pred_node.pop()
                            pred_node.pop()
                            pred_node.append(res[0])
                            node.prepend(res[1])
                            if pred_node.num_gates > 1 and set(pred_node[-2].tqs) == set(pred_node[-1].tqs):
                                # fuse two SU(4)s cause they act on the same qubits
                                qubits = sorted(pred_node[-1].tqs)
                                su4 = UnivGate(Circuit([pred_node[-2], pred_node[-1]]).unitary()).on(qubits)
                                circ.insert(circ.index(pred_node[-2]), su4)
                                circ.remove(pred_node[-2])
                                circ.remove(pred_node[-1])
                                pred_node.pop()
                                pred_node.pop()
                                pred_node.append(su4)
                            if pred_node.num_nonlocal_gates < num_2q_threshold:
                                # dag.nodes[pred_node]['resolved'] = True
                                dag.attrs['resolved'][pred_node] = True
        if succ_nodes := [node for node in dag.successors(node_idx) if node.num_nonlocal_gates <= num_2q_threshold + 1]:
            for succ_node in succ_nodes:
                if succ_node.num_gates > 1 and set(succ_node[1].qregs).issubset(set(node.qubits)):
                    # although the "subset" condition is not necessary, it can help improve running performance
                    circ_tmp = circ.clone()
                    circ_tmp.remove(succ_node[0])
                    gate_dependency = circ_tmp.to_dag()
                    if all([(pred in node) for pred in
                            gate_dependency.predecessors(node_index(gate_dependency, succ_node[1]))]):
                        if res := approximately_commutative(succ_node[0], succ_node[1]):
                            circ.insert(circ.index(succ_node[0]), res[0])
                            circ.remove(succ_node[0])
                            circ.insert(circ.index(succ_node[1]), res[1])
                            circ.remove(succ_node[1])
                            succ_node.pop(0)
                            succ_node.pop(0)
                            succ_node.prepend(res[1])
                            node.append(res[0])
                            if succ_node.num_gates > 1 and set(succ_node[0].tqs) == set(succ_node[1].tqs):
                                # fuse two SU(4)s cause they act on the same qubits
                                qubits = sorted(succ_node[0].tqs)
                                su4 = UnivGate(Circuit([succ_node[0], succ_node[1]]).unitary()).on(qubits)
                                circ.insert(circ.index(succ_node[0]), su4)
                                circ.remove(succ_node[0])
                                circ.remove(succ_node[1])
                                succ_node.pop(0)
                                succ_node.pop(0)
                                succ_node.prepend(su4)
                            if succ_node.num_nonlocal_gates < num_2q_threshold:
                                dag.attrs['resolved'][succ_node] = True
        dag.attrs['resolved'][node] = True

    return dag


def num_nontrivial_swaps(circ: Circuit) -> int:
    from phoenix.basic.gates import SWAPGate
    from phoenix.utils.graphs import node_index
    """Count the number of SWAP gates that cannot be absorbed by their adjacent 2Q gates"""
    dag = Circuit([g for g in circ if g.num_qregs > 1]).to_dag()
    swaps = [g for g in circ if isinstance(g, SWAPGate)]
    num_swaps = len(swaps)
    for swap in swaps:
        if ((successors := dag.successors(node_index(dag, swap)))
                and (len(successors) == 1)
                and (set(successors[0].qregs) == set(swap.qregs))):
            num_swaps -= 1
            continue
        if ((predecessors := dag.predecessors(node_index(dag, swap)))
                and (len(predecessors) == 1)
                and (set(predecessors[0].qregs) == set(swap.qregs))):
            num_swaps -= 1
    return num_swaps
