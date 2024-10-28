from synthesis_sd import *
from utils.hardware import *
from synthesis_FT import assign_time_parameter
from functools import partial
from utils.scheduler import Scheduler
import random
import pdb

from qiskit import QuantumCircuit

def pauli_single_gates(qc, pauli_map, ps, left=True):
    if left == True:
        for i in range(len(ps)):
            if ps[i] == 'X':
                qc.u(np.pi/2, 0, np.pi, pauli_map[i])
            elif ps[i] == 'Y':
                qc.u(np.pi/2, -np.pi/2, np.pi/2, pauli_map[i])
    else:
        for i in range(len(ps)):
            if ps[i] == 'X':
                qc.u(np.pi/2, 0, np.pi, pauli_map[i])
            elif ps[i] == 'Y':
                qc.u(-np.pi/2, -np.pi/2, np.pi/2, pauli_map[i])

def synthesis_initial(pauli_layers, pauli_map=None, graph=None, qc=None, arch='manhattan'):
    assign_time_parameter(pauli_layers, 1)
    lnq = len(pauli_layers[0][0][0]) # logical qubits
    if graph == None:
        G, C = load_graph(arch, dist_comp=True) # G is adj, C is dist
        graph = pGraph(G, C)
    if pauli_map == None:
        pauli_map = dummy_qubit_mapping(graph, lnq)
        # pauli_map = [0, 1, 2, 3, 13, 5, 6, 7, 8, 9, 10, 11, 12, 4]
    else:
        add_pauli_map(graph, pauli_map)
    pnq = len(graph) # physical qubits
    if qc == None:
        qc = QuantumCircuit(pnq)
    return pauli_map, graph, qc

def try_block(n_qubits : int, block : list, scheduler : Scheduler, swap_coefficient : int):
    level = [-1 for i in range(n_qubits)]
    prior = ['' for i in range(n_qubits)]
    # level 0: always I
    # level 1: always X, Y or Z
    # level 2: not always the same pauli
    for pauli_string in block:
        for wire, pauli_op in enumerate(pauli_string.ps):
            if prior[wire] == '':
                if pauli_op == 'I':
                    level[wire] = 0
                    prior[wire] = 'I'
                elif pauli_op == 'X' or pauli_op == 'Y' or pauli_op == 'Z':
                    level[wire] = 1
                    prior[wire] = pauli_op
                else:
                    raise Exception('None I, X, Y or Z character in ' + pauli_string.ps)
                continue
            
            if level[wire] == 2:
                continue
            
            if pauli_op == 'I':
                if level[wire] == 1:
                    level[wire] = 2
            elif pauli_op == 'X' or pauli_op == 'Y' or pauli_op == 'Z':
                if level[wire] == 0:
                    level[wire] = 2
                elif level[wire] == 1 and prior[wire] != pauli_op:
                    level[wire] = 2
            else:
                raise Exception('None I, X, Y or Z character in ' + pauli_string.ps)
        
    # assign level 1 qubits as flower_head
    # assign level 2 qubits as stalk
    flower_head = []
    stalk = []
    for i, l in enumerate(level):
        if l == 1:
            flower_head.append(i)
        elif l == 2:
            stalk.append(i)
    
    # do MST algorithm 3 times:
    # 1st time: connect flower_head
    # 2nd time: connect stalk
    # 3rd time: connect flower_head and stalk by only one edge
    
    centor = scheduler.find_centor(stalk)
    
    if stalk == []:
        stalk = flower_head[-1:]
        flower_head = flower_head[:-1]
        centor = stalk[0]
    root_tree_nodes, edges1 = scheduler.gather_root_tree(stalk, centor)
    edges2 = scheduler.gather_leaf_tree_bfs(flower_head, root_tree_nodes, len(block), use_bridge=False, swap_coefficient=swap_coefficient)
    find_root = True
    for pauli_string in block:
        if find_root == True:
            root = stalk[0]
            for i in stalk:
                if pauli_string.ps[i] != 'I':
                    root = i
                    break
            scheduler.Tree_init(edges1 + edges2, root)
        
        # the left side of a pauli string circuit
        scheduler.enable_cancel = True
        for i in flower_head + stalk:
            pauli = pauli_string.ps[i]
            if pauli == 'I':
                pass
            elif pauli == 'Z':
                pass
            elif pauli == 'X':
                scheduler.add_instruction('Logical_left_X', i)
            elif pauli == 'Y':
                scheduler.add_instruction('Logical_left_Y', i)
            else:
                raise Exception('Illegal pauli operator: ' + pauli)
        
        scheduler.tree.refresh()
        
        save_instructions = []
        for i in range(len(scheduler.tree.node_list)):
            node = scheduler.tree.node_list[i]
            if node.idx_after_swap < n_qubits and pauli_string.ps[node.idx_after_swap] == 'I':
                continue
            if node.parent_after_swap != -1:
                if node.parent_after_swap >= n_qubits or pauli_string.ps[node.parent_after_swap] != 'I':
                    # node.parent_after_swap >= n_qubits means it's a bridge
                    # node.parent_after_swap is not an 'I' means we just CX to it.
                    scheduler.add_instruction('Logical_CNOT', (node.idx_after_swap, node.parent_after_swap))
                    save_instructions.append(('Logical_CNOT', (node.idx_after_swap, node.parent_after_swap)))
                else:
                    # otherwise, the parent is an 'I' that we need to swap to go through.
                    scheduler.add_instruction('Logical_SWAP', (node.idx_after_swap, node.parent_after_swap))
                    save_instructions.append(('Logical_SWAP', (node.idx_after_swap, node.parent_after_swap)))
                    scheduler.tree.swap_two_nodes(node.parent_after_swap, node.idx_after_swap)
            else:
                scheduler.add_instruction('Logical_RZ', node.idx_after_swap)
        
        scheduler.clear_uncompiled_logical_instructions()
        # the right side of a pauli string circuit
        scheduler.enable_cancel = True
        for ir in reversed(save_instructions):
            scheduler.add_instruction(ir[0], ir[1])
        
        for i in reversed(flower_head + stalk):
            pauli = pauli_string.ps[i]
            if pauli == 'I' or pauli == 'Z':
                pass
            elif pauli == 'X':
                scheduler.add_instruction('Logical_right_X', i)
            elif pauli == 'Y':
                scheduler.add_instruction('Logical_right_Y', i)
            else:
                raise Exception('Illegal pauli operator: ' + pauli)
        scheduler.clear_uncompiled_logical_instructions()

def similarity(level1, level2):
    common = 0
    ls1 = 0
    ls2 = 0
    for l1, l2 in zip(level1, level2):
        if l1 == 1 and l2 == 1:
            common = common + 1
        if l1 == 1:
            ls1 = ls1 + 1
        if l2 == 1:
            ls2 = ls2 + 1
    return 0 if common == 0 else float(common) / (ls1 + ls2 - common)

def synthesis_lookahead_bfs(pauli_layers, pauli_map=None, graph=None, qc=None, arch='manhattan', use_bridge=False, swap_coefficient=3, k=10):
    pauli_map, graph, qc = synthesis_initial([[block] for block in pauli_layers], pauli_map, graph, qc, arch)
    scheduler = Scheduler(pauli_map, graph, qc)
    n_qubits = len(pauli_layers[0][0].ps)
    block_cnt = 0
    ps_cnt = 0
    
    for block in pauli_layers:
        block_cnt = block_cnt + 1
    print(block_cnt)
    
    level_list = []
    for block in pauli_layers:
        level = [-1 for i in range(n_qubits)]
        prior = ['' for i in range(n_qubits)]
        for pauli_string in block:
            for wire, pauli_op in enumerate(pauli_string.ps):
                if prior[wire] == '':
                    if pauli_op == 'I':
                        level[wire] = 0
                        prior[wire] = 'I'
                    elif pauli_op == 'X' or pauli_op == 'Y' or pauli_op == 'Z':
                        level[wire] = 1
                        prior[wire] = pauli_op
                    else:
                        raise Exception('None I, X, Y or Z character in ' + pauli_string.ps)
                
                if level[wire] == 2:
                    continue
                
                if pauli_op == 'I':
                    if level[wire] == 1:
                        level[wire] = 2
                elif pauli_op == 'X' or pauli_op == 'Y' or pauli_op == 'Z':
                    if level[wire] == 0:
                        level[wire] = 2
                    elif level[wire] == 1 and prior[wire] != pauli_op:
                        level[wire] = 2
                else:
                    raise Exception('None I, X, Y or Z character in ' + pauli_string.ps)
        
        level_list.append(level)

    last_level = None
    while len(pauli_layers) > 0:
        if last_level == None:
            selected_index = list(range(min(k, len(pauli_layers))))
        else:
            sorted_index = sorted(list(range(len(pauli_layers))), key=lambda i: -similarity(last_level, level_list[i]))
            selected_index = sorted_index[:k]
        
        cost_list = []
        for index in selected_index:
            block = pauli_layers[index]
            test_scheduler = Scheduler(None, None, None, from_other_scheduler=scheduler)
            try_block(n_qubits, block, test_scheduler, swap_coefficient=swap_coefficient)
            cost = test_scheduler.collect_CNOT_cost_in_one_block()
            cost_list.append((index, cost))
        sorted_cost_list = sorted(cost_list, key=lambda pair: pair[1])
        index = sorted_cost_list[0][0]
        
        try_block(n_qubits, pauli_layers[index], scheduler, swap_coefficient=swap_coefficient)
        del pauli_layers[index]
        last_level = level_list[index]
        del level_list[index]

    # debug(scheduler)
    
    return scheduler.qc, metrics(scheduler, n_qubits)

def metrics(scheduler, n_qubits):
    return {
        'n_qubits': n_qubits,
        'IR_total': scheduler.total_logical_instruction,
        'IR_remain': len(scheduler.instruction_list),
        'IR_cancel_ratio': (scheduler.total_logical_instruction - len(scheduler.instruction_list)) / scheduler.total_logical_instruction,
        'tetris_swap_count': scheduler.total_swap_cnt,
        'tetris_cx_count' : scheduler.total_cx_cnt,
        'tetris_bridge_count': scheduler.total_bridge_cnt,
    }

def debug(scheduler):
    # pdb.set_trace()
    total_cnot = 0
    for x, y, z in scheduler.record:
        total_cnot = total_cnot + z
    print(scheduler.record)
    print(total_cnot)
    print(scheduler.instruction_list)

