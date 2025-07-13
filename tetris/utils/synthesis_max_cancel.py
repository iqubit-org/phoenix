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

def is_permutation_with_one_extra_element(list1, list2):
    if len(list1) != len(list2) + 1:
        return False, -1  # Lists have different lengths, so they can't meet the condition

    element_counts = {}
    
    # Count elements in list1
    for item in list1:
        element_counts[item] = element_counts.get(item, 0) + 1

    # Subtract elements from list2
    for item in list2:
        if item not in element_counts:
            return False, -1  # Found an element in list2 that's not in list1
        element_counts[item] -= 1
        if element_counts[item] < 0:
            return False, -1  # Element count is negative, indicating a mismatch

    # Check for exactly one extra element
    extra_element_count = sum(count for count in element_counts.values() if count == 1)
    if extra_element_count == 1:
        return True, sum(key for key, count in element_counts.items() if count == 1)
    return False, -1


def synthesis_max_cancel(pauli_layers, pauli_map=None, graph=None, qc=None, arch='manhattan', use_bridge=False):
    pauli_map, graph, qc = synthesis_initial([[block] for block in pauli_layers], pauli_map, graph, qc, arch)
    scheduler = Scheduler(pauli_map, graph, qc)
    n_qubits = len(pauli_layers[0][0].ps)
    rdy_for_bridge = [0 for i in range(n_qubits)]
    block_cnt = 0
    
    l_single_gates_cnt = 0
    ps_cnt = 0
    
    for block in pauli_layers:
        block_cnt = block_cnt + 1
        for pauli_string in block:
            for wire, pauli_op in enumerate(pauli_string.ps):
                if pauli_op == 'X' or pauli_op == 'Y' or pauli_op == 'Z':
                    rdy_for_bridge[wire] = block_cnt
    # print(block_cnt)
    # print(rdy_for_bridge)
    block_cnt = 0
    last_ps_common_part = []
    for block in pauli_layers:
        ps_cnt = ps_cnt + len(block)
        block_cnt = block_cnt + 1
        # bridgable == True means the data qubit is already measured and can be treat like an ancillary qubit
        for wire, r in enumerate(rdy_for_bridge):
            if block_cnt > r:
                scheduler.notify_ancilla(wire)
                # print(block_cnt)
        
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
        current_common_part = []
        
        for i, l in enumerate(level):
            if l == 1: # common part
                flower_head.append(i)
                current_common_part.append(i)
            elif l == 2:
                stalk.append(i)
        
        flag, element = is_permutation_with_one_extra_element(current_common_part, last_ps_common_part)
        if flag:
            # if true, means we can cancel across the block:
            last_ps_common_part.append(element)
        else:
            last_ps_common_part = current_common_part

            # the left side of a pauli string circuit
        scheduler.enable_cancel = True
        for i in flower_head:
            pauli = block[0].ps[i]
            if pauli == 'I':
                raise Exception('Illegal pauli operator I at common part: ' + pauli)
            elif pauli == 'Z':
                pass
            elif pauli == 'X':
                l_single_gates_cnt = l_single_gates_cnt + 1
                scheduler.add_instruction('Logical_left_X', i)
            elif pauli == 'Y':
                l_single_gates_cnt = l_single_gates_cnt + 1
                scheduler.add_instruction('Logical_left_Y', i)
            else:
                raise Exception('Illegal pauli operator: ' + pauli)
        
        for i in range(len(last_ps_common_part) - 1):
            common_1 = last_ps_common_part[i]
            common_2 = last_ps_common_part[i + 1]
            scheduler.add_instruction('Logical_CNOT', (common_1, common_2))
        
        if element != -1:
            stalk = stalk + [element]
        centor = scheduler.find_centor(stalk)
        
        root_tree_nodes, edges1 = scheduler.gather_root_tree(stalk, centor)
        find_root = True

        for pauli_string in block:
            if find_root == True:
                root = stalk[0]
                for i in stalk:
                    if pauli_string.ps[i] != 'I':
                        root = i
                        break
                scheduler.Tree_init(edges1, root)
            
            # the left side of a pauli string circuit
            scheduler.enable_cancel = True
            for i in stalk:
                if i == element:
                    continue
                pauli = pauli_string.ps[i]
                if pauli == 'I':
                    pass
                elif pauli == 'Z':
                    pass
                elif pauli == 'X':
                    l_single_gates_cnt = l_single_gates_cnt + 2
                    scheduler.add_instruction('Logical_left_X', i)
                elif pauli == 'Y':
                    l_single_gates_cnt = l_single_gates_cnt + 2
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
            scheduler.enable_cancel = False
            for ir in reversed(save_instructions):
                scheduler.add_instruction(ir[0], ir[1])
            
            for i in reversed(stalk):
                if i == element:
                    continue
                pauli = pauli_string.ps[i]
                if pauli == 'I' or pauli == 'Z':
                    pass
                elif pauli == 'X':
                    scheduler.add_instruction('Logical_right_X', i)
                elif pauli == 'Y':
                    scheduler.add_instruction('Logical_right_Y', i)
                else:
                    raise Exception('Illegal pauli operator: ' + pauli)
    

        for i in reversed(range(len(last_ps_common_part) - 1)):
            common_1 = last_ps_common_part[i]
            common_2 = last_ps_common_part[i + 1]
            scheduler.add_instruction('Logical_CNOT', (common_1, common_2))
        
        for i in reversed(flower_head):
            pauli = block[0].ps[i]
            if pauli == 'I':
                raise Exception('Illegal pauli operator I at common part: ' + pauli)
            elif pauli == 'Z':
                pass
            elif pauli == 'X':
                l_single_gates_cnt = l_single_gates_cnt + 1
                scheduler.add_instruction('Logical_right_X', i)
            elif pauli == 'Y':
                l_single_gates_cnt = l_single_gates_cnt + 1
                scheduler.add_instruction('Logical_right_Y', i)
            else:
                raise Exception('Illegal pauli operator: ' + pauli)
    
    # print(scheduler.pauli_map)
    scheduler.clear_uncompiled_logical_instructions()

    # debug(scheduler)
    
    return scheduler.qc, metrics(scheduler, n_qubits, ps_cnt)

def metrics(scheduler, n_qubits, ps_cnt):
    return {
        'n_qubits': n_qubits,
        'IR_total': scheduler.total_logical_instruction,
        'IR_remain': len(scheduler.instruction_list),
        'IR_cancel_ratio': (scheduler.total_logical_instruction - len(scheduler.instruction_list)) / scheduler.total_logical_instruction,
        'tetris_swap_count': scheduler.total_swap_cnt,
        'tetris_cx_count' : scheduler.total_cx_cnt,
        'tetris_bridge_count': scheduler.total_bridge_cnt,
        'pauli string count': ps_cnt,
    }

def debug(scheduler):
    # pdb.set_trace()
    total_cnot = 0
    for x, y, z in scheduler.record:
        total_cnot = total_cnot + z
    print(scheduler.record)
    print(total_cnot)
    print(scheduler.instruction_list)

