import pdb
import numpy as np
from benchmark.mypauli import pauliString
from utils.parallel_bl import *

def bridge_friendly_sort(n_qubits, blocks, level_list, not_sorted_qubits):
    length = len(blocks)
    num_Is = [0 for i in range(n_qubits)]
    for i in range(length): # scan each block
        for j, k in enumerate(level_list[i]): # in block[i], check the level k of each qubit j in that block
            if j in not_sorted_qubits and k == 0:
                num_Is[j] = num_Is[j] + 1
    if np.max(num_Is) == 0: # no need to sort
        return gate_count_oriented_scheduling(blocks)#[[block] for block in blocks]
    base_qubit = np.argmax(num_Is)

    new_blocks_front = []
    for i in range(length):
        if level_list[i][base_qubit] != 0:
            new_blocks_front.append(blocks[i])
    new_blocks_back = []
    level_list_back = []
    for i in range(length):
        if level_list[i][base_qubit] == 0:
            new_blocks_back.append(blocks[i])
            level_list_back.append(level_list[i])

    not_sorted_qubits.remove(base_qubit)
    
    result = gate_count_oriented_scheduling(new_blocks_front) + bridge_friendly_sort(n_qubits, new_blocks_back, level_list_back, not_sorted_qubits)
    return result
    
            

def bridge_friendly_block_scheduling(parr):
    n_qubits = len(parr[0][0]) # num_qubits
    # parr is a list of pauli_block
    # calculate the level for each pauli_block
    level_list = []
    for block in parr:
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
        level_list.append(level)
    
    return bridge_friendly_sort(n_qubits, parr, level_list, [i for i in range(n_qubits)])
