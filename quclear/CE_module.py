'''Copyright © 2025 UChicago Argonne, LLC All right reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://github.com/revilooliver/QuCLEAR/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import numpy as np
import re
import qiskit
import math
from typing import List
from itertools import product
from collections import defaultdict, OrderedDict
from itertools import combinations

from qiskit import *
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli as convert_Pauli_op
from qiskit.quantum_info import Clifford

import logging

# Configure logging
# Configure module-level logger
logger = logging.getLogger(__name__)


def construct_qcc_circuit(entanglers: List[str], params: List[float], truncation=None, barrier=False):
    '''This function defines the exponential building block for hamtilbonian simulation. 
    
    Args:
        entanglers: list storing Pauli words for construction of qcc_circuit.
        params: parameters for the rotations in blocks
    Returns:
        qcc_circuit
    '''
    if truncation != None:
        if len(entanglers) > truncation:
            num_blocks = truncation
        else:
            num_blocks = len(entanglers)
    else:
        num_blocks = len(entanglers)
    
    num_qubits = len(entanglers[0])
    qcc_circuit = QuantumCircuit(num_qubits)
    for i in range(num_blocks):
        circuit = QuantumCircuit(num_qubits)
        key = entanglers[i]
        if all(c == 'I' for c in key):
            continue
        coupler_map = []
        # We first construct coupler_map according to the key.
        for j in range(num_qubits):
            if key[num_qubits-1-j] != 'I':
                coupler_map.append(j)
                
        # Then we construct the circuit.
        if len(coupler_map) == 1:
            # there is no CNOT gate.
            c = coupler_map[0]
            if key[num_qubits-1-c] == 'X':
                circuit.h(c)
                circuit.rz(params[i], c)
                circuit.h(c)
            elif key[num_qubits-1-c] == 'Y': 
                circuit.sdg(c)
                circuit.h(c)
                circuit.rz(params[i], c)
                circuit.h(c)
                circuit.s(c)
            elif key[num_qubits-1-c] == 'Z':
                circuit.rz(params[i], c)
                
            qcc_circuit.compose(circuit, inplace=True)
        else:
            # Here we would need CNOT gate.
            for j in coupler_map:
                if key[num_qubits-1-j] == 'X':
                    circuit.h(j)
                elif key[num_qubits-1-j] == 'Y':
                    circuit.sdg(j)
                    circuit.h(j)
                    
            for j in range(len(coupler_map) - 1):
                circuit.cx(coupler_map[j], coupler_map[j+1])
                
            param_gate = QuantumCircuit(num_qubits)
            param_gate.rz(params[i], coupler_map[-1])
            
            qcc_circuit.compose(circuit, inplace=True)
            qcc_circuit.compose(param_gate, inplace=True)
            qcc_circuit.compose(circuit.inverse(), inplace=True)
        if barrier is True:
            qcc_circuit.barrier()
    
    return qcc_circuit


def construct_extracted_subcircuit(entangler: str, param: float):
    '''This function defines the extracted building block for hamtilbonian simulation. 
    
    Args:
        entangler: Pauli words for construction of optimized building block
        param: parameter associated with the building block
    Returns:
        qcc_subcircuit
    '''
    
    num_qubits = len(entangler)
    circuit = QuantumCircuit(num_qubits)
    key = entangler
    coupler_map = []
    # We first construct coupler_map according to the key.
    for j in range(num_qubits):
        if key[num_qubits-1-j] != 'I':
            coupler_map.append(j)
            
    # Then we construct the circuit.
    if len(coupler_map) == 1:
        # there is no CNOT gate.
        c = coupler_map[0]
        if key[num_qubits-1-c] == 'X':
            circuit.h(c)
            circuit.rz(param, c)
            # circuit.h(c)
        elif key[num_qubits-1-c] == 'Y':
            circuit.sdg(c)
            circuit.h(c)
            circuit.rz(param, c)
            # circuit.h(c)
            # circuit.s(c)
        elif key[num_qubits-1-c] == 'Z':
            circuit.rz(param, c)
    else:
        # Here we would need CNOT gate.
        for j in coupler_map:
            if key[num_qubits-1-j] == 'X':
                circuit.h(j)
            elif key[num_qubits-1-j] == 'Y':
                circuit.sdg(j)
                circuit.h(j)
                
        for j in range(len(coupler_map) - 1):
            circuit.cx(coupler_map[j], coupler_map[j+1])
            
        param_gate = QuantumCircuit(num_qubits)
        param_gate.rz(param, coupler_map[-1])
        
        # circuit.compose(circuit, inplace=True)
        circuit.compose(param_gate, inplace=True)
    
    return circuit

def construct_Clifford_subcircuit(entangler: str):
    '''This function defines the clifford subcircuit for hamtilbonian simulation. 
    
    Args:
        entangler: Pauli words for construction of optimized building block
    Returns:
        clifford_subcircuit
    '''
    
    num_qubits = len(entangler)
    circuit = QuantumCircuit(num_qubits)
    key = entangler
    coupler_map = []
    # We first construct coupler_map according to the key.
    for j in range(num_qubits):
        if key[num_qubits-1-j] != 'I':
            coupler_map.append(j)
            
    # Then we construct the circuit.
    if len(coupler_map) == 1:
        # there is no CNOT gate.
        c = coupler_map[0]
        if key[num_qubits-1-c] == 'X':
            circuit.h(c)
        elif key[num_qubits-1-c] == 'Y':
            circuit.h(c)
            circuit.s(c)
    else:
        # Here we would need CNOT gate.
        for j in range(len(coupler_map)-2, -1, -1):
            # print(len(coupler_map), j, j+1)
            circuit.cx(coupler_map[j], coupler_map[j+1])
            
        for j in coupler_map:
            if key[num_qubits-1-j] == 'X':
                circuit.h(j)
            elif key[num_qubits-1-j] == 'Y':
                circuit.h(j)
                circuit.s(j)
    # clifford_subcircuit = circuit.inverse()
    
    return circuit

def extract_pauli_string(s):
    # Check for specific prefixes and remove them
    if s.startswith("-i"):
        return s[2:]
    elif s.startswith("-"):
        return s[1:]
    elif s.startswith("i"):
        return s[1:]
    else:
        return s
    
def construct_sq_subcircuit(entangler: str):
    '''This function defines the single qubit clifford subcircuit for hamtilbonian simulation. 
    
    Args:
        entangler: Pauli words for construction of optimized building block
    Returns:
        clifford_subcircuit
    '''
    # entangler = extract_pauli_string(entangler)
    num_qubits = len(entangler)
    circuit = QuantumCircuit(num_qubits)
    key = entangler
    # We first construct coupler_map according to the key.
    for j in range(num_qubits):
        if key[num_qubits - 1 - j] == 'X':
            circuit.h(j)
        elif key[num_qubits - 1 - j] == 'Y':
            circuit.h(j)
            circuit.s(j)
    return circuit


def push_sq_pauli(entangler: str, current_pauli: str):
    '''This function generates the pauli string after pushing the single qubit gates from current_pauli 

    Args:
        entangler: the string being pushed through
        current_pauli: the string with single qubit gates
    Returns:
        pauli_withsign
    '''
    X_dict = {
        #pushing H gate through the following paulis
        "X": [1, "Z"],
        "Y": [-1, "Y"],
        "Z": [1, "X"],
        "I": [1, "I"],
    }

    Y_dict = {
        #pushing S, then H gate through the paulis
        "X": [1, "Y"],  #"S": [1, "Y"], H: [-1, Y]
        "Y": [1, "Z"],  #"S": [-1, "X"], H: [1, Z]
        "Z": [1, "X"], # "S": [1, "Z"],  "H": [1, "X"],
        "I": [1, "I"],
    }
    updated_sign = 1
    updated_pauli = ""
    for idx in range(len(current_pauli)):
        sq_char = current_pauli[idx]
        pushed_char = entangler[idx]
        if sq_char == 'X':
            #we should push a single qubit H through
            sign, char = X_dict[pushed_char]
            updated_sign = updated_sign * sign
            updated_pauli += char
        elif sq_char == 'Y':
            sign, char = Y_dict[pushed_char]
            updated_sign = updated_sign * sign
            updated_pauli += char
        else:
            updated_pauli += pushed_char #the sign remains unchanged
    return str(updated_sign), updated_pauli

#this function generates the best CNOT tree structure given two pauli strings, maximize the cancellation
#TODO: these is a special case for the base entangler only contains one non identity string.
def find_single_tree(base_entangler_inv: str, match_entangler_inv: str):

    '''This function generates the best CNOT tree circuit for the base_entangler, maximizing the minimization of match_entangler.

    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        match_entangler: the target entangler that we are matching and minimizing
    Returns:
        CNOT_tree: the CNOT tree circuit
    '''
    num_qubits  = len(base_entangler_inv)
    I_list = []
    X_list = []
    Y_list = []
    Z_list = []
    N_list = []
    match_entangler = match_entangler_inv[::-1]
    base_entangler = base_entangler_inv[::-1]
    #first iterate over the base_entangler and find the non I index:
    for i in range(num_qubits):
        if base_entangler[i] != 'I':
            if match_entangler[i] == 'I':
                I_list.append(i)
            elif match_entangler[i] == 'X':
                X_list.append(i)
            elif match_entangler[i] == 'Y':
                Y_list.append(i)
            elif match_entangler[i] == 'Z':
                Z_list.append(i)
            else:
                raise Exception("Incorrect letter in entangler spotted", match_entangler)
        else:
            N_list.append(i)
    #Based on the lists, construct the quantum circuit:
    root_list = []
    qc = QuantumCircuit(num_qubits)
    I_root = X_root = Y_root = Z_root = N_root = final_root = -1
    #iterate over the I list:
    if len(I_list) == 1:
        I_root = I_list[0]
    elif len(I_list) > 1:
        for i_idx in range(len(I_list) - 1):
            qc.cx(I_list[i_idx], I_list[i_idx + 1])
        I_root = I_list[-1]
    #iterate over the Z list:
    if len(Z_list) == 1:
        Z_root = Z_list[0]
    elif len(Z_list) > 1:
        for z_idx in range(len(Z_list) - 1):
            qc.cx(Z_list[z_idx], Z_list[z_idx + 1])
        Z_root = Z_list[-1]
    #iterate over the X list:
    if len(X_list) == 1:
        X_root = X_list[0]
    elif len(X_list) > 1:
        for x_idx in range(len(X_list) - 1):
            qc.cx(X_list[x_idx], X_list[x_idx + 1])
        X_root = X_list[-1]
    #iterate over the Y list:
    if len(Y_list) == 1:
        Y_root = Y_list[0]
    elif len(Y_list) > 1:
        for y_idx in range(len(Y_list) - 1):
            qc.cx(Y_list[y_idx], Y_list[y_idx + 1])
        Y_root = Y_list[-1]

    # Function to connect the roots based on priority and set the final root
    def connect_roots(qc, root_name, roots_dict, priorities):
        root_value = roots_dict[root_name]
        if root_value > -1:
            for other_root_name in priorities:
                if roots_dict[other_root_name] > -1:
                    qc.cx(root_value, roots_dict[other_root_name])
                    # roots_dict[other_root_name] = -1
                    roots_dict[root_name] = -1
                    return roots_dict[other_root_name]
        return -1

    # Initialize the root dictionary
    roots_dict = {
        "Z_root": Z_root,
        "I_root": I_root,
        "Y_root": Y_root,
        "X_root": X_root,
        "N_root": N_root
    }

    final_root = None

    # Connection priorities
    priority_connections = {
        "Z_root": ["Y_root", "X_root", "I_root"],
        "I_root": ["X_root", "Y_root"],
        "Y_root": ["X_root"],
        # "X_root": ["N_root"]
    }

    # Connect roots based on priority
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if root_name in priority_connections:
            last_connected = connect_roots(qc, root_name, roots_dict, priority_connections[root_name])
            if last_connected != -1:
                final_root = last_connected

    # If any roots are still unconnected, set final_root
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if roots_dict[root_name] > -1:
            final_root = roots_dict[root_name]
            # print("final, root, rootes_dict[root_name]", final_root, roots_dict[root_name])


    # Ensure final_root is set correctly if not already set
    if final_root is None:
        #final_root = X_root
        raise Exception("final root not set")


    return final_root, qc

# #TODO: these is a special case for the base entangler only contains one non identity string.
def find_leaves(sorted_entanglers_params_inv: List[List[str]], curr_pauli, updated_paulis, qc_tree, tree_list, commute_idx: int, pauli_idx: int, append_clifford, depth = 0, max_depth = 800):
    X_leaves = []
    Y_leaves = []
    Z_leaves = []
    I_leaves = []
    if depth > max_depth:
        print(depth)
        for index in tree_list[1:]:
            qc_tree.cx(index, tree_list[0])
        final_root = tree_list[0]
        return final_root
    try:
        compare_pauli = updated_paulis[(commute_idx, pauli_idx)]
    except:
        initial_compare_pauli_inv = sorted_entanglers_params_inv[commute_idx][pauli_idx][0]
        temp_pauli = update_paulis([initial_compare_pauli_inv], clifford_circuit = append_clifford, parameters = False)
        pushed_sign, compare_pauli_inv = push_sq_pauli(entangler = temp_pauli[0], current_pauli = curr_pauli)
        compare_pauli = compare_pauli_inv[::-1]
        updated_paulis[(commute_idx, pauli_idx)] = compare_pauli

    #should update the compare pauli
    next_commute_idx, next_pauli_idx = gen_next_pauli_idx(sorted_entanglers_params_inv, commute_idx, pauli_idx)
    # counter += 1 First test not limiting lookahead size.
    I_root = X_root = Y_root = Z_root = final_root = -1
    for index in tree_list:
        if compare_pauli[index] == 'X':
            X_leaves.append(index)
        elif compare_pauli[index] == 'Y':
            Y_leaves.append(index)
        elif compare_pauli[index] == 'Z':
            Z_leaves.append(index)
        else:
            I_leaves.append(index)

    if next_commute_idx == None:
        #if there is no next pauli, connect all the leaves
        for index in range(len(tree_list) - 1):
            qc_tree.cx(tree_list[index], tree_list[index + 1])
        return tree_list[-1]
    if len(X_leaves) == 1:
        X_root = X_leaves[0]
    elif len(X_leaves) > 1:
        X_root = find_leaves(sorted_entanglers_params_inv, curr_pauli, updated_paulis, qc_tree, X_leaves, next_commute_idx, next_pauli_idx, append_clifford, depth + 1, max_depth)
    if len(Y_leaves) == 1:
        Y_root = Y_leaves[0]
    elif len(Y_leaves) > 1:
        Y_root = find_leaves(sorted_entanglers_params_inv, curr_pauli, updated_paulis, qc_tree, Y_leaves, next_commute_idx, next_pauli_idx, append_clifford, depth + 1, max_depth)
    if len(Z_leaves) == 1:
        Z_root = Z_leaves[0]
    elif len(Z_leaves) > 1:
        Z_root = find_leaves(sorted_entanglers_params_inv, curr_pauli, updated_paulis, qc_tree, Z_leaves, next_commute_idx, next_pauli_idx, append_clifford, depth + 1, max_depth)
    if len(I_leaves) == 1:
        I_root = I_leaves[0]
    elif len(I_leaves) > 1:
        I_root = find_leaves(sorted_entanglers_params_inv, curr_pauli, updated_paulis, qc_tree, I_leaves, next_commute_idx, next_pauli_idx, append_clifford, depth + 1, max_depth)
    #print(next_commute_idx, next_pauli_idx, X_root, Y_root, Z_root, I_root)
    #Connect all the roots together:
    # Function to connect the roots based on priority and set the final root
    def connect_roots(qc, root_name, roots_dict, priorities):
        root_value = roots_dict[root_name]
        if root_value > -1:
            for other_root_name in priorities:
                if roots_dict[other_root_name] > -1:
                    qc.cx(root_value, roots_dict[other_root_name])
                    # roots_dict[other_root_name] = -1
                    roots_dict[root_name] = -1
                    return roots_dict[other_root_name]
        return -1

    # Initialize the root dictionary
    roots_dict = {
        "Z_root": Z_root,
        "I_root": I_root,
        "Y_root": Y_root,
        "X_root": X_root,
    }

    final_root = None

    # Connection priorities
    priority_connections = {
        "Z_root": ["Y_root", "X_root", "I_root"],
        "I_root": ["X_root", "Y_root"],
        "Y_root": ["X_root"],
    }

    # Connect roots based on priority
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if root_name in priority_connections:
            last_connected = connect_roots(qc_tree, root_name, roots_dict, priority_connections[root_name])
            if last_connected != -1:
                final_root = last_connected

    # If any roots are still unconnected, set final_root
    for root_name in ["Z_root", "I_root", "Y_root", "X_root"]:
        if roots_dict[root_name] > -1:
            final_root = roots_dict[root_name]
            # print("final, root, rootes_dict[root_name]", final_root, roots_dict[root_name])


    # Ensure final_root is set correctly if not already set
    if final_root is None:
        #final_root = X_root
        raise Exception("final root not set")

    return final_root

def pauli_strings_commute(pauli_str1, pauli_str2):
    """
    Determine if two Pauli strings commute.
    
    :param pauli_str1: A string representing the first Pauli operator.
    :param pauli_str2: A string representing the second Pauli operator.
    :return: True if the Pauli strings commute, False otherwise.
    """
    if len(pauli_str1) != len(pauli_str2):
        raise ValueError("Pauli strings must be of the same length.", pauli_str1, pauli_str2)
    
    commute = True  # Assume they commute until proven otherwise
    
    anticommute_count = 0
    
    for i in range(len(pauli_str1)):
        if pauli_str1[i] != pauli_str2[i] and pauli_str1[i] != 'I' and pauli_str2[i] != 'I':
            # Found anti-commuting Pauli matrices
            commute = False
            anticommute_count += 1

    if anticommute_count % 2 == 0:
        commute = True
    
    return commute

def convert_commute_sets(Paulis: List[str], params: List[float]) -> List[List[str]]:
    '''This function converts the Paulis to commute sets.
    
    Args:
        Paulis: list storing Pauli words for construction of optimized circuit.
    Returns:
        paulis_sets: list of Paul sets that commutes
    '''
    current_set = []
    paulis_sets = []
    
    for idx in range(len(Paulis)):
        pauli = Paulis[idx]
        if all(c == 'I' for c in pauli):
            #This is a pauli string with all identity, should be removed
            continue
        if not current_set:
            current_set.append([pauli, params[idx]])
        else:
            can_be_added = True
            for current_pauli in current_set:
                if not pauli_strings_commute(current_pauli[0], pauli):
                    can_be_added = False
                    break
            if can_be_added:
                current_set.append([pauli, params[idx]])
            else:
                paulis_sets.append(current_set)
                current_set = [[pauli, params[idx]]]

    if current_set:
        paulis_sets.append(current_set)

    return paulis_sets

def pauli_weight(pauli: str) -> int:
    weight = 0
    weight_paulis = {'X', 'Y', 'Z'}
    
    for char in pauli:
        if char in weight_paulis:
            weight += 1
    
    return weight

def calculate_opt_weight_fast(base_entangler_inv: str, match_entangler_inv: str) -> int:
    '''This function generates the best CNOT tree circuit for the base_entangler and calculates the weight of match-entangler after optimization
    , maximizing the minimization of match_entangler.
    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        match_entangler: the target entangler that we are matching and minimizing
    Returns:
        opt_weight: the weight of the match_entangler_inv after optimization
    '''
    #TODO: this should be optimized to simply based on the two strings instead of based on circuits.
    #assume we have already pushed the single qubit gates
    sq_index, init_cx_tree = find_single_tree(base_entangler_inv, match_entangler_inv)
    extracted_cx_tree = init_cx_tree.inverse()
    optimized_paulis = update_paulis([match_entangler_inv], extracted_cx_tree, parameters = False)#construct_opt_pauli(entangler = match_entangler_inv, clifford_circuit = extracted_cx_tree)
    return pauli_weight(optimized_paulis[0]), optimized_paulis[0]


def gen_next_pauli_idx(sorted_entanglers_params, commute_idx, pauli_idx):
    # Current inner list
    current_list = sorted_entanglers_params[commute_idx]
    
    # If the inner index is not at the last element, increment the inner index
    if pauli_idx < len(current_list) - 1:
        return commute_idx, pauli_idx + 1
    else:
        # If the inner index is at the last element, move to the next outer list
        next_outer_index = commute_idx + 1
        
        # If the outer index is at the last list, wrap around to the first list
        if next_outer_index >= len(sorted_entanglers_params):
            next_outer_index = None
            
        return next_outer_index, 0
    
def split_pauli_string(pauli_string: str) -> tuple:
    # Define valid Pauli characters
    valid_characters = {'X', 'Y', 'Z', 'I'}
    
    # Initialize sign and pauli string
    sign = ''
    
    # Check for invalid starting signs and raise error if present
    if pauli_string.startswith('-i') or pauli_string.startswith('i'):
        raise ValueError("Pauli string cannot start with '-i' or 'i'.")
    
    # Check for valid starting sign '-'
    if pauli_string.startswith('-'):
        sign = '-1'
        pauli_string = pauli_string[1:]
    else:
        sign = '+1'
    
    # Check if the remaining string contains only valid Pauli characters
    if all(char in valid_characters for char in pauli_string):
        return sign, pauli_string
    else:
        raise ValueError("Invalid Pauli string.")
    

def update_paulis(Paulis_params_list: List[List[str]], clifford_circuit, parameters = False):
    #in this update, the sign doesn't matter
    evolved_Paulis_list = []
    if parameters == True:
        for Pauli_str, parmas in Paulis_params_list:
            Pauli_op = convert_Pauli_op(Pauli_str)
            evolved_Pauli_op = Pauli_op.evolve(clifford_circuit)
            evolved_sign, evolved_Pauli = split_pauli_string(evolved_Pauli_op.to_label())
            evolved_Paulis_list.append(evolved_Pauli)
    else:
        for Pauli_str in Paulis_params_list:
            Pauli_op = convert_Pauli_op(Pauli_str)
            evolved_Pauli_op = Pauli_op.evolve(clifford_circuit)
            evolved_sign, evolved_Pauli = split_pauli_string(evolved_Pauli_op.to_label())
            evolved_Paulis_list.append(evolved_Pauli)

    return evolved_Paulis_list

def update_pauli_param(Pauli_param: List[str], clifford_circuit):
    #in this update, the sign matters
    Pauli_str = Pauli_param[0]
    param = Pauli_param[1]
    Pauli_op = convert_Pauli_op(Pauli_str)
    evolved_Pauli_op = Pauli_op.evolve(clifford_circuit)
    evolved_sign, evolved_Pauli = split_pauli_string(evolved_Pauli_op.to_label())
    if evolved_sign == '-1':
        param = -param
    return [evolved_Pauli, param]


def find_best_pauli_index_threshold(base_entangler: str, commute_sets: List[List[str]], append_clifford, threshold = 1) -> int:
    '''This function finds the best pauli entangler in a set of commuting paulis.
    Args:
        base_entangler: the base entangler that searchers for the CNOT tree structure
        commute_entanglers: the target entangler that we are matching and minimizing
    Returns:
        ordered_entanglers: the ordered commute_entanglers after optimization
    '''
    ordered_entanglers = commute_sets.copy()
    min_weight = float('inf')
    min_index = None
    for idx, entangler in enumerate(ordered_entanglers):
        updated_entanglers = update_paulis(Paulis_params_list = [entangler], clifford_circuit = append_clifford, parameters = True)
        opt_sign, pushed_pauli = push_sq_pauli(entangler = updated_entanglers[0], current_pauli = base_entangler)
        weight, optimized_pauli_with_sign = calculate_opt_weight_fast(base_entangler, pushed_pauli)
        logger.debug('weight:%s, entangler_set:%s', weight, entangler)
        if weight <= threshold:
            return idx
        if weight < min_weight:
            min_weight = weight
            min_index = idx

    return min_index

def convert_order_sets(Paulis: List[str], params: List[float]) -> List[List[str]]:
    current_set = []
    paulis_sets = []
    for idx in range(len(Paulis)):
        #current_set.append([Paulis[idx], params[idx]])
        paulis_sets.append([[Paulis[idx], params[idx]]])
    return paulis_sets



def CE_recur_tree_seq(entanglers: List[str], params: List[float], barrier=False, threshold = 1):
    '''This function defines the optimized fully connected tree block for hamtiltonian simulation in commute list format, also considering lookahead
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        opt_qc, append_clifford, opt_paulis, opt_params
    '''
    commute_sets = convert_order_sets(Paulis= entanglers, params = params)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    append_clifford = Clifford(append_clifford)
    # opt_params = params.copy()
    sorted_entanglers_params = []

    #sort all the paulis based on their weight:#TODO: need to resolve when two cases have the same weight
    for commute_list in commute_sets:
        sorted_list = sorted(commute_list, key=lambda x: pauli_weight(x[0]))
        sorted_entanglers_params.append(sorted_list)

    logging.debug("start_sorted_list: %s", sorted_entanglers_params)
    next_pauli = 0
    # Iterate over all the lists of commuting entanglers that need optimization
    for commute_idx, sorted_list in enumerate(sorted_entanglers_params):
        # updated_commute_list = [sorted_list[0]]  # Initialize with the first Pauli string
        # remaining_commute_paulis = sorted_list[1:].copy()
        
        for pauli_idx in range(len(sorted_list)):
            #here we start process for each current pauli
            curr_pauli = sorted_entanglers_params[commute_idx][pauli_idx][0]
            curr_param = sorted_entanglers_params[commute_idx][pauli_idx][1]
            # print("curr_pauli", curr_pauli, curr_param, type(curr_param))
            sq_qc = construct_sq_subcircuit(curr_pauli)  # Construct the single qubit subcircuit
            #find the best next pauli:
            #based on the pauli_idx determine which should be the next pauli, or the pauli in the next commuting list
            if pauli_idx == len(sorted_list) - 1: # if this pauli is the last one in a commuting list, find the pauli in the next list
                if commute_idx == len(sorted_entanglers_params) - 1:
                    next_pauli = None
                else:
                    if len(sorted_entanglers_params[commute_idx + 1]) > 1:
                        next_pauli_index = find_best_pauli_index_threshold(base_entangler = curr_pauli, commute_sets= sorted_entanglers_params[commute_idx + 1], append_clifford = append_clifford, threshold = threshold)
                        element = sorted_entanglers_params[commute_idx + 1].pop(next_pauli_index)
                        # Insert the element at the beginning of the list
                        sorted_entanglers_params[commute_idx + 1].insert(0, element)
            else: 
                #UPDATE THE next commuting PAULIS:
                next_pauli_index = find_best_pauli_index_threshold(base_entangler = curr_pauli, commute_sets= sorted_entanglers_params[commute_idx][pauli_idx + 1:], append_clifford = append_clifford, threshold = threshold)
                # Remove the element at the specified index
                element = sorted_entanglers_params[commute_idx].pop(pauli_idx + 1 + next_pauli_index)
                # Insert the element at the beginning of the list
                sorted_entanglers_params[commute_idx].insert(pauli_idx + 1, element)
                # next_pauli = sorted_entanglers_params[commute_idx][0]
            logging.debug("after_search_for_the best next: %s", sorted_entanglers_params)
            logging.debug("next_pauli: %s", next_pauli)
            #up to this step we haven't extracted any clifford circuit, just analysis with look up table, should be fast
            if next_pauli == None:      
                #extract for the last block:
                extracted_clif = construct_Clifford_subcircuit(curr_pauli)
                #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
                append_clifford = extracted_clif.compose(append_clifford.to_circuit())
                #The extracted circuit for the current block with index
                extracted_qc = construct_extracted_subcircuit(entangler = curr_pauli, param = curr_param)
                #Add the extracted circuit to the optimized circuit
                logging.debug("final paulis: %s", curr_pauli)

                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(extracted_qc, inplace = True)

            else:
                #print("before single tree", curr_pauli, updated_entanglers)
                init_cx_tree = QuantumCircuit(len(entanglers[0]))
                next_commute_idx, next_pauli_idx = gen_next_pauli_idx(sorted_entanglers_params,commute_idx, pauli_idx)
                tree_list = [len(curr_pauli) - 1 - i for i in range(len(curr_pauli)) if curr_pauli[i] != 'I']
                sq_index = find_leaves(sorted_entanglers_params, curr_pauli = curr_pauli, updated_paulis = {}, qc_tree = init_cx_tree, tree_list = tree_list, commute_idx = next_commute_idx, pauli_idx = next_pauli_idx, append_clifford = append_clifford)
                #sq_index, init_cx_tree = find_single_tree_lookahead_adapt(base_entangler_inv = curr_pauli, match_entangler_inv = updated_entanglers[0], lookahead_entanglers_inv = updated_entanglers)
                init_clif = sq_qc.inverse()
                init_clif.compose(init_cx_tree, inplace = True)
                extracted_cx_tree = init_cx_tree.inverse()
                extracted_clif = init_clif.inverse()
                extracted_clif = Clifford(extracted_clif)
                #Add the extracted_clifford to the append clifford
                append_clifford = extracted_clif.compose(append_clifford)

                init_clif.rz(curr_param, sq_index)

                #Add the extracted circuit to the optimized circuit
                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(init_clif, inplace = True)

                #Use append_clifford to update the next pauli:
                # print(pauli_idx, len(sorted_list), sorted_list, sorted_entanglers_params)
                if pauli_idx == len(sorted_list) - 1: 
                    # print("before update", sorted_entanglers_params[commute_idx + 1][0])
                    sorted_entanglers_params[commute_idx + 1][0] = update_pauli_param(Pauli_param = sorted_entanglers_params[commute_idx + 1][0], clifford_circuit = append_clifford)
                else:
                    # print("before update", sorted_entanglers_params[commute_idx][pauli_idx + 1])
                    sorted_entanglers_params[commute_idx][pauli_idx + 1] = update_pauli_param(Pauli_param = sorted_entanglers_params[commute_idx][pauli_idx + 1], clifford_circuit = append_clifford)
    return opt_qc, append_clifford, sorted_entanglers_params

def CE_recur_tree(entanglers: List[str], params: List[float], barrier=False, threshold = 1):
    '''This function defines the optimized fully connected tree block for hamtiltonian simulation in commute list format, also considering lookahead
    
    Args:
        entanglers: list storing Pauli words for construction of optimized qcc_circuit.
        params: parameters for the rotations
        barrier: barriers between blocks of gates
    Returns:
        opt_qc, append_clifford, opt_paulis, opt_params
    '''
    commute_sets = convert_commute_sets(Paulis= entanglers, params = params)

    opt_qc = QuantumCircuit(len(entanglers[0]))
    append_clifford = QuantumCircuit(len(entanglers[0]))
    append_clifford = Clifford(append_clifford)
    # opt_params = params.copy()
    sorted_entanglers_params = []

    #sort all the paulis based on their weight:#TODO: need to resolve when two cases have the same weight
    for commute_list in commute_sets:
        sorted_list = sorted(commute_list, key=lambda x: pauli_weight(x[0]))
        sorted_entanglers_params.append(sorted_list)

    logging.debug("start_sorted_list: %s", sorted_entanglers_params)
    next_pauli = 0
    # Iterate over all the lists of commuting entanglers that need optimization
    for commute_idx, sorted_list in enumerate(sorted_entanglers_params):
        # updated_commute_list = [sorted_list[0]]  # Initialize with the first Pauli string
        # remaining_commute_paulis = sorted_list[1:].copy()
        
        for pauli_idx in range(len(sorted_list)):
            #here we start process for each current pauli
            curr_pauli = sorted_entanglers_params[commute_idx][pauli_idx][0]
            curr_param = sorted_entanglers_params[commute_idx][pauli_idx][1]
            # print("curr_pauli", curr_pauli, curr_param, type(curr_param))
            sq_qc = construct_sq_subcircuit(curr_pauli)  # Construct the single qubit subcircuit
            #find the best next pauli:
            #based on the pauli_idx determine which should be the next pauli, or the pauli in the next commuting list
            if pauli_idx == len(sorted_list) - 1: # if this pauli is the last one in a commuting list, find the pauli in the next list
                if commute_idx == len(sorted_entanglers_params) - 1:
                    next_pauli = None
                else:
                    if len(sorted_entanglers_params[commute_idx + 1]) > 1:
                        next_pauli_index = find_best_pauli_index_threshold(base_entangler = curr_pauli, commute_sets= sorted_entanglers_params[commute_idx + 1], append_clifford = append_clifford, threshold = threshold)
                        element = sorted_entanglers_params[commute_idx + 1].pop(next_pauli_index)
                        # Insert the element at the beginning of the list
                        sorted_entanglers_params[commute_idx + 1].insert(0, element)
            else: 
                #UPDATE THE next commuting PAULIS:
                next_pauli_index = find_best_pauli_index_threshold(base_entangler = curr_pauli, commute_sets= sorted_entanglers_params[commute_idx][pauli_idx + 1:], append_clifford = append_clifford, threshold = threshold)
                # Remove the element at the specified index
                element = sorted_entanglers_params[commute_idx].pop(pauli_idx + 1 + next_pauli_index)
                # Insert the element at the beginning of the list
                sorted_entanglers_params[commute_idx].insert(pauli_idx + 1, element)
                # next_pauli = sorted_entanglers_params[commute_idx][0]
            logging.debug("after_search_for_the best next: %s", sorted_entanglers_params)
            logging.debug("next_pauli: %s", next_pauli)
            #up to this step we haven't extracted any clifford circuit, just analysis with look up table, should be fast
            if next_pauli == None:      
                #extract for the last block:
                extracted_clif = construct_Clifford_subcircuit(curr_pauli)
                #Add the extracted clifford to the beginning of the append_clifford at the end of the circuit
                append_clifford = extracted_clif.compose(append_clifford.to_circuit())
                #The extracted circuit for the current block with index
                extracted_qc = construct_extracted_subcircuit(entangler = curr_pauli, param = curr_param)
                #Add the extracted circuit to the optimized circuit
                logging.debug("final paulis: %s", curr_pauli)

                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(extracted_qc, inplace = True)

            else:
                #print("before single tree", curr_pauli, updated_entanglers)
                init_cx_tree = QuantumCircuit(len(entanglers[0]))
                next_commute_idx, next_pauli_idx = gen_next_pauli_idx(sorted_entanglers_params,commute_idx, pauli_idx)
                tree_list = [len(curr_pauli) - 1 - i for i in range(len(curr_pauli)) if curr_pauli[i] != 'I']
                sq_index = find_leaves(sorted_entanglers_params, curr_pauli = curr_pauli, updated_paulis = {}, qc_tree = init_cx_tree, tree_list = tree_list, commute_idx = next_commute_idx, pauli_idx = next_pauli_idx, append_clifford = append_clifford)
                #sq_index, init_cx_tree = find_single_tree_lookahead_adapt(base_entangler_inv = curr_pauli, match_entangler_inv = updated_entanglers[0], lookahead_entanglers_inv = updated_entanglers)
                init_clif = sq_qc.inverse()
                init_clif.compose(init_cx_tree, inplace = True)
                extracted_cx_tree = init_cx_tree.inverse()
                extracted_clif = init_clif.inverse()
                extracted_clif = Clifford(extracted_clif)
                #Add the extracted_clifford to the append clifford
                append_clifford = extracted_clif.compose(append_clifford)

                init_clif.rz(curr_param, sq_index)

                #Add the extracted circuit to the optimized circuit
                if barrier == True:
                    opt_qc.barrier()
                opt_qc.compose(init_clif, inplace = True)

                #Use append_clifford to update the next pauli:
                # print(pauli_idx, len(sorted_list), sorted_list, sorted_entanglers_params)
                if pauli_idx == len(sorted_list) - 1: 
                    # print("before update", sorted_entanglers_params[commute_idx + 1][0])
                    sorted_entanglers_params[commute_idx + 1][0] = update_pauli_param(Pauli_param = sorted_entanglers_params[commute_idx + 1][0], clifford_circuit = append_clifford)
                else:
                    # print("before update", sorted_entanglers_params[commute_idx][pauli_idx + 1])
                    sorted_entanglers_params[commute_idx][pauli_idx + 1] = update_pauli_param(Pauli_param = sorted_entanglers_params[commute_idx][pauli_idx + 1], clifford_circuit = append_clifford)
    return opt_qc, append_clifford, sorted_entanglers_params

