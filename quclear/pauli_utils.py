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

import qiskit
import itertools
import pickle
import math

from typing import Any
from typing import Callable
from collections import defaultdict
from qiskit.dagcircuit import DAGOutNode, DAGOpNode

from qiskit import *
from typing import List
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

class PushOperator:
    '''Class for finding checks and pushing operations through in symbolic form.'''
    @staticmethod
    def x(op2):
        '''Pushes x through op2.'''
        ops = {
            "X": [1, "X"],
            "Y": [-1, "X"],
            "Z": [-1, "X"],
            "H": [1, "Z"],
            "S": [1, "Y"],
            "SDG": [-1, "Y"]
        }
        return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")
    
    @staticmethod
    def y(op2):
        '''Pushes y through op2.'''
        ops = {
            "X": [-1, "Y"],
            "Y": [1, "Y"],
            "Z": [-1, "Y"],
            "H": [-1, "Y"],
            "S": [-1, "X"],
            "SDG": [1, "X"]
        }
        return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")
    
    @staticmethod        
    def z(op2):
        '''Pushes z through op2.'''
        ops = {
            "X": [-1, "Z"],
            "Y": [-1, "Z"],
            "Z": [1, "Z"],
            "H": [1, "X"],
            "S": [1, "Z"],
            "SDG": [1, "Z"],
            "RZ": [1, "Z"]
        }
        return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")

    @staticmethod
    def cx(op1):
        '''Pushes op1 through cx.'''
        ops = {
            ("I", "I"): [1, "I", "I"],
            ("I", "X"): [1, "I", "X"],
            ("I", "Y"): [1, "Z", "Y"],
            ("I", "Z"): [1, "Z", "Z"],
            ("X", "I"): [1, "X", "X"],
            ("X", "X"): [1, "X", "I"],
            ("X", "Y"): [1, "Y", "Z"],
            ("X", "Z"): [-1, "Y", "Y"],
            ("Y", "I"): [1, "Y", "X"],
            ("Y", "X"): [1, "Y", "I"],
            ("Y", "Y"): [-1, "X", "Z"],
            ("Y", "Z"): [1, "X", "Y"],
            ("Z", "I"): [1, "Z", "I"],
            ("Z", "X"): [1, "Z", "X"],
            ("Z", "Y"): [1, "I", "Y"],
            ("Z", "Z"): [1, "I", "Z"]
        }
        return ops.get(tuple(op1), None) or Exception(f"{op1[0]} , {op1[1]} wasn't a Pauli element.")

    @staticmethod
    def swap(op1):
        '''Passes op1 through swap.'''
        return [1] + list(reversed(op1))

    
def get_weight(pauli_string):
    '''Gets the weight of a Pauli string. Returns: int'''
    count = 0
    for character in pauli_string:
        if character != "I":
            count += 1
    return count

class ChecksResult:
    def __init__(self, p2_weight, p1_str, p2_str):
        self.p2_weight = p2_weight
        self.p1_str = p1_str
        self.p2_str = p2_str
        
class CheckOperator:
    '''Stores the check operation along with the phase. operations is a list of strings.'''

    def __init__(self, phase: int, operations: List[str]):
        self.phase = phase
        self.operations = operations

class TempCheckOperator(CheckOperator):
    '''A temporary class for storing the check operation along with the phase and other variables.'''

    def __init__(self, phase: int, operations: List[str]):
        super().__init__(phase, operations)
        self.layer_idx = 1

class ChecksFinder:
    '''Finds checks symbolically.'''

    def __init__(self, number_of_qubits: int, circ):
        self.circ_reversed = circ.inverse()
        self.number_of_qubits = number_of_qubits

    def find_checks_sym(self, pauli_group_elem: List[str]) -> ChecksResult:
        '''Finds p1 and p2 elements symbolically.'''
        circ_reversed = self.circ_reversed
        pauli_group_elem_ops = list(pauli_group_elem)
        p2 = CheckOperator(1, pauli_group_elem_ops)
        p1 = CheckOperator(1, ["I" for _ in range(len(pauli_group_elem))])
        temp_check_reversed = TempCheckOperator(1, list(reversed(pauli_group_elem_ops)))

        circ_dag = circuit_to_dag(circ_reversed)
        layers = list(circ_dag.multigraph_layers())
        num_layers = len(layers)

        while True:
            layer = layers[temp_check_reversed.layer_idx]
            for node in layer:
                if isinstance(node, DAGOpNode):
                    self.handle_operator_node(node, temp_check_reversed)
            if self.should_return_result(temp_check_reversed, num_layers):
                p1.phase = temp_check_reversed.phase
                p1.operations = list(reversed(temp_check_reversed.operations))
                return self.get_check_strs(p1, p2)
            temp_check_reversed.layer_idx += 1

    def handle_operator_node(self, node, temp_check_reversed: TempCheckOperator):
        '''Handles operations for nodes of type "op".'''
        current_qubits = self.get_current_qubits(node)
        current_ops = [temp_check_reversed.operations[qubit] for qubit in current_qubits]
        node_op = node.name.upper()
        self.update_current_ops(current_ops, node_op, temp_check_reversed, current_qubits)

    def should_return_result(self, temp_check_reversed: TempCheckOperator, num_layers: int) -> bool:
        '''Checks if we have reached the last layer.'''
        return temp_check_reversed.layer_idx == num_layers - 1

    @staticmethod
    def update_current_ops(op1: List[str], op2: str, temp_check_reversed: TempCheckOperator, current_qubits: List[int]):
        '''Finds the intermediate check. Always push op1 through op2. '''
        result = ChecksFinder.get_result(op1, op2)
        temp_check_reversed.phase *= result[0]
        for idx, op in enumerate(result[1:]):
            temp_check_reversed.operations[current_qubits[idx]] = op

    @staticmethod
    def get_result(op1: List[str], op2: str) -> List[str]:
        '''Obtain the result based on the values of op1 and op2.'''
        if len(op1) == 1:
            return ChecksFinder.single_qubit_operation(op1[0], op2)
        else:
            return ChecksFinder.double_qubit_operation(op1, op2)

    @staticmethod
    def single_qubit_operation(op1: str, op2: str) -> List[str]:
        '''Process the single qubit operations.'''
        if op1 == "X":
            return PushOperator.x(op2)
        elif op1 == "Y":
            return PushOperator.y(op2)
        elif op1 == "Z":
            return PushOperator.z(op2)
        elif op1 == "I":
            return [1, "I"]
        else:
            raise ValueError(f"{op1} is not I, X, Y, or Z.")

    @staticmethod
    def double_qubit_operation(op1: List[str], op2: str) -> List[str]:
        '''Process the double qubit operations.'''
        if op2 == "CX":
            return PushOperator.cx(op1)
        elif op2 == "SWAP":
            return PushOperator.swap(op1)
        else:
            raise ValueError(f"{op2} is not cx or swap.")

    @staticmethod
    def get_check_strs(p1: CheckOperator, p2: CheckOperator) -> ChecksResult:
        '''Turns p1 and p2 to strings results.'''
        p1_str = ChecksFinder.get_formatted_str(p1)
        p2_str = ChecksFinder.get_formatted_str(p2)
        check_result = ChecksResult(get_weight(p2.operations), p1_str, p2_str)
        return check_result

    @staticmethod
    def get_formatted_str(check_operator: CheckOperator) -> str:
        '''Format the phase and operations into a string.'''
        operations = check_operator.operations
        phase = check_operator.phase
        phase_str = f"+{phase}" if len(str(phase)) == 1 else str(phase)
        operations.insert(0, phase_str)
        return "".join(operations)
    
    @staticmethod
    def get_current_qubits(node):
        '''Finding checks: Symbolic: get the current qubits whose operations that will be passed through.'''
        # We have to check for single or two qubit gates.
        if node.name in ["x", "y", "z", "h", "s", "sdg", "rz"]:
            return [node.qargs[0]._index]
        elif node.name in ["cx", "swap"]:
            return [node.qargs[0]._index, node.qargs[1]._index]
        else:
            assert False, "Overlooked a node operation."
            
            
def append_paulis_to_circuit(circuit, pauli_string):
    """
    Appends Pauli operations to the quantum circuit based on the pauli_string input.
    """
    for index, char in enumerate(reversed(pauli_string)):
        if char == 'I':
            circuit.i(index)
        elif char == 'X':
            circuit.x(index)
        elif char == 'Y':
            circuit.y(index)
        elif char == 'Z':
            circuit.z(index)
            
def append_control_paulis_to_circuit(circuit, pauli_string, ancilla_index, mapping):
    """
    Appends controlled Pauli operations to the quantum circuit based on the pauli_string input.
    """
    for orign_index, char in enumerate(reversed(pauli_string)):
        index = mapping[orign_index]
        if char == 'X':
            circuit.cx(ancilla_index, index)
        elif char == 'Y':
            circuit.cy(ancilla_index, index)
        elif char == 'Z':
            circuit.cz(ancilla_index, index)

def verify_circuit_with_pauli_checks(circuit, left_check, right_check):
    """
    Verifies that the original circuit is equivalent to a new circuit that includes left and right Pauli checks.
    The equivalence is verified by comparing the unitary matrix representations of both circuits.
    """
    assert len(circuit.qubits) == len(left_check) == len(right_check), "Number of qubits in circuit and checks must be equal."

    verification_circuit = QuantumCircuit(len(circuit.qubits))
    
    append_paulis_to_circuit(verification_circuit, left_check)
    verification_circuit.compose(circuit, inplace=True)
    append_paulis_to_circuit(verification_circuit, right_check)

    original_operator = Operator(circuit)
    verification_operator = Operator(verification_circuit)

    return verification_circuit, original_operator.equiv(verification_operator)

def pauli_strings_commute(pauli_str1, pauli_str2):
    """
    Determine if two Pauli strings commute.
    
    :param pauli_str1: A string representing the first Pauli operator.
    :param pauli_str2: A string representing the second Pauli operator.
    :return: True if the Pauli strings commute, False otherwise.
    """
    if len(pauli_str1) != len(pauli_str2):
        raise ValueError("Pauli strings must be of the same length.")
    
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