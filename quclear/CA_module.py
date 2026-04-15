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

from typing import List
from qiskit import *
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli as convert_Pauli_op
from qiskit.quantum_info import Clifford
from .vqe_utils import MeasureCircuit, evaluation
from .CE_module import split_pauli_string


def extract_CNOT_network(append_clifford: QuantumCircuit):
    #decompose the swap to CNOT
    append_clifford = transpile(append_clifford, basis_gates = ['swap', "cx", "h"])
    dag = circuit_to_dag(append_clifford.inverse())
    num_qubits = len(append_clifford.qubits)
    cnot_network = QuantumCircuit(num_qubits)
    hadamard_counts = [0] * num_qubits
    for node in dag.topological_op_nodes(): 
        if node.name == 's' or node.name == 'sdg':
            raise Exception("Circuit contains s or sdg gate")
        if node.name == 'h':
            hadamard_counts[node.qargs[0]._index] += 1
        if node.name == 'swap':
            temp_count = hadamard_counts[node.qargs[0]._index]
            hadamard_counts[node.qargs[0]._index] = hadamard_counts[node.qargs[1]._index]
            hadamard_counts[node.qargs[1]._index] = temp_count
            cnot_network.cx(node.qargs[0]._index, node.qargs[1]._index)
            cnot_network.cx(node.qargs[1]._index, node.qargs[0]._index)
            cnot_network.cx(node.qargs[0]._index, node.qargs[1]._index)
        if node.name == 'cx':
            control_qubit = node.qargs[0]._index 
            target_qubit = node.qargs[1]._index
            if hadamard_counts[node.qargs[0]._index] != hadamard_counts[node.qargs[1]._index]:
                raise Exception("Incorrect hadamard gate count")
            if hadamard_counts[node.qargs[0]._index] % 2 == 1:
                #Switch the control and the target qubit for even layers of hadamard gates
                temp = control_qubit 
                control_qubit = target_qubit
                target_qubit = temp
            cnot_network.cx(control_qubit, target_qubit)
    return cnot_network.inverse(), hadamard_counts

def CA_post_QAOA(opt_qc: QuantumCircuit, hadamard_counts: List[int]):
    # Add single qubit Hadamard gates according to the hadamard counts
    for idx in range(len(hadamard_counts)):
        if hadamard_counts[idx] % 2 == 1:
            opt_qc.h(idx)
    opt_qc.measure_active()
    return opt_qc

def apply_cnot(binary_value, control_index, target_index):
    # Convert binary string to a list of characters for easy manipulation
    binary_list = list(binary_value)
    # Apply CNOT: If the control qubit is 1, flip the target qubit
    if binary_list[control_index] == '1':
        binary_list[target_index] = '0' if binary_list[target_index] == '1' else '1'
    
    # Convert list back to binary string
    return ''.join(binary_list)

def update_probabilities(prob_dist, circuit_dag):
    updated_states = {}
    for state in prob_dist.keys():
        new_state = state
        for node in circuit_dag.topological_op_nodes():
            if node.name == 'cx':
                control_qubit = len(state) - node.qargs[0]._index - 1
                target_qubit = len(state) - node.qargs[1]._index - 1
                new_state = apply_cnot(new_state, control_qubit, target_qubit)
        updated_states[new_state] = prob_dist[state]
    return updated_states

def update_observables(Paulis: List[str], clifford_circuits: List[QuantumCircuit]):
    '''This function calcu;ates the pauli observable after extracting the clifford subcircuit.

    Args:
        Paulis: Pauli words for the observable
        clifford_circuit: the clifford subcircuit for extraction
    Returns:
        updated_Paulis
    '''
    num_qubits = len(Paulis[0])
    updated_Paulis = []
    updated_signs = []
    for clifford in clifford_circuits:
        append_clifford = Clifford(clifford)
        for idx, Pauli_str in enumerate(Paulis):
            Pauli_op = convert_Pauli_op(Pauli_str)
            evolved_Pauli_op = Pauli_op.evolve(append_clifford)
            evolved_sign, evolved_Pauli = split_pauli_string(evolved_Pauli_op.to_label())

            # print(updated_Paulis, pauli_result.p1_str, pauli_result.p2_str)
            updated_signs.append(evolved_sign)
            updated_Paulis.append(evolved_Pauli)

    return updated_signs, updated_Paulis 


def sim_expect_value(qc: QuantumCircuit, observable: str, shots: int=10000000):
    '''This function simulates the circuit and calculates the expectationvalue of the observable

    Args:
        qc: quantum circuit to be simulated
        observable: the pauli observable
    Returns:
        expectation_val: the expecation value
    '''
    pauli_commute = [[(observable, (1+0j))]]
    #generate individual measurment circuits that changes the basis
    meas_qcs = []
    for i in range(0, len(pauli_commute)):
        temp_qc = MeasureCircuit(pauli_commute[i], num_qubits = len(observable), num_qargs = len(observable))
        meas_qcs.append(temp_qc)
    qc_meas =  qc.compose(meas_qcs[0])
    qc_meas.measure_active()  
    simulator = AerSimulator()
    result = simulator.run(qc_meas, shots = shots).result()
    expectation_val = evaluation(result.get_counts(), shots = shots, Pauli = observable) 
    return expectation_val
