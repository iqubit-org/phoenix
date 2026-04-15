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
import itertools
from qiskit.circuit import QuantumCircuit

def read_from_file(filename):
    #read from a file, return the hamiltonian list
    Hamiltonian_list = []
    f = open(filename,"r") # Here we read in the qubit Hamiltonian from the .txt file
    for line in f:
        a = line.strip().split(',')
        c = (str(a[0]), float(a[1]))
        Hamiltonian_list.append(c)
    return Hamiltonian_list

def MeasureCircuit(Sum_Op, num_qubits, num_qargs):
    # Determine how many commute groups are in the SummedOp
    num_terms = len(Sum_Op)

    # Find the Paulis with least number of I in it(shoud be 0).
    # The problem is here. In this case, not not all the subgroups have a term with no I in it. So we have to loop over all
    # of the terms to construct a Pauli string.
    Pauli = ''

    for i in range(num_qubits):
        intermed = []
        for j in range(num_terms):
            intermed.append(Sum_Op[j][0][i])
        if 'X' in intermed:
            Pauli += 'X'
        elif 'Y' in intermed:
            Pauli += 'Y'
        else:
            Pauli += 'Z'

    if len(Pauli) != num_qubits:
        raise Exception('The length does not equal, traverse has problem.')

    Pauli_string = Pauli[::-1]  # This has reversed the order.
    # Now Pauli_string is the target that we should use to construct the measurement circuit.

    qc = QuantumCircuit(num_qargs)
#     qc.barrier()
    print(Pauli_string)
    
    for i in range(num_qubits):
        if Pauli_string[i] == 'X':
            qc.u(np.pi / 2, 0, np.pi, i)
        if Pauli_string[i] == 'Y':
            qc.u(np.pi / 2, 0, np.pi / 2, i)
        else:
            None

    return qc

def evaluation(d: dict, shots: int, Pauli: str):
    # This Pauli_string is in arbitrary form of I, X, Y and Z.
    # Determine the number of qubits, which is also related to the number of measurement outcomes.
    num_qubits = len(Pauli)
    Pauli_string = ''
    for i in Pauli:
        if i == 'I':
            Pauli_string += i
        else:
            Pauli_string += 'Z'

    def kbits(n):
        result = []
        for k in range(0, n + 1):
            for bits in itertools.combinations(range(n), k):
                s = ['0'] * n
                for bit in bits:
                    s[bit] = '1'
                result.append(''.join(s))
        return result

    # Generate all binary strings of N bits.
    outcomes = kbits(num_qubits)

    def get_from(d: dict, key: str):
        value = 0
        if key in d:
            value = d[key]
        return value

    # Here we compute the expectation value.
    expectation_value = 0
    for i in outcomes:
        intermediate = 0
        for j in range(num_qubits):
            if (Pauli_string[j] == 'Z') and (i[j] == '1'):
                intermediate += 1
            else:
                None

        if (intermediate % 2) == 0:
            expectation_value += get_from(d, i)
        else:
            expectation_value -= get_from(d, i)

    expectation_value = expectation_value / shots

    return expectation_value

def apply_checking_circuit(qc, ctrl_bits, ancilla_bits, side = None, phase_z = None, phase_y = None, x = None):
    if len(ctrl_bits) != len(ancilla_bits):
        print("Size mismatch")
        return None
    if side == 'front':
        for i in ancilla_bits:
            qc.h(i)
        if x is True:
            qc.x(ctrl_bits)
        qc.ry(phase_y, ctrl_bits)
        qc.rz(phase_z, ctrl_bits)
        for j,k in zip(ctrl_bits, ancilla_bits):
            qc.cz(j, k)
        qc.rz(-phase_z, ctrl_bits)
        qc.ry(-phase_y, ctrl_bits)
        if x is True:
            qc.x(ctrl_bits)
    elif side == 'end':
        qc.rz(-phase_z, ctrl_bits)
        qc.ry(-phase_y, ctrl_bits)
        for j,k in zip(ctrl_bits, ancilla_bits):
            qc.cz(j, k)
        qc.ry(phase_y, ctrl_bits)
        qc.rz(phase_z, ctrl_bits)
        for i in ancilla_bits:
            qc.h(i)
    else:
        print("Side undefined")