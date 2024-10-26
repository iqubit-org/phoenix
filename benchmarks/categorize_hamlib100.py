"""
This script is to generate .json and .qasm files from the original 100_representative.json file which is 100 selected benchmarks from HamLib.
"""

import sys

sys.path.append('..')

import os
import json
from phoenix.utils.functions import infidelity
from phoenix.models import HamiltonianModel


QASM_DIR = 'hamlib100qasm'
JSON_DIR = 'hamlib100json'


with open('100_representative.json', 'r') as f:
    data = json.load(f)


for ham in data:
    program_name = '{}-{}'.format(ham['ham_problem'], ham['ham_instance'].strip('/'))
    program_name = program_name.split(',')[0]
    program_name = program_name.replace('ham_', '')
    program_name = program_name.replace('ham-', '')
    print(program_name)

    qasm_fname = os.path.join(QASM_DIR, ham['ham_category'], program_name + '.qasm')
    json_fname = os.path.join(JSON_DIR, ham['ham_category'], program_name + '.json')


    # save to json
    with open(json_fname, 'w') as f:
        json_body = {
            'num_qubits': ham['ham_qubits'],
            'num_terms': ham['ham_terms'],
            'paulis': ham['ham_hamlib_hamiltonian_terms'],
            'coeffs': ham['ham_hamlib_hamiltonian_coefficients']
        }
        json.dump(json_body, f, indent=4)

    # save to qasm
    ham_model = HamiltonianModel(ham['ham_hamlib_hamiltonian_terms'], ham['ham_hamlib_hamiltonian_coefficients'])
    circ = ham_model.generate_circuit()
    circ.to_qasm(qasm_fname)

    # import numpy as np
    # print(np.max(ham['ham_hamlib_hamiltonian_coefficients']), np.min(ham['ham_hamlib_hamiltonian_coefficients']))

    # if ham['ham_qubits'] <= 10:
    #     u_ideal = ham_model.unitary_evolution()
    #     u_trotter = circ.unitary()
    #     print(program_name)
    #     print('infidelity:', infidelity(u_ideal, u_trotter))
    #     print()
