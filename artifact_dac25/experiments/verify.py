import sys

sys.path.append('../..')

import os
import json
import numpy as np
from phoenix import Circuit
from scipy import linalg
from qiskit.quantum_info import SparsePauliOp, Pauli
from phoenix.utils.functions import infidelity
from natsort import natsorted

input_json_dpath = '../benchmarks/uccsd'
output_dpath = sys.argv[1] if len(sys.argv) > 1 else '../benchmarks/uccsd_qasm'


def ideal_evolution(json_fname):
    with open(json_fname, 'r') as f:
        data = json.load(f)
    # front_x = [tensor_1_slot(Pauli('X').to_matrix(), data['num_qubits'], i) for i in data['front_x_on']]
    ham = SparsePauliOp(data['paulis'], data['coeffs'])
    evol = linalg.expm(-1j * ham.to_matrix())
    # evol = linalg.expm(-1j * ham.to_matrix()) @ cirq.dot(*front_x)
    return evol


print('output_dpath:', output_dpath)
for fname in natsorted(os.listdir(input_json_dpath)):
    input_json_fname = os.path.join(input_json_dpath, fname)
    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))

    with open(input_json_fname, 'r') as f:
        data = json.load(f)
    
    if data['num_qubits'] > 10:
        print('Skipping', fname, 'due to num_qubits > 10')
        continue

    u_ideal = linalg.expm(-1j * SparsePauliOp(data['paulis'], data['coeffs']).to_matrix())

    circ_opt = Circuit.from_qasm(fname=output_fname)
    u_opt = circ_opt.unitary()

    print(fname)
    print('\tinfidelity u_ideal v.s. u_opt:', infidelity(u_ideal, u_opt))
