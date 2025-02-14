import sys

sys.path.append('..')

import qiskit
import cirq
import os
import json
import numpy as np
from phoenix import Circuit
from phoenix.utils.ops import tensor_1_slot
from scipy import linalg
from qiskit.quantum_info import SparsePauliOp, Pauli
from phoenix.utils.functions import infidelity
from natsort import natsorted

# input_qasm_dpath = './output_uccsd/phoenix/all2all'

input_json_dpath = '../benchmarks/uccsd_json'
input_qasm_dpath = '../benchmarks/uccsd_qasm'

output_dpath = './output_uccsd/phoenix/all2all'
# output_dpath = './output_uccsd/tket/all2all'


# output_dpath = 'qiskit_post_opt'

# output_dpath = 'qiskit_opt'

# output_dpath = './output_uccsd/tetris/all2all'

# output_dpath = 'tket_opt'
# output_dpath = 'tket_post_opt'

def ideal_evolution(json_fname):
    with open(json_fname, 'r') as f:
        data = json.load(f)

    front_x = [tensor_1_slot(Pauli('X').to_matrix(), data['num_qubits'], i) for i in data['front_x_on']]
    ham = SparsePauliOp(data['paulis'], data['coeffs'])
    evol = linalg.expm(-1j * ham.to_matrix()) @ cirq.dot(*front_x)
    return evol


print('output_dpath:', output_dpath)
for fname in natsorted(os.listdir(input_qasm_dpath), reverse=True):
    input_fname = os.path.join(input_qasm_dpath, fname)
    output_fname = os.path.join(output_dpath, fname)
    input_json_fname = os.path.join(input_json_dpath, fname.replace('.qasm', '.json'))

    if qiskit.QuantumCircuit.from_qasm_file(input_fname).num_qubits > 10:
        continue

    u_ideal = ideal_evolution(input_json_fname)

    circ = Circuit.from_qasm(fname=input_fname)
    u_ref = circ.unitary()

    circ_opt = Circuit.from_qasm(fname=output_fname)
    u_opt = circ_opt.unitary()

    print(fname)
    print('\tinfidelity u_ideal v.s. u_opt:', infidelity(u_ideal, u_opt))
    print('\tinfidelity u_ref v.s. u_opt:', infidelity(u_ref, u_opt))
