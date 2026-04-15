"""Verify that the Pauli operators and QASM code are equivalent."""
import sys

sys.path.append('../')

import os
import json
import numpy as np
from functools import reduce
from qiskit.quantum_info import Pauli
from phoenix.models import HamiltonianModel
from phoenix import gates, Circuit
from phoenix.utils.functions import infidelity
from phoenix.utils.ops import tensor_1_slot

circ_dir = 'hamlib_qasm/'


categories = [
    'binaryoptimization',
    'discreteoptimization',
    'condensedmatter',
    'chemistry',
]
qasm_fnames = []
for dir in categories:
    qasm_fnames.extend([os.path.join(circ_dir, dir, fname) for fname in os.listdir(os.path.join(circ_dir, dir)) if fname.endswith('.qasm')])
json_fnames = [fname.replace('qasm', 'json') for fname in qasm_fnames]

for qasm_fname, json_fname in zip(qasm_fnames, json_fnames):
    if not (os.path.exists(qasm_fname) and os.path.exists(json_fname)):
        continue
    print()
    print('verifying {} and {} ...'.format(qasm_fname, json_fname))

    with open(json_fname, 'r') as f:
        data = json.load(f)
    circ = Circuit.from_qasm(fname=qasm_fname)

    if data['num_qubits'] > 10:
        print('skipping due to large number of qubits')
        continue
    if len(circ) > 10000:
        print('skipping due to large number of gates')
        continue

    ham = HamiltonianModel(data['paulis'], data['coeffs'])
    u_ideal = ham.unitary_evolution()

    u_ref = circ.unitary()

    circ_trotter = ham.generate_circuit()
    u_trotter = circ_trotter.unitary()

    print('infidelity (ideal v.s. trotter):', infidelity(u_ideal, u_trotter))
    print('infidelity (ideal v.s. ref):', infidelity(u_ideal, u_ref))
    print('infidelity (trotter v.s. ref):', infidelity(u_trotter, u_ref))
