import qiskit
import os
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from natsort import natsorted

# input_dpath = './output_uccsd/phoenix/all2all'
input_dpath = '../benchmarks/uccsd_qasm'

# output_dpath = './output_uccsd/phoenix/all2all'
output_dpath = './output_uccsd/tket/all2all'

# output_dpath = 'qiskit_post_opt'

# output_dpath = 'qiskit_opt'

# output_dpath = './output_uccsd/tetris/all2all'

# output_dpath = 'tket_opt'
# output_dpath = 'tket_post_opt'

import sys

sys.path.append('..')
from phoenix.utils.functions import infidelity

for fname in natsorted(os.listdir(input_dpath), reverse=True):
    input_fname = os.path.join(input_dpath, fname)
    output_fname = os.path.join(output_dpath, fname)

    if qiskit.QuantumCircuit.from_qasm_file(input_fname).num_qubits > 10:
        continue

    circ = circuit_from_qasm(open(input_fname).read())

    circ_opt = circuit_from_qasm(open(output_fname).read())

    u = cirq.unitary(circ)
    v = cirq.unitary(circ_opt)

    print(fname, infidelity(u, v))
