import os
import cirq
import qiskit
import pytket
import pytket.qasm
import pytket.passes
import numpy as np
from qiskit.quantum_info import Operator
from natsort import natsorted

import matplotlib.pyplot as plt

num_2q_gates_tket = []
num_2q_gates_tket2 = []
num_2q_gates_tket3 = []
depth_2q_tket = []
depth_2q_tket2 = []
depth_2q_tket3 = []

fnames = natsorted(os.listdir('tket_post_opt_manhattan2'))
for fname in fnames:
    fname_tket = os.path.join('tket_post_opt_manhattan', fname)
    fname_tket2 = os.path.join('tket_post_opt_manhattan2', fname)
    fname_tket3 = os.path.join('tket_post_opt_manhattan3', fname)

    circ_tket = qiskit.QuantumCircuit.from_qasm_file(fname_tket)
    circ_tket2 = qiskit.QuantumCircuit.from_qasm_file(fname_tket2)
    circ_tket3 = qiskit.QuantumCircuit.from_qasm_file(fname_tket3)

    print(f'fname: {fname}')
    # print('#gates\t {}\t {}'.format(circ_tket.size(), circ_tket2.size()))
    # print('depth\t {}\t {}'.format(circ_tket.depth(), circ_tket2.depth()))
    print('#2Q\t {}\t {}\t {}'.format(circ_tket.num_nonlocal_gates(),
                                 circ_tket2.num_nonlocal_gates(),
                                 circ_tket3.num_nonlocal_gates()))
    print('dep(2q)\t {}\t {}\t {}'.format(
        circ_tket2.depth(lambda intr: intr.operation.num_qubits > 1),
        circ_tket2.depth(lambda intr: intr.operation.num_qubits > 1),
        circ_tket3.depth(lambda intr: intr.operation.num_qubits > 1)
    ))

    num_2q_gates_tket.append(circ_tket.num_nonlocal_gates())
    num_2q_gates_tket2.append(circ_tket2.num_nonlocal_gates())
    num_2q_gates_tket3.append(circ_tket3.num_nonlocal_gates())
    depth_2q_tket.append(circ_tket.depth(lambda intr: intr.operation.num_qubits > 1))
    depth_2q_tket2.append(circ_tket2.depth(lambda intr: intr.operation.num_qubits > 1))
    depth_2q_tket3.append(circ_tket3.depth(lambda intr: intr.operation.num_qubits > 1))

plt.figure(figsize=(13, 5))
ind = np.arange(len(fnames))
width = 0.3
plt.bar(ind - width, num_2q_gates_tket, width, label='tket')
plt.bar(ind, num_2q_gates_tket2, width, label='tket2')
plt.bar(ind + width, num_2q_gates_tket3, width, label='tket3')
plt.xticks(ind, [fname.split('.')[0] for fname in fnames], rotation=45, fontweight='bold')
plt.xlabel('Circuit')
plt.ylabel('Number of 2-qubit gates')
plt.legend()
plt.tight_layout()
plt.show()
