import sys

sys.path.append('..')

from phoenix import Circuit
from qiskit import QuantumCircuit
import os

dpath = './chem_qasm/'

for fname in os.listdir(dpath):
    fname = os.path.join(dpath, fname)
    # circ = Circuit.from_qasm(fname=fname)
    circ = QuantumCircuit.from_qasm_file(fname)
    if circ.num_nonlocal_gates() > 20000:
        # remove this file
        os.remove(fname)
