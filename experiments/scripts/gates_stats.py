import os
import sys
from qiskit import QuantumCircuit

filename = sys.argv[1]

qc = QuantumCircuit.from_qasm_file(filename)

print(qc.count_ops())
