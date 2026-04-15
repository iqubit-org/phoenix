# ZX YY ZZ YX



import qiskit
import numpy as np
qc = qiskit.QuantumCircuit(2)
qc.rzx(np.random.rand(), 0, 1)
qc.ryy(np.random.rand(), 0, 1)
qc.rzz(np.random.rand(), 0, 1)
qc.sdg(0)
qc.h(0)
qc.s(0)
qc.h(1)
qc.rzz(np.random.rand(), 0, 1)
qc.h(1)
qc.sdg(0)
qc.h(0)
qc.s(0)

print(qc)
import canopus
print(canopus.utils.canonical_coordinate(canopus.utils.qc2mat(qc)))

qc = qiskit.transpile(qc, optimization_level=3, basis_gates=['cx', 'u'])
print(qc)
