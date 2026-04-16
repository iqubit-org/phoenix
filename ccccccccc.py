import canopus
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.random import random_circuit
import numpy as np


u = UnitaryGate(unitary_group.rvs(2))
v = UnitaryGate(unitary_group.rvs(2))

su4 = UnitaryGate(np.kron(u.to_matrix(), v.to_matrix()))


qc = QuantumCircuit(2)
# qc.append(u, [0])
# qc.append(v, [1])
qc.append(su4, [0, 1])


qc = random_circuit(5, 20, max_operands=2)

print(qc)

qc_ = canopus.rebase_to_canonical(qc)
print(qc_)

print(qc_.count_ops())
