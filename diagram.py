from phoenix.synthesis import ordering
from phoenix.models import BSF
from phoenix import Circuit, gates
import numpy as np

############################################################################################################
n = 5
m = 8
paulis = ['X', 'Y', 'Z']

circ = Circuit()
np.random.seed(123)
for _ in range(m):
    circ.append(
        gates.Clifford2QGate('Z', 'X').on(np.random.choice(range(n), 2, replace=False).tolist()))


print(circ.to_qiskit().draw(fold=300))

print(ordering.left_end_empty_layers(circ), ordering.right_end_empty_layers(circ))

############################################################################################################

tab = BSF([
    'ZYY',
    'ZZY',
    'XYY',
    'XZY',
])

print(tab.mat)

tab_ = BSF([
    'ZYI',
    'ZZI',
    'XYI',
    'XZI',
])

print(tab_.mat)





