import numpy as np
from itertools import combinations

np.random.seed(42)  # For reproducibility
qubits = np.random.choice(range(20), 11, replace=False)
qubits = sorted(qubits)
print(qubits)

qubit_pairs = np.array(list(combinations(qubits, 2))).tolist()
# qubit_pairs = resort_indices(qubit_pairs)
qubit_pairs = sorted(qubit_pairs, key=lambda idx: (idx[0] % 2, idx[1]))
