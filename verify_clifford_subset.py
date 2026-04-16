import numpy as np
from qiskit.quantum_info import Clifford, SparsePauliOp
from phoenix_qiskit.basics import CNOTEquivCliffordGate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import QuantumCircuit

def get_clifford_from_pauli(pauli_str):
    # Use the user's definition: Clifford2QGate
    # This is an involution (order 2), not a rotation (order 8)
    return Clifford(CNOTEquivCliffordGate(pauli_str[0], pauli_str[1]))

# pauli_pairs = [
#     "XX",
#     "XY",
#     "XZ",
#     "YZ",
#     "ZX"
# ]

pauli_pairs = [
    "XX",
    "YY",
    "ZZ",
    "ZX",
    "XZ"
]

# pauli_pairs = [
#     "XX",
#     "XY",
#     "XZ",
#     "YX",
#     "YY",
#     "YZ",
#     "ZX",
#     "ZY",
#     "ZZ",
# ]





generators = []
print("Generators:")
for p in pauli_pairs:
    c = get_clifford_from_pauli(p)
    generators.append(c)
    print(f"C({p[0]}, {p[1]})")

# BFS to generate group
visited = set()
queue = [Clifford(np.eye(4))] # Identity
visited.add(queue[0].to_matrix().tobytes())

count = 0
while count < len(queue):
    current = queue[count]
    count += 1
    
    for gen in generators:
        next_cliff = current.compose(gen)
        h = next_cliff.to_matrix().tobytes()
        if h not in visited:
            visited.add(h)
            queue.append(next_cliff)
            
    if count % 1000 == 0:
        print(f"Discovered {len(visited)} elements so far...")

print(f"Final Group Size: {len(visited)}")
if len(visited) == 11520:
    print("Universal: YES")
else:
    print("Universal: NO")
