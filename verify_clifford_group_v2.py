import numpy as np
from qiskit.quantum_info import Clifford, Pauli
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, CXGate, IGate

# Find mapping Cliffords
def find_mapping(source_pauli_char, target_pauli_char):
    # Find 1Q Clifford C such that C * source * C^dag = target
    # Brute force over 24 single qubit Cliffords
    
    gens = [Clifford(HGate()), Clifford(SGate())]
    found = []
    queue = [Clifford(IGate())]
    visited = set()
    
    while queue:
        curr = queue.pop(0)
        h = curr.to_matrix().tobytes()
        if h in visited:
            continue
        visited.add(h)
        found.append(curr)
        if len(found) >= 24:
            break
        for g in gens:
            queue.append(curr.compose(g))
            
    s = Pauli(source_pauli_char)
    t = Pauli(target_pauli_char)
    
    for c in found:
        # Use evolve to compute C P C^dag
        if s.evolve(c) == t:
            return c
    return None

# Cache mappings
mappings_to_Z = {} # P -> Z
mappings_from_X = {} # X -> P

for p in ['X', 'Y', 'Z']:
    mappings_to_Z[p] = find_mapping(p, 'Z')
    mappings_from_X[p] = find_mapping('X', p)

def get_clifford_gate_circuit(p1, p2, reverse=False):
    qc = QuantumCircuit(2)
    ctrl = 0 if not reverse else 1
    targ = 1 if not reverse else 0
    
    u_ctrl = mappings_to_Z[p1]
    v_targ = mappings_from_X[p2]
    
    # Pre: U on ctrl, Vdag on targ.
    qc.append(u_ctrl.to_circuit(), [ctrl])
    qc.append(v_targ.adjoint().to_circuit(), [targ])
    
    qc.cx(ctrl, targ)
    
    # Post: Udag on ctrl, V on targ.
    qc.append(u_ctrl.adjoint().to_circuit(), [ctrl])
    qc.append(v_targ.to_circuit(), [targ])
    
    return Clifford(qc)

generators_fixed = []
for p1 in ['X', 'Y', 'Z']:
    for p2 in ['X', 'Y', 'Z']:
        generators_fixed.append(get_clifford_gate_circuit(p1, p2, reverse=False))

def get_group_size(generators):
    found = set()
    queue = []
    eye = Clifford(np.eye(4))
    
    def get_hash(c):
        return (c.symplectic_matrix.tobytes(), c.phase.tobytes())

    start_hash = get_hash(eye)
    found.add(start_hash)
    queue.append(eye)
    
    count = 0
    while queue:
        curr = queue.pop(0)
        count += 1
        if count % 2000 == 0:
            print(f"Found {len(found)} elements...")
            
        for g in generators:
            new_c = curr.compose(g)
            h = get_hash(new_c)
            if h not in found:
                found.add(h)
                queue.append(new_c)
                
    return len(found)

if __name__ == "__main__":
    print(f"Generators: {len(generators_fixed)}")
    size = get_group_size(generators_fixed)
    print(f"Group size: {size}")
    if size == 11520:
        print("Universal!")
    else:
        print("Not universal.")

