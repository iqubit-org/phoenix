import numpy as np
from qiskit.quantum_info import Clifford, SparsePauliOp

def get_clifford_gate(p1_char, p2_char, reverse=False):
    # p1_char: Pauli on control (q0)
    # p2_char: Pauli on target (q1)
    
    # Labels: "II", "I"+p1, p2+"I", p2+p1
    # Note: Qiskit label "BA" means B on q1, A on q0.
    # If reverse=False (Control q0, Target q1):
    # Term 1: (I+P1) on q0, I on q1 -> I on q1, (I+P1) on q0 -> "II", "I"+p1
    # Term 2: (I-P1) on q0, P2 on q1 -> P2 on q1, (I-P1) on q0 -> p2+"I", p2+p1 (with minus)
    
    labels = ["II", "I"+p1_char, p2_char+"I", p2_char+p1_char]
    coeffs = [0.5, 0.5, 0.5, -0.5]
    
    if reverse:
        # Control q1, Target q0
        # P1 on q1, P2 on q0
        # Term 1: (I+P1) on q1, I on q0 -> (I+P1) on q1, I on q0 -> "II", p1+"I"
        # Term 2: (I-P1) on q1, P2 on q0 -> (I-P1) on q1, P2 on q0 -> "I"+p2, p1+p2 (with minus)
        labels = ["II", p1_char+"I", "I"+p2_char, p1_char+p2_char]
    
    op = SparsePauliOp(labels, coeffs)
    # Convert to matrix and then Clifford
    # Note: SparsePauliOp.to_matrix() returns a numpy matrix.
    # Clifford(matrix) works if it is a valid Clifford.
    return Clifford(op.to_matrix())

paulis = ['X', 'Y', 'Z']
generators_fixed = [] 

for p1 in paulis:
    for p2 in paulis:
        try:
            c = get_clifford_gate(p1, p2, reverse=False)
            generators_fixed.append(c)
        except Exception as e:
            print(f"Error creating C({p1}, {p2}): {e}")

def get_group_size(generators):
    found = set()
    queue = []
    
    eye = Clifford(np.eye(4))
    
    def get_hash(c):
        # Use symplectic matrix and phase as unique identifier
        return (c.symplectic_matrix.tobytes(), c.phase.tobytes())

    start_hash = get_hash(eye)
    found.add(start_hash)
    queue.append(eye)
    
    count = 0
    while queue:
        curr = queue.pop(0)
        count += 1
        if count % 1000 == 0:
            print(f"Found {len(found)} elements...")
            
        for g in generators:
            new_c = curr.compose(g)
            h = get_hash(new_c)
            if h not in found:
                found.add(h)
                queue.append(new_c)
                
    return len(found)

if __name__ == "__main__":
    print(f"Generators (fixed direction): {len(generators_fixed)}")
    print("Calculating group size...")
    size = get_group_size(generators_fixed)
    print(f"Group size (fixed direction): {size}")
    
    if size == 11520:
        print("Result: The set generates the full 2-qubit Clifford group Sp(4,2).")
    else:
        print(f"Result: The set generates a subgroup of size {size}.")

