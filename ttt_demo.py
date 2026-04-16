import json

with open('./benchmarks/uccsd_json/NH_frz_JW_sto3g.json', 'r') as f:
    data = json.load(f)

num_qubits = data['num_qubits']
paulis = data['paulis']
coeffs = data['coeffs']


# Use phoenix_qiskit to work out a compilation example
from phoenix_qiskit import Hamiltonian, compile_hamiltonian_simulation
import numpy as np

print(f"Loaded Hamiltonian data: {len(paulis)} terms, {num_qubits} qubits")

# Create Hamiltonian instance
# Note: coeffs from JSON might be a list of floats.
ham = Hamiltonian(paulis, coeffs)

print("Compiling Hamiltonian simulation circuit...")
# Compile with first-order Trotterization
qc = compile_hamiltonian_simulation(ham, time=1.0, order=1)

print(qc)

print("Compilation complete!")
print(f"Compiled Circuit Stats:")
print(f"  - Depth: {qc.depth()}")
print(f"  - Gate count: {qc.count_ops()}")
print(f"  - Qubits: {qc.num_qubits}")

# Verify Clifford gates are present (indicating simplification worked)
clifford_gates = [inst.operation.name for inst in qc.data if hasattr(inst.operation, 'name') and inst.operation.name.startswith('c')]
if clifford_gates:
    print(f"  - Clifford 2Q gates used: {len(clifford_gates)}")
    print(f"  - Sample: {clifford_gates[:5]}")
else:
    print("  - No Clifford 2Q gates found (simplification might not have triggered or only local terms).")



from canopus.utils import print_circ_info

print_circ_info(qc)
