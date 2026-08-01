import phoenix
from qiskit import QuantumCircuit
import numpy as np
from rich.console import Console
from qiskit.quantum_info import Operator

console = Console()

paulis = [
    "YYYXZI",
    "ZZYXZI",
    "IYYZXY",
    "IZYZXX",
]
paulis = [p[::-1] for p in paulis]
coeffs = np.random.rand(len(paulis)) / 100
ham = phoenix.Hamiltonian(paulis, coeffs)

console.rule("Naive synthesis")
qc = ham.generate_circuit(decompose=False)
phoenix.utils.print_circ_info(qc)
print(qc.draw(fold=-1))
u = Operator(qc).to_matrix()

console.rule("Support Grouping")
qc = phoenix.compile_hamiltonian_simulation(ham, optimize=False, grouping='support')
phoenix.utils.print_circ_info(qc)
phoenix.utils.print_circ_info(phoenix.optimize_phoenix_circuit_by_qiskit(qc))
print(qc.draw(fold=-1))
print('infidelity:', phoenix.utils.infidelity(Operator(qc).to_matrix(), u))

console.rule("Holistic Grouping")
qc = phoenix.compile_hamiltonian_simulation(ham, optimize=False, grouping='holistic', terminal='replay', rho_threshold=1.0)
phoenix.utils.print_circ_info(qc)
phoenix.utils.print_circ_info(phoenix.optimize_phoenix_circuit_by_qiskit(qc))
print(qc.draw(fold=-1))
print('infidelity:', phoenix.utils.infidelity(Operator(qc).to_matrix(), u))
