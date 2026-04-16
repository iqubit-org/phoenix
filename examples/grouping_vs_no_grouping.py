import phoenix
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


ham = phoenix.Hamiltonian(['XXXZIYZI', 'YXXZIYYI', 'ZXXZIYZI'], [-0.0125, -0.0125, -0.0125])


ham = phoenix.Hamiltonian(['XXXZIYZI',
                            'YXXZIYYI',
                            'ZXXZIYZI',
                            'IXXZIYZZ',
                            'IXXZIYYY',
                            'IXXZIYZY',], [-0.0125, -0.0125, -0.0125, -0.0125, -0.0125, -0.0125])


ham.print_tableau()
u = ham.unitary_evolution()
qc_phoenix = phoenix.compile_hamiltonian_simulation(ham, optimize=False)
phoenix.utils.print_circ_info(qc_phoenix, title='Phoenix synthesized circuit')
print(qc_phoenix)

print('Infidelity', phoenix.utils.infidelity(u, Operator(qc_phoenix).to_matrix()))

