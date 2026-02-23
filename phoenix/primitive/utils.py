from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian



@dataclass
class SimplificationStep:
    clifford: CNOTEquivCliffordGate | fSwapEquivCliffordGate
    local_hamiltonian: Hamiltonian
    qubits: tuple[int, int]


def constr_circuit_from_simp_steps(ham: Hamiltonian, steps: list[SimplificationStep]) -> QuantumCircuit:
    qc_pre = QuantumCircuit(ham.num_qubits)
    qc_post = QuantumCircuit(ham.num_qubits)

    for step in steps:
        qc_post.append(PauliEvolutionGate(step.local_hamiltonian), range(ham.num_qubits))
        qc_post.append(step.clifford, step.qubits)

        qc_pre.append(step.clifford, step.qubits)
    
    qc_post = qc_post.reverse_ops()
    qc_pre.append(PauliEvolutionGate(ham), range(ham.num_qubits)) # TODO: PauliEvolutionGate(ham) kernel to be optimized/rebased?

    return qc_pre.compose(qc_post).decompose('PauliEvolution')

