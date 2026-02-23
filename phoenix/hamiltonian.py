import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from scipy import linalg


class Hamiltonian(SparsePauliOp):
    """
    A customized Hamiltonian class inheriting from qiskit.quantum_info.SparsePauliOp.
    """
    def __init__(
        self,
        data: PauliList | SparsePauliOp | Pauli | list | str,
        coeffs: np.ndarray | None = None,
        *,
        ignore_pauli_phase: bool = False,
        copy: bool = True,
    ):
        # ! Note that pauli list is in little-endian order in Qiskit
        temp_op = SparsePauliOp(data, coeffs, ignore_pauli_phase=ignore_pauli_phase, copy=copy)
        simplified = temp_op.simplify()        
        super().__init__(simplified.paulis, simplified.coeffs, ignore_pauli_phase=False, copy=False)

    def unitary_evolution(self, t: float = 1.0) -> np.ndarray:
        """Generate the corresponding unitary evolution operator."""
        return linalg.expm(-1j * self.to_matrix() * t)

    def normalize(self) -> 'Hamiltonian':
        """Return a normalized version of the Hamiltonian."""
        norm = self.norm()
        if norm == 0:
            return self
        return Hamiltonian(self.paulis, self.coeffs / norm)

    def norm(self) -> float:
        """Return the norm of the Hamiltonian (sum of absolute coefficients)."""
        # Note: The original implementation used spectral norm of each term * coeff.
        # Since Pauli matrices have spectral norm 1, this simplifies to sum(|coeff|).
        return np.sum(np.abs(self.coeffs))
    
    def group_same_weights(self) -> list['Hamiltonian']:
        """Group Pauli strings by their nontrivial parts."""
        from .primitive.grouping import group_paulis_and_coeffs
        
        return [Hamiltonian(pls, coes) for pls, coes in group_paulis_and_coeffs(self.paulis.to_labels(), self.coeffs).values()]
    
    def tableau(self, arrange='xz', with_phase=False) -> np.ndarray:
        """Return the tableau representation of the Hamiltonian."""
        if arrange == 'xz':
            parts = [self.paulis.x, self.paulis.z]
        elif arrange == 'zx':
            parts = [self.paulis.z, self.paulis.x]
        else:
            raise ValueError("arrange must be 'xz' or 'zx'")

        if with_phase:
            parts.append(np.expand_dims(self.paulis.phase, axis=-1))
        return np.hstack(parts).astype(int)

    def print_tableau(self, arrange='xz'):
        from prettytable import PrettyTable

        table = PrettyTable()
        if arrange == 'xz':
            table.field_names = ['Pauli', 'X part', 'Z part', 's']
        elif arrange == 'zx':
            table.field_names = ['Pauli', 'Z part', 'X part', 's']
        else:
            raise ValueError("arrange must be 'xz' or 'zx'")
        for pauli, xz, sign in zip(self.paulis.to_labels(), self.tableau(arrange=arrange, with_phase=True), self.paulis.phase):
            x_part = xz[:self.num_qubits].astype(str)
            z_part = xz[self.num_qubits:2*self.num_qubits].astype(str)
            if arrange == 'xz':
                table.add_row([pauli[::-1], ' '.join(x_part), ' '.join(z_part), sign])
            else:
                table.add_row([pauli[::-1], ' '.join(z_part), ' '.join(x_part), sign])

        print(table)

    @property
    def total_weight(self) -> int:
        if not self.size:  # it is an empty tableau
            return 0
        if ''.join(self.paulis.to_labels()).count('I') == self.size * self.num_qubits:  # all are I
            return 0
        if not self.num_nonlocal_paulis:  # only 1Q rotations
            return 1
        mat = self.with_ops[self.which_nonlocal_paulis]
        return np.bitwise_or.reduce(mat, axis=0).sum()

    @property
    def with_ops(self) -> np.ndarray:
        return self.paulis.x | self.paulis.z
    
    @property
    def active_qubits(self):
        """Which qubits involve non-identity operations."""
        return np.where(np.any(self.with_ops, axis=0))[0]

    @property
    def num_nonlocal_paulis(self) -> int:
        return np.sum(self.with_ops.sum(axis=1) > 1)
    
    @property
    def num_local_paulis(self) -> int:
        return np.sum(self.with_ops.sum(axis=1) <= 1)
    
    @property
    def reverse(self) -> 'Hamiltonian':
        """Reverse the order of Pauli exponentiations"""
        return Hamiltonian(self.paulis[::-1], self.coeffs[::-1])

    @property
    def which_nonlocal_paulis(self) -> np.ndarray:
        return np.where(self.with_ops.sum(axis=1) > 1)[0]

    @property
    def which_local_paulis(self) -> np.ndarray:
        return np.where(self.with_ops.sum(axis=1) <= 1)[0]

    def apply_clifford(self, cliff, *qubits, inplace=False, frame='s') -> 'Hamiltonian':
        qc = QuantumCircuit(self.num_qubits)
        qc.append(cliff, qubits)
        
        if inplace:
            self.paulis =  self.paulis.evolve(qc, frame=frame)
            return self
        else:
            return Hamiltonian(self.paulis.evolve(qc, frame=frame), self.coeffs)

    def separate_local_nonlocal(self) -> tuple['Hamiltonian', 'Hamiltonian']:
        """
        Separate Hamiltonian into local (weight <= 1) and non-local parts.
        Returns (local_ham, nonlocal_ham).
        """
        weights = np.sum(self.with_ops, axis=1)
        local_mask = weights <= 1
        nonlocal_mask = ~local_mask
        
        local_ham = Hamiltonian(self.paulis[local_mask], self.coeffs[local_mask])
        nonlocal_ham = Hamiltonian(self.paulis[nonlocal_mask], self.coeffs[nonlocal_mask])
        
        return local_ham, nonlocal_ham

    def to_pauli_evolution_gate(self) -> PauliEvolutionGate:
        return PauliEvolutionGate(self)
    
    def generate_circuit(self, time: float | Parameter = 1.0) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.append(PauliEvolutionGate(self, time), range(self.num_qubits))
        return qc.decompose()
