from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate
import qiskit.quantum_info as qi


class Clifford2QGate(Gate):
    def __init__(self, pauli_0: str, pauli_1: str, label: str | None = None, *, duration=None, unit="dt"):
        """UCG Clifford2Q gate."""
        assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
        super().__init__(f"c{pauli_0.lower()}{pauli_1.lower()}", 2, [], label=label, duration=duration, unit=unit)
        self.pauli_0, self.pauli_1 = pauli_0, pauli_1

    def _define(self):
        """Define the gate in terms of CNOTs and single qubit gates."""
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library import HGate, SGate, CXGate
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)

        if self.name == 'cxx':
            rules = [
                (HGate(), [q[0]], []),
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[0]], []),
            ]
        elif self.name == 'cxy':
            rules = [
                (HGate(), [q[0]], []),
                (SGate().inverse(), [q[1]], []),  # sdg
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[0]], []),
                (SGate(), [q[1]], []),  # s
            ]
        elif self.name == 'cxz':
            rules = [
                (HGate(), [q[0]], []),
                (HGate(), [q[1]], []),
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[0]], []),
                (HGate(), [q[1]], []),
            ]
        elif self.name == 'cyx':
            rules = [
                (SGate().inverse(), [q[0]], []),  # sdg
                (HGate(), [q[0]], []),
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[0]], []),
                (SGate(), [q[0]], []),  # s
            ]
        elif self.name == 'cyy':
            rules = [
                (SGate().inverse(), [q[0]], []),  # sdg
                (HGate(), [q[0]], []),
                (SGate().inverse(), [q[1]], []),  # sdg
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[0]], []),
                (SGate(), [q[0]], []),  # s
                (SGate(), [q[1]], []),  # s
            ]
        elif self.name == 'cyz':
            rules = [
                (SGate().inverse(), [q[0]], []),  # sdg
                (HGate(), [q[0]], []),
                (HGate(), [q[1]], []),
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[0]], []),
                (SGate(), [q[0]], []),  # s
                (HGate(), [q[1]], []),
            ]
        elif self.name == 'czx':
            rules = [
                (CXGate(), [q[0], q[1]], []),
            ]
        elif self.name == 'czy':
            rules = [
                (SGate().inverse(), [q[1]], []),  # sdg
                (CXGate(), [q[0], q[1]], []),
                (SGate(), [q[1]], []),  # s
            ]
        elif self.name == 'czz':
            rules = [
                (HGate(), [q[1]], []),
                (CXGate(), [q[0], q[1]], []),
                (HGate(), [q[1]], []),
            ]
        else:
            raise ValueError(f"Unsupported Clifford2QGate: {self.name}")

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the U1 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        I = qi.Pauli('I')
        P0, P1 = qi.Pauli(self.pauli_0), qi.Pauli(self.pauli_1)
        mat = qi.SparsePauliOp([I ^ I, P0 ^ I, I ^ P1, P0 ^ P1],
                               [1 / 2, 1 / 2, 1 / 2, -1 / 2]).to_matrix()
        return qi.Operator(mat).reverse_qargs().to_matrix()

    def inverse(self, annotated: bool = False):
        return Clifford2QGate(self.pauli_0, self.pauli_1)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, Clifford2QGate) and self.pauli_0 == other.pauli_0 and self.pauli_1 == other.pauli_1
