import qiskit.quantum_info as qi
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit._utils import with_gate_array
from qiskit.circuit.library import CXGate, HGate, SdgGate, SGate
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key


@with_gate_array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]])
class fSwapGate(SingletonGate):
    r"""fSWAP gate.

    .. math::

        fSWAP = (S^\dagger \otimes S^\dagger) \cdot \text{iSWAP}
            =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -1
            \end{pmatrix}

    .. code-block:: text

             ┌───┐┌───┐     ┌───┐ ┌───┐ ┌─────┐
        q_0: ┤ S ├┤ H ├──■──┤ X ├─┤ H ├─┤ Sdg ├
             ├───┤└───┘┌─┴─┐└─┬─┘┌┴───┴┐└─────┘
        q_1: ┤ S ├─────┤ X ├──■──┤ Sdg ├───────
             └───┘     └───┘     └─────┘       
    
    Reference: https://arxiv.org/abs/1807.07112
    """

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("fswap", 2, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (SGate(), [q[0]], []),
            (HGate(), [q[0]], []),
            (SGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (HGate(), [q[0]], []),
            (SdgGate(), [q[0]], []),
            (SdgGate(), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated=False):
        return fSwapGate()  # self-inverse

    def reverse_ops(self):
        return self.copy()

    def __eq__(self, other):
        return isinstance(other, fSwapGate)


# Pre-define rules and gates to avoid overhead
_GATE_CACHE = {
    "H": HGate(),
    "S": SGate(),
    "Sdg": SdgGate(),
    "CX": CXGate(),
    'fSwap': fSwapGate()
}

_CLIFFORD_RULES = {
    'cxx': [('H', 0), ('CX', (0, 1)), ('H', 0)],
    'cxy': [('H', 0), ('Sdg', 1), ('CX', (0, 1)), ('H', 0), ('S', 1)],
    'cxz': [('H', 0), ('H', 1), ('CX', (0, 1)), ('H', 0), ('H', 1)],
    'cyx': [('Sdg', 0), ('H', 0), ('CX', (0, 1)), ('H', 0), ('S', 0)],
    'cyy': [('Sdg', 0), ('H', 0), ('Sdg', 1), ('CX', (0, 1)),
            ('H', 0), ('S', 0), ('S', 1)],
    'cyz': [('Sdg', 0), ('H', 0), ('H', 1), ('CX', (0, 1)),
            ('H', 0), ('S', 0), ('H', 1)],
    'czx': [('CX', (0, 1))],
    'czy': [('Sdg', 1), ('CX', (0, 1)), ('S', 1)],
    'czz': [('H', 1), ('CX', (0, 1)), ('H', 1)],
    'fsxx': [('H', 0), ('H', 1), ('fSwap', (0, 1)), ('H', 0), ('H', 1)],
    'fsxy': [('H', 0), ('Sdg', 1), ('H', 1), ('fSwap', (0, 1)),
             ('H', 0), ('H', 1), ('S', 1)],
    'fsxz': [('H', 0), ('fSwap', (0, 1)), ('H', 0)],
    'fsyx': [('Sdg', 0), ('H', 0), ('H', 1), ('fSwap', (0, 1)),
             ('H', 0), ('S', 0), ('H', 1)],
    'fsyy': [('Sdg', 0), ('H', 0), ('Sdg', 1), ('H', 1), ('fSwap', (0, 1)),
             ('H', 0), ('S', 0), ('H', 1), ('S', 1)],
    'fsyz': [('Sdg', 0), ('H', 0), ('fSwap', (0, 1)), ('H', 0), ('S', 0)],
    'fszx': [('H', 1), ('fSwap', (0, 1)), ('H', 1)],
    'fszy': [('Sdg', 1), ('H', 1), ('fSwap', (0, 1)), ('H', 1), ('S', 1)],
    'fszz': [('fSwap', (0, 1))],
}


class CNOTEquivCliffordGate(Gate):
    r"""
    A custom 2-qubit Clifford gate defined by its conjugation properties.
    
    This gate implements a Clifford operation $C$ such that $C^\dagger (Z \otimes Z) C = P_0 \otimes P_1$,
    where $P_0, P_1 \in \{X, Y, Z\}$.
    
    It is equivalent to a CNOT up to single-qubit basis changes.
    """

    def __init__(self, pauli_0: str, pauli_1: str, label: str | None = None):
        pauli_0, pauli_1 = pauli_0.upper(), pauli_1.upper()
        if pauli_0 not in ['X', 'Y', 'Z'] or pauli_1 not in ['X', 'Y', 'Z']:
            raise ValueError(f"Invalid Paulis: {pauli_0}, {pauli_1}")

        self.pauli_0 = pauli_0
        self.pauli_1 = pauli_1

        # Name convention: c<p0><p1> e.g. cxy
        name = f"c{pauli_0.lower()}{pauli_1.lower()}"
        super().__init__(name, 2, [], label=label)

    def _define(self):
        """Define the gate decomposition in terms of standard gates."""
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)

        try:
            instructions = _CLIFFORD_RULES[self.name]
        except KeyError as exc:
            raise ValueError(f"Unsupported Clifford2QGate: {self.name}") from exc

        for gate_key, targets in instructions:
            gate = _GATE_CACHE[gate_key]
            if isinstance(targets, tuple):
                # Multi-qubit gate (CX)
                qc.append(gate, [q[i] for i in targets])
            else:
                # Single-qubit gate
                qc.append(gate, [q[targets]])

        self.definition = qc

    def _build_circuit(self) -> QuantumCircuit:
        """
        Return the circuit implementing the gate.
        Kept for backward compatibility and explicit circuit retrieval.
        """
        if self.definition is None:
            self._define()
        return self.definition

    def __array__(self, dtype=None, copy=None):
        """Return the matrix representation."""
        # Construct using SparsePauliOp for clarity and correctness
        # The gate corresponds to 0.5 * (II + P0_on_q0 + P1_on_q1 - P0_on_q0 * P1_on_q1)
        # Note: Qiskit tensor order is q1^q0. That is, P0 is on q0 -> Label "I" + P0; P1 is on q1 -> Label P1 + "I"

        p0 = self.pauli_0
        p1 = self.pauli_1
        op = qi.SparsePauliOp.from_list([
            ("II", 0.5),
            (f"I{p0}", 0.5),
            (f"{p1}I", 0.5),
            (f"{p1}{p0}", -0.5)
        ])

        return op.to_matrix()

    def inverse(self, annotated: bool = False):
        """Return the inverse gate (self-inverse)."""
        self.copy()
        return CNOTEquivCliffordGate(self.pauli_0, self.pauli_1)

    def reverse_ops(self):
        return self.copy()

    def __eq__(self, other):
        return (isinstance(other, CNOTEquivCliffordGate) and
                self.pauli_0 == other.pauli_0 and
                self.pauli_1 == other.pauli_1)


class fSwapEquivCliffordGate(Gate):
    """A custom 2-qubit Clifford gate equivalent to fSWAP up to single-qubit basis changes."""

    def __init__(self, pauli_0: str, pauli_1: str, label: str | None = None):
        pauli_0, pauli_1 = pauli_0.upper(), pauli_1.upper()
        if pauli_0 not in ['X', 'Y', 'Z'] or pauli_1 not in ['X', 'Y', 'Z']:
            raise ValueError(f"Invalid Paulis: {pauli_0}, {pauli_1}")

        self.pauli_0 = pauli_0
        self.pauli_1 = pauli_1

        # Name convention: fs<p0><p1> e.g. fsxy
        name = f"fs{pauli_0.lower()}{pauli_1.lower()}"
        super().__init__(name, 2, [], label=label)

    def _define(self):
        """Define the gate decomposition in terms of standard gates."""
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)

        try:
            instructions = _CLIFFORD_RULES[self.name]
        except KeyError as exc:
            raise ValueError(f"Unsupported fSwapEquivClifford: {self.name}") from exc

        for gate_key, targets in instructions:
            gate = _GATE_CACHE[gate_key]
            if isinstance(targets, tuple):
                # Multi-qubit gate (fSwap)
                qc.append(gate, [q[i] for i in targets])
            else:
                # Single-qubit gate
                qc.append(gate, [q[targets]])

        self.definition = qc

    def _build_circuit(self) -> QuantumCircuit:
        """
        Return the circuit implementing the gate.
        Kept for backward compatibility and explicit circuit retrieval.
        """
        if self.definition is None:
            self._define()
        return self.definition

    def __array__(self, dtype=None, copy=None):
        """Return the matrix representation."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")

        # Build the matrix by composing the circuit
        if self.definition is None:
            self._define()
        return qi.Operator(self.definition).data

    def inverse(self, annotated: bool = False):
        """Return the inverse gate."""
        # iSWAP† ≠ iSWAP, so the inverse requires conjugating with the inverse
        # For simplicity, return a gate that when decomposed gives the inverse circuit
        inv_gate = fSwapEquivCliffordGate(self.pauli_0, self.pauli_1)
        # Override the definition to be the inverse
        inv_gate._inverse = True
        return inv_gate

    def reverse_ops(self):
        return self.copy()

    def __eq__(self, other):
        return (isinstance(other, fSwapEquivCliffordGate) and
                self.pauli_0 == other.pauli_0 and
                self.pauli_1 == other.pauli_1)
