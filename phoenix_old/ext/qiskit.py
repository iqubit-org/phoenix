from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
import qiskit.quantum_info as qi
from math import pi
import numpy as np
from qiskit.circuit.parameterexpression import ParameterValueType



class CNOTEquivClifford(Gate):
    def __init__(self, pauli_0: str, pauli_1: str, label: str | None = None):
        """UCG Clifford2Q gate."""
        pauli_0, pauli_1 = pauli_0.upper(), pauli_1.upper()
        assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
        super().__init__(f"c{pauli_0.lower()}{pauli_1.lower()}", 2, [], label=label)
        self.pauli_0, self.pauli_1 = pauli_0, pauli_1


    def _build_circuit(self) -> QuantumCircuit:
        """Return the circuit implementing the CNOT-equivalent Clifford."""
        from qiskit.circuit.library import HGate, SGate, CXGate, SdgGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)

        gates = {
            "H": HGate(),
            "S": SGate(),
            "Sdg": SdgGate(),
            "CX": CXGate(),
        }

        rule_map: dict[str, list[tuple[str, int | tuple[int, int]]]] = {
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
        }

        try:
            instructions = rule_map[self.name]
        except KeyError as exc:
            raise ValueError(f"Unsupported Clifford2QGate: {self.name}") from exc

        for gate_key, targets in instructions:
            gate = gates[gate_key]
            if isinstance(targets, tuple):
                qargs = [q[targets[0]], q[targets[1]]]
            else:
                qargs = [q[targets]]
            qc._append(gate, qargs, [])

        return qc

    def _define(self):
        """Define the gate in terms of CNOTs and single qubit gates."""
        self.definition = self._build_circuit()

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
        return CNOTEquivClifford(self.pauli_0, self.pauli_1)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CNOTEquivClifford) and self.pauli_0 == other.pauli_0 and self.pauli_1 == other.pauli_1


class iSwapEquivClifford(Gate):
    def __init__(self, pauli_0: str, pauli_1: str, label: str | None = None):
        """iSWAP-equivalent Clifford gate."""
        pauli_0, pauli_1 = pauli_0.upper(), pauli_1.upper()
        assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
        super().__init__(f"is{pauli_0.lower()}{pauli_1.lower()}", 2, [], label=label)
        self.pauli_0, self.pauli_1 = pauli_0, pauli_1

    def _wrt_iswap(self):
        wrt0 = []
        if self.pauli_0 == 'X':
            wrt0.append('H')
        elif self.pauli_0 == 'Y':
            wrt0.extend(['H', 'S'])

        wrt1 = []
        if self.pauli_1 == 'X':
            wrt1.append('H')
        elif self.pauli_1 == 'Y':
            wrt1.extend(['H', 'S'])

        return wrt0, wrt1


    def _build_circuit(self) -> QuantumCircuit:
        """Return the circuit implementing the gate."""
        from qiskit.circuit.library import HGate, SGate, SdgGate, iSwapGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)

        gates = {
            "H": HGate(),
            "S": SGate(),
            "Sdg": SdgGate(),
            "iSwap": iSwapGate(),
        }

        rule_map: dict[str, list[tuple[str, int | tuple[int, int]]]] = {
            'isxx': [('H', 0), ('H', 1), ('iSwap', (0, 1)), ('H', 0), ('H', 1)],
            'isxy': [('H', 0), ('Sdg', 1), ('H', 1), ('iSwap', (0, 1)),
                     ('H', 0), ('H', 1), ('S', 1)],
            'isxz': [('H', 0), ('iSwap', (0, 1)), ('H', 0)],
            'isyx': [('Sdg', 0), ('H', 0), ('H', 1), ('iSwap', (0, 1)),
                     ('H', 0), ('S', 0), ('H', 1)],
            'isyy': [('Sdg', 0), ('H', 0), ('Sdg', 1), ('H', 1), ('iSwap', (0, 1)),
                     ('H', 0), ('S', 0), ('H', 1), ('S', 1)],
            'isyz': [('Sdg', 0), ('H', 0), ('iSwap', (0, 1)), ('H', 0), ('S', 0)],
            'iszx': [('H', 1), ('iSwap', (0, 1)), ('H', 1)],
            'iszy': [('Sdg', 1), ('H', 1), ('iSwap', (0, 1)), ('H', 1), ('S', 1)],
            'iszz': [('iSwap', (0, 1))],
        }

        try:
            instructions = rule_map[self.name]
        except KeyError as exc:
            raise ValueError(f"Unsupported iSwapEquivClifford: {self.name}") from exc

        for gate_key, targets in instructions:
            gate = gates[gate_key]
            if isinstance(targets, tuple):
                qargs = [q[targets[0]], q[targets[1]]]
            else:
                qargs = [q[targets]]
            qc._append(gate, qargs, [])

        return qc

    def _define(self):
        self.definition = self._build_circuit()

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return qi.Operator(self._build_circuit()).reverse_qargs().to_matrix()

    def inverse(self, annotated: bool = False):
        return super().inverse()

    def __eq__(self, other):
        return isinstance(other, iSwapEquivClifford) and self.pauli_0 == other.pauli_0 and self.pauli_1 == other.pauli_1



class CanonicalGate(Gate):
    r"""Canonical representation of any 2-qubit gate.

    **Circuit symbol:**

    .. code-block:: text
          ┌─────────────┐
        ──┤0            ├──
          │  Can(a,b,c) │
        ──┤1            ├──
          └─────────────┘

    .. math::
        \mathrm{Can}(a, b, c) = e^{- i \frac{\pi}{2}(a XX + b YY + c ZZ)} where 0.5 ≥ a ≥ b ≥ |c|
    """

    def __init__(
            self,
            a: ParameterValueType, b: ParameterValueType, c: ParameterValueType,
            label: str | None = None
    ):
        super().__init__("can", 2, [a, b, c], label=label)

    def inverse(self, annotated: bool = False):
        return CanonicalGate(-self.params[0], -self.params[1], -self.params[2])

    def _define(self):
        """
        gate can(theta, phi, lam) q0,q1 {
            rxx(theta) q0, q1;
            ryy(phi) q0, q1;
            rzz(lam) q0, q1;
        }

        gate can(theta, phi, lam) q0,q1 {
            u3(1.5*pi, 0.0, 1.5*pi) q1;
            u3(0.5*pi, 1.5*pi, 0.5*pi) q0;
            cx q1, q0;
            u3(1.5*pi, theta + pi, 0.5*pi) q1;
            u3(pi, 0.0, phi + pi) q0;
            cx q1, q0;
            u3(0.5*pi, 0.0, 0.5*pi) q1;
            u3(0.0, 1.5*pi, lam + 0.5*pi) q0;
            cx q1, q0;
        }
        """
        from qiskit.circuit.library import UGate, CXGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (UGate(1.5 * pi, 0.0, 1.5 * pi), [q[1]], []),
            (UGate(0.5 * pi, 1.5 * pi, 0.5 * pi), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (UGate(1.5 * pi, self.params[0] * pi + pi, 0.5 * pi), [q[1]], []),
            (UGate(pi, 0.0, self.params[1] * pi + pi), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (UGate(0.5 * pi, 0.0, 0.5 * pi), [q[1]], []),
            (UGate(0.0, 1.5 * pi, self.params[2] * pi + 0.5 * pi), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the U1 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        a, b, c = (float(param) for param in self.params)
        mat = canonical_unitary(a, b, c)
        return qi.Operator(mat).reverse_qargs().to_matrix()

    def __eq__(self, other):
        if isinstance(other, CanonicalGate):
            return self._compare_parameters(other)
        return False




def canonical_unitary(a: float, b: float, c: float) -> np.ndarray:
    r"""Return the unitary matrix of the canonical gate \mathrm{Can}(a, b, c) = e^{- i \frac{\pi}{2}(a XX + b YY + c ZZ)}.

    Args:
        a (float): parameter a
        b (float): parameter b
        c (float): parameter c

    Returns:
        np.ndarray: 4x4 unitary matrix
    """

    half_pi = 0.5 * np.pi
    x = a * half_pi
    y = b * half_pi
    z = c * half_pi

    x_minus_y = x - y
    x_plus_y = x + y
    cosm = np.cos(x_minus_y)
    cosp = np.cos(x_plus_y)
    sinm = np.sin(x_minus_y)
    sinp = np.sin(x_plus_y)

    eim = np.exp(-1j * z)
    eip = np.exp(1j * z)

    return np.array(
        [
            [eim * cosm, 0.0, 0.0, -1j * eim * sinm],
            [0.0, eip * cosp, -1j * eip * sinp, 0.0],
            [0.0, -1j * eip * sinp, eip * cosp, 0.0],
            [-1j * eim * sinm, 0.0, 0.0, eim * cosm],
        ],
        dtype=np.complex128,
    )
