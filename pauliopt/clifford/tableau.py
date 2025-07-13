from typing import List

import numpy as np

from pauliopt.circuits import (
    _get_qubits_qiskit,
    _get_phase_qiskit,
    QISKIT_CONVERSION,
    Circuit,
    AbstractCircuit,
)
from pauliopt.gates import (
    CliffordGate,
    SingleQubitClifford,
    TwoQubitClifford,
    Gate,
    H,
    V,
    Vdg,
    S,
    Sdg,
    CX,
    CY,
    CZ,
)


def mult_paulis(p1, p2, sign1, sign2, n_qubits):
    """
    Small helper function to multiply two Pauli strings and correctly update the sign.

    Args:
        p1 (np.ndarray): Pauli string 1
        p2 (np.ndarray): Pauli string 2
        sign1 (int): Sign of Pauli string 1
        sign2 (int): Sign of Pauli string 2
        n_qubits (int): Number of qubits in the Pauli strings

    Returns:
        np.ndarray: Pauli string 1 * Pauli string 2
    """
    x_1 = p1[:n_qubits].copy()
    z_1 = p1[n_qubits:].copy()
    x_2 = p2[:n_qubits].copy()
    z_2 = p2[n_qubits:].copy()

    x_1_z_2 = z_1 * x_2
    z_1_x_2 = x_1 * z_2

    ac = (x_1_z_2 + z_1_x_2) % 2

    x_1 = (x_1 + x_2) % 2
    z_1 = (z_1 + z_2) % 2

    x_1_z_2 = ((x_1_z_2 + x_1 + z_1) % 2) * ac
    sign_change = int(((np.sum(ac) + 2 * np.sum(x_1_z_2)) % 4) > 1)
    new_sign = (sign1 + sign2 + sign_change) % 4
    new_p1 = np.concatenate([x_1, z_1])
    return new_p1, new_sign


class CliffordTableau:
    """
    Class for storing and manipulating Clifford tableau.
    A Clifford tableau is a representation of a Clifford circuit as a
    2n x 2n binary matrix, where n is the number of qubits. The first n rows
    represent the stabilizers, and the last n rows represent the destabilizers.
    The first n columns represent the X operators, and the last n columns
    represent the Z operators.
    The sign of the operator in row i is given by the i-th entry of
    the sign vector.

    The clifford is initialized as the identity matrix with a zero sign vector.

    Args:
        n_qubits (int): Number of qubits in the clifford.


    A more readable representation of the clifford is given by the string:
    ```python
    >>> from pauliopt.clifford.clifford import CliffordTableau
    >>> ct = CliffordTableau(2)
    >>> print(ct)
    # Expected Output:
    # X/Z I/I | +
    # I/I X/Z | +
    >>> ct.append_h(0)
    >>> print(ct)
    # Expected Output:
    # Z/X I/I | +
    # I/I X/Z | +
    ```

    To get the raw $2n \times 2n$ matrix representation of the clifford, use:
    ```python
    >>> ct.clifford
    # Expected Output:
    # array([[0, 0, 1, 0],
    #        [0, 1, 0, 0],
    #        [1, 0, 0, 0],
    #        [0, 0, 0, 1]], dtype=uint8)
    ```
    To get the sign vector, use:
    ```python
    >>> ct.signs
    # Expected Output:
    # array([0, 0, 0, 0], dtype=uint8)
    ```

    """

    def __init__(self, n_qubits):
        self.tableau = np.eye(2 * n_qubits, dtype=np.uint8)
        self.signs = np.zeros((2 * n_qubits), dtype=np.uint8)
        self.n_qubits = n_qubits

    def __str__(self) -> str:
        out = str(self.string_repr(sep=" "))
        return out

    @staticmethod
    def from_tableau(tableau, signs):
        """
        Create a CliffordTableau from a clifford and sign vector.

        Args:
            tableau (np.ndarray): $2n \times 2n$ binary matrix representing the clifford.
            signs (np.ndarray): $2n$-dimensional binary vector representing the sign vector.

        Returns:
            CliffordTableau: CliffordTableau object.
        """
        n_qubits = tableau.shape[0] // 2
        if not (
            tableau.shape == (2 * n_qubits, 2 * n_qubits)
            and signs.shape == (2 * n_qubits,)
        ):
            raise ValueError(
                "Tableau and signs must have shape "
                "(2 * n_qubits, 2 * n_qubits) and (2 * n_qubits,)"
            )
        ct = CliffordTableau(n_qubits)
        ct.tableau = tableau
        ct.signs = signs
        return ct

    @staticmethod
    def from_qiskit_tableau(qiskit_ct: "qiskit.quantum_info.Clifford"):
        """
        Create a CliffordTableau from a qiskit Clifford object.

        Args:
            qiskit_ct (qiskit.quantum_info.Clifford): Clifford object.

        Returns:
            CliffordTableau: CliffordTableau object.
        """
        n_qubits = qiskit_ct.num_qubits
        ct = CliffordTableau(n_qubits)
        ct.tableau = qiskit_ct.symplectic_matrix
        ct.signs = qiskit_ct.phase
        return ct

    def string_repr(self, sep=" ", sign_sep="| "):
        """
        Get a string representation of the clifford.

        Args:
            sep (str): Separator between the pauli operators
            sign_sep (str): Separator between operators and sign.

        """
        out = ""
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                x_str = ["I", "X", "Z", "Y"][int(self._x_out(i, j))]
                z_str = ["I", "X", "Z", "Y"][int(self._z_out(i, j))]
                out += f"{x_str}/{z_str}" + sep
            out += sign_sep + f"{'+' if self.signs[i] == 0 else '-'} \n"
        return out

    def _x_out(self, row, col):
        """
        Get the X operator in row `row` and column `col`.

        Args:
            row (int): Row index.
            col (int): Column index.
        """
        return self.tableau[row, col] + 2 * self.tableau[row, col + self.n_qubits]

    def _z_out(self, row, col):
        """
        Get the Z operator in row `row` and column `col`.

        Args:
            row (int): Row index.
            col (int): Column index.
        """
        return (
            self.tableau[row + self.n_qubits, col]
            + 2 * self.tableau[row + self.n_qubits, col + self.n_qubits]
        )

    @property
    def x_matrix(self):
        """
        Binary matrix representing the X-Basis of the clifford tableau.
        :return:
        """
        x_matrx = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                x_matrx[i, j] = self._x_out(i, j)
        return x_matrx

    @property
    def z_matrix(self):
        """
        Binary matrix representing the Z-Basis of the clifford tableau.
        :return:
        """
        z_matrx = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                z_matrx[i, j] = self._z_out(i, j)
        return z_matrx

    def _xor_row(self, i, j):
        """
        XOR the value of row j to row i and adjust the signs accordingly.

        Args:
            i (int): Row index.
            j (int): Row index.
        """
        row_i = self.tableau[i, :]
        row_j = self.tableau[j, :]
        sign_i = self.signs[i]
        sign_j = self.signs[j]

        n = self.n_qubits
        row_j, sign_j = mult_paulis(row_i, row_j, sign_i, sign_j, n)
        self.insert_pauli_row(row_j, sign_j, i)

    def prepend_h(self, qubit):
        """
        Prepend a Hadamard gate to the clifford.

        Args:
            qubit (int): Qubit the hadamard gate is applied to.
        """
        idx0, idx1 = qubit, self.n_qubits + qubit
        self.signs[[idx0, idx1]] = self.signs[[idx1, idx0]]
        self.tableau[[idx0, idx1], :] = self.tableau[[idx1, idx0], :]

    def append_h(self, qubit):
        """
        Append a Hadamard gate to the clifford.

        Args:
            qubit (int): Qubit the hadamard gate is applied to.
        """
        idx0, idx1 = qubit, self.n_qubits + qubit
        self.signs += self.tableau[:, idx0] * self.tableau[:, idx1]
        self.signs %= 2
        self.tableau[:, [idx0, idx1]] = self.tableau[:, [idx1, idx0]]

    def prepend_s(self, qubit):
        """
        Prepend a S gate to the clifford.

        Args:
            qubit (int): Qubit the S gate is applied to.
        """
        self._xor_row(qubit, qubit + self.n_qubits)

    def append_s(self, qubit):
        """
        Append a S gate to the clifford.

        Args:
            qubit (int): Qubit the S gate is applied to.
        """
        idx0, idx1 = qubit, self.n_qubits + qubit
        self.signs += self.tableau[:, idx0] * self.tableau[:, idx1]
        self.signs %= 2
        self.tableau[:, idx1] ^= self.tableau[:, idx0]

    def prepend_cnot(self, control, target):
        """
        Prepend a CNOT gate to the clifford.

        Args:
            control (int): Control qubit.
            target (int): Target qubit.
        """
        self.signs = self.signs.astype(int)
        self._xor_row(control, target)
        self._xor_row(target + self.n_qubits, control + self.n_qubits)

    def append_cnot(self, control, target):
        """
        Append a CNOT gate to the clifford.

        Args:
            control (int): Control qubit.
            target (int): Target qubit.
        """
        x_ia = self.tableau[:, control]
        x_ib = self.tableau[:, target]

        z_ia = self.tableau[:, self.n_qubits + control]
        z_ib = self.tableau[:, self.n_qubits + target]

        control_n = control + self.n_qubits
        target_n = target + self.n_qubits

        self.tableau[:, target] ^= self.tableau[:, control]
        self.tableau[:, control_n] ^= self.tableau[:, target_n]

        self.signs += x_ia * z_ib * (x_ib + z_ia + 1)
        self.signs %= 2

    def insert_pauli_row(self, pauli, p_sign, row):
        """
        Insert a Pauli row into the clifford.

        Args:
            pauli (np.array): Pauli to be inserted.
            p_sign (int): Sign of the Pauli.
            row (int): Row to insert the Pauli.

        """
        self.tableau[row] = pauli.copy()
        self.signs[row] = p_sign

    def inverse(self):
        """
        Invert the clifford.


        Note: this is will create a deep copy of the clifford.

        Returns:
            CliffordTableau: Inverted clifford.

        """
        n_qubits = self.n_qubits
        assert self.tableau.shape == (2 * n_qubits, 2 * n_qubits)

        x2x = self.tableau[:n_qubits, :n_qubits].copy()
        z2z = self.tableau[n_qubits:, n_qubits:].copy()

        x2z = self.tableau[:n_qubits, n_qubits:].copy()
        z2x = self.tableau[n_qubits:, :n_qubits].copy()

        top_row = np.hstack((z2z.T, x2z.T))
        bottom_row = np.hstack((z2x.T, x2x.T))
        new_tableau = np.vstack((top_row, bottom_row))

        ct_new = CliffordTableau.from_tableau(new_tableau, self.signs.copy())
        ct_intermediate = self.apply(ct_new)
        ct_new.signs = (ct_new.signs + ct_intermediate.signs) % 2

        return ct_new

    def apply(self, other: "CliffordTableau"):
        """
        Apply a CliffordTableau to the current clifford.

        Note: this is will create a deep copy of the clifford.

        Args:
            other (CliffordTableau): CliffordTableau to apply.

        Returns:
            CliffordTableau: Applied CliffordTableau.
        """
        new_tableau = (self.tableau @ other.tableau) % 2

        phase = (other.tableau.dot(self.signs) + other.signs) % 2

        # Correcting for phase due to Pauli multiplication
        ifacts = np.zeros(2 * self.n_qubits, dtype=int)

        for k in range(2 * self.n_qubits):
            row2 = other.tableau[k]
            x2 = other.tableau[k, : self.n_qubits]
            z2 = other.tableau[k, self.n_qubits :]

            # Adding a factor of i for each Y in the image of an operator under the
            # first operation, since Y=iXZ

            ifacts[k] += np.sum(x2 * z2)

            # Adding factors of i due to qubit-wise Pauli multiplication

            for j in range(self.n_qubits):
                x = 0
                z = 0
                for i in range(2 * self.n_qubits):
                    if row2[i]:
                        x1 = self.tableau[i, j]
                        z1 = self.tableau[i, j + self.n_qubits]
                        if (x == 1 or z == 1) and (x1 == 1 or z1 == 1):
                            # determine the phase change due to the product of Pauli matrices,
                            # accounting for the possibilities of i, -i, and no phase change
                            val = np.mod(np.abs(3 * z1 - x1) - np.abs(3 * z - x) - 1, 3)
                            if val == 0:
                                ifacts[k] += 1
                            elif val == 1:
                                ifacts[k] -= 1
                        x ^= x1
                        z ^= z1

        p = np.mod(ifacts, 4) // 2
        phase = np.mod(phase + p, 2)

        return CliffordTableau.from_tableau(new_tableau, phase)

    def prepend_gate(self, gate: CliffordGate) -> None:
        for h_s_cx_gate in reversed(gate.get_h_s_cx_decomposition()):
            if h_s_cx_gate.name == "H":
                assert isinstance(h_s_cx_gate, SingleQubitClifford)
                self.prepend_h(h_s_cx_gate.qubit)
            elif h_s_cx_gate.name == "S":
                assert isinstance(h_s_cx_gate, SingleQubitClifford)
                self.prepend_s(h_s_cx_gate.qubit)
            elif h_s_cx_gate.name == "CX":
                assert isinstance(h_s_cx_gate, TwoQubitClifford)
                self.prepend_cnot(h_s_cx_gate.control, h_s_cx_gate.target)
            else:
                raise TypeError(f"Invalid H, S, CX decomposition of {gate.name}")

    def append_gate(self, gate: CliffordGate) -> None:
        for h_s_cx_gate in gate.get_h_s_cx_decomposition():
            if h_s_cx_gate.name == "H":
                assert isinstance(h_s_cx_gate, SingleQubitClifford)
                self.append_h(h_s_cx_gate.qubit)
            elif h_s_cx_gate.name == "S":
                assert isinstance(h_s_cx_gate, SingleQubitClifford)
                self.append_s(h_s_cx_gate.qubit)
            elif h_s_cx_gate.name == "CX":
                assert isinstance(h_s_cx_gate, TwoQubitClifford)
                self.append_cnot(h_s_cx_gate.control, h_s_cx_gate.target)
            else:
                raise TypeError(f"Invalid H, S, CX decomposition of {gate.name}")


class CliffordRegion(AbstractCircuit):
    """Circuit, that specifically consists only out of clifford gates."""

    def __init__(self, n_qubits, _gates: List[Gate] = None) -> None:
        super().__init__(n_qubits, _gates=_gates)

    @property
    def gates(self):
        return self._gates

    def _check_gate(self, gate):
        n_qubits = self.n_qubits
        if not isinstance(gate, CliffordGate):
            raise TypeError(
                f"{gate} is not a valid gate. All gates must be clifford gates."
            )

        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"{gate.qubits} are not unique.")

        if any(not (0 <= qubit < n_qubits) for qubit in gate.qubits):
            msg = f"{gate} acts out of range for {n_qubits} qubit circuit."
            raise ValueError(msg)

    @staticmethod
    def from_qiskit(qc: "qiskit.QuantumCircuit"):
        circ = CliffordRegion(qc.num_qubits)

        for inst in qc:
            qubits = _get_qubits_qiskit(inst.qubits, qc.qregs[0])
            phase = _get_phase_qiskit(inst.operation.params)
            circ.add_gate(QISKIT_CONVERSION[inst.operation.name](qubits, phase))

        return circ

    def to_tableau(self, append: bool = True) -> CliffordTableau:

        ct = CliffordTableau(self.n_qubits)

        for gate in self._gates:
            assert isinstance(gate, CliffordGate)
            if append:
                ct.append_gate(gate)
            else:
                ct.prepend_gate(gate)
        return ct

    def __iadd__(self, other: "Circuit"):
        for gate in other._gates:
            self.add_gate(gate)
        return self

    def __add__(self, other: "Circuit"):
        if self.n_qubits != other.n_qubits:
            print(self.n_qubits, other.n_qubits)
            raise Exception("Can only concatenate circuits with same number of qubits.")
        return CliffordRegion(self.n_qubits, self._gates + other._gates)

    def h(self, qubit):
        qubits = (qubit,)
        self.add_gate(H(*qubits))

    def v(self, qubit):
        qubits = (qubit,)
        self.add_gate(V(*qubits))

    def vdg(self, qubit):
        qubits = (qubit,)
        self.add_gate(Vdg(*qubits))

    def s(self, qubit):
        qubits = (qubit,)
        self.add_gate(S(*qubits))

    def sdg(self, qubit):
        qubits = (qubit,)
        self.add_gate(Sdg(*qubits))

    def cx(self, control, target):
        qubits = (control, target)
        self.add_gate(CX(*qubits))

    def cy(self, control, target):
        qubits = (control, target)
        self.add_gate(CY(*qubits))

    def cz(self, control, target):
        qubits = (control, target)
        self.add_gate(CZ(*qubits))
