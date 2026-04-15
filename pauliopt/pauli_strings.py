from enum import Enum

PAULI_DICT = {
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("X", "I"): (1, "X"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Y", "I"): (1, "Y"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
    ("Z", "I"): (1, "Z"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("I", "I"): (1, "I"),
}


class Pauli(Enum):
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

    def __add__(self, other):
        if isinstance(other, Pauli):
            summands = [(1, self), (1, other)]
            return SummedPauliOp(summands, 1)
        elif isinstance(other, SummedPauliOp):
            if other.n_qubits != 1:
                raise ValueError("SummedPauliOp must have n_qubits=1")
            summands = [(1, self)] + other.summands
            return SummedPauliOp(summands, 1)
        else:
            raise ValueError("Cannot add Pauli to {}".format(type(other)))

    def __matmul__(self, other):
        if isinstance(other, Pauli):
            sign, res = PAULI_DICT[(self.value, other.value)]
            res = Pauli(res)
            return SummedPauliOp([(sign, res)], 1)
        elif isinstance(other, SummedPauliOp):
            if other.n_qubits != 1:
                raise ValueError("SummedPauliOp must have n_qubits=1")
            summands = []
            for sign, pauli in other.summands:
                new_sign, res = PAULI_DICT[(self.value, pauli.value)]
                res = Pauli(res)
                summands.append((sign * new_sign, res))
            return SummedPauliOp(summands, 1)
        elif isinstance(other, float):
            return SummedPauliOp([(other, self)], 1)
        else:
            raise ValueError("Cannot add Pauli to {}".format(type(other)))

    def __xor__(self, other):
        if isinstance(other, Pauli):
            return SummedPauliOp([(1, [self, other])], 2)


class SummedPauliOp:
    def __init__(self, summands, n_qubits):
        self.summands = summands
        self.n_qubits = n_qubits


I = Pauli.I
X = Pauli.X
Y = Pauli.Y
Z = Pauli.Z
