"""
Quantum Gate
"""
import cirq
import numpy as np
from typing import List, Union
from math import pi
from numpy import ndarray
from scipy import linalg
from numpy import exp, sin, cos, sqrt
from copy import copy


class Gate:
    """Single-qubit gate and controlled two-qubit gate class"""

    def __init__(self, data: ndarray, name: str = None, *args, **kwargs):
        """

        Args:
            data: unitary matrix operating on target qubits, ndarray type
            tqs: target qubit indices
            cqs: control qubit indices, optional
            name: name of the quantum gate, optional
        """
        from phoenix.utils.functions import is_power_of_two

        self.name = name
        self._targ_qubits = []
        self._ctrl_qubits = []
        if not is_power_of_two(data.shape[0]) or not np.allclose(data @ data.conj().T, np.identity(data.shape[0])):
            raise ValueError('data is not valid quantum gate operator')
        self.n_qubits = int(np.log2(data.shape[0]))  # initial n_qubits is defined by data.shape
        self.data = data.astype(complex)

        # parameters for transforming to QASM
        self.angle = kwargs.get('angle', None)
        self.angles = kwargs.get('angles', None)
        self.params = kwargs.get('params', None)
        self.exponent = kwargs.get('exponent', None)

    def __hash__(self):
        return hash(id(self))

    def copy(self):
        return copy(self)

    def on(self, tqs: Union[List[int], int], cqs: Union[List[int], int] = None):
        """
        Operate on specific qubits.

        Args:
            tqs: target qubit indices
            cqs: control qubit indices, optional

        Returns: Quantum Gate with  qubit indices.
        """
        if isinstance(tqs, int):
            tqs = [tqs]

        if cqs is None:
            cqs = []
        elif isinstance(cqs, int):
            cqs = [cqs]

        if not tqs:
            raise ValueError('target qubits are not specified')

        if set(tqs) & set(cqs):
            raise ValueError('target qubits and control qubits are overlapped')

        g = self.copy()

        if isinstance(g, UnivGate) and len(tqs) != g.n_qubits:
            raise ValueError('number of target qubits does not coincide with the gate definition')

        if g.n_qubits > 1:  # only single-qubit gate can be expanded to identical tensor-product gate
            if len(tqs) > g.n_qubits or len(tqs) < g.n_qubits:
                raise ValueError('number of target qubits does not coincide with the gate definition')

        g.n_qubits = len(tqs)  # new number of target qubits
        g._targ_qubits = tqs
        if cqs:
            g._ctrl_qubits = cqs
        return g

    def __call__(self, *args, **kwargs):
        return self.on(*args, **kwargs)

    def __repr__(self) -> str:
        prefix = self.name
        if self.angle:
            prefix += '({:.2f}π)'.format(self.angle / pi)
        elif self.angles:
            prefix += '({})'.format(','.join(['{:.2f}π'.format(a / pi) for a in self.angles]))
        elif self.params:
            prefix += '({})'.format(','.join([str(p) for p in self.params]))
        elif self.exponent:
            prefix += '({})'.format(self.exponent)

        tqs_str = str(self._targ_qubits[0]) if len(self._targ_qubits) == 1 else '|'.join(
            [str(tq) for tq in self._targ_qubits])
        cqs_str = str(self._ctrl_qubits[0]) if len(self._ctrl_qubits) == 1 else '|'.join(
            [str(cq) for cq in self._ctrl_qubits])
        if not self._targ_qubits:
            return prefix
        if not self._ctrl_qubits:
            return '{}{{{}}}'.format(prefix, tqs_str)
        return '{}{{{}←{}}}'.format(prefix, tqs_str, cqs_str)

    def math_repr(self):
        tqs_str = str(self._targ_qubits[0]) if len(self._targ_qubits) == 1 else '|'.join(
            [str(tq) for tq in self._targ_qubits])
        cqs_str = str(self._ctrl_qubits[0]) if len(self._ctrl_qubits) == 1 else '|'.join(
            [str(cq) for cq in self._ctrl_qubits])
        g_name = self.name
        if g_name == 'SDG':
            g_name = 'S^\dagger'
        if g_name == 'TDG':
            g_name = 'T^\dagger'
        if not self._ctrl_qubits:
            return '${}_{{{}}}$'.format(g_name, tqs_str)
        return '${}_{{{}←{}}}$'.format(g_name, tqs_str, cqs_str)

    @property
    def tq(self):
        if len(self._targ_qubits) > 1:
            raise ValueError('Gate {} has more than 1 target qubit'.format(self.name))
        if not self._targ_qubits:
            raise ValueError('Gate {} has no target qubit'.format(self.name))
        return self._targ_qubits[0]

    @property
    def cq(self):
        if len(self._ctrl_qubits) > 1:
            raise ValueError('Gate {} has more than 1 control qubit'.format(self.name))
        if not self._ctrl_qubits:
            raise ValueError('Gate {} has no control qubit'.format(self.name))
        return self._ctrl_qubits[0]

    @property
    def tqs(self):
        return self._targ_qubits

    @property
    def cqs(self):
        return self._ctrl_qubits

    @property
    def qregs(self):
        return self._targ_qubits if not self._ctrl_qubits else self._ctrl_qubits + self._targ_qubits

    @property
    def num_qregs(self):
        """Number of qubits operated by the gate (both target and control qubits)"""
        return len(self.qregs)

    def hermitian(self):
        """
        Hermitian conjugate of the origin gate

        Returns: a new Gate instance
        """
        if np.allclose(self.data.conj().T, self.data):
            return self.copy()

        g = self.copy()
        g.data = self.data.conj().T
        if g.name.endswith('†'):
            g.name = g.name[:-1]
        else:
            g.name = g.name + '†'
        return g


class UnivGate(Gate):
    """Universal quantum gate"""

    def __init__(self, data, name=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.name = name if name else 'U'


# Fixed Gate
class XGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[0. + 0.j, 1. + 0.j],
                                   [1. + 0.j, 0. + 0.j]]), name='X', *args, **kwargs)


class YGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[0. + 0.j, -1.j],
                                   [1.j, 0. + 0.j]]), name='Y', *args, **kwargs)


class ZGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, -1. + 0.j]]), name='Z', *args, **kwargs)


class IGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.identity(2).astype(complex), name='I', *args, **kwargs)


class SGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 1.j]]), name='S', *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return SDG.on(self.tqs, self.cqs)
        return SDGGate()


class SDGGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. - 1.j]]), name='SDG', *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return S.on(self.tqs, self.cqs)
        return SGate()


class TGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, np.exp(1.j * pi / 4)]]), name='T', *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return TDG.on(self.tqs, self.cqs)
        return TDGGate()


class TDGGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, np.exp(- 1.j * pi / 4)]]), name='TDG', *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return T.on(self.tqs, self.cqs)
        return TGate()


class HGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 1. + 0.j],
                                   [1. + 0.j, -1. + 0.j]]) / sqrt(2), name='H', *args, **kwargs)


class SWAPGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]), name='SWAP', *args, **kwargs)


class SQSWGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1, 0, 0, 0],
                                   [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                                   [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
                                   [0, 0, 0, 1]
                                   ]), name='SQSW', *args, **kwargs)


class ISWAPGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 1.j, 0. + 0.j],
                                   [0. + 0.j, 1.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]), name='ISWAP', *args, **kwargs)


class SQiSWGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1, 0, 0, 0],
                                   [0, 1 / sqrt(2), 1j / sqrt(2), 0],
                                   [0, 1j / sqrt(2), 1 / sqrt(2), 0],
                                   [0, 0, 0, 1]
                                   ]), name='SQiSW', *args, **kwargs)


class BGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[np.cos(np.pi / 8), 0, 0, 1j * np.sin(np.pi / 8)],
                                   [0, np.cos(3 * np.pi / 8), 1j * np.sin(3 * np.pi / 8), 0],
                                   [0, 1j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                                   [1j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)]
                                   ]), name='B', *args, **kwargs)


class SycamoreGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(FSim(np.pi / 2, np.pi / 6).data, name='SYC', *args, **kwargs)


# Rotation Gate

class RX(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * X.data), name='RX', angle=theta, *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return RX(-self.angle).on(self.tqs, self.cqs)
        return RX(-self.angle)


class RY(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * Y.data), name='RY', angle=theta, *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return RY(-self.angle).on(self.tqs, self.cqs)
        return RY(-self.angle)


class RZ(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * Z.data), name='RZ', angle=theta, *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return RZ(-self.angle).on(self.tqs, self.cqs)
        return RZ(-self.angle)


class PhaseShift(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(np.array([[1. + 0., 0. + 0.],
                                   [0. + 0., np.exp(1j * theta)]]), name='P', angle=theta, *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return PhaseShift(-self.angle).on(self.tqs, self.cqs)
        return PhaseShift(-self.angle)


class GlobalPhase(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(np.array([[np.exp(1j * theta), 0. + 0.],
                                   [0. + 0., np.exp(1j * theta)]]), name='GlobalPhase', angle=theta, *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return GlobalPhase(-self.angle).on(self.tqs, self.cqs)
        return GlobalPhase(-self.angle)


class XPow(Gate):
    """X power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (X.data - I.data)), name='XPow', exponent=exponent,
                         *args, **kwargs)
        assert np.allclose(self.data, np.exp(1j * exponent * pi / 2) * RX(pi * exponent).data)

    def hermitian(self):
        if self.tqs:
            return XPow(-self.exponent).on(self.tqs, self.cqs)
        return XPow(-self.exponent)


class YPow(Gate):
    """Y power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (Y.data - I.data)), name='YPow', exponent=exponent,
                         *args, **kwargs)
        assert np.allclose(self.data, np.exp(1j * exponent * pi / 2) * RY(pi * exponent).data)

    def hermitian(self):
        if self.tqs:
            return YPow(-self.exponent).on(self.tqs, self.cqs)
        return YPow(-self.exponent)


class ZPow(Gate):
    """Z power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (Z.data - I.data)), name='ZPow', exponent=exponent,
                         *args, **kwargs)
        assert np.allclose(self.data, np.exp(1j * exponent * pi / 2) * RZ(pi * exponent).data)

    def hermitian(self):
        if self.tqs:
            return ZPow(-self.exponent).on(self.tqs, self.cqs)
        return ZPow(-self.exponent)


class U1(Gate):
    """U1 gate"""

    def __init__(self, lamda, *args, **kwargs):
        super().__init__(np.array([[1. + 0., 0. + 0.],
                                   [0. + 0., np.exp(1j * lamda)]]), name='U1', angle=lamda, *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return U1(-self.angle).on(self.tqs, self.cqs)
        return U1(-self.angle)


class U2(Gate):
    """U2 gate"""

    def __init__(self, phi, lamda, *args, **kwargs):
        super().__init__(np.array([[1, - exp(1j * lamda)],
                                   [exp(1j * phi), exp(1j * (phi + lamda))]]) / sqrt(2),
                         name='U2', angles=(phi, lamda), *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return U3(-pi / 2, -self.angles[1], -self.angles[0]).on(self.tqs, self.cqs)
        return U3(-pi / 2, -self.angles[1], -self.angles[0])


class U3(Gate):
    """U3 gate"""

    def __init__(self, theta, phi, lamda, *args, **kwargs):
        super().__init__(np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
                                   [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]]),
                         name='U3', angles=(theta, phi, lamda), *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return U3(-self.angles[0], -self.angles[2], -self.angles[1]).on(self.tqs, self.cqs)
        return U3(-self.angles[0], -self.angles[2], -self.angles[1])


class RXX(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * np.kron(X.data, X.data)), name='RXX', angle=theta, *args,
                         **kwargs)

    def hermitian(self):
        if self.tqs:
            return RXX(-self.angle).on(self.tqs, self.cqs)
        return RXX(-self.angle)


class RYY(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * np.kron(Y.data, Y.data)), name='RYY', angle=theta, *args,
                         **kwargs)

    def hermitian(self):
        if self.tqs:
            return RYY(-self.angle).on(self.tqs, self.cqs)
        return RYY(-self.angle)


class RZZ(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * np.kron(Z.data, Z.data)), name='RZZ', angle=theta, *args,
                         **kwargs)

    def hermitian(self):
        if self.tqs:
            return RZZ(-self.angle).on(self.tqs, self.cqs)
        return RZZ(-self.angle)


class FSim(Gate):
    def __init__(self, theta, phi, *args, **kwargs):
        super().__init__(np.array([[1, 0, 0, 0],
                                   [0, cos(theta), -1j * sin(theta), 0],
                                   [0, -1j * sin(theta), cos(theta), 0],
                                   [0, 0, 0, exp(-1j * phi)]]), name='FSim', angles=(theta, phi), *args, **kwargs)

    def hermitian(self):
        if self.tqs:
            return FSim(-self.angles[0], -self.angles[1]).on(self.tqs, self.cqs)
        return FSim(-self.angles[0], -self.angles[1])


class Clifford2QGate(Gate):
    """Use universal controlled gates to represent generic 2Q Clifford gates"""

    def __init__(self, pauli_0: str, pauli_1: str):
        assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
        import qiskit.quantum_info as qi
        self.pauli_0, self.pauli_1 = pauli_0, pauli_1
        I = qi.Pauli('I')
        P0, P1 = qi.Pauli(pauli_0), qi.Pauli(pauli_1)
        super().__init__(qi.SparsePauliOp([I ^ I, P0 ^ I, I ^ P1, P0 ^ P1],
                                          [1 / 2, 1 / 2, 1 / 2, -1 / 2]).to_matrix(),
                         name='C({}, {})'.format(pauli_0, pauli_1))

    def hermitian(self):
        return self.copy()

    # def __eq__(self, other):
    #     is_same_type = self.pauli_0 == other.pauli_0 and self.pauli_1 == other.pauli_1
    #     if self.tqs:
    #         return is_same_type and self.tqs == other.tqs and self.cqs == other.cqs
    #     return is_same_type

    # def __hash__(self):
    #     return hash((self.pauli_0, self.pauli_1, tuple(self.tqs), tuple(self.cqs)))


class Canonical(Gate):
    r"""
    Canonical gate with respect to Weyl chamber

    .. math::
        \textrm{Can}(\theta_1, \theta_2, \theta_3) = e^{- i \frac{1}{2}(\theta_1 XX + \theta_2 YY + \theta_3 ZZ)}
    """

    def __init__(self, theta1, theta2, theta3, *args, **kwargs):
        super().__init__(linalg.expm(-1j / 2 * (theta1 * np.kron(X.data, X.data) +
                                                theta2 * np.kron(Y.data, Y.data) +
                                                theta3 * np.kron(Z.data, Z.data))),
                         name='Can', angles=(theta1, theta2, theta3), *args, **kwargs)

    # def __init__(self, t1, t2, t3, *args, **kwargs):
    #     """
    #     Canonical gate with respect to Weyl chamber

    #     .. math::
    #         \textrm{Can}(t_1, t_2, t_3) = e^{- i \frac{\pi}{2}(t_1 XX + t_2 YY + t_3 ZZ)}
    #     """
    #     super().__init__(linalg.expm(-1j * pi / 2 * (t1 * np.kron(X.data, X.data) +
    #                                                  t2 * np.kron(Y.data, Y.data) +
    #                                                  t3 * np.kron(Z.data, Z.data))),
    #                      name='Can', params=(t1, t2, t3), *args, **kwargs)

    @property
    def weyl_coordinates(self) -> ndarray:
        r"""
        Reduce the rotation angles to coordinates of Weyl chamber.

        Herein we cohere to such a tetrahedron with coordinates:
            (x, y, z) ~ e^{- i \frac{\pi}{2}(x XX + y YY + z ZZ)}
            where (x, y, z) ∈ {1/2 ≥ x ≥ y ≥ z ≥ 0} ∪ {1/2 ≥ (1-x) ≥ y ≥ z ≥ 0}

        According to the series of symmetry characteristics of Weyl gate.
        """
        from phoenix.utils.functions import replace_close_to_zero_with_zero

        def xyz_satisfied(x, y, z):
            def within_left(x, y, z):
                x, y, z = round(x, 4), round(y, 4), round(z, 4)
                if 0.5 >= x >= y >= z >= 0:
                    return True
                return False

            def within_right(x, y, z):
                one_minus_x = round(1 - x, 4)
                y, z = round(y, 4), round(z, 4)
                if 0.5 >= one_minus_x >= y >= z >= 0:
                    return True
                return False

            return within_left(x, y, z) or within_right(x, y, z)

        coordinates = - 2 / pi * self.kak_coefficients
        x, y, z = replace_close_to_zero_with_zero(coordinates)  # there must be 1/2 >= -x >= -y >= |z|

        if x < 0 and y < 0:
            x, y = -x, -y
        if x < 0 and y == 0:
            x = -x
        if x == 0 and y < 0:
            y = -y

        if z < 0:
            if np.allclose(x, 0.5):
                z = -z
            else:
                # x, y, z = 1 - y, x, -z
                x, y, z = 1 - x, y, -z

        if z == 0 and 0.5 < x < 1:
            x = 1 - x

        assert xyz_satisfied(x, y, z), "Not (x, y, z) ∈ {1/2 >= x >= y >= z >= 0} ∪ {1/2 >= (1-x) >= y >= z >= 0}"

        return np.array([x, y, z])

    @property
    def kak_coefficients(self) -> ndarray:
        r"""Sort of different from weyl_coordinates(), herein we cohere to such a tetrahedron with coordinates:
            (x, y, z) ~ e^{i \frac{\pi}{2}(x XX + y YY + z ZZ)}


        0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
        """
        from phoenix.utils.functions import replace_close_to_zero_with_zero

        def xyz_satisfied(x, y, z):
            x, y, z = round(x, 4), round(y, 4), round(z, 4)
            if x >= y >= np.abs(z):
                return True
            return False

        coeffs = - 0.5 * np.array(self.angles)
        if xyz_satisfied(*coeffs):
            return coeffs
        return replace_close_to_zero_with_zero(cirq.kak_decomposition(self.data).interaction_coefficients)

    def hermitian(self):
        if self.tqs:
            return Can(-self.angles[0], -self.angles[1], -self.angles[2]).on(self.tqs, self.cqs)
        return Can(-self.angles[0], -self.angles[1], -self.angles[2])


# Non-operation Gate

class Barrier:
    def __init__(self):
        raise NotImplementedError


class Measurement:
    def __init__(self):
        raise NotImplementedError


X = XGate()
Y = YGate()
Z = ZGate()
I = IGate()
S = SGate()
SDG = SDGGate()
T = TGate()
TDG = TDGGate()
H = HGate()
SWAP = SWAPGate()
ISWAP = ISWAPGate()
SQSW = SQSWGate()
SQiSW = SQiSWGate()
B = BGate()
SYC = SycamoreGate()

Can = Canonical  # its alias: Canonical gate
P = PhaseShift

PAULI_ROTATION_GATES = ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz']
ROTATION_GATES = ['rx', 'ry', 'rz', 'u1', 'u2' 'u3', 'rxx', 'ryy', 'rzz', 'can']
FIXED_GATES = ['x', 'y', 'z', 'i', 'id', 'h', 's', 't', 'sdg', 'tdg', 'cx', 'cz', 'swap', 'ch', 'iswap', 'ccx', 'ccz']
CONTROLLABLE_GATES = ['x', 'y', 'z', 'h', 'swap', 'rx', 'ry', 'rz', 'u3']
HERMITIAN_GATES = ['x', 'y', 'z', 'h', 'swap']

RYY_DEF = """gate ryy(param0) q0,q1 {
    rx(pi/2) q0;
    rx(pi/2) q1;
    cx q0, q1;
    rz(param0) q1;
    cx q0, q1;
    rx(-pi/2) q0;
    rx(-pi/2) q1;
}"""

CAN_DEF_BY_CNOT = """gate can (param0, param1, param2) q0,q1 {
    u3(1.5*pi, 0.0, 1.5*pi) q0;
    u3(0.5*pi, 1.5*pi, 0.5*pi) q1;
    cx q0, q1;
    u3(1.5*pi, param0 + pi, 0.5*pi) q0;
    u3(pi, 0.0, param1 + pi) q1;
    cx q0, q1;
    u3(0.5*pi, 0.0, 0.5*pi) q0;
    u3(0.0, 1.5*pi, param2 + 0.5*pi) q1;
    cx q0, q1;
}"""

CAN_DEF_BY_ISING = """gate can (param0, param1, param2) q0,q1 {
    rxx(param0) q0, q1;
    ryy(param1) q0, q1;
    rzz(param2) q0, q1;
}"""

ISWAP_DEF_BY_CNOT = """gate iswap a,b {
    s a;
    s b;
    h a;
    cx a, b;
    cx b, a;
    h b;
}"""

CXX_DEF_BY_CNOT = """gate cxx a,b {
h a;
cx a, b;
h a;
}"""

CXY_DEF_BY_CNOT = """gate cxy a,b {
h a;
sdg b;
cx a, b;
h a;
s b;
}"""

CXZ_DEF_BY_CNOT = """gate cxz a,b {
h a;
h b;
cx a, b;
h a;
h b;
}"""

CYX_DEF_BY_CNOT = """gate cyx a,b {
sdg a;
h a;
cx a, b;
h a;
s a;
}"""

CYY_DEF_BY_CNOT = """gate cyy a,b {
sdg a;
h a;
sdg b;
cx a, b;
h a;
s a;
s b;
}"""

CYZ_DEF_BY_CNOT = """gate cyz a,b {
sdg a;
h a;
h b;
cx a, b;
h a;
s a;
h b;
}"""

CZX_DEF_BY_CNOT = """gate czx a,b {
cx a, b;
}"""

CZY_DEF_BY_CNOT = """gate czy a,b {
sdg b;
cx a, b;
s b;
}"""

CZZ_DEF_BY_CNOT = """gate czz a,b {
h b;
cx a, b;
h b;
}"""
