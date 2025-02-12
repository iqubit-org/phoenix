"""
Operator-related Utils functions
"""
from typing import List, Tuple, Union, Optional
from math import sqrt, atan2

import cirq
import numpy as np
from scipy import linalg
from phoenix.basic import gates
from phoenix.utils.functions import is_power_of_two, replace_close_to_zero_with_zero

M = np.array([[1, 0, 0, 1j],
              [0, 1j, 1, 0],
              [0, 1j, -1, 0],
              [1, 0, 0, -1j]]) / sqrt(2)

M_DAG = M.conj().T

A = np.array([[1, 1, -1, 1],
              [1, 1, 1, -1],
              [1, -1, -1, -1],
              [1, -1, 1, 1]])


def exp_xx(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.X.data, gates.X.data))


def exp_yy(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.Y.data, gates.Y.data))


def exp_zz(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.Z.data, gates.Z.data))


def exp_xy(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.X.data, gates.Y.data))


def exp_yx(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.Y.data, gates.X.data))


def exp_xz(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.X.data, gates.Z.data))


def exp_zx(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.Z.data, gates.X.data))


def exp_yz(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.Y.data, gates.Z.data))


def exp_zy(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(gates.Z.data, gates.Y.data))


def tensor_1_slot(U: np.ndarray, n: int, tq: int) -> np.ndarray:
    """
    Given a 2x2 matrix, compute the matrix expanded to the whole Hilbert space (totally n qubits).

    Args:
        U: matrix with size [2,2].
        n: total number of qubit subspaces.
        tq: target qubit index.

    Returns:
        Matrix, expanded via tensor product, with size [2^n, 2^n].
    """
    if tq not in range(n):
        raise ValueError('the qubit index is out of range')
    ops = [np.identity(2)] * n
    ops[tq] = U
    return cirq.kron(*ops)

def tensor_slots(U: np.ndarray, n: int, indices: List[int]) -> np.ndarray:
    """
    Given a matrix, compute the matrix expanded to the whole Hilbert space (totally n qubits).

    Args:
        U: matrix with size
        n: total number of qubit subspaces
        indices: target qubit indices

    Returns:
        Matrix, expanded via tensor product, with size [2^n, 2^n].
    """
    if not is_power_of_two(U.shape[0]):
        raise ValueError(f"Dimension of input matrix need should be power of 2, but get {U.shape[0]}")
    m = int(np.log2(U.shape[0]))
    if len(indices) != m or max(indices) >= n:
        raise ValueError(f'input indices {indices} does not consist with dimension of input matrix U')
    if m == 1:
        return tensor_1_slot(U, n, indices[0])
    else:
        arr_list = [U] + [np.identity(2)] * (n - m)
        res = cirq.kron(*arr_list).reshape([2] * 2 * n)
        idx = np.repeat(-1, n)
        for i, k in enumerate(indices):
            idx[k] = i
        idx[idx < 0] = range(m, n)
        idx_latter = [i + n for i in idx]
        return res.transpose(idx.tolist() + idx_latter).reshape(2 ** n, 2 ** n)


def times_two_matrix(U: np.ndarray, V: np.ndarray) -> Optional[np.ndarray]:
    """Calculate the coefficient a, s.t., U = a V. If a does not exist, return None."""
    assert U.shape == V.shape, "input matrices should have the same dimension"
    idx1 = np.flatnonzero(replace_close_to_zero_with_zero(U))
    idx2 = np.flatnonzero(replace_close_to_zero_with_zero(V))
    try:
        if np.allclose(idx1, idx2):
            return U.ravel()[idx1[0]] / V.ravel()[idx2[0]]
    except ValueError:
        return None


# def is_equiv_unitary(U: np.ndarray, V: np.ndarray) -> bool:
#     """Distinguish whether two unitary operators are equivalent, regardless of the global phase."""
#     if U.shape != V.shape:
#         raise ValueError(f'U and V have different dimensions: {U.shape}, {V.shape}.')
#     d = U.shape[0]
#     if not np.allclose(U @ U.conj().T, np.identity(d)):
#         raise ValueError('U is not unitary')
#     if not np.allclose(V @ V.conj().T, np.identity(d)):
#         raise ValueError('V is not unitary')
#     Uf = U.ravel()
#     Vf = V.ravel()
#     idx_Uf = np.flatnonzero(Uf.round(6))  # considering some precision
#     idx_Vf = np.flatnonzero(Vf.round(6))
#     try:
#         if np.allclose(idx_Uf, idx_Vf):
#             coes = Uf[idx_Uf] / Vf[idx_Vf]
#             return np.allclose(coes / coes[0], np.ones(len(idx_Uf)))
#         else:
#             return False
#     except ValueError:
#         return False


def is_equiv_unitary(U: np.ndarray, V: np.ndarray) -> bool:
    """Distinguish whether two unitary operators are equivalent, regardless of the global phase."""
    U, V = match_global_phase(U, V)
    return np.allclose(U, V, atol=1e-7)


def match_global_phase(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    Args:
        a: A numpy array.
        b: Another numpy array.

    Returns:
        A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape or a.size == 0:
        return np.copy(a), np.copy(b)

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])


def so4_to_magic_su2s(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose 1 SO(4) operator into 2 SU(2) operators with Magic matrix transformation: U = Mdag @ kron(A, B) @ M.

    Args:
        U: a SO(4) matrix.

    Returns:
        two SU(2) matrices, or, raise error.
    """
    if not is_so4(U):
        raise ValueError('Input matrix is not in SO(4)')
    # KPD is definitely feasible when the input matrix is in SO(4)
    return kron_decomp(M @ U @ M_DAG)


def is_so4(U: np.ndarray) -> bool:
    """Distinguish if a matrix is in SO(4) (4-dimension Special Orthogonal group)."""
    if U.shape != (4, 4):
        raise ValueError('U should be a 4*4 matrix')
    return np.allclose(U @ U.conj().T, np.identity(4)) and np.allclose(linalg.det(U), 1)


def is_tensor_prod(U: np.ndarray) -> bool:
    """Distinguish whether a 4x4 matrix is the tensor product of two 2x2 matrices."""
    a, b = kron_decomp(U)
    if a is None:
        return False
    elif np.allclose(U, np.kron(a, b)):
        return True
    return False


def kron_decomp(M: np.ndarray, method: str = 'nearest') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Kronecker product decomposition (KPD) algorithm for 4x4 4*4 matrix.

    Args:
        M: 4x4 matrix
        method: 'nearest', 'new', or 'cirq'. Default is 'nearest'.
            The 'new' method is not absolutely robust (without tolerance) sometimes.

    References:
        [1] Crooks, Gavin E. "Gates, states, and circuits." Gates states and circuits (2020).
        [2] New Kronecker product decompositions and its applications. https://www.researchinventy.com/papers/v1i11/F0111025030.pdf
        [3] https://quantumai.google/reference/python/cirq/kron_factor_4x4_to_2x2s
    """
    if M.shape != (4, 4):
        raise ValueError('Input matrix should be a 4*4 matrix')

    def nearest_kron_decomp(M: np.ndarray):
        """
        Acquire nearest KPD of a 4x4 matrix via Pitsianis-Van Loan algorithm.
        """
        if M.shape != (4, 4):
            raise ValueError('Input matrix should be a 4*4 matrix')
        M = M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
        u, d, vh = linalg.svd(M)
        A = np.sqrt(d[0]) * u[:, 0].reshape(2, 2)
        B = np.sqrt(d[0]) * vh[0, :].reshape(2, 2)
        return A, B

    def new_kron_decomp(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M00, M01, M10, M11 = M[:2, :2], M[:2, 2:], M[2:, :2], M[2:, 2:]
        K = np.vstack([M00.ravel(), M01.ravel(), M10.ravel(), M11.ravel()])
        if np.linalg.matrix_rank(K) != 1:
            return None, None

        # If K is full-rank, the input matrix is in form of tensor product
        l = [not np.allclose(np.zeros(4), K[i]) for i in range(4)]
        idx = l.index(True)  # the first non-zero block
        B = K[idx]
        A = np.array([])
        for i in range(4):
            if l[i]:
                a = times_two_matrix(K[i], B)
            else:
                a = 0
            A = np.append(A, a)
        A = A.reshape(2, 2)
        B = B.reshape(2, 2)
        return A, B

    def cirq_kron_decomp(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g, A, B = cirq.kron_factor_4x4_to_2x2s(M)
        return g * A, B

    if method == 'nearest':
        return nearest_kron_decomp(M)
    elif method == 'new':
        return new_kron_decomp(M)
    elif method == 'cirq':
        return cirq_kron_decomp(M)


def params_zyz(U: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    r"""
    ZYZ decomposition of a 2x2 unitary matrix.

    .. math::
        U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)

    Args:
        U: 2x2 unitary matrix

    Returns:
        `\alpha`, `\theta`, `\phi`, `\lambda`, four phase angles.
    """
    if U.shape != (2, 2):
        raise ValueError('Input matrix should be a 2*2 matrix')
    coe = linalg.det(U) ** (-0.5)
    alpha = - np.angle(coe)
    v = coe * U
    v = replace_close_to_zero_with_zero(v)
    theta = 2 * atan2(abs(v[1, 0]), abs(v[0, 0]))
    phi_lam_sum = 2 * np.angle(v[1, 1])
    phi_lam_diff = 2 * np.angle(v[1, 0])
    phi = (phi_lam_sum + phi_lam_diff) / 2
    lam = (phi_lam_sum - phi_lam_diff) / 2
    alpha, theta, phi, lam = replace_close_to_zero_with_zero([alpha, theta, phi, lam])
    return alpha, (theta, phi, lam)


def params_u3(U: np.ndarray, return_phase=False) -> Union[
    Tuple[float, float, float, float], Tuple[float, Tuple[float, float, float]]]:
    r"""
    Obtain the U3 parameters of a 2x2 unitary matrix.

    .. math::
        U = exp(i p) U3(\theta, \phi, \lambda)

    Args:
        U: 2x2 unitary matrix
        return_phase: whether return the global phase `p`.

    Returns:
        Global phase `p` and three parameters `\theta`, `\phi`, `\lambda` of a standard U3 gate.
    """
    alpha, (theta, phi, lam) = params_zyz(U)
    phase = alpha - (phi + lam) / 2
    phase, theta, phi, lam = replace_close_to_zero_with_zero(np.array([phase, theta, phi, lam]))
    if return_phase:
        return phase, (theta, phi, lam)
    return theta, phi, lam


def controlled_unitary_matrix(U: np.ndarray, num_ctrl: int = 1) -> np.ndarray:
    """Construct the controlled-unitary matrix based on input unitary matrix."""
    proj_0, proj_1 = np.diag([1, 0]), np.diag([0, 1])
    for _ in range(num_ctrl):
        ident = cirq.kron(*[np.identity(2)] * int(np.log2(U.shape[0])))
        U = np.kron(proj_0, ident) + np.kron(proj_1, U)
    return U


def multiplexor_matrix(n: int, tq: int, *args) -> np.ndarray:
    """
    Construct a quantum multiplexor in form of matrix.

    Args:
        n: total qubit index range (0 ~ n-1)
        tq: target qubit index
        *args: matrix components of the multiplexor

    Returns:
        Matrix, in type of np.ndarray.
    """
    if not len(args) == 2 ** (n - 1):
        raise ValueError(f'Number of input matrix components is not equal to {n}')
    qubits = list(range(n - 1))
    qubits.insert(tq, n - 1)
    U = linalg.block_diag(*[mat for mat in args])
    U = U.reshape([2] * 2 * n)
    U = U.transpose(qubits + [q + n for q in qubits]).reshape(2 ** n, 2 ** n)
    return U


def circuit_to_unitary(circ, backend=None) -> np.ndarray:
    if backend is None:
        return circ.unitary()
    elif backend == 'qiskit':
        from qiskit.quantum_info import Operator
        circ_qiskit = circ.to_qiskit().reverse_bits()
        return Operator(circ_qiskit).to_matrix()
    elif backend == 'cirq':
        import cirq
        return cirq.unitary(circ.to_cirq())
    else:
        raise ValueError("Unsupported backend {}".format(backend))


def kak_coefficients(U: np.ndarray):
    if U.shape != (4, 4):
        raise ValueError('U should be a 4*4 matrix')
    d = U.shape[0]
    if not np.allclose(U @ U.conj().T, np.identity(d)):
        raise ValueError('U is not unitary')
    return replace_close_to_zero_with_zero(cirq.kak_decomposition(U).interaction_coefficients)


def weyl_coordinates(U: np.ndarray):
    coordinates = - 2 / np.pi * kak_coefficients(U)
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
            x, y, z = 1 - x, y, -z

    if z == 0 and 0.5 < x < 1:
        x = 1 - x

    return np.array([x, y, z])
