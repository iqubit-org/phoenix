"""
Other Utils functions
"""
import warnings
import numpy as np
from math import pi
from typing import List, Tuple


def limit_angle(a: float) -> float:
    """Limit equivalent rotation angle into (-pi, pi]."""
    if a >= 0:
        r = a % (2 * pi)
        if r >= 0 and r <= pi:
            return r
        else:
            return r - 2 * pi
    else:
        r = (-a) % (2 * pi)
        if r >= 0 and r <= pi:
            return -r
        else:
            return 2 * pi - r


def is_power_of_two(num) -> bool:
    """Check whether a number is power of 2 or not."""
    return (num & (num - 1) == 0) and num != 0


def calculate_elementary_permutations(initial_rank, target_rank) -> List[Tuple[int, int]]:
    """Calculate the series of elementary permutations to transform the initial rank to the target rank"""
    elementary_permutations = []
    while initial_rank != target_rank:
        for i in range(len(initial_rank)):
            if initial_rank[i] != target_rank[i]:
                # find the indices of the two elements that need to be swapped
                index1 = i
                index2 = initial_rank.index(target_rank[i])
                # change two elements
                initial_rank[index1], initial_rank[index2] = initial_rank[index2], initial_rank[index1]
                # add the permutation relationship of the swap to the output list
                elementary_permutations.append((index1, index2))
    return elementary_permutations


def infidelity(u: np.ndarray, v: np.ndarray) -> float:
    """Infidelity between two matrices"""
    if u.shape != v.shape:
        raise ValueError('u and v must have the same shape.')
    d = u.shape[0]
    return 1 - np.abs(np.trace(u.conj().T @ v)) / d


def spectral_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Spectral distance between two matrices"""
    print(u.shape, v.shape)
    if u.shape != v.shape:
        raise ValueError('u and v must have the same shape.')

    return np.linalg.norm(u - v, ord=2)


def average_case_error(u: np.ndarray, v: np.ndarray) -> float:
    """Normalized Frobenius norm between two matrices"""
    if u.shape != v.shape:
        raise ValueError('u and v must have the same shape.')
    d = u.shape[0]

    return np.linalg.norm(u - v, ord='fro') / np.sqrt(d)


def diamond_norm_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Diamond norm distance between two matrices"""
    if u.shape != v.shape:
        raise ValueError('u and v must have the same shape.')
    if np.log2(u.shape[0]) > 6:
        warnings.warn('Dimension is too large for the diamond norm calculation.')

    import qutip

    return qutip.metrics.dnorm(qutip.to_super(qutip.Qobj(u)) - qutip.to_super(qutip.Qobj(v)))


def replace_close_to_zero_with_zero(arr) -> np.ndarray:
    arr = np.array(arr)
    close_to_zero = np.isclose(arr, 0)
    arr[close_to_zero] = 0
    return arr


def to_special_unitary(u: np.ndarray) -> np.ndarray:
    """Convert a unitary matrix to a special unitary matrix."""
    if u.shape[0] != u.shape[1]:
        raise ValueError('Input matrix should be a square matrix')

    def extract_phase_angle(u):
        d = u.shape[0]
        coe = np.linalg.det(u) ** (1 / d)
        alpha = np.angle(coe)
        return alpha

    alpha = extract_phase_angle(u)
    return u * np.exp(-1j * alpha)
