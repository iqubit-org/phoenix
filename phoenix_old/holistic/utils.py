import numpy as np
from phoenix_old.models.paulis import BSF
from itertools import combinations


def _heuristic_bsf_cost_func(bsf: BSF) -> float:
    r"""
    Heuristic cost for a 3-qubit Pauli Tableau, the smaller the simpler.

    .. math::
        \mathrm{cost} := \sum_g  w_g * n_g^2 +
            \frac{1}{|G|} \sum_{\langle i,j \rangle} \left( \lVert r_x^{(i)} \lor r_z^{(i)} \lor r_x^{(j)} \lor r_z^{(j)} \rVert
            + \frac{1}{2}  (\lVert r_x^{(i)} \lor r_x^{(j)} \rVert + \lVert r_z^{(i)} \lor r_z^{(j)} \rVert) \right)
    """
    cost = 0.0
    unique_rows, indices, inverse_indices, counts = np.unique(bsf.with_ops, axis=0,
                                                              return_index=True,
                                                              return_inverse=True,
                                                              return_counts=True)
    num_groups = len(unique_rows)

    """
    [[0 0 1 1 1]
     [0 1 1 1 1]
     [1 0 0 0 1]
     [1 0 1 1 1]
     [1 1 0 1 1]
     [1 1 1 0 0]
     [1 1 1 1 0]]
    [9 1 3 0 6 4 2]
    [3 1 6 2 5 5 4 3 5 0]
    [1 1 1 2 1 3 1]
    """
    if bsf.which_nonlocal_paulis.size > 1:
        row_combs = np.array(list(combinations(bsf.which_nonlocal_paulis, 2))).T
        cost += np.sum(np.bitwise_or(bsf.with_ops[row_combs[0]], bsf.with_ops[row_combs[1]]).sum(axis=1)**2) / num_groups
        cost += np.sum(np.bitwise_or(bsf.x[row_combs[0]], bsf.x[row_combs[1]]).sum(axis=1)**2) * 0.5 / num_groups
        cost += np.sum(np.bitwise_or(bsf.z[row_combs[0]], bsf.z[row_combs[1]]).sum(axis=1)**2) * 0.5 / num_groups
    cost += (unique_rows.sum(axis=1) * counts ** 2).sum()
    return cost

heuristic_bsf_cost = np.vectorize(_heuristic_bsf_cost_func)
