from pauliopt.pauli.pauli_polynomial import PauliPolynomial
import numpy as np


def remove_collapsed_pauli_gadgets(remaining_poly):
    return list(filter(lambda x: x.angle != 2 * np.pi and x.angle != 0, remaining_poly))


def find_matching_parity_right(idx, remaining_poly):
    gadget = remaining_poly[idx]
    for idx_right, gadget_right in enumerate(remaining_poly[idx + 1 :]):
        if all([p_1 == p_2 for p_1, p_2 in zip(gadget.paulis, gadget_right.paulis)]):
            return idx + idx_right + 1
    return None


def is_commuting_region(idx, idx_right, remaining_poly, allow_acs=False):
    if allow_acs:
        return True
    for k in range(idx, idx_right):
        if not remaining_poly[idx].commutes(remaining_poly[k]):
            return False
    return True


def propagate_phase_gadgets(remaining_poly, allow_acs=False):
    converged = True
    for idx, gadget in enumerate(remaining_poly):
        idx_right = find_matching_parity_right(idx, remaining_poly)
        if idx_right is None:
            continue
        if not is_commuting_region(idx, idx_right, remaining_poly, allow_acs=allow_acs):
            continue

        remaining_poly[idx_right].angle = remaining_poly[idx_right].angle + gadget.angle
        remaining_poly[idx].angle = 0.0
        converged = False
    return converged


def simplify_pauli_polynomial(pp: PauliPolynomial, allow_acs=False):
    remaining_poly = [gadget.copy() for gadget in pp.pauli_gadgets]
    converged = False
    while not converged:
        remaining_poly = remove_collapsed_pauli_gadgets(remaining_poly)
        converged = propagate_phase_gadgets(remaining_poly, allow_acs=allow_acs)

    pp_ = PauliPolynomial(pp.num_qubits)
    for gadget in remaining_poly:
        pp_ >>= gadget
    return pp_
