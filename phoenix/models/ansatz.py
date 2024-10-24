import numpy as np
from phoenix.basic import gates
from phoenix.basic.circuits import Circuit
from phoenix.models.hamiltonians import HamiltonianModel
from typing import Union, Tuple


class AnsatzGenerator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def constr_circuit(self, *args, **kwargs):
        raise NotImplementedError


class TrotterCNOT(AnsatzGenerator):
    def __init__(self, ham: HamiltonianModel, t: float = 1, num_steps: int = 1, order: int = 1):
        self.H = ham
        self.t = t
        self.num_steps = num_steps
        self.order = order

    def constr_circuit(self, grouping=None) -> Circuit:
        circ = Circuit()
        scale = self.t / self.num_steps / self.order
        paulis, coeffs = self.H.paulis_and_coeffs(grouping)
        for _ in range(self.num_steps):
            if self.order == 1:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ += _paulistr_to_circuit_cnot(paulistr, coeff * scale)
            elif self.order == 2:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ += _paulistr_to_circuit_cnot(paulistr, coeff * scale)
                for paulistr, coeff in reversed(list(zip(paulis, coeffs))):
                    circ += _paulistr_to_circuit_cnot(paulistr, coeff * scale)
            else:
                raise ValueError("Not implemented for order > 2")
        return circ


class TrotterSU4(AnsatzGenerator):
    def __init__(self, ham: HamiltonianModel, t: float = 1, num_steps: int = 1, order: int = 1):
        self.H = ham
        self.t = t
        self.num_steps = num_steps
        self.order = order

    def constr_circuit(self, grouping=None) -> Circuit:
        paulis, coeffs = self.H.paulis_and_coeffs(grouping, agg='canonical')
        circ = Circuit()
        scale = self.t / self.num_steps / self.order
        for _ in range(self.num_steps):
            if self.order == 1:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ += _paulistr_to_circuit_su4(paulistr, np.array(coeff) * scale)
            elif self.order == 2:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ += _paulistr_to_circuit_su4(paulistr, np.array(coeff) * scale)
                for paulistr, coeff in reversed(list(zip(paulis, coeffs))):
                    circ += _paulistr_to_circuit_su4(paulistr, np.array(coeff) * scale)
            else:
                raise ValueError("Not implemented for order > 2")
        return circ


def _paulistr_to_circuit_cnot(paulistr: str, coeff: float) -> Circuit:
    assert coeff.imag == 0, "Imaginary coefficients are not supported"
    coeff = coeff.real
    theta = 2 * coeff
    indices = np.where(np.array(list(paulistr)) != 'I')[0].tolist()
    circ = Circuit()
    if len(indices) > 2:
        raise ValueError("Not implemented for more than 2 nontrivial indices")
    if len(indices) == 0:
        raise ValueError("Not implemented for no nontrivial indices")
    if len(indices) == 1:
        if paulistr[indices[0]] == 'X':
            circ.append(
                *[gates.H.on(i) for i in indices],
                gates.RZ(theta).on(indices),
                *[gates.H.on(i) for i in indices]
            )
        elif paulistr[indices[0]] == 'Y':
            circ.append(
                *[gates.SDG.on(i) for i in indices],
                *[gates.H.on(i) for i in indices],
                gates.RZ(theta).on(indices),
                *[gates.H.on(i) for i in indices],
                *[gates.S.on(i) for i in indices]
            )
        elif paulistr[indices[0]] == 'Z':
            circ.append(gates.RZ(theta).on(indices))
        else:
            raise ValueError(f"Invalid Pauli string {paulistr}")
    if len(indices) == 2:
        if paulistr[indices[0]] == paulistr[indices[1]] == 'X':
            circ.append(
                *[gates.H.on(i) for i in indices],
                gates.X.on(*indices), gates.RZ(theta).on(indices[0]), gates.X.on(*indices),
                *[gates.H.on(i) for i in indices]
            )
        elif paulistr[indices[0]] == paulistr[indices[1]] == 'Y':
            circ.append(
                *[gates.SDG.on(i) for i in indices],
                *[gates.H.on(i) for i in indices],
                gates.X.on(*indices), gates.RZ(theta).on(indices[0]), gates.X.on(*indices),
                *[gates.H.on(i) for i in indices],
                *[gates.S.on(i) for i in indices]
            )
        elif paulistr[indices[0]] == paulistr[indices[1]] == 'Z':
            # return gate.RZZ(theta).on(indices)
            circ.append(gates.X.on(*indices), gates.RZ(theta).on(indices[0]), gates.X.on(*indices))

        else:
            raise ValueError(f"Invalid Pauli string {paulistr}")
    return circ


def _paulistr_to_circuit_su4(paulistr: Union[Tuple[str], str], coeff: Union[Tuple[float], float]) -> Circuit:
    circ = Circuit()
    if isinstance(paulistr, tuple):
        assert len(paulistr) == 3
        indices_xx = np.where(np.array(list(paulistr[0])) != 'I')[0].tolist()
        indices_yy = np.where(np.array(list(paulistr[1])) != 'I')[0].tolist()
        indices_zz = np.where(np.array(list(paulistr[2])) != 'I')[0].tolist()
        assert indices_xx == indices_yy == indices_zz
        theta = [c * 2 for c in coeff]
        circ.append(gates.Can(*theta).on(indices_xx))
    else:
        theta = coeff * 2
        indices = np.where(np.array(list(paulistr)) != 'I')[0].tolist()
        assert len(indices) == 1
        if paulistr[indices[0]] == 'X':
            circ.append(
                *[gates.H.on(i) for i in indices],
                gates.RZ(theta).on(indices),
                *[gates.H.on(i) for i in indices]
            )
        elif paulistr[indices[0]] == 'Y':
            circ.append(
                *[gates.SDG.on(i) for i in indices],
                *[gates.H.on(i) for i in indices],
                gates.RZ(theta).on(indices),
                *[gates.H.on(i) for i in indices],
                *[gates.S.on(i) for i in indices]
            )
        elif paulistr[indices[0]] == 'Z':
            circ.append(gates.RZ(theta).on(indices))
        else:
            raise ValueError(f"Invalid Pauli string {paulistr}")
    return circ
