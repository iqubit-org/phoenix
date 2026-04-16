import numpy as np
from phoenix_old.basic import gates
from phoenix_old.basic.circuits import Circuit
from phoenix_old.models.hamiltonians import HamiltonianModel
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

    def constr_circuit_old(self, grouping=None) -> Circuit:
        circ = Circuit()
        scale = self.t / self.num_steps / self.order
        paulis, coeffs = self.H.paulis_and_coeffs(grouping)
        for _ in range(self.num_steps):
            if self.order == 1:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ.compose(_paulistr_to_circuit_cnot(paulistr, coeff * scale))
            elif self.order == 2:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ.compose(_paulistr_to_circuit_cnot(paulistr, coeff * scale))
                for paulistr, coeff in reversed(list(zip(paulis, coeffs))):
                    circ.compose(_paulistr_to_circuit_cnot(paulistr, coeff * scale))
            else:
                raise ValueError("Not implemented for order > 2")
        return circ
    
    # def constr_circuit(self) -> QuantumCircuit:
    #     paulis, coeffs = self.H.paulis_and_coeffs()
    #     from qiskit.opflow import PauliSumOp
    #     from qiskit.circuit.library import PauliEvolutionGate
    #     # EvolutionSynthesis
    #     op = PauliSumOp.from_list([(p, c) for p, c in zip(paulis, coeffs)])
    #     evo_gate = PauliEvolutionGate(op, time=self.t, num_time_slices=self.num_steps, synthesis_order=self.order)
    #     return evo_gate.to_circuit()

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
                    circ.compose(_paulistr_to_circuit_su4(paulistr, np.array(coeff) * scale))
            elif self.order == 2:
                for paulistr, coeff in zip(paulis, coeffs):
                    circ.compose(_paulistr_to_circuit_su4(paulistr, np.array(coeff) * scale))
                for paulistr, coeff in reversed(list(zip(paulis, coeffs))):
                    circ.compose(_paulistr_to_circuit_su4(paulistr, np.array(coeff) * scale))
            else:
                raise ValueError("Not implemented for order > 2")
        return circ

from qiskit import QuantumCircuit

def _paulistr_to_circuit_cnot(paulistr: str, coeff: float) -> QuantumCircuit:
    assert coeff.imag == 0, "Imaginary coefficients are not supported"
    theta = 2 * coeff.real
    indices = np.where(np.array(list(paulistr)) != 'I')[0].tolist()
    qc = QuantumCircuit(len(paulistr))
    if len(indices) > 2:
        raise ValueError("Not implemented for more than 2 nontrivial indices")
    if not indices:
        raise ValueError("Not implemented for no nontrivial indices")

    if len(indices) == 1:
        idx = indices[0]
        if paulistr[idx] == 'X':
            qc.rx(theta, idx)
        elif paulistr[idx] == 'Y':
            qc.ry(theta, idx)
        elif paulistr[idx] == 'Z':
            qc.rz(theta, idx)
        else:
            raise ValueError(f"Invalid Pauli string {paulistr}")

    if len(indices) == 2:
        ctrl, targ = indices
        if paulistr[targ] == paulistr[ctrl] == 'X':
            qc.rxx(theta, *indices)
        elif paulistr[targ] == paulistr[ctrl] == 'Y':
            qc.ryy(theta, *indices)
        elif paulistr[targ] == paulistr[ctrl] == 'Z':
            qc.rzz(theta, *indices)
        else:
            raise ValueError(f"Invalid Pauli string {paulistr}")

    return qc




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
