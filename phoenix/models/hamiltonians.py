import numpy as np
import qiskit.quantum_info as qi
from typing import List, Union, Tuple, Dict

from scipy import linalg
from copy import deepcopy
from itertools import chain
from phoenix.basic.circuits import Circuit
from phoenix.models.paulis import BSF
from phoenix.models.cliffords import Clifford2Q
from phoenix.utils.passes import concatenate_subcircuits
from rich.console import Console

console = Console()


class HamiltonianModel:
    def __init__(self, paulis, coeffs, *args, **kwargs):
        self.paulis = list(paulis)
        self.coeffs = np.array(coeffs)
        self.num_qubits = len(paulis[0])

    def pauli_sum(self) -> qi.SparsePauliOp:
        """Representation of the Hamiltonian as a sum of Pauli operators"""
        return qi.SparsePauliOp(self.paulis, self.coeffs)

    def paulis_and_coeffs(self, *args, **kwargs) -> Tuple[List, np.ndarray]:
        """Return the list of Pauli strings and the corresponding coefficients"""
        return self.paulis, self.coeffs

    def to_matrix(self) -> np.ndarray:
        """Return the matrix representation of the Hamiltonian"""
        return self.pauli_sum().to_matrix()

    def unitary_evolution(self, t: float = 1.0) -> np.ndarray:
        """Generator the corresponding unitary evolution operator"""
        return linalg.expm(-1j * self.to_matrix() * t)

    def normalize(self) -> 'HamiltonianModel':
        """Return a normalized version of the Hamiltonian"""
        norm = self.norm()
        ham = deepcopy(self)
        ham.coeffs = self.coeffs / norm
        return ham

    def norm(self) -> float:
        """Return the norm of the Hamiltonian, i.e., sum of all spectral norms of each iterm"""
        norm = 0
        for p, c in zip(self.paulis, self.coeffs):
            norm += np.abs(c) * linalg.norm(qi.Pauli(p).to_matrix(), 2)
        return norm

    def to_bsf(self) -> BSF:
        paulis, coeffs = self.paulis_and_coeffs()
        return BSF(paulis, coeffs)

    def group_paulis_and_coeffs(self) -> Dict[Tuple, Tuple[List[str], np.ndarray]]:
        """Group Pauli strings (with coefficients) by their nontrivial parts."""
        from phoenix.synthesis.grouping import group_paulis_and_coeffs
        return group_paulis_and_coeffs(self.paulis, self.coeffs)

    def generate_circuit(self, order: int = 1) -> Circuit:
        """
        According to Pauli strings and coefficients, generate the corresponding circuit with CNOT basis.

        Args:
            order: the order of Trotter-Suzuki decomposition (default: 1)
        """

        def trotterize(pls, coes, ord: int = 1, scale: float = 1) -> Circuit:
            """Trotterizes Pauli strings and coefficients into a quantum circuit in CNOT basis."""
            if ord == 1:
                return HamiltonianModel(pls, coes * scale).to_bsf().as_cnot_circuit()
            elif ord == 2:
                return HamiltonianModel(pls + pls[::-1],
                                        np.hstack((coes, coes[::-1])) / 2 * scale).to_bsf().as_cnot_circuit()
            else:
                if ord % 2:
                    raise ValueError("Not implemented for odd order > 2")
                k = ord / 2  # k is an integer > 1
                p_k = 1 / (4 - 4 ** (1 / (2 * k - 1)))
                return (trotterize(pls, coes, ord - 2, p_k * scale) + trotterize(pls, coes, ord - 2, p_k * scale) +
                        trotterize(pls, coes, ord - 2, (1 - 4 * p_k) * scale) +
                        trotterize(pls, coes, ord - 2, p_k * scale) + trotterize(pls, coes, ord - 2, p_k * scale))

        return trotterize(self.paulis, self.coeffs, order)

    def reconfigure(self) -> List[Union[BSF, Clifford2Q]]:
        """Reconfigure the Hamiltonian into a series of 2-weight Hamiltonians interleaved with Clifford2Q operators"""
        """
        {(0, 1, 2): ['XYZII'], (0, 1): ['XXIII', 'YYIII', 'ZZIII'], (2, 3): ['IIXXI', 'IIYYI', 'IIZZI'], (3, 4): ['IIIXX', 'IIIYY', 'IIIZZ'],
            (1, 2): ['IXXII', 'IYYII', 'IZZII'], (0,): ['ZIIII'], (1,): ['IZIII'], (2,): ['IIZII'], (3,): ['IIIZI'], (4,): ['IIIIZ']}
        """
        from phoenix.synthesis.simplification import simplify_bsf

        groups: Dict[Tuple[int], Tuple[List[str], np.ndarray]] = self.group_paulis_and_coeffs()
        config = []

        for idx, (paulis, coeffs) in groups.items():
            res = []
            bsf = BSF(paulis, coeffs)
            bsf_, cliffords_with_locals = simplify_bsf(bsf)  # ! simplify the BSF

            # res.append(bsf_)
            # for cliff, local_bsf in reversed(cliffords_with_locals):
            #     res.insert(0, cliff)
            #     res.append(cliff)
            #     if local_bsf.num_paulis > 0:
            #         res.append(local_bsf)

            if cliffords_with_locals:  # because it might be an empty list
                post_cliffs_with_locals = [[cliff, local_bfs] if local_bfs.num_paulis > 0 else [cliff] for
                                           cliff, local_bfs
                                           in reversed(cliffords_with_locals)]
                res.extend([cliff for cliff, _ in cliffords_with_locals])  # pre-cliffs
                res.append(bsf_)
                res.extend([item for sublist in post_cliffs_with_locals for item in sublist])
            else:
                res.append(bsf_)

            config.extend(res)

        return config

    def phoenix_reconfigure(self) -> List[List[Union[BSF, Clifford2Q]]]:
        """
        In contrast to `reconfigure_and_generate_circuit`, this method uses phoenix_reconfigure() to
            generate simplified subcircuits and perform ordering for these subscircuits.
        If order_blocks is False, this method is equivalent to reconfigure_and_generate_circuit(order=1).

        Args:
            by: the basis ('cnot' or 'su4') of the generated circuit (default: 'cnot')
            order_blocks: whether to order the blocks (default: True) to lower circuit depth
            efficient: whether to use efficient ordering (default: False), i.e., without exploiting Clifford2Q cancellation
        """
        from phoenix.synthesis.simplification import simplify_bsf

        groups: Dict[Tuple[int], Tuple[List[str], np.ndarray]] = self.group_paulis_and_coeffs()
        configs = []

        for idx, (paulis, coeffs) in groups.items():
            res = []
            bsf = BSF(paulis, coeffs)
            bsf_, cliffords_with_locals = simplify_bsf(bsf)  # ! simplify the BSF

            if cliffords_with_locals:  # because it might be an empty list
                post_cliffs_with_locals = [[cliff, local_bfs] if local_bfs.num_paulis > 0 else [cliff] for
                                           cliff, local_bfs
                                           in reversed(cliffords_with_locals)]
                res.extend([cliff for cliff, _ in cliffords_with_locals])  # pre-cliffs
                res.append(bsf_)
                res.extend([item for sublist in post_cliffs_with_locals for item in sublist])
            else:
                res.append(bsf_)

            configs.append(res)  # ! in phoenix_reconfigure(), this is .append() instead of .extend()

        return configs

    def reconfigure_and_generate_circuit(self, by: str = 'cnot', order: int = 1) -> Circuit:
        """
        Generate SU(4)-based circuit according to result of reconfiguring the Hamiltonian.

        Args:
            by: the basis ('cnot' or 'su4') of the generated circuit (default: 'cnot')
            order: the order of Trotter-Suzuki decomposition (default: 1)
        """

        def reverse_config(config: List[Union[BSF, Clifford2Q]]) -> List[Union[BSF, Clifford2Q]]:
            res = []
            for item in reversed(config):
                if isinstance(item, BSF):
                    res.append(item.reverse())
                if isinstance(item, Clifford2Q):
                    res.append(item)
            return res

        def rescale_config(config: List[Union[BSF, Clifford2Q]], scale: float) -> List[Union[BSF, Clifford2Q]]:
            res = []
            for item in config:
                if isinstance(item, BSF):
                    res.append(BSF(item.paulis, item.coeffs * scale, item.signs))
                if isinstance(item, Clifford2Q):
                    res.append(item)
            return res

        def trotter_config(config: List[Union[BSF, Clifford2Q]], ord: int = 1, scale: float = 1) -> Circuit:
            """Similar to `trotterize` but trotterize (BSF with Clifford2Q) instead of (Paulis, Coeffs)"""
            from phoenix.synthesis import utils
            if ord == 1:
                return utils.config_to_circuit(rescale_config(config, scale))
            elif ord == 2:
                return (utils.config_to_circuit(rescale_config(config, scale)) +
                        utils.config_to_circuit(rescale_config(reverse_config(config), scale)))
            else:
                if ord % 2:
                    raise ValueError("Not implemented for odd order > 2")
                k = ord / 2
                p_k = 1 / (4 - 4 ** (1 / (2 * k - 1)))
                return (trotter_config(config, ord - 2, p_k * scale) + trotter_config(config, ord - 2, p_k * scale) +
                        trotter_config(config, ord - 2, (1 - 4 * p_k) * scale) +
                        trotter_config(config, ord - 2, p_k * scale) + trotter_config(config, ord - 2, p_k * scale))

        # config = self.reconfigure()
        return trotter_config(self.reconfigure(), order)

    def phoenix_circuit(self, by='cnot', order_blocks: bool = True, efficient: bool = False) -> Circuit:
        """
        In contrast to `reconfigure_and_generate_circuit`, this method uses phoenix_reconfigure() to
            generate simplified subcircuits and perform ordering for these subscircuits.
        If order_blocks is False, this method is equivalent to reconfigure_and_generate_circuit(order=1).

        Args:
            by: the basis ('cnot' or 'su4') of the generated circuit (default: 'cnot')
            order_blocks: whether to order the blocks (default: True) to lower circuit depth
            efficient: whether to use efficient ordering (default: False), i.e., without exploiting Clifford2Q cancellation
        """
        from phoenix.synthesis import ordering, utils

        # grouping and group-wise simplification
        configs = self.phoenix_reconfigure()

        # ordering
        blocks = [utils.config_to_circuit(config, by=by) for config in configs]
        if order_blocks:
            circ = ordering.order_blocks(blocks, efficient=efficient)
        else:
            circ = concatenate_subcircuits(blocks)

        return circ
    


class Heisenberg1D(HamiltonianModel):
    # ! It is a periodic 1D Heisenberg model
    def __init__(self, Jx, Jy, Jz, h):
        self.couplings = np.array([Jx, Jy, Jz])
        self.frees = np.array(h)
        self.num_qubits = len(h)
        self._constr_paulis_and_coeffs()

    def _constr_paulis_and_coeffs(self, tile=False):
        """
        Construct the list of Pauli strings and the corresponding coefficients for Heisenberg 1D model.

        :param tile: whether to tile the couplings (default: False)
            If tile is True, generate Pauli lists like ['XXI', 'YYI', 'ZZI', 'IXX', 'IYY', 'IZZ', 'XIX', 'YIY', 'ZIZ']
            If tile is False, generate Pauli lists like ['XXI', 'IXX', 'XIX', 'YYI', 'IYY', 'YIY', 'ZZI', 'IZZ', 'ZIZ']
        """
        I_str = 'I' * self.num_qubits

        def XX_str(i, j):
            res = list(I_str)
            res[i] = 'X'
            res[j] = 'X'
            return ''.join(res)

        def YY_str(i, j):
            res = list(I_str)
            res[i] = 'Y'
            res[j] = 'Y'
            return ''.join(res)

        def ZZ_str(i, j):
            res = list(I_str)
            res[i] = 'Z'
            res[j] = 'Z'
            return ''.join(res)

        def Z_str(i):
            res = list(I_str)
            res[i] = 'Z'
            return ''.join(res)

        if tile:
            coupling_paulis = list(chain.from_iterable(
                [[XX_str(i % self.num_qubits, (i + 1) % self.num_qubits),
                  YY_str(i % self.num_qubits, (i + 1) % self.num_qubits),
                  ZZ_str(i % self.num_qubits, (i + 1) % self.num_qubits)] for i in range(self.num_qubits)]
            ))
            coupling_coeffs = np.tile(self.couplings, self.num_qubits)
        else:
            coupling_paulis = [XX_str(i % self.num_qubits, (i + 1) % self.num_qubits) for i in
                               range(self.num_qubits)] + [YY_str(i % self.num_qubits, (i + 1) % self.num_qubits) for
                                                          i in range(self.num_qubits)] + [
                                  ZZ_str(i % self.num_qubits, (i + 1) % self.num_qubits) for i in
                                  range(self.num_qubits)]
            coupling_coeffs = np.repeat(self.couplings, self.num_qubits)

        free_paulis = [Z_str(i) for i in range(self.num_qubits)]

        self.paulis = coupling_paulis + free_paulis
        self.coeffs = np.hstack((coupling_coeffs, self.frees))

    def paulis_and_coeffs(self, grouping=None, agg=None) -> Tuple[
        List[Union[Tuple[str], str]], List[Union[Tuple[float], float]]]:
        """
        Return the list of Pauli strings and the corresponding coefficients

        Args:

            agg: inner aggregation approach (default: None means each element of paulis is a string)
        """
        I_str = 'I' * self.num_qubits

        def XX_str(i, j):
            res = list(I_str)
            res[i] = 'X'
            res[j] = 'X'
            return ''.join(res)

        def YY_str(i, j):
            res = list(I_str)
            res[i] = 'Y'
            res[j] = 'Y'
            return ''.join(res)

        def ZZ_str(i, j):
            res = list(I_str)
            res[i] = 'Z'
            res[j] = 'Z'
            return ''.join(res)

        def Z_str(i):
            res = list(I_str)
            res[i] = 'Z'
            return ''.join(res)

        coupling_paulis = []
        coupling_coeffs = []

        if agg is None:
            if grouping is None:
                for i in range(self.num_qubits - 1):
                    coupling_paulis.append(XX_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[0])
                    coupling_paulis.append(YY_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[1])
                    coupling_paulis.append(ZZ_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[2])
            elif grouping == "even-odd":
                for i in range(0, self.num_qubits - 1, 2):
                    coupling_paulis.append(XX_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[0])
                    coupling_paulis.append(YY_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[1])
                    coupling_paulis.append(ZZ_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[2])
                for i in range(1, self.num_qubits - 1, 2):
                    coupling_paulis.append(XX_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[0])
                    coupling_paulis.append(YY_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[1])
                    coupling_paulis.append(ZZ_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[2])
            elif grouping == "xyz":
                for i in range(self.num_qubits - 1):
                    coupling_paulis.append(XX_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[0])
                for i in range(self.num_qubits - 1):
                    coupling_paulis.append(YY_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[1])
                for i in range(self.num_qubits - 1):
                    coupling_paulis.append(ZZ_str(i, i + 1))
                    coupling_coeffs.append(self.couplings[2])
            else:
                raise ValueError('Invalid grouping method')

        elif agg == "canonical":
            if grouping is None:
                for i in range(self.num_qubits - 1):
                    coupling_paulis.append((XX_str(i, i + 1), YY_str(i, i + 1), ZZ_str(i, i + 1)))
                    coupling_coeffs.append(tuple(self.couplings))
            elif grouping == "even-odd":
                for i in range(0, self.num_qubits - 1, 2):
                    coupling_paulis.append((XX_str(i, i + 1), YY_str(i, i + 1), ZZ_str(i, i + 1)))
                    coupling_coeffs.append(tuple(self.couplings))
                for i in range(1, self.num_qubits - 1, 2):
                    coupling_paulis.append((XX_str(i, i + 1), YY_str(i, i + 1), ZZ_str(i, i + 1)))
                    coupling_coeffs.append(tuple(self.couplings))
            else:
                raise ValueError('Invalid grouping method')
        else:
            raise ValueError('Invalid aggregation method')

        free_paulis = [Z_str(i) for i in range(self.num_qubits)]

        return coupling_paulis + free_paulis, coupling_coeffs + self.frees.tolist()

    def normalize(self) -> 'Heisenberg1D':
        norm = self.norm()
        return Heisenberg1D(*(self.couplings / norm), self.frees / norm)
