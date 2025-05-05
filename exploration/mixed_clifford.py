import sys

sys.path.append('../')

import numpy as np
from phoenix.synthesis import simplification, grouping
from phoenix import utils, models
from phoenix import Circuit, gates
from iswapclifford import Clifford2QiSWAP, simplify_bsf_by_iswap, simplify_bsf_by_mixed
from rich.console import Console

console = Console()


def random_pauli(n: int) -> str:
    return ''.join(np.random.choice(['X', 'Y', 'Z', 'I'], n, replace=True))


def test_cnot_clifford(bsf):
    print()
    console.rule('Testing CNOT clifford family ...', style='bold blue')
    bsf, cliffords_with_locals = simplification.simplify_bsf(bsf)
    console.print('Result: {} Cliffords are required'.format(len(cliffords_with_locals)))
    return bsf, cliffords_with_locals


def test_iswap_clifford(bsf):
    print()
    console.rule('Testing iSWAP clifford family ...', style='bold blue')
    bsf, cliffords_with_locals = simplify_bsf_by_iswap(bsf)
    console.print('Result: {} Cliffords are required'.format(len(cliffords_with_locals)))
    return bsf, cliffords_with_locals


def test_mixed_clifford(bsf):
    print()
    console.rule('Testing mixed clifford family ...', style='bold purple')
    bsf, cliffords_with_locals = simplify_bsf_by_mixed(bsf)
    console.print('Result: {} Cliffords are required'.format(len(cliffords_with_locals)))
    return bsf, cliffords_with_locals


if __name__ == "__main__":
    # np.random.seed(123)
    paulis = [random_pauli(4) for _ in range(20)]
    paulis = [pauli for pauli in paulis if pauli.count('I') == 0]
    coeffs = np.random.rand(len(paulis))
    print(paulis)
    # ham = models.HamiltonianModel(paulis, coeffs)
    # circ = ham.phoenix_circuit()
    # print(circ.to_cirq())

    bsf = models.BSF(paulis, coeffs)
    console.print('initial paulis: {} and cost: {}'.format(bsf.paulis,
                                                           simplification.heuristic_bsf_cost(bsf)), style='bold red')
    test_cnot_clifford(bsf)
    test_iswap_clifford(bsf)
    test_mixed_clifford(bsf)
