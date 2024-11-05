import sys

sys.path.append('..')

from phoenix.models import BSF, Clifford2Q, HamiltonianModel
from phoenix import gates
from phoenix.utils.functions import infidelity
import numpy as np
import cirq
from copy import deepcopy


def test_bsf_phase_minus():
    # paulis = ''.join(np.random.choice(['X', 'Y', 'Z'], 2, replace=True))
    paulis = 'XZ'
    print('paulis:', paulis)
    tab = BSF(paulis, 0.2)

    print('initial paulis:', tab.paulis)

    circ = tab.as_cnot_circuit()
    print(circ.to_cirq())

    print('tab.signs:', tab.signs)
    tab1 = deepcopy(tab)
    tab1 = tab1.apply_clifford_2q(Clifford2Q('Z', 'X'), 0, 1)

    print('tab1.paulis:', tab1.paulis)
    print('tab1.signs:', tab1.signs)
    circ1 = tab1.as_cnot_circuit()

    circ1.prepend(gates.X.on(1, 0))
    circ1.append(gates.X.on(1, 0))

    print(circ1.to_cirq())

    print(infidelity(circ.unitary(), circ1.unitary()))

    cirq.testing.assert_allclose_up_to_global_phase(circ.unitary(), circ1.unitary(), atol=1e-8)


def test_bsf_phase():
    paulis = ''.join(np.random.choice(['X', 'Y', 'Z'], 2, replace=True))
    print('paulis:', paulis)
    tab = BSF(paulis, 0.2)

    print('initial paulis:', tab.paulis)

    circ = tab.as_cnot_circuit()
    print(circ.to_cirq())

    print('tab.signs:', tab.signs)
    tab1 = deepcopy(tab)
    tab1 = tab1.apply_clifford_2q(Clifford2Q('Z', 'X'), 0, 1)
    tab1 = tab1.apply_h(0)
    tab1 = tab1.apply_h(1)
    tab1 = tab1.apply_s(0)
    tab1 = tab1.apply_sdg(1)
    tab1 = tab1.apply_clifford_2q(Clifford2Q('Z', 'Z'), 0, 1)

    print('tab1.paulis:', tab.paulis)
    print('tab1.signs:', tab.signs)
    circ1 = tab1.as_cnot_circuit()

    circ1.prepend(gates.Z.on(1, 0))
    circ1.append(gates.Z.on(1, 0))

    circ1.prepend(gates.S.on(0))
    circ1.append(gates.SDG.on(0))

    circ1.prepend(gates.SDG.on(1))
    circ1.append(gates.S.on(1))

    circ1.prepend(gates.H.on(0))
    circ1.prepend(gates.H.on(1))
    circ1.append(gates.H.on(0))
    circ1.append(gates.H.on(1))

    circ1.prepend(gates.X.on(1, 0))
    circ1.append(gates.X.on(1, 0))

    print(circ1.to_cirq())

    print(circ.unitary().round(4))
    print(circ1.unitary().round(4))

    print(infidelity(circ.unitary(), circ1.unitary()))

    cirq.testing.assert_allclose_up_to_global_phase(circ.unitary(), circ1.unitary(), atol=1e-8)


def test_simplify_pauli_group():
    paulis = ['YXZZXXXZ', 'XXXZYXXZ', 'XXYZXXXZ', 'YXXZZXXZ', 'ZXXZYXXZ', 'ZXYZXXXZ']
    coeffs = np.array([0.01 * i for i in range(1, len(paulis) + 1)])
    ham = HamiltonianModel(paulis, coeffs)
    u = ham.unitary_evolution()
    circ = ham.generate_circuit()
    circ_su4 = ham.reconfigure_and_generate_circuit('su4')
    print('infidelity (initial v.s. ideal)', infidelity(circ.unitary(), u))
    print('infidelity (su4 circuit v.s. ideal)', infidelity(circ_su4.unitary(), u))
