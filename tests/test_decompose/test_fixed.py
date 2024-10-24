from phoenix import gates
from phoenix import decompose


def assert_equivalent_unitary(U, V):
    try:
        import cirq
        cirq.testing.assert_allclose_up_to_global_phase(U, V, atol=1e-5)
    except ModuleNotFoundError:
        from phoenix.utils.operations import is_equiv_unitary
        assert is_equiv_unitary(U, V)


def ceshi_decompose(g, decomp_func):
    from phoenix.utils.operations import tensor_slots, controlled_unitary_matrix
    from functools import reduce
    import numpy as np

    n = g.num_qregs
    qregs_sorted = sorted(g.qregs)
    qregs_rewiring = {p: q for p, q in zip(qregs_sorted, range(n))}
    qregs_rewired = [qregs_rewiring[q] for q in g.qregs]

    if g.n_qubits > int(np.log2(g.data.shape[0])) == 1:
        data = reduce(np.kron, [g.data] * g.n_qubits)
    else:
        data = g.data

    if g.cqs:
        U = controlled_unitary_matrix(data, len(g.cqs))
        U = tensor_slots(U, n, qregs_rewired)
    else:
        U = tensor_slots(data, n, qregs_rewired)

    circ = decomp_func(g)
    print(circ)
    print(circ.qubits)
    assert_equivalent_unitary(U, circ.unitary())


##########################################
# Pauli-related gates decomposition
##########################################
def test_X_1_02():
    ceshi_decompose(gates.X.on([1], [0, 2]), decompose.ccx_decompose)


def test_Y_1_0():
    ceshi_decompose(gates.Y.on(1, 0), decompose.cy_decompose)


def test_Z_1_0():
    ceshi_decompose(gates.Z.on(1, 0), decompose.cz_decompose)


def test_RX_2_0():
    ceshi_decompose(gates.RX(1.2).on(2, 0), decompose.crx_decompose)


def test_RY_2_0():
    ceshi_decompose(gates.RY(1.2).on(2, 0), decompose.cry_decompose)


def test_RZ_2_0():
    ceshi_decompose(gates.RZ(1.2).on(2, 0), decompose.crz_decompose)


##########################################
# S/H/T-related gates decomposition
##########################################
def test_H_1_0():
    ceshi_decompose(gates.H.on(1, 0), decompose.ch_decompose)


def test_S_1_0():
    ceshi_decompose(gates.S.on(1, 0), decompose.cs_decompose)


def test_T_1_0():
    ceshi_decompose(gates.T.on(1, 0), decompose.ct_decompose)


##########################################
# SWAP-related decomposition
##########################################
def test_swap_01():
    ceshi_decompose(gates.SWAP.on([0, 1]), decompose.swap_decompose)


def test_swap_01_2():
    ceshi_decompose(gates.SWAP.on([0, 1], [2]), decompose.cswap_decompose)
