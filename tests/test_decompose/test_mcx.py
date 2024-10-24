import numpy as np
from phoenix import Circuit, gates
from phoenix import decompose

from phoenix.utils import operations

rng = np.random.default_rng()


##################################################
# There must be n-2 >= m !!!
# - n: number of available qubits
# - m: number of control qubits in MCX
##################################################


def test_cccx(shuffle: bool = False):
    g = gates.X.on(4, [0, 1, 2])

    if shuffle:
        g_qregs = rng.permutation(g.qregs).tolist()
        g = g.on(g_qregs[0], g_qregs[1:])
    print('testing mcx_decompose on', g)

    u = Circuit([g]).unitary(with_dummy=True)

    circ = decompose.mcx_decompose(g, list(range(5)))

    assert operations.is_equiv_unitary(u, circ.unitary(with_dummy=True))

    print(circ.to_cirq())


def test_cccx_1(shuffle: bool = False):
    g = gates.X.on(7, [0, 1, 2])

    if shuffle:
        g_qregs = rng.permutation(g.qregs).tolist()
        g = g.on(g_qregs[0], g_qregs[1:])
    print('testing mcx_decompose on', g)

    u = Circuit([g]).unitary(with_dummy=True)

    circ = decompose.mcx_decompose(g, list(range(8)))

    assert operations.is_equiv_unitary(u, circ.unitary(with_dummy=True))

    print(circ.to_cirq())


def test_ccccx(shuffle: bool = False):
    g = gates.X.on(6, [0, 1, 2, 3])

    if shuffle:
        g_qregs = rng.permutation(g.qregs).tolist()
        g = g.on(g_qregs[0], g_qregs[1:])
    print('testing mcx_decompose on', g)

    u = Circuit([g]).unitary(with_dummy=True)

    circ = decompose.mcx_decompose(g, list(range(7)))

    assert operations.is_equiv_unitary(u, circ.unitary())

    print(circ.to_cirq())


def test_ccccx_1(shuffle: bool = False):
    g = gates.X.on(5, [0, 1, 2, 3])

    if shuffle:
        g_qregs = rng.permutation(g.qregs).tolist()
        g = g.on(g_qregs[0], g_qregs[1:])
    print('testing mcx_decompose on', g)

    u = Circuit([g]).unitary(with_dummy=True)

    circ = decompose.mcx_decompose(g, list(range(6)))

    assert operations.is_equiv_unitary(u, circ.unitary())

    print(circ.to_cirq())


def test_cccccx(shuffle: bool = False):
    g = gates.X.on(8, [0, 1, 2, 3, 4])

    if shuffle:
        g_qregs = rng.permutation(g.qregs).tolist()
        g = g.on(g_qregs[0], g_qregs[1:])
    print('testing mcx_decompose on', g)

    u = Circuit([g]).unitary(with_dummy=True)

    circ = decompose.mcx_decompose(g, list(range(9)))

    assert operations.is_equiv_unitary(u, circ.unitary())

    print(circ.to_cirq())


def test_cccccx_1(shuffle: bool = False):
    g = gates.X.on(6, [0, 1, 2, 3, 4])

    if shuffle:
        g_qregs = rng.permutation(g.qregs).tolist()
        g = g.on(g_qregs[0], g_qregs[1:])
    print('testing mcx_decompose on', g)

    u = Circuit([g]).unitary(with_dummy=True)

    circ = decompose.mcx_decompose(g, list(range(7)))

    assert operations.is_equiv_unitary(u, circ.unitary())

    print(circ.to_cirq())
