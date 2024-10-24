"""Test Weyl gate"""
import cirq
import numpy as np
from scipy import stats
from phoenix import gates
from phoenix import decompose


def get_weyl_coordinates(g):
    assert g.num_qregs == 2
    circ = decompose.can_decompose(g)
    can = next(iter([g for g in circ.gates if isinstance(g, gates.Can)]))
    return can.weyl_coordinates


def xyz_satisfied(x, y, z):
    """Whethter the coordinates are reduced to (x, y, z) ∈ {1/2 >= x >= y >= z >= 0} ∪ {1/2 >= (1-x) >= y >= z >= 0}"""

    def within_left(x, y, z):
        x, y, z = round(x, 4), round(y, 4), round(z, 4)
        if 0.5 >= x >= y >= z >= 0:
            return True
        return False

    def within_right(x, y, z):
        one_minus_x = round(1 - x, 4)
        y, z = round(y, 4), round(z, 4)
        if 0.5 >= one_minus_x >= y >= z >= 0:
            return True
        return False

    return within_left(x, y, z) or within_right(x, y, z)


def test_swap():
    swap = gates.SWAP.on([0, 1])
    print(get_weyl_coordinates(swap))
    assert np.allclose(get_weyl_coordinates(swap), (1 / 2, 1 / 2, 1 / 2))


def test_iswap():
    iswap = gates.ISWAP.on([0, 1])
    print(get_weyl_coordinates(iswap))
    assert np.allclose(get_weyl_coordinates(iswap), (1 / 2, 1 / 2, 0))


def test_sqsw():
    sqsw = gates.SQSW.on([0, 1])
    print(get_weyl_coordinates(sqsw))
    assert np.allclose(get_weyl_coordinates(sqsw), (1 / 4, 1 / 4, 1 / 4))


def test_sqswdg():
    sqswdg = gates.SQSW.on([0, 1]).hermitian()
    print(get_weyl_coordinates(sqswdg))
    assert np.allclose(get_weyl_coordinates(sqswdg), (3 / 4, 1 / 4, 1 / 4))


def test_sqisw():
    sqisw = gates.SQiSW.on([0, 1])
    print(get_weyl_coordinates(sqisw))
    assert np.allclose(get_weyl_coordinates(sqisw), (1 / 4, 1 / 4, 0))


def test_cx():
    cx = gates.UnivGate(cirq.unitary(cirq.CX)).on([0, 1])
    print(get_weyl_coordinates(cx))
    assert np.allclose(get_weyl_coordinates(cx), (1 / 2, 0, 0))


def test_cz():
    cz = gates.UnivGate(cirq.unitary(cirq.CZ)).on([0, 1])
    print(get_weyl_coordinates(cz))
    assert np.allclose(get_weyl_coordinates(cz), (1 / 2, 0, 0))


def test_syc():
    syc = gates.SYC.on([0, 1])
    print(get_weyl_coordinates(syc))
    assert np.allclose(get_weyl_coordinates(syc), (1 / 2, 1 / 2, 1 / 12))


def test_cv():
    cv = gates.UnivGate(np.block([
        [np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), cirq.unitary(cirq.XPowGate(exponent=0.5))]
    ])).on([0, 1])
    print(get_weyl_coordinates(cv))
    assert np.allclose(get_weyl_coordinates(cv), (1 / 4, 0, 0))


def test_b():
    b = gates.B.on([0, 1])
    print(get_weyl_coordinates(b))
    assert np.allclose(get_weyl_coordinates(b), (1 / 2, 1 / 4, 0))


def test_random():
    for _ in range(10):
        x = np.random.rand() / 2
        y = x - np.random.rand() / 10
        while y < 0:
            y = x - np.random.rand() / 10
        z = y - np.random.rand() / 10
        while z < 0:
            z = y - np.random.rand() / 10
        assert np.allclose(gates.Can(x * np.pi, y * np.pi, z * np.pi).weyl_coordinates, (x, y, z))

    for _ in range(10):
        x = 1 - np.random.rand() / 2
        one_minus_x = 1 - x
        y = one_minus_x - np.random.rand() / 10
        while y < 0:
            y = one_minus_x - np.random.rand() / 10
        z = y - np.random.rand() / 10
        while z < 0:
            z = y - np.random.rand() / 10
        assert np.allclose(gates.Can(x * np.pi, y * np.pi, z * np.pi).weyl_coordinates, (x, y, z))

    for _ in range(20):
        g = gates.UnivGate(stats.unitary_group.rvs(4)).on([0, 1])
        assert xyz_satisfied(*get_weyl_coordinates(g))
