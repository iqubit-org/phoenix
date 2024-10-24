from phoenix import decompose
from phoenix import gates
from scipy.stats import unitary_group
from phoenix.utils.operations import controlled_unitary_matrix, multiplexor_matrix, tensor_slots
import numpy as np
from scipy import linalg

rand_unitary = unitary_group.rvs


def assert_equivalent_unitary(U, V):
    try:
        import cirq
        cirq.testing.assert_allclose_up_to_global_phase(U, V, atol=1e-5)
    except ModuleNotFoundError:
        from phoenix.utils.operations import is_equiv_unitary
        assert is_equiv_unitary(U, V)


def test_euler_decompose():
    # ZYZ basis Euler decomposition
    print('ZYZ-basis Euler decomposition')
    U = rand_unitary(2, random_state=123)
    g = gates.UnivGate(U, 'U').on(0)
    circ_zyz = decompose.euler_decompose(g)
    print(circ_zyz)
    assert_equivalent_unitary(U, circ_zyz.unitary())

    # U3 basis Euler decomposition
    print('U3-basis Euler decomposition')
    circ_u3 = decompose.euler_decompose(g, basis='u3')
    print(circ_u3)
    assert_equivalent_unitary(U, circ_u3.unitary())
    print()


def test_tensor_product_decompose():
    print('Tensor product decomposition')
    XY = np.kron(gates.X.data, gates.Y.data)
    g = gates.UnivGate(XY, 'XY').on([0, 1])
    circ = decompose.tensor_product_decompose(g)
    print(circ)
    assert_equivalent_unitary(XY, circ.unitary())
    print()


def test_abc_decompose():
    # special case:
    print('ABC decomposition: CRz(pi)')
    g = gates.RZ(np.pi).on(1, 0)
    circ = decompose.abc_decompose(g)
    print(circ)
    assert_equivalent_unitary(controlled_unitary_matrix(g.data), circ.unitary())

    # arbitrary CU gate
    print('ABC decomposition: arbitrary CU gate')
    U = rand_unitary(2, random_state=123)
    g = gates.UnivGate(U, 'U').on(1, 0)
    circ = decompose.abc_decompose(g)
    print(circ)
    assert_equivalent_unitary(controlled_unitary_matrix(U), circ.unitary())
    print()


def test_kak_decompose():
    # KAK decomposition (to CNOT + U3 gates)
    print('KAK decomposition')
    g = gates.UnivGate(rand_unitary(4, random_state=123), 'U').on([0, 1])
    circ = decompose.kak_decompose(g)
    print(circ)
    print()
    assert_equivalent_unitary(g.data, circ.unitary())


def test_can_decompose():
    # Canonical decomposition (to Canonical + U3 gates)
    print('Canonical decomposition')
    g = gates.UnivGate(rand_unitary(4, random_state=123), 'U').on([0, 1])
    circ = decompose.can_decompose(g)
    print(circ)
    print()
    assert_equivalent_unitary(g.data, circ.unitary())


def test_demultiplex_pair():
    n = 2
    U1 = rand_unitary(2 ** (n - 1), random_state=123)
    U2 = rand_unitary(2 ** (n - 1), random_state=1234)
    circ = decompose.demultiplex_pair(U1, U2, tqs=list(range(1, n)), cq=0)
    assert_equivalent_unitary(linalg.block_diag(U1, U2), circ.unitary())


def test_demultiplex_puali():
    print('Demultiplexing Pauli Multiplexor')
    np.random.seed(123)
    n = 5
    rads = np.random.rand(2 ** (n - 1))
    sigma = 'Z'
    rot_sigma = getattr(gates, f'R{sigma}')
    cqs = list(range(n))
    tq = cqs.pop(1)
    U = multiplexor_matrix(n, tq, *[rot_sigma(rad).data for rad in rads])
    circ = decompose.demultiplex_pauli(sigma, tq, cqs, *rads, permute_cnot=True)
    print(circ)
    print()
    assert_equivalent_unitary(U, circ.unitary())


def test_qs_decompose():
    print('Quantum Shannon decomposition')
    n = 4
    U = rand_unitary(2 ** n, random_state=123)
    g = gates.UnivGate(U, 'U').on(list(range(n)))
    circ = decompose.qs_decompose(g)
    print(circ)
    print()
    assert_equivalent_unitary(U, circ.unitary())


def test_cu_decompose():
    print('m-control n-target CU decomposition')
    cqs = [0, 2, 4, 5]  # arbitrary order is OK
    tqs = [1, 6]
    m = len(cqs)
    n = len(tqs)
    U = rand_unitary(2 ** n, random_state=123)
    circ = decompose.cu_decompose(gates.UnivGate(U, 'U').on(tqs, cqs))
    print(circ)
    print()
    assert_equivalent_unitary(
        tensor_slots(controlled_unitary_matrix(U, m), max(cqs + tqs) + 1, cqs + tqs),
        circ.unitary(with_dummy=True)
    )
