import cirq.testing
from phoenix import Circuit
from phoenix.transforms.circuit_pass import fuse_blocks, fuse_neighbor_u3
from phoenix.transforms.circuit_pass import sequential_partition


def test_fuse_blocks_2q():
    circ = Circuit.from_qasm(fname='../../input/cx-basis/bit_adder/rd53_251.qasm')
    # circ = gene_random_circuit(8, 300)
    blocks_2q = sequential_partition(circ, 2)
    circ_from_blocks_2q = fuse_blocks(blocks_2q)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_from_blocks_2q.unitary(),
        atol=1e-5
    )


def test_fuse_blocks_3q():
    circ = Circuit.from_qasm(fname='../../input/cx-basis/bit_adder/rd53_251.qasm')

    # circ = gene_random_circuit(8, 300)
    blocks_3q = sequential_partition(circ, 3)
    circ_from_blocks_3q = fuse_blocks(blocks_3q)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_from_blocks_3q.unitary(),
        atol=1e-5
    )


def test_fuse_neighbor_u3():
    circ = Circuit.from_qasm(fname='../../input/cx-basis/bit_adder/rd53_251.qasm')
    circ_fused = fuse_neighbor_u3(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_fused.unitary(),
        atol=1e-5
    )
