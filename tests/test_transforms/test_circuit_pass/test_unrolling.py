import cirq
from phoenix import circuits, gates
from phoenix import transforms


def test_unroll_su4_to_cnot():
    """Partition --> Fuse --> Unroll --> Check"""
    circ = circuits.Circuit([
        gates.H.on(0), gates.H.on(2), gates.H.on(5),
        gates.Z.on(0),
        gates.X.on(2, 1), gates.X.on(5, 4),
        gates.X.on(1, 0), gates.X.on(3, 2),
        gates.H.on(2), gates.H.on(3),
        gates.X.on(2, 1), gates.X.on(5, 3),
        gates.Z.on(3),
        gates.X.on(3, 4),
        gates.X.on(0, 3)
    ])

    blocks = transforms.circuit_pass.sequential_partition(circ, grain=2)
    fused = transforms.circuit_pass.fuse_blocks(blocks)
    circ_unrolled = transforms.circuit_pass.unroll_su4(fused, by='cnot')
    cirq.testing.assert_allclose_up_to_global_phase(
        circ_unrolled.unitary(),
        circ.unitary(),
        atol=1e-7
    )


def test_unroll_su4_to_can():
    """Partition --> Fuse --> Unroll --> Check"""
    circ = circuits.Circuit([
        gates.H.on(0), gates.H.on(2), gates.H.on(5),
        gates.Z.on(0),
        gates.X.on(2, 1), gates.X.on(5, 4),
        gates.X.on(1, 0), gates.X.on(3, 2),
        gates.H.on(2), gates.H.on(3),
        gates.X.on(2, 1), gates.X.on(5, 3),
        gates.Z.on(3),
        gates.X.on(3, 4),
        gates.X.on(0, 3)
    ])

    blocks = transforms.circuit_pass.sequential_partition(circ, grain=2)
    fused = transforms.circuit_pass.fuse_blocks(blocks)
    circ_unrolled = transforms.circuit_pass.unroll_su4(fused, by='can')
    cirq.testing.assert_allclose_up_to_global_phase(
        circ_unrolled.unitary(),
        circ.unitary(),
        atol=1e-7
    )
