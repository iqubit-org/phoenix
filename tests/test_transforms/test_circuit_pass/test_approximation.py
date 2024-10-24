import cirq
from phoenix.utils.arch import gene_random_circuit
from phoenix.transforms.circuit_pass import approx_to_cnot, approx_to_su4


def test_approx_to_cnot():
    """
    Test approximate synthesis using CNOT + U3 gates
    Use a random circuit with 4 qubits and 100 gates
    For computational efficiency, set partitioning grain as default (3), set max_synthesis_size as default (3)
    """
    circ = gene_random_circuit(4, 100)
    circ_approx = approx_to_cnot(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_approx.unitary(),
        atol=1e-5
    )


def test_approx_to_su4():
    """
    Test approximate synthesis using SU(4) gates, i.e., Canonical + U3
    Use a random circuit with 4 qubits and 100 gates (rewired to [0,1,2,3,6])
    For computational efficiency, set partitioning grain as default (3), set max_synthesis_size as default (3)
    """
    circ = gene_random_circuit(4, 100)
    circ = circ.rewire({
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 6
    })  # non-continuous qubit indices
    circ_approx = approx_to_su4(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_approx.unitary(),
        atol=1e-5
    )
