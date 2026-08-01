import numpy as np
import pytest
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator

from phoenix.compiler import compile_hamiltonian_simulation
from phoenix.hamiltonian import Hamiltonian
from phoenix.primitive.holistic import holistic_compile, peel_forward


def test_explicit_adaptive_settings_match_the_production_default():
    ham = Hamiltonian(["IXX", "XYZ"], [0.2, -0.3])

    default = peel_forward(ham)
    explicit = peel_forward(ham, rho_threshold=0.35)

    assert [(gate.name, qubits) for gate, qubits in default.moves] == [
        (gate.name, qubits) for gate, qubits in explicit.moves
    ]
    assert default.emissions == explicit.emissions


def test_density_threshold_endpoints_recover_fixed_emission_policies():
    ham = Hamiltonian(["XX"], [0.2])

    aggressive = peel_forward(ham, rho_threshold=1.0)
    weight_one = peel_forward(ham, rho_threshold=0.0)

    assert len(aggressive.moves) == 0
    assert len(weight_one.moves) == 1
    assert all(label.count("I") == 1 for _time, labels, _coeffs in weight_one.emissions for label in labels)


def test_default_density_policy_emits_sparse_weight_two_rows():
    ham = Hamiltonian(["IIIIXX", "IIXXII", "XXIIII"], [0.1, 0.2, 0.3])

    result = peel_forward(ham)

    assert len(result.moves) == 0
    assert all(label.count("I") == 4 for _time, labels, _coeffs in result.emissions for label in labels)


def test_default_density_policy_peels_dense_weight_two_rows():
    result = peel_forward(Hamiltonian(["XX"], [0.2]))

    assert len(result.moves) == 1
    assert all(label.count("I") == 1 for _time, labels, _coeffs in result.emissions for label in labels)


@pytest.mark.parametrize("rho_threshold", [0.0, 0.35, 1.0])
def test_all_density_threshold_arms_preserve_the_target_unitary(rho_threshold):
    ham = Hamiltonian(["XX"], [0.2])
    actual = holistic_compile(
        ham,
        terminal="replay",
        rho_threshold=rho_threshold,
    )
    expected = PauliEvolutionGate(ham)

    assert np.allclose(Operator(actual).data, Operator(expected).data)


@pytest.mark.parametrize("rho_threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_density_threshold_is_rejected(rho_threshold):
    with pytest.raises(ValueError, match="rho_threshold"):
        peel_forward(Hamiltonian(["X"], [0.2]), rho_threshold=rho_threshold)


@pytest.mark.parametrize(
    "entry_point", [peel_forward, holistic_compile, compile_hamiltonian_simulation]
)
def test_retired_emit_max_weight_keyword_is_not_accepted(entry_point):
    with pytest.raises(TypeError, match="emit_max_weight"):
        entry_point(Hamiltonian(["X"], [0.2]), emit_max_weight=1)
