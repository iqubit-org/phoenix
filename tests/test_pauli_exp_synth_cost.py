import pytest

from phoenix.utils import pauli_exp_synth_cost


@pytest.mark.parametrize(
    ("paulis", "expected_cost"),
    [
        ([], 0),
        (["XX"], 2),
        (["xx", "YY"], 2),
        (["XX", "YY", "ZZ"], 3),
        (["XY", "YZ", "ZX"], 3),
        (["ZX", "YY", "ZZ", "YX"], 2),
    ],
)
def test_generic_cost_uses_support_graph_rank(paulis, expected_cost):
    assert pauli_exp_synth_cost(paulis) == expected_cost


def test_numeric_coefficients_combine_repeated_terms():
    assert pauli_exp_synth_cost(["XX", "XX"], [1.0, -1.0]) == 0


def test_numeric_coefficients_can_lower_full_support_rank():
    paulis = ["XX", "XY", "YX", "YY", "ZZ"]

    assert pauli_exp_synth_cost(paulis) == 3
    assert pauli_exp_synth_cost(paulis, [1.0, 1.0, 1.0, 1.0, 1.0]) == 2


@pytest.mark.parametrize("pauli", ["X", "XYZ", "XI", "AB", 42])
def test_invalid_pauli_label_is_rejected(pauli):
    with pytest.raises(ValueError, match="Invalid 2Q Pauli label"):
        pauli_exp_synth_cost([pauli])


def test_coefficient_count_must_match_pauli_count():
    with pytest.raises(ValueError, match="number of coefficients"):
        pauli_exp_synth_cost(["XX", "YY"], [1.0])


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), 1.0 + 1.0j])
def test_coefficients_must_be_finite_and_real(coefficient):
    with pytest.raises(ValueError, match="finite real numbers"):
        pauli_exp_synth_cost(["XX"], [coefficient])
