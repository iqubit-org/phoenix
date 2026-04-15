from __future__ import annotations

import argparse
from itertools import permutations

import sympy as sp


AXES = ("X", "Y", "Z")
AXIS_TO_INDEX = {axis: idx for idx, axis in enumerate(AXES)}


def build_interaction_matrix(labels: list[str], coeffs: list[sp.Expr]) -> sp.Matrix:
    matrix = sp.zeros(3, 3)
    for label, coeff in zip(labels, coeffs):
        left, right = label
        matrix[AXIS_TO_INDEX[left], AXIS_TO_INDEX[right]] += coeff
    return matrix


def support_edges(labels: list[str]) -> list[tuple[str, str]]:
    return sorted({(label[0], label[1]) for label in labels})


def max_matching_size(edges: list[tuple[str, str]]) -> int:
    edge_set = set(edges)
    best = 0
    for size in range(1, 4):
        for left_subset in permutations(AXES, size):
            for right_subset in permutations(AXES, size):
                if all((left_subset[i], right_subset[i]) in edge_set for i in range(size)):
                    best = max(best, size)
    return best


def left_right_dims(labels: list[str]) -> tuple[int, int]:
    return len({label[0] for label in labels}), len({label[1] for label in labels})


def parse_coeffs(raw_coeffs: list[str] | None, num_labels: int) -> list[sp.Expr]:
    if raw_coeffs is None:
        return [sp.Symbol(f"a{i + 1}", real=True) for i in range(num_labels)]
    if len(raw_coeffs) != num_labels:
        raise ValueError("The number of coefficients must match the number of labels.")
    return [sp.sympify(coeff) for coeff in raw_coeffs]


def validate_labels(labels: list[str]) -> list[str]:
    normalized = []
    for label in labels:
        current = label.upper()
        if len(current) != 2 or any(axis not in AXIS_TO_INDEX for axis in current):
            raise ValueError(f"Invalid 2Q Pauli label: {label!r}")
        normalized.append(current)
    return normalized


def is_numeric_matrix(matrix: sp.Matrix) -> bool:
    return all(entry.free_symbols == set() for entry in matrix)


def classify_generic_cnot(generic_rank: int) -> str:
    if generic_rank <= 0:
        return "0-CNOT class"
    if generic_rank <= 2:
        return "at-most-2-CNOT class"
    return "generic-3-CNOT class"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze H = sum_i a_i P_i for distinct or repeated 2Q Pauli terms using the "
            "interaction-matrix formalism."
        )
    )
    parser.add_argument(
        "labels",
        nargs="+",
        help="2Q Pauli labels, e.g. XX YY ZZ or ZX YY ZZ",
    )
    parser.add_argument(
        "--coeffs",
        nargs="*",
        help="Optional coefficients. If omitted, symbolic coefficients a1, a2, ... are used.",
    )
    args = parser.parse_args()

    labels = validate_labels(args.labels)
    coeffs = parse_coeffs(args.coeffs, len(labels))
    matrix = build_interaction_matrix(labels, coeffs)
    edges = support_edges(labels)
    generic_rank = max_matching_size(edges)
    left_dim, right_dim = left_right_dims(labels)

    print("Labels:", labels)
    print("Coefficients:", coeffs)
    print()
    print("Interaction matrix J (rows/cols ordered as X, Y, Z):")
    print(sp.pretty(matrix))
    print()
    print("Support edges:", edges)
    print("Left-axis span dimension:", left_dim)
    print("Right-axis span dimension:", right_dim)
    print("Generic rank(J) from support graph:", generic_rank)
    print("Generic class:", classify_generic_cnot(generic_rank))
    print()

    determinant = sp.factor(matrix.det())
    print("det(J):", determinant)
    if generic_rank == 3:
        print("Perfect matching detected: yes")
    else:
        print("Perfect matching detected: no")

    if left_dim <= 2 or right_dim <= 2:
        print("Structural rule: rank(J) <= 2, so c3 = 0 and the block needs at most 2 CNOTs.")

    if is_numeric_matrix(matrix):
        exact_rank = int(matrix.rank())
        print("Exact numeric rank(J):", exact_rank)
        if exact_rank <= 2:
            print("Exact consequence: c3 = 0.")
        else:
            print("Exact consequence: generic 3-CNOT behavior is active for these coefficients.")


if __name__ == "__main__":
    main()
