from typing import List

import networkx as nx

from pauliopt.circuits import Circuit
from pauliopt.gates import CX, H, V, Vdg, Sdg
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli_strings import I, X, Y, Z, Pauli
from pauliopt.topologies import Topology
from pauliopt.utils import is_cutting
from pauliopt.clifford.tableau import CliffordRegion
from pauliopt.clifford.tableau_synthesis import synthesize_tableau


def pick_row(pp: PauliPolynomial, columns_to_use, qubits_to_use):
    qubit_scores = []
    weight_i = 10
    for q in qubits_to_use:
        i_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == I])
        x_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == X])
        y_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == Y])
        z_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == Z])
        score = (
            weight_i * i_score
            + max([x_score, y_score, z_score])
            - min([x_score, y_score, z_score])
        )
        qubit_scores.append((q, score))
    return max(qubit_scores, key=lambda x: x[1])[0]


def update_gadget_single_column(
    pp: PauliPolynomial, qc: Circuit, q: int, p: Pauli, columns_to_use
):
    if p == X:
        gate = H(q)
        pp.propagate(gate, columns_to_use)
        qc.h(q)
    elif p == Y:
        gate = V(q)
        pp.propagate(gate, columns_to_use)
        qc.v(q)
    elif p == Z:
        pass  # Nothing to do here
    else:
        raise ValueError("Invalid Pauli")


def find_common_paulis(q, pp: PauliPolynomial, columns_to_use):
    common_paulis = []
    for col in columns_to_use:
        if pp[col][q] != I:
            common_paulis.append(pp[col][q])
    common_paulis = list(set(common_paulis))
    if len(common_paulis) == 1:
        return common_paulis[0]
    return None


def update_single_qubits(pp: PauliPolynomial, c: Circuit, qubits: list, columns_to_use):
    change = False
    for q in qubits:
        p = find_common_paulis(q, pp, columns_to_use)
        if p is not None:
            update_gadget_single_column(pp, c, q, p, columns_to_use)
            change = True
    return change


def is_compatible(pp: PauliPolynomial, q1, q2, columns_to_use):
    return find_compatible_pair(pp, q1, q2, columns_to_use) is not None


def find_compatible_pair(pp: PauliPolynomial, q1, q2, columns_to_use):
    for p1 in [X, Y, Z]:
        for p2 in [X, Y, Z]:
            found_pair = True
            for l in columns_to_use:
                p_gdt = pp[l][q1]
                p_gdt2 = pp[l][q2]
                a_valid = p_gdt in [I, p1]
                b_valid = p_gdt2 in [I, p2]
                if a_valid != b_valid:
                    found_pair = False
                    break
            if found_pair:
                return p1, p2

    return None


def pick_best_pair(pp, G: nx.Graph, columns_to_use, qubits):
    for q_1, q_2 in G.edges:
        if q_1 in qubits and q_2 in qubits:
            pairs = find_compatible_pair(pp, q_1, q_2, columns_to_use)
            if pairs is not None:
                return pairs, (q_1, q_2)
    return None


def filter_identity_qubits(pp: PauliPolynomial, qubits, columns_to_use):
    non_identity_qubits = []
    for q in qubits:
        if pp[columns_to_use[0]][q] != I:
            non_identity_qubits.append(q)
    return non_identity_qubits


def update_pair_qubits(
    pp: PauliPolynomial, c: Circuit, topology, qubits, columns_to_use
):
    non_visited_qubits = [q for q in qubits]
    non_visited_qubits = filter_identity_qubits(pp, non_visited_qubits, columns_to_use)
    qubit_pairs = pick_best_pair(
        pp,
        topology.to_nx.subgraph(non_visited_qubits),
        columns_to_use,
        non_visited_qubits,
    )
    (p1, p2), (q_1, q_2) = qubit_pairs
    non_visited_qubits.remove(q_1)
    non_visited_qubits.remove(q_2)
    update_gadget_single_column(pp, c, q_1, p1, columns_to_use)
    update_gadget_single_column(pp, c, q_2, p2, columns_to_use)

    pp.propagate(CX(q_1, q_2), columns_to_use)
    c.cx(q_1, q_2)


def partition_pauli_polynomial(pp: PauliPolynomial, row: int, columns_to_use: list):
    col_i = []
    col_x = []
    col_y = []
    col_z = []
    for col in columns_to_use:
        if pp.pauli_gadgets[col][row] == X:
            col_x.append(col)
        elif pp.pauli_gadgets[col][row] == Y:
            col_y.append(col)
        elif pp.pauli_gadgets[col][row] == Z:
            col_z.append(col)
        elif pp.pauli_gadgets[col][row] == I:
            col_i.append(col)
        else:
            raise ValueError("Invalid Pauli in Partition")
    return col_i, col_x, col_y, col_z


def max_partition_pauli_polynomial(pp: PauliPolynomial, row: int, columns_to_use: list):
    cols = [[X], [Y], [Z]]
    cols_i = []
    for col in columns_to_use:
        if pp.pauli_gadgets[col][row] == X:
            cols[0].append(col)
        elif pp.pauli_gadgets[col][row] == Y:
            cols[1].append(col)
        elif pp.pauli_gadgets[col][row] == Z:
            cols[2].append(col)
        elif pp.pauli_gadgets[col][row] == I:
            cols_i.append(col)
        else:
            raise ValueError("Invalid Pauli in Weighted")

    cols.sort(key=lambda x: len(x))

    return cols_i, cols[-1][1:], cols[-1][0], cols[0][1:] + cols[1][1:]


def identity_partition_pauli_polynomial(
    pp: PauliPolynomial, row: int, columns_to_use: list
):
    col_i = []
    cols = []
    for col in columns_to_use:
        if pp.pauli_gadgets[col][row] == I:
            col_i.append(col)
        elif pp.pauli_gadgets[col][row] == X:
            cols.append(col)
        elif pp.pauli_gadgets[col][row] == Y:
            cols.append(col)
        elif pp.pauli_gadgets[col][row] == Z:
            cols.append(col)
        else:
            raise ValueError("Invalid Pauli")
    return col_i, cols


def bipartition_pauli_polynomial(pp: PauliPolynomial, row: int, columns_to_use: list):
    col_i, col_x, col_y, col_z = partition_pauli_polynomial(pp, row, columns_to_use)
    cols = []
    if not col_x:
        return col_i, X, Z, col_y, Z, col_z
    if not col_y:
        return col_i, X, Y, col_x, Z, col_z

    if len(col_x) != 0 and len(col_y) != 0:
        cols.append((X, Y, col_x + col_y, Z, col_z, len(col_x) + len(col_y)))
    if len(col_x) != 0 and len(col_z) != 0:
        cols.append((X, Z, col_x + col_z, Y, col_y, len(col_x) + len(col_z)))
    if len(col_y) != 0 and len(col_z):
        cols.append((Y, Z, col_y + col_z, X, col_x, len(col_y) + len(col_z)))

    if cols:
        type_two_1, type_two_2, cols_2, type_col1, col1, _ = max(
            cols, key=lambda x: x[-1]
        )
    else:
        raise Exception("Invalid State")
    return col_i, type_two_1, type_two_2, cols_2, type_col1, col1


def zy_partition_pauli_polynomial(pp: PauliPolynomial, row: int, columns_to_use: list):
    col_i, col_x, col_y, col_z = partition_pauli_polynomial(pp, row, columns_to_use)
    return col_i, col_z + col_y, col_x


def propagate_circuit(
    pp: PauliPolynomial, circuit: CliffordRegion, sub_columns: List[int] = None
):
    if sub_columns is None:
        sub_columns = list(range(pp.num_gadgets))
    for gate in reversed(circuit.gates):
        pp.propagate(gate, sub_columns)


def pauli_polynomial_steiner_gray_clifford(pp: PauliPolynomial, topo: Topology):
    perm_gadgets = []
    permutation = {k: k for k in range(pp.num_qubits)}
    G = topo.to_nx

    def identity_recurse(columns_to_use, qubits_to_use):
        """Determines row and row_next for recursion, removes all identity operators on both row and row_next"""
        qc_out = Circuit(pp.num_qubits)
        qc_prop = CliffordRegion(pp.num_qubits)
        # always check to remove columns here, this should prevent some of the strange reintroduction of rotations
        qc_out += check_columns(columns_to_use)
        if not columns_to_use or not qubits_to_use:
            return qc_out, qc_prop
        G_sub = G.subgraph(qubits_to_use)

        non_cutting = [q for q in qubits_to_use if not is_cutting(q, G_sub)]

        row = pick_row(pp, columns_to_use, non_cutting)
        row_neighbors = list(G_sub.neighbors(row))
        assert row_neighbors

        col_i, col_rest_1 = identity_partition_pauli_polynomial(pp, row, columns_to_use)

        remaining_qubits = [q for q in qubits_to_use if q != row]
        qc_i, qc_prop_i = identity_recurse(col_i, remaining_qubits)
        qc_out += qc_i
        propagate_circuit(pp, qc_prop_i, col_rest_1)
        # prepend propagated gates
        qc_prop = qc_prop_i + qc_prop

        row_next = pick_row(pp, col_rest_1, row_neighbors)
        # identity is empty on `row`, identify region with largest pauli
        _, col_max, pauli_max, col_rest_2 = max_partition_pauli_polynomial(
            pp, row, col_rest_1
        )

        # find column identities on `row_next` in `col_max`
        col_i_swap, col_rest_swap = identity_partition_pauli_polynomial(
            pp, row_next, col_max
        )
        col_rest_3 = col_rest_swap + col_rest_2

        # this swap maximizes identity region that is recursed into identity_recurse
        # we also know that this swap removes rotations from a non-cutting vertex
        qc_swap, qc_prop_swap = swap_row(col_i_swap, row, row_next, pauli_max)
        # Add swapping gates to output and propagate gates
        # propagating qc_prop_swap jumbles paulis on `row`
        qc_out += qc_swap
        propagate_circuit(pp, qc_prop_swap, col_rest_3)

        qc_prop = qc_prop_swap + qc_prop

        # find new identities on `row` in `col_rest` and
        col_i, col_rest_4 = identity_partition_pauli_polynomial(pp, row, col_rest_3)
        # immediately recurse removing `row`
        qc_i_re, qc_prop_i_re = identity_recurse(col_i_swap + col_i, remaining_qubits)
        qc_out += qc_i_re

        propagate_circuit(pp, qc_prop_i_re, col_rest_4)
        qc_prop = qc_prop_i_re + qc_prop

        # now if row and row_next have no identities, we p_recurse
        col_i_row, col_rest_5 = identity_partition_pauli_polynomial(pp, row, col_rest_4)

        col_i_row_next, col_rest_6 = identity_partition_pauli_polynomial(
            pp, row_next, col_rest_5
        )

        # identity is empty on `row`, identify region with largest pauli
        _, col_max, pauli_max, col_rest_7 = max_partition_pauli_polynomial(
            pp, row, col_rest_6
        )
        col_rest_8 = col_i_row + col_i_row_next + col_rest_7
        # basically only called for perfect conditions, otherwise getting rid of I's makes more sense

        qc_p, qc_prop_p = p_recurse(col_max, qubits_to_use, row, row_next, pauli_max)
        qc_out += qc_p
        propagate_circuit(pp, qc_prop_p, col_rest_8)
        qc_prop = qc_prop_p + qc_prop
        # otherwise we continue removing identities

        qc_last, qc_prop_last = identity_recurse(col_rest_8, qubits_to_use)
        qc_out += qc_last
        qc_prop = qc_prop_last + qc_prop

        return qc_out, qc_prop

    def p_recurse(columns_to_use, qubits_to_use, row, row_next, rec_type):
        """Always receives columns where `row` and `row_next` do not contain identities because functions here can reintroduce entanglement if identities exist"""
        assert rec_type in [X, Y, Z]
        # no check columns because all entries are non_identity
        qc_out = Circuit(pp.num_qubits)
        qc_prop = CliffordRegion(pp.num_qubits)

        if not columns_to_use or not qubits_to_use:
            return qc_out, qc_prop

        if rec_type == X:
            qc_out.h(row)
            pp.propagate(H(row), columns_to_use)
        elif rec_type == Y:
            qc_out.v(row)
            pp.propagate(Vdg(row), columns_to_use)

        col_i_row, _ = identity_partition_pauli_polynomial(pp, row, columns_to_use)
        col_i, col_x, col_y, col_z = partition_pauli_polynomial(
            pp, row_next, columns_to_use
        )

        # should never receive identity
        assert not col_i_row
        assert not col_i

        if not col_x and not col_y:
            qc_one, qc_prop_one = simplify_one_pauli(
                col_z, qubits_to_use, row, row_next, Z
            )
            qc_out += qc_one
            qc_prop = qc_prop_one + qc_prop
        elif not col_x and not col_z:
            qc_one, qc_prop_one = simplify_one_pauli(
                col_y, qubits_to_use, row, row_next, Y
            )
            qc_out += qc_one
            qc_prop = qc_prop_one + qc_prop
        elif not col_y and not col_z:
            qc_one, qc_prop_one = simplify_one_pauli(
                col_x, qubits_to_use, row, row_next, X
            )
            qc_out += qc_one
            qc_prop = qc_prop_one + qc_prop
        elif not col_x:
            qc_two, qc_prop_two = simplify_two_pauli(
                col_y + col_z, qubits_to_use, row, row_next, Y, Z
            )
            qc_out += qc_two
            qc_prop = qc_prop_two + qc_prop
        elif not col_y:
            qc_two, qc_prop_two = simplify_two_pauli(
                col_x + col_z, qubits_to_use, row, row_next, X, Z
            )
            qc_out += qc_two
            qc_prop = qc_prop_two + qc_prop
        elif not col_z:
            qc_two, qc_prop_two = simplify_two_pauli(
                col_x + col_y, qubits_to_use, row, row_next, X, Y
            )
            qc_out += qc_two
            qc_prop = qc_prop_two + qc_prop
        else:
            qc_two, qc_prop_two = simplify_two_pauli(
                col_y + col_z, qubits_to_use, row, row_next, Y, Z
            )
            qc_out += qc_two
            propagate_circuit(pp, qc_prop_two, col_x)
            qc_prop = qc_prop_two + qc_prop
            # this may introduce identity to the circuit, so identity recurse here

            qc_i, qc_prop_i = identity_recurse(col_x, qubits_to_use)
            qc_out += qc_i
            qc_prop = qc_prop_i + qc_prop

        if rec_type == X:
            qc_prop.add_gate(H(row))
        elif rec_type == Y:
            qc_prop.add_gate(Vdg(row))

        return qc_out, qc_prop

    def simplify_two_pauli(
        columns_to_use, qubits_to_use, row, row_next, rec_type_1, rec_type_2
    ):
        qc_out = Circuit(pp.num_qubits)
        qc_prop = CliffordRegion(pp.num_qubits)
        if not columns_to_use or not qubits_to_use:
            return qc_out, qc_prop

        if rec_type_1 == X and rec_type_2 == Y:
            qc_out.h(row_next)
            pp.propagate(H(row_next), columns_to_use)
        elif rec_type_1 == X and rec_type_2 == Z:
            qc_out.s(row_next)
            pp.propagate(Sdg(row_next), columns_to_use)

        qc_out.cx(row, row_next)
        pp.propagate(CX(row, row_next), columns_to_use)

        columns_to_use = [col for col in columns_to_use if pp[col][row_next] in [Z, Y]]

        remaining_qubits = [q for q in qubits_to_use if q != row]
        # identity_recurse here because `next_row`` can contain identity
        qc_iden, qc_prop_iden = identity_recurse(columns_to_use, remaining_qubits)
        qc_out += qc_iden

        qc_prop = qc_prop_iden + qc_prop
        qc_prop.add_gate(CX(row, row_next))

        if rec_type_1 == X and rec_type_2 == Y:
            qc_prop.add_gate(H(row_next))
        elif rec_type_1 == X and rec_type_2 == Z:
            qc_prop.add_gate(Sdg(row_next))
        return qc_out, qc_prop

    def simplify_one_pauli(columns_to_use, qubits_to_use, row, row_next, rec_type):
        qc_out = Circuit(pp.num_qubits)
        qc_prop = CliffordRegion(pp.num_qubits)
        if not columns_to_use or not qubits_to_use:
            return qc_out, qc_prop

        if rec_type == X:
            qc_out.h(row_next)
            pp.propagate(H(row_next), columns_to_use)
        elif rec_type == Y:
            qc_out.v(row_next)
            pp.propagate(Vdg(row_next), columns_to_use)

        qc_out.cx(row, row_next)
        pp.propagate(CX(row, row_next), columns_to_use)

        columns_to_use = [col for col in columns_to_use if pp[col][row_next] in [Z]]

        remaining_qubits = [q for q in qubits_to_use if q != row]

        qc_iden, qc_prop_iden = identity_recurse(columns_to_use, remaining_qubits)
        qc_out += qc_iden
        qc_prop = qc_prop_iden + qc_prop

        qc_prop.add_gate(CX(row, row_next))

        if rec_type == X:
            qc_prop.add_gate(H(row_next))
        elif rec_type == Y:
            qc_prop.add_gate(Vdg(row_next))

        return qc_out, qc_prop

    def swap_row(columns_to_use, row, row_next, pauli_type):
        """
        Converts `row,row_next` to `ZI` and converts it to `IZ`. Swaps `row` and `row_next`
        based on pauli_type, returns added gates for swapping and gates for propagation
        """
        qc_out = Circuit(pp.num_qubits)
        qc_prop = CliffordRegion(pp.num_qubits)
        if not columns_to_use:
            return qc_out, qc_prop
        if pauli_type == X:
            qc_out.h(row)
            pp.propagate(H(row), columns_to_use)
        elif pauli_type == Y:
            qc_out.v(row)
            pp.propagate(Vdg(row), columns_to_use)
        elif pauli_type == Z:
            pass

        qc_out.cx(row_next, row)
        qc_out.cx(row, row_next)

        pp.propagate(CX(row_next, row), columns_to_use)
        pp.propagate(CX(row, row_next), columns_to_use)

        qc_prop.add_gate(CX(row, row_next))
        qc_prop.add_gate(CX(row_next, row))

        if pauli_type == X:
            qc_prop.add_gate(H(row))
        elif pauli_type == Y:
            qc_prop.add_gate(Vdg(row))
        elif pauli_type == Z:
            pass

        return qc_out, qc_prop

    def check_columns(columns_to_use):
        qc = Circuit(pp.num_qubits)
        to_remove = []
        for col in columns_to_use:
            if pp[col].num_legs() == 1:
                row = [q for q in range(pp.num_qubits) if pp[col][q] != I][0]
                col_type = pp[col][row]

                if col_type == X:
                    qc.h(row)
                elif col_type == Y:
                    qc.v(row)

                qc.rz(pp.pauli_gadgets[col].angle, row)
                perm_gadgets.append(col)
                to_remove.append(col)

                if col_type == X:
                    qc.h(row)
                elif col_type == Y:
                    qc.vdg(row)
        for col in to_remove:
            columns_to_use.remove(col)
        return qc

    circ_out = Circuit(pp.num_qubits)
    columns_to_use = list(range(pp.num_gadgets))
    circ_out += check_columns(columns_to_use)
    circ_recurse, circ_prop = identity_recurse(
        columns_to_use, list(range(pp.num_qubits))
    )
    circ_prop, permutation = synthesize_tableau(
        circ_prop.to_tableau(), topo, include_swaps=False
    )

    circ_out = circ_out + circ_recurse + circ_prop
    circ_out.final_permutation = circ_prop.final_permutation
    permutation = [permutation[i] for i in range(pp.num_qubits)]
    return circ_out, perm_gadgets, permutation
