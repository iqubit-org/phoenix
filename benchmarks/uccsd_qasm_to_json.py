"""
Convert standard UCCSD QASM files to JSON files for the UCCSD benchmarking.

Each UCCSD circuit is like:
         ┌───┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
    q_0: ┤ X ├┤0             ├┤0             ├┤0             ├┤0             ├
         ├───┤│              ││              ││              ││              │
    q_1: ┤ X ├┤1             ├┤1             ├┤1             ├┤1             ├
         └───┘│              ││              ││              ││              │
    q_2: ─────┤2             ├┤2             ├┤2             ├┤2             ├
         ┌───┐│              ││              ││              ││              │
    q_3: ┤ X ├┤3             ├┤3             ├┤3             ├┤3             ├
         ├───┤│  Evolution^1 ││  Evolution^1 ││  Evolution^1 ││  Evolution^1 │
    q_4: ┤ X ├┤4             ├┤4             ├┤4             ├┤4             ├
         ├───┤│              ││              ││              ││              │
    q_5: ┤ X ├┤5             ├┤5             ├┤5             ├┤5             ├
         └───┘│              ││              ││              ││              │
    q_6: ─────┤6             ├┤6             ├┤6             ├┤6             ├
              │              ││              ││              ││              │
    q_7: ─────┤7             ├┤7             ├┤7             ├┤7             ├
              └──────────────┘└──────────────┘└──────────────┘└──────────────┘

in which each Evolution layer is a list of Pauli strings with the same weight,
E.g.,

q_0: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

q_1: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

q_2: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

q_3: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     ┌─────────────────┐                                                 ┌──────────────────┐       ┌───┐
q_4: ┤ U(π/2,-π/2,π/2) ├──■───────────────────────────────────■──────────┤ U(-π/2,-π/2,π/2) ├───────┤ H ├─────────■────────────────────────────■──
     ├─────────────────┤┌─┴─┐                               ┌─┴─┐        ├──────────────────┤┌──────┴───┴──────┐┌─┴─┐                        ┌─┴─┐
q_5: ┤ U(π/2,-π/2,π/2) ├┤ X ├──■──────────────────■─────────┤ X ├────────┤ U(-π/2,-π/2,π/2) ├┤ U(π/2,-π/2,π/2) ├┤ X ├──■──────────────────■──┤ X ├
     ├─────────────────┤└───┘┌─┴─┐┌────────────┐┌─┴─┐┌──────┴───┴───────┐└──────┬───┬───────┘└─────────────────┘└───┘┌─┴─┐┌────────────┐┌─┴─┐├───┤
q_6: ┤ U(π/2,-π/2,π/2) ├─────┤ X ├┤ P(-0.0001) ├┤ X ├┤ U(-π/2,-π/2,π/2) ├───────┤ H ├────────────────────────────────┤ X ├┤ P(-0.0001) ├┤ X ├┤ H ├
     └─────────────────┘     └───┘└────────────┘└───┘└──────────────────┘       └───┘                                └───┘└────────────┘└───┘└───┘
q_7: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


In the standard UCCSD QASM files (from tket_benchmarking https://github.com/CQCL/tket_benchmarking/tree/master/active/chem_qasm),
gates are:
    - U3(π,0,π): X gate (only occurs in the first layer on some qubits)
    - U3(π/2,0,π): Hadamard gate (X = H Z H)
    - U3(-π/2,-π/2,π/2): "S H S" gate, (Y = S H S Z SDG H SDG)
    - U3(π/2,-π/2,π/2): "SDG H SDG" gate
    - CX: CNOT gate
    - U1(θ): Rz(θ) gate
"""

import sys

sys.path.append('..')

import os
import json
import numpy as np
import networkx as nx
from math import pi
from functools import reduce
from operator import add
from natsort import natsorted
from phoenix import Circuit, gates
from phoenix.utils import passes


class PhasePolynomial:
    def __init__(self, n, coeff, qubits):
        self.num_qubits = n
        # self.qubits = sorted(qubits)
        self.qubits = list(qubits)
        self.coeff = coeff
        self.transformations = []

    def __repr__(self):
        paulis = ['I' if i not in self.qubits else 'Z' for i in range(self.num_qubits)]
        for qubit, pauli in self.transformations:
            paulis[qubit] = pauli
        return ''.join(paulis)

    def add_rotation_qubit(self, qubit):
        if qubit not in self.qubits:
            self.qubits.append(qubit)
            # self.qubits.sort()

    def add_transformation(self, qubit, pauli):
        assert pauli in ['X', 'Y']
        if (qubit, pauli) not in self.transformations:
            self.transformations.append((qubit, pauli))


qasm_dir = './chem_qasm/'
json_dir = './chem_json/'

fnames = natsorted([fname for fname in os.listdir(qasm_dir) if fname.endswith('.qasm')])

for fname in fnames:
    qasm_fname = os.path.join(qasm_dir, fname)
    json_fname = os.path.join(json_dir, fname.replace('.qasm', '.json'))

    if os.path.exists(json_fname):
        continue

    circ = Circuit.from_qasm(fname=qasm_fname)

    print('converting {} to {} ...'.format(qasm_fname, json_fname))

    dag = circ.to_dag('networkx')

    # pop the first layer of X gates
    x_gates = [g for g in dag if isinstance(g, gates.U3) and np.allclose(g.angles, [pi, 0, pi])]
    front_x_on = [g.tq for g in x_gates]
    for g in x_gates:
        dag.remove_node(g)

    # contract CNOTs & Rz to Z...Z(θ)
    while rz_gates := [node for node in dag if isinstance(node, gates.U1)]:
        rz = rz_gates[0]
        phase_poly = PhasePolynomial(n=circ.num_qubits, coeff=rz.angle / 2, qubits=rz.tqs)
        dag = nx.contracted_nodes(dag, phase_poly, rz, self_loops=False)

        # the CNOT to be contracted into phase_poly is the one whose "tq" is in the qubits of phase_poly
        # moreover, herein only consider the simplest structure of CNOT tree (V-shape)
        while pred_cnots := [g for g in list(dag.predecessors(phase_poly)) if
                             isinstance(g, gates.XGate) and len(g.qregs) == 2 and
                             g.tq == phase_poly.qubits[-1] and g.cq not in phase_poly.qubits]:
            succ_cnots = [g for g in list(dag.successors(phase_poly)) if isinstance(g, gates.XGate) and
                          len(g.qregs) == 2 and g.tq == phase_poly.qubits[-1] and g.cq not in phase_poly.qubits]
            pred_cx = pred_cnots[0]
            succ_cx = succ_cnots[0]
            assert pred_cx.qregs == succ_cx.qregs, "phase_poly({}) pred_cx.qregs = {}, succ_cx.qregs = {}".format(
                phase_poly.qubits, pred_cx.qregs, succ_cx.qregs)
            phase_poly.add_rotation_qubit(pred_cx.tq)
            phase_poly.add_rotation_qubit(pred_cx.cq)
            dag = nx.contracted_nodes(dag, phase_poly, pred_cx, self_loops=False)
            dag = nx.contracted_nodes(dag, phase_poly, succ_cx, self_loops=False)

    # identify Pauli transformation (H, S-H-S, ...)
    nx.set_node_attributes(dag, False, 'resolved')
    while phase_polies := [node for node in dag if
                           isinstance(node, PhasePolynomial) and not dag.nodes[node]['resolved']]:
        phase_poly = phase_polies[0]
        dag.nodes[phase_poly]['resolved'] = True
        # find the neighbors of phase_poly
        while pred_u3s := [g for g in list(dag.predecessors(phase_poly)) if isinstance(g, gates.U3)]:
            pred_u3 = pred_u3s[0]
            succ_u3 = [g for g in list(dag.successors(phase_poly)) if isinstance(g, gates.U3) and g.tq == pred_u3.tq][0]
            # print(pred_u3, succ_u3)
            if np.allclose(pred_u3.angles, [pi / 2, 0, pi]) and np.allclose(succ_u3.angles, [pi / 2, 0, pi]):
                # H Z H --> X
                phase_poly.add_transformation(pred_u3.tq, 'X')
                dag = nx.contracted_nodes(dag, phase_poly, pred_u3, self_loops=False)
                dag = nx.contracted_nodes(dag, phase_poly, succ_u3, self_loops=False)
            elif np.allclose(pred_u3.angles, [pi / 2, -pi / 2, pi / 2]) and np.allclose(succ_u3.angles,
                                                                                        [-pi / 2, -pi / 2, pi / 2]):
                # S H S Z SDG H SDG --> Y
                phase_poly.add_transformation(pred_u3.tq, 'Y')
                dag = nx.contracted_nodes(dag, phase_poly, pred_u3, self_loops=False)
                dag = nx.contracted_nodes(dag, phase_poly, succ_u3, self_loops=False)
            else:
                raise ValueError(f"Unmatched gates: {pred_u3} and {succ_u3}")

    nodes = reduce(add, passes.dag_to_layers(dag))
    # print(nodes)

    paulis = [str(node) for node in nodes]
    coeffs = [node.coeff for node in nodes]

    data = {
        'num_qubits': circ.num_qubits,
        'front_x_on': front_x_on,
        'paulis': paulis,
        'coeffs': coeffs
    }

    with open(json_fname, 'w') as f:
        json.dump(data, f, indent=4)
