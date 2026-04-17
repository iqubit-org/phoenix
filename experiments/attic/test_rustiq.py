import json
from rustiq import pauli_network_synthesis, Metric
from rustiq.utils import entangling_count, entangling_depth, convert_circuit

import sys


def rustiq_pass_rcount(paulis):
    # circ = pauli_network_synthesis(paulis, Metric.COUNT) # preserve_order is True by default
    circ = pauli_network_synthesis(paulis, Metric("count"))  # preserve_order is True by default

    # because the returned circuit is a Clifford circuit on one side, we need to double the entangling count and depth
    return entangling_count(circ) * 2, entangling_depth(circ) * 2


def rustiq_pass_rdepth(paulis):
    # circ = pauli_network_synthesis(paulis, Metric.DEPTH)
    circ = pauli_network_synthesis(paulis, Metric("depth"))
    return entangling_count(circ) * 2, entangling_depth(circ) * 2


with open(sys.argv[1], "r") as f:
    data = json.load(f)


num_2q_rcount, depth_2q_rcount = rustiq_pass_rcount(data["paulis"])
num_2q_rdepth, depth_2q_rdepth = rustiq_pass_rdepth(data["paulis"])


print(f"Rustiq pass (rcount): {num_2q_rcount} 2Q gates, {depth_2q_rcount} depth")
print(f"Rustiq pass (rdepth): {num_2q_rdepth} 2Q gates, {depth_2q_rdepth} depth")
