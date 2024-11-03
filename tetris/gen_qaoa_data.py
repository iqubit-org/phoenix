from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevSuperFastMapper, BravyiKitaevMapper
from benchmark.mypauli import pauliString

import pickle
import ast
import pdb
import networkx as nx
import matplotlib.pyplot as plt


n_vertices = [15, 16, 17, 18, 19, 20]
for n_v in n_vertices:
    # Generate a power-law cluster graph
    # n: number of nodes
    # m: number of random edges to add for each new node
    # p: probability of adding a triangle after adding a random edge
    # G = nx.powerlaw_cluster_graph(n=n_v, m=4, p=0.05)
    G = nx.erdos_renyi_graph(n=n_v, p=0.05)

    pauli_blocks = []
    for (u, v) in list(G.edges()):
        ps = ['I' for i in range(n_v)]
        ps[u] = 'Z'
        ps[v] = 'Z'
        block = [pauliString(''.join(ps))]
        pauli_blocks.append(block)
    with open('data/qaoa/erdos_renyi_graph/{0}.pickle'.format(n_v), 'wb') as f:
        pickle.dump(pauli_blocks, f)
