import numpy as np
from qiskit_optimization.applications import max_cut, tsp, vertex_cover, vehicle_routing, clique, exact_cover, graph_partition, number_partition, set_packing, stable_set, knapsack
from .mypauli import pauliString
from qiskit.quantum_info import Pauli
import ipdb

def get_operator(weight_matrix):
    """Generate Hamiltonian for the max-cut problem of a graph.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.

    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli(z_p, x_p)])
                shift -= 0.5 * weight_matrix[i, j]
    return pauli_list, shift

w1 = np.array([[0., 1., 1., 1.], [1., 0., 1., 0.], [1., 1., 0., 1.], [1., 0., 1., 0.]])
import random
def random_adjacency(n, seed=10):
    random.seed(seed)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            w[i,j] = random.randint(0, 1)
    for i in range(n):
        for j in range(0, i-1):
            w[i, j] = w[j, i]
    return w

import networkx as nx
def rand_reg(d, n, seed=None):
    return rand_regular(d, n, seed)
def rand_regular(d, n, seed=None):
    G = nx.random_regular_graph(d, n, seed)
    return nx.adjacency_matrix(G).todense()
def rand_er(n, p, seed=None):
    G = nx.erdos_renyi_graph(n, p, seed)
    return nx.adjacency_matrix(G).todense()

def gene_qaoa_oplist(w, model=max_cut):
    ipdb.set_trace()
    qubitOp, _ = get_operator(w)
    oplist = []
    # ps = qubitOp.to_dict()['paulis']
    ps = qubitOp
    for j in ps:
        oplist.append([pauliString(j['label'], real=j['coeff']['real'], imag=j['coeff']['imag'])])
    return oplist
    
def mc_oplist(w=w1):
    return gene_qaoa_oplist(w, max_cut)

def tsp_oplist(n, seed=None):
    w = tsp.random_tsp(n, seed=seed) # w contains a distance matrix, not adjacent matrix
    return gene_qaoa_oplist(w, tsp)
    
def vc_oplist(w=w1):
    return gene_qaoa_oplist(w, vertex_cover)
    
def ec_oplist(w=w1):
    return gene_qaoa_oplist(w, exact_cover)
    
def gp_oplist(w=w1):
    return gene_qaoa_oplist(w, graph_partition)
