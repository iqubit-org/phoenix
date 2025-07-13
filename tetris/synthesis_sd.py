# super-conducting synthesis
import numpy as np
from tetris.arch import *

max_size = 1000

# G is adjacency matrix
def simple_dfs_path(G, path, st, hops):
    n = G.shape[0]
    if hops == 1:
        return [st]
    for i in range(n):
        if i not in path and G[st, i] > 0:
            r = simple_dfs_path(G, path+[st], i, hops - 1)
            if len(r) == hops - 1:
                return [st] + r

def simple_initial(G, nq):
    ewt = -1
    n = G.shape[0]
    for i in range(n):
        for j in range(n):
            if G[i, j] >= ewt:
                eid = i
                ewt = G[i, j]
    return simple_dfs_path(G, [], eid, nq)

def ps2nodes(ps):
    r = []
    for i in range(len(ps)):
        if ps[i] != 'I':
            r.append(i)
    return r

def init_nodes(G, path):
    nq = len(path)
    n = G.shape[0]
    r = []
    r1 = []
    for i in range(n):
        node = pNode(i)
        r.append(node)
        for j in range(n):
            if G[i,j] > 0:
                node.add_adjacent(j)
        if i in path:
            r1.append(node)
    return r, r1


# def construct_graph(G):
#     n = G.shape[0]
#     graph = []
#     for i in range(n):
#         nd = pNode(i)
#         for j in range(n):
#             if G[i, j] == 1:
#                 nd.add_adjacent(j)
#     return graph

# local move only consider local best step, not necessarily global optimal
# nmt: matched nodes set
def local_move(graph, cost_matrix, nmt, pauli_map, src, target):
    p = []
    for i in src.adj:
        if i in nmt:
            if i == target:
                return True
            else:
                continue
        else:
            p.append((i, cost_matrix[i, target.idx]))

# find maximal matched branch
# move other nodes here
# pauli_map: logical qubits --> physical qubits
# block_cover: logical qubits range
def map_cover(graph, pauli_map, block_cover):
    max_cp = None
    max_l = -1
    for i in block_cover:
        r = max_dfs_path(graph, block_cover, [], graph[pauli_map[i]])
        if len(r) > max_l:
            max_l = len(r)
            max_cp = r
    # max_cp should not be None, then I got the diameter of the cover.
    # be careful with the SWAP insertion
    if len(max_cp) == 1: # means all nodes are isolated
        pass
    pass

# always assume all pauli qubits are on a connected path
# input: G, 
# nodes mapping of pauli qubits PN, Pauli string blocks
# a trivial version: 
# assume no holes in qubits cover.

# from qiskit import QuantumCircuit

# def simple_synthesis(G, pauli_map, pauli_layers):
#     for i in pauli_layers:
#         for j in i: # j is a pauli string array
#             # check maximum qubits cover
#             block_cover = []
#             for k in j:
#                 for l in psn_nodes(k):
#                     if l not in block_cover:
#                         block_cover.append(l)
#             block_cover = sorted(block_cover)
#             # block mapping, perfer local CNOT
#             map_cover(G, pauli_map, block_cover)

# class 

def compute_block_cover(pauli_block):
    blo_cov = []
    for i in pauli_block:
        for l in ps2nodes(i.ps):
            if l not in blo_cov:
                blo_cov.append(l)
    return blo_cov

def compute_block_interior(pauli_block):
    blo_cov = ps2nodes(pauli_block[0].ps)
    for i in pauli_block[1:]:
        blo_cov = list(set(blo_cov) & set(ps2nodes(i.ps)))
    return blo_cov

def max_dfs_path(graph, cover, start, path=[]):
    max_cp = []
    max_l = -1
    for i in start.adj:
        if i not in cover:
            continue
        if i in path:
            continue
        else:
            r = max_dfs_path(graph, cover, graph[i], path=path + [start.idx])
            if len(r) > max_l:
                max_l = len(r)
                max_cp = r
    if max_l == -1:
        return [start.idx]
    else:
        return [start.idx] + max_cp

def logical_list_physical(pauli_map, l):
    return [pauli_map[i] for i in l]

# def physical_list_logical(l):
    # return [i.lqb for i in l]
def physical_list_logical(graph, l):
    return [graph[i].lqb for i in l]

def swap_nodes(pauli_map, a, b):
    t = a.lqb
    a.lqb = b.lqb
    b.lqb = t
    if a.lqb != None:
        pauli_map[a.lqb] = a.idx
    if b.lqb != None:
        pauli_map[b.lqb] = b.idx

# here cover are physical qubits from logical cover
def max_dfs_tree(graph, cover, start, path=[]):
    max_cp = []
    for i in start.adj:
        if i not in cover:
            continue
        if i in path:
            continue
        else:
            r = max_dfs_tree(graph, cover, graph[i], path=path + max_cp + [start.idx])
            max_cp += r
    return [start.idx] + max_cp

def dummy_qubit_mapping(graph, nq):
    for i in range(nq):
        graph[i].lqb = i
    return list(range(nq))

def find_short_node(graph, pauli_map, nc, dp):
    minid0 = -1
    minid1 = -1
    mindist = max_size
    for i in nc:
        for j in dp:
            d = graph.C[pauli_map[i], j]
            if d < mindist:
                mindist = d
                minid0 = i
                minid1 = graph[j].lqb
    return minid0, minid1

def add_pauli_map(graph, pauli_map):
    for i in range(len(pauli_map)):
        graph[pauli_map[i]].lqb = i

def connect_node(graph, pauli_map, pid0, pid1, ins):
    minid = -1
    mindist = max_size
    for i in graph[pid0].adj:
        if graph.C[i, pid1] < mindist:
            minid = i
            mindist = graph.C[i, pid1]
    if minid == pid1:
        return
    else:
        ins.append(['swap',(pid0, minid)])
        swap_nodes(pauli_map, graph[pid0], graph[minid])
        connect_node(graph, pauli_map, minid, pid1, ins)

def try_connect_node_1(graph, pid0, pid1, ins):
    minid = -1
    mindist = max_size
    cpid = pid0
    while minid != pid1:
        for i in graph[cpid].adj:
            if graph.C[i, pid1] < mindist:
                minid = i
                mindist = graph.C[i, pid1]
        if minid == pid1:
            break
        else:
            ins.append(['swap',(pid0, minid)])
            cpid = minid

def try_connect_node_2(graph, pauli_map, pid0, pid1, ins, xlist):
    minid = -1
    mindist = max_size
    cpid = pid0
    while minid != pid1:
        if cpid in xlist:
            return -1
        for i in graph[cpid].adj:
            if graph.C[i, pid1] < mindist:
                minid = i
                mindist = graph.C[i, pid1]
        if minid in xlist:
            return -1
        if minid == pid1:
            break
        else:
            ins.append(['swap',(pid0, minid)])
            swap_nodes(pauli_map, graph[pid0], graph[minid])
            cpid = minid
    return 0