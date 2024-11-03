from arch import *
from functools import partial
import numpy as np
from qiskit import QuantumCircuit, transpile
from synthesis_FT import assign_time_parameter
from synthesis_SC import block_opt_SC

def compute_neighbor(G):
    pnq = len(G)
    r = [[] for i in range(pnq)]
    for i in G.data:
        r[i.idx].append(i.adj)
        r1 = []
        for j in i.adj:
            r1 += G[j].adj
        r1 = list(set(r1) - {i} - set(i.adj))
        r[i.idx].append(r1)
        nr1 = len(i.adj)
        nr2 = len(r1)
        r[i.idx].append(nr1 + nr2)
    for i in range(pnq):
        r[i][0] = sorted(r[i][0], key=lambda x: -r[x][2])
        r[i][1] = sorted(r[i][1], key=lambda x: -r[x][2])
    return r
        
def qaim_place(graph, pauli_layers):
    lnq = len(pauli_layers[0][0][0])
    pnq = len(graph)
    parr = [i[0][0] for i in pauli_layers]
    count_occur = [0 for i in range(lnq)]
    l_neighbor = [[] for i in range(lnq)]
    l_qubit_order = list(range(lnq))
    for i in parr:
        j0 = i.ps.find('Z')
        j1 = i.ps.find('Z', j0+1)
        l_neighbor[j0].append(j1)
        l_neighbor[j1].append(j0)
        count_occur[j0] += 1
        count_occur[j1] += 1
    for i in range(lnq):
        l_neighbor[i] = list(set(l_neighbor[i]))
    l_qubit_order = sorted(l_qubit_order, key=lambda x: -count_occur[x])
    cr = compute_neighbor(graph)
    p_qubit_order = list(range(pnq))
    p_qubit_order = sorted(p_qubit_order, key=lambda x: -cr[x][2])
    graph[p_qubit_order[0]].lqb = l_qubit_order[0]
    m = [-1 for i in range(lnq)]
    m[l_qubit_order[0]] = p_qubit_order[0]
    l_qubit_mapped = [l_qubit_order[0]]
    p_qubit_mapped = [p_qubit_order[0]]
    p_qubit_order = p_qubit_order[1:]
    i = 1
    while i < lnq:
        ist = list(set(l_neighbor[l_qubit_order[i]]).intersection(l_qubit_mapped))
        if ist == []:
            m[l_qubit_order[i]] = p_qubit_order[0]
            p_qubit_mapped.append(p_qubit_order[0])
            p_qubit_order = p_qubit_order[1:]
            l_qubit_mapped.append(l_qubit_order[i])
        else:
            pid = -1
            pdist = 10000
            slots = [] # physical slots
            for j in ist:
                slots += list(set(cr[m[j]][0]+cr[m[j]][1]) - set(p_qubit_mapped))
            slots = list(set(slots))
            if slots != []:
                for j in slots:
                    tdist = 0
                    for k in ist:
                        tdist += graph.C[m[k], j]
                    if tdist < pdist:
                        pdist = tdist
                        pid = j
            else:
                pid = p_qubit_order[0]
            m[l_qubit_order[i]] = pid
            p_qubit_mapped.append(pid)
            p_qubit_order.remove(pid)
            l_qubit_mapped.append(l_qubit_order[i])
        i += 1
    return m
        

def swap_nodes(pauli_map, a, b):
    t = a.lqb
    a.lqb = b.lqb
    b.lqb = t
    if a.lqb != None:
        pauli_map[a.lqb] = a.idx
    if b.lqb != None:
        pauli_map[b.lqb] = b.idx
        
def dummy_qubit_mapping(graph, psl):
    nq = len(psl[0][0][0])
    #a = [1,2,3,4,5,6,8,9,10,11,12,13]
    m = list(range(nq))
    return m
    
def dummy_place(graph, psl):
    nq = len(psl[0][0][0])
    #a = [1,2,3,4,5,6,8,9,10,11,12,13]
    m = list(range(nq))
    return m
        
def ps2nodes(ps):
    r = []
    for i in range(len(ps)):
        if ps[i] != 'I':
            r.append(i)
    return r
    
def add_pauli_map(graph, pauli_map):
    for i in range(len(pauli_map)):
        graph[pauli_map[i]].lqb = i

def synthesis_initial1(pauli_layers, pauli_map=None, graph=None, qc=None, arch='manhattan'):
    # lnq , logical qubits number
    if graph == None:
        G, C = load_graph(arch, dist_comp=True, len_func=lambda x:1) # G is adj, C is dist
        graph = pGraph(G, C)
    if pauli_map == None:
        pauli_map = qaim_place(graph, pauli_layers) # dummy_qubit_mapping(graph, lnq)
    add_pauli_map(graph, pauli_map)
    # print(pauli_map)
    pnq = len(graph) # physical qubits
    if qc == None:
        qc = QuantumCircuit(pnq)
    return pauli_map, graph, qc

    
def synthesis_initial2(pauli_layers, pauli_map=None, graph=None, qc=None, arch='manhattan'):
    # lnq , logical qubits number
    if graph == None:
        G, C = load_graph(arch, dist_comp=True, len_func=lambda x:1) # G is adj, C is dist
        graph = pGraph(G, C)
    if pauli_map == None:
        pauli_map = dummy_place(graph, pauli_layers)
    add_pauli_map(graph, pauli_map)
    # print(pauli_map)
    pnq = len(graph) # physical qubits
    if qc == None:
        qc = QuantumCircuit(pnq)
    return pauli_map, graph, qc

    
def synth_qaoa1(pauli_layers, pauli_map=None, graph=None, qc=None, arch='manhattan', gamma=0.5, beta=0.5):
    lnq = len(pauli_layers[0][0][0])
    assign_time_parameter(pauli_layers, gamma)
    pauli_map, graph, qc = synthesis_initial1(pauli_layers, pauli_map, graph, qc, arch)
    qc = block_opt_SC(pauli_layers, graph=graph, pauli_map=pauli_map, qc=qc, arch=arch)
    for i in range(lnq):
        qc.rx(beta, pauli_map[i])
    return qc

def qiskit_synthesis(ps_layers, graph=None, qc=None, arch='manhattan', pauli_map=None, gamma=0.5, beta=0.5):
    from qiskit.aqua.operators.legacy import evolution_instruction
    from qiskit.quantum_info import Pauli
    lnq = len(ps_layers[0][0][0])
    pauli_map, graph, qc = synthesis_initial1(ps_layers, pauli_map, graph, qc, arch)
    for i in pauli_map:
        qc.h(i)
    for i in ps_layers:
        for j in i:
            for k in j:
                ns = ps2nodes(k.ps)
                if len(ns) != 2:
                    continue
                qc.cx(pauli_map[ns[0]], pauli_map[ns[1]])
                qc.rz(gamma, pauli_map[ns[1]])
                qc.cx(pauli_map[ns[0]], pauli_map[ns[1]]) # I only output the logical embedding in the physical device, I will not handle connectivity
                # one problem: why use evolution instruction will cause fatten output
                # for i in ps_layers:
                # for j in i:
                    # for k in j:
                        # qc.append(evolution_instruction([[0.001, Pauli.from_label(k.ps[::-1])]], 1, 1), [qc.qubits[i] for i in pauli_map[::1]])
    for i in pauli_map:
        qc.rx(beta, i)
    # basis_gates=['u3', 'swap', 'cx']
    return qc