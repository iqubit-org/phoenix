from utils.parallel_bl import mutual, pXOR, parallel_order_size_bl, pDiff, lexi_order_bl
from benchmark.mypauli import pauliString

def assign_time_parameter(ps_layers, time_parameter):
    for i in ps_layers:
        for j in i:
            for k in range(len(j)):
                j[k].real += time_parameter
                j[k].coeff += time_parameter

from qiskit import QuantumCircuit, transpile

def simple_cancel(circ):
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    dag = circuit_to_dag(circ)
    # 1-qubit optimization
    for qi in circ.qubits:
        while True:
            k = list(dag.nodes_on_wire(qi))
            i = 0
            fl = False
            while i < len(k)-1:
                if k[i].name == 'h' and k[i+1].name == 'h':
                    dag.remove_op_node(k[i])
                    dag.remove_op_node(k[i+1])
                    fl = True
                    i += 2
                elif k[i].name == 'sdg' and k[i+1].name == 's':
                    dag.remove_op_node(k[i])
                    dag.remove_op_node(k[i+1])
                    fl = True
                    i += 2
                elif k[i].name == 's' and k[i+1].name == 'sdg':
                    dag.remove_op_node(k[i])
                    dag.remove_op_node(k[i+1])
                    fl = True
                    i += 2
                else:
                    i += 1
            if fl == False:
                break
    # from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import CXCancellation
    pm = CXCancellation()
    dag1 = pm.run(dag)
    return dag_to_circuit(dag1)

def simple_cancel_loop(circ):
    while True:
        t0 = sum(circ.count_ops().values())
        circ1 = simple_cancel(circ)
        t1 = sum(circ1.count_ops().values())
        # print('before:',t0,', after:', t1)
        if t1 == t0:
            return circ1
        else:
            circ = circ1

def count_gates(ol):
    c = ol.count_ops()
    t0 = sum(c.values())
    if 'cx' in c:
        t1 = c['cx']
    else:
        t1 = 0
    return t1, t0-t1

def comp_baseline(bl, ol):
    t0, t1 = count_gates(bl)
    t3, t4 = count_gates(ol)
    print('CNOT reduction:', -t3+t0, ", CNOT count:", t0)
    print('Single reduction:', -t4+t1, ", Single count:", t1)

def ps2nodes(ps):
    n = len(ps)
    nodes = []
    for i in range(n):
        if ps[i] != 'I':
            nodes.append(i)
    return nodes

# nq^2 complexity
def get_chain(src_leaf, graph, dph):
    for i in graph[src_leaf]:
        if i not in dph.keys():
            dph[i] = dph[src_leaf] - 1
        else:
            dph[i] = min(dph[src_leaf] - 1, dph[i])
        get_chain(i, graph, dph)

def complement_tree1(nodes, graph):
    if len(nodes) == 0:
        return [], [], -1
    elif len(nodes) == 1:
        return [], [], nodes[0]
    # find isolated nodes
    tn = [] # target node
    for i in nodes:
        tn += graph[i]
    sn = [] # source node
    for i in nodes:
        if len(graph[i]) > 0:
            sn.append(i)
    iln = list(set(nodes)-set(tn+sn))
    slf = list(set(sn)-set(tn)) # source leaf
    tlf = list(set(tn)-set(sn)) # target leaf
    if len(iln) > 0:
        root = iln[-1]
        if len(iln) > 1:
            slf.append(iln[0])
            for i in range(len(iln)-1):
                graph[iln[i]].append(iln[i+1])
        for i in tlf:
            graph[i].append(root)
    else:
        root = tlf[-1]
        for i in range(len(tlf)-1):
            graph[tlf[i]].append(root)
    dph = {}
    for i in slf:
        dph[i] = 0
        get_chain(i, graph, dph)
    cnotset = []
    for i in nodes:
        for j in graph[i]:
            cnotset.append((i, j))
    from functools import cmp_to_key
    def __key(e1):
        return dph[e1[0]]
    cnotset = sorted(cnotset, key=__key)
    root_depth = -dph[root]
    pch = [[] for i in range(root_depth+1)]
    for i in nodes:
        # print(root_depth+dph[i])
        pch[root_depth+dph[i]].append(i)
    return pch, cnotset, root

class pauli_tree:
    def __init__(self, graph, root, parameter):
        self.tree = graph
        self.root = root
        self.parameter = parameter

def mutual_pos(ps1, ps2):
    pos = []
    for i in range(len(ps1)):
        if ps1[i] == ps2[i] and ps1[i] != "I":
            pos.append(i)
    return pos

def reorder_layer(ps_layers):
    psl = []
    nq = len(ps_layers[0][0][0])
    for i in ps_layers:
        plb = i        
        while len(plb) > 0:
            plb1 = []
            plb2 = []
            pb = plb[0]
            plb2.append(pb)
            pso = 'I'*nq
            pso = pXOR(pso, pb.ps)
            for j in plb[1:]:
                if pDiff(j.ps, pso) == True:
                    plb2.append(j)
                    pso = pXOR(pso, j.ps)
                else:
                    plb1.append(j)
            plb = plb1
            psl.append(plb2)
    return psl

def reorder_layer1(ps_layers):
    psl = []
    nq = len(ps_layers[0][0][0])
    def _key(pb):
        s = 0
        ps = pb[0][0].ps
        for i in ps:
            s *= 4
            if i == 'I':
                s += 0
            elif i == 'X':
                s += 1
            elif i == 'Y':
                s += 2
            elif i == 'Z':
                s += 3
        return -s
    # ps_layers = sorted(ps_layers, key=_key)
    for i in ps_layers:
        plb = []
        for j in i:
            for k in j:
                plb.append([k])
        psl.append(plb)
    ps_layers = psl
    psl = []
    # print('l:', ps_layers)
    for i in ps_layers:
        plb = i
        while len(plb) > 0:
            plb1 = []
            plb2 = []
            pb = plb[0]
            plb2.append(pb)
            pso = 'I'*nq
            pso = pXOR(pso, pb[0].ps)
            for j in plb[1:]:
                if pDiff(j[0].ps, pso) == True:
                    plb2.append(j)
                    pso = pXOR(pso, j[0].ps)
                else:
                    plb1.append(j)
            plb = plb1
            psl.append(plb2)
    # print('l1:', psl)
    return psl

def max_chain(pt): # pt is ordered cx sequence
    pc = [[pt[0]]]
    for i in pt:
        fl = False
        for j in pc:
            if i[0] == j[-1][0] or i[0] == j[-1][1] or i[1] == j[-1][0] or i[1] == j[-1][1]:
                j.append(i)
                fl = True
        if fl == False:
            pc.append([i])
    return pc

def simple_seq_synthesis(ps_layers, time_parameter=1):
    assign_time_parameter(ps_layers, time_parameter)
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    for i in ps_layers:
        for j in i:
            for k in j:
                pos = []
                ps = k.ps
                for l in range(nq):
                    if ps[l] != "I":
                        pos.append(l)
                for l in range(nq):
                    if ps[l] == 'X':
                        qc.h(l)
                    elif ps[l] == 'Y':
                        qc.s(l)
                        qc.h(l)
                for l in range(len(pos)-1):
                    qc.cx(pos[l], pos[l+1])
                qc.rz(2*k.real, pos[-1])
                for l in range(len(pos)-1,0,-1):
                    qc.cx(pos[l-1], pos[l])
    return qc

def syn_pauli_string(qc, nq, ps0, cnotset, root, coeff):
    # if len(cnotset) > max(len(ps0)-ps0.count('I')-1,0):
    #     print(ps0)
    # qc.barrier()
    for k in range(nq):
        if ps0[k] == 'X':
            qc.h(k)
        if ps0[k] == 'Y':
            qc.s(k)
            qc.h(k)
    for k in reversed(cnotset):
        qc.cx(k[0], k[1])
    # qc.rz(2*i0[0].real, root)
    if root >= 0:
        qc.rz(2*coeff, root)
    for k in cnotset:
        qc.cx(k[0], k[1])
    for k in range(nq):
        if ps0[k] == 'X':
            qc.h(k)
        if ps0[k] == 'Y':                    
            qc.h(k)
            qc.sdg(k)

def find_consecutive(pos):
    if len(pos) == 0:
        return pos
    t = pos[0]
    chain = []
    pc = [t]
    for i in pos[1:]:
        if i == t + 1:
            pc.append(i)
        else:
            chain.append(pc)
            pc = [i]
        t = i
    chain.append(pc)
    return chain

def init_two_layer(qc, nq, pl0, pl1):
    for i0 in pl0:
        ps0 = i0[0].ps
        psg = [[] for i2 in range(nq)]
        for i1 in pl1:
            ps1 = i1[0].ps
            pos = mutual_pos(ps0, ps1)
            pcos = find_consecutive(pos)
            for i2 in pcos:
                for i3 in range(len(i2) - 1):
                    psg[i2[i3]].append(i2[i3+1])
            for i2 in range(len(pcos)-1):
                psg[pcos[i2][-1]].append(pcos[i2+1][-1])
            # ne = len(pos)
            # for i2 in range(ne - 1):
            #     psg[pos[i2]].append(pos[i2+1])
        psn = ps2nodes(ps0)
        pch, cnotset, root = complement_tree1(psn, psg)
        syn_pauli_string(qc, nq, ps0, cnotset, root, i0[0].real)
    ptc = []
    for i0 in pl1:
        ps0 = i0[0].ps
        psg = [[] for i2 in range(nq)]
        for i1 in pl0:
            ps1 = i1[0].ps
            pos = mutual_pos(ps0, ps1)
            pcos = find_consecutive(pos)
            for i2 in pcos:
                for i3 in range(len(i2) - 1):
                    psg[i2[i3]].append(i2[i3+1])
            for i2 in range(len(pcos)-1):
                psg[pcos[i2][-1]].append(pcos[i2+1][-1])
        psn = ps2nodes(ps0)
        pch, cnotset, root = complement_tree1(psn, psg)
        syn_pauli_string(qc, nq, ps0, cnotset, root, i0[0].real)
        ptc.append(pch)
    return ptc

def complement_tree2(nodes, link):
    # assert current graph is a chain
    if len(nodes) == 0:
        return [], -1
    elif len(nodes) == 1:
        return [], nodes[0]
    remain = []
    for i in nodes:
        if i not in link:
            remain.append(i)
    if link != []:
        r1 = []
        r2 = []
        for i in remain:
            if i > link[-1]:
                r2.append(i)
            else:
                r1.append(i)
    else:
        r1 = []
        r2 = remain
    pch = link + r2
    cnotset = []
    for i in range(len(pch)-1):
        cnotset.append((pch[i],pch[i+1]))
    root = pch[-1]
    if r1 != []:
        for i in range(len(r1)-1):
            cnotset.append((r1[i],r1[i+1]))
        cnotset.append((r1[-1], root))
    cnotset = cnotset[::-1]
    return cnotset, root

def complement_tree3(nodes, link):
    # assert current graph is a chain
    if len(nodes) == 0:
        return [], -1
    elif len(nodes) == 1:
        return [], nodes[0]
    remain = []
    for i in nodes:
        if i not in link:
            remain.append(i)
    if link != []:
        r1 = []
        r2 = []
        for i in remain:
            if i > link[-1]:
                r2.append(i)
            else:
                r1.append(i)
    else:
        r1 = []
        r2 = remain
    cnotset = []
    for i in range(len(link)-1):
        cnotset.append((link[i],link[i+1]))
    if r2 != []:
        root = r2[-1]
        if link != []:
            cnotset.append((link[-1], root))
    else:
        root = link[-1]
    for i in range(len(r2)-1):
        cnotset.append((r2[i],r2[i+1]))        
    if r1 != []:
        for i in range(len(r1)-1):
            cnotset.append((r1[i],r1[i+1]))
        cnotset.append((r1[-1], root))
    cnotset = cnotset[::-1]
    return cnotset, root

def simple_seq_synthesis1(ps_layers, time_parameter=1):
    assign_time_parameter(ps_layers, time_parameter)
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    psli = ps_layers[0]
    i = 1
    nl = len(ps_layers)
    while i < nl:
        cn_prev = 0
        pos_prev = mutual_pos(ps_layers[i][0][0].ps, psli[0][0].ps)
        cn_prev = len(pos_prev)
        cn_next = 0
        if i < nl - 1:
            pos_next = mutual_pos(ps_layers[i][0][0].ps, ps_layers[i+1][0][0].ps)
            cn_next = len(pos_next)
        if cn_prev >= cn_next:
            pos = pos_prev
        else:
            pos = pos_next
        psn = ps2nodes(ps_layers[i][0][0].ps)
        cnotset, root = complement_tree2(psn, pos)
        syn_pauli_string(qc, nq, ps_layers[i][0][0].ps, cnotset, root, ps_layers[i][0][0].real)
        for i1 in ps_layers[i][1:]:
            psn = ps2nodes(i1[0].ps)
            cnotset, root = complement_tree2(psn, [])
            syn_pauli_string(qc, nq, i1[0].ps, cnotset, root, i1[0].real)
        i += 1
    return qc

import numpy as np
# Assert the schedule is like:
# every block only has one Pauli String, no padding happens.
def construct_cost_matrix(ps_layers):
    ne = len(ps_layers)
    cost_matrix = np.zeros((ne, ne))
    for i in range(1, ne):
        pos = mutual_pos(ps_layers[i][0][0].ps, ps_layers[i-1][0][0].ps)
        ps = ps_layers[i][0][0].ps
        cost_matrix[i][i-1] = len(pos)
        cost_matrix[i-1][i] = len(pos)
    return cost_matrix

def construct_cost_matrix_line(ps_layers):
    ne = len(ps_layers)
    cost_matrix = [0 for i in range(ne)]
    for i in range(1, ne):
        pos = mutual_pos(ps_layers[i][0][0].ps, ps_layers[i-1][0][0].ps)
        cost_matrix[i - 1] = len(pos)
    return cost_matrix

def construct_cost_matrix1(ps_layers):
    ne = len(ps_layers)
    cost_matrix = np.zeros((ne, ne))
    for i in range(1, ne):
        pos = mutual_pos(ps_layers[i][0][0].ps, ps_layers[i-1][0][0].ps)
        ps = ps_layers[i][0][0].ps
        # sd = 0
        # for j in pos:
        #     if ps[j] == 'X' or ps[j] == 'Y':
        #         sd += 1
        cost_matrix[i][i-1] = max(len(pos) - 1, 0)
        cost_matrix[i-1][i] = max(len(pos) - 1, 0)
    return cost_matrix

def max_singlet_pairs(cost_matrix):
    ne = cost_matrix.shape[0]
    node_list = list(range(ne))
    edge = []
    while True:
        max_val = -1
        max_id = 0
        for i in range(len(node_list) - 1):
            if node_list[i+1] == node_list[i] + 1:
                if cost_matrix[node_list[i]][node_list[i+1]] > max_val:
                    max_id = i
                    max_val = cost_matrix[node_list[i]][node_list[i+1]]
        if max_val == -1:
            for i in node_list:
                edge.append((i,i))
            break
        else:
            edge.append((node_list[max_id], node_list[max_id]+1))
            node_list = node_list[:max_id]+node_list[max_id+2:]
    # sort edge
    def _key(e):
        return e[0]
    edge = sorted(edge, key=_key)
    return edge

def max_singlet_pairs_line(cost_matrix):
    ne = len(cost_matrix)
    node_list = list(range(ne))
    edge = []
    while True:
        max_val = -1
        max_id = 0
        for i in range(len(node_list) - 1):
            if node_list[i+1] == node_list[i] + 1:
                if cost_matrix[node_list[i]] > max_val:
                    max_id = i
                    max_val = cost_matrix[node_list[i]]
        if max_val == -1:
            for i in node_list:
                edge.append((i,i))
            break
        else:
            edge.append((node_list[max_id], node_list[max_id]+1))
            node_list = node_list[:max_id]+node_list[max_id+2:]
    # sort edge
    def _key(e):
        return e[0]
    edge = sorted(edge, key=_key)
    return edge

def max_match_synthesis(ps_layers, tree_compl):
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    cost_matrix = construct_cost_matrix_line(ps_layers)
    # find max pairs
    edge = max_singlet_pairs_line(cost_matrix)
    # print(edge)
    for i in edge:
        if i[0] == i[1]:
            psn = ps2nodes(ps_layers[i[0]][0][0].ps)
            cnotset, root = tree_compl(psn, [])
            syn_pauli_string(qc, nq, ps_layers[i[0]][0][0].ps, cnotset, root, ps_layers[i[0]][0][0].real)
            for j in ps_layers[i[0]][1:]:
                psn = ps2nodes(j[0].ps)
                cnotset, root = tree_compl(psn, [])
                syn_pauli_string(qc, nq, j[0].ps, cnotset, root, j[0].real)
        else:
            pos = mutual_pos(ps_layers[i[0]][0][0].ps, ps_layers[i[1]][0][0].ps)
            psn = ps2nodes(ps_layers[i[0]][0][0].ps)
            cnotset, root = tree_compl(psn, pos)
            syn_pauli_string(qc, nq, ps_layers[i[0]][0][0].ps, cnotset, root, ps_layers[i[0]][0][0].real)
            for j in ps_layers[i[0]][1:]:
                psn = ps2nodes(j[0].ps)
                cnotset, root = tree_compl(psn, [])
                syn_pauli_string(qc, nq, j[0].ps, cnotset, root, j[0].real)
            psn = ps2nodes(ps_layers[i[1]][0][0].ps)
            cnotset, root = tree_compl(psn, pos)
            syn_pauli_string(qc, nq, ps_layers[i[1]][0][0].ps, cnotset, root, ps_layers[i[1]][0][0].real)
            for j in ps_layers[i[1]][1:]:
                psn = ps2nodes(j[0].ps)
                cnotset, root = tree_compl(psn, [])
                syn_pauli_string(qc, nq, j[0].ps, cnotset, root, j[0].real)
    return qc

def block_opt_FT(ps_layers, time_parameter=1):
    assign_time_parameter(ps_layers, time_parameter)
    qc = max_match_synthesis(ps_layers, complement_tree2)
    return qc

def simple_seq_synthesis3(ps_layers, time_parameter=1):
    assign_time_parameter(ps_layers, time_parameter)
    qc = max_match_synthesis(ps_layers, complement_tree3)
    return qc

def prop_synthesis_single(ps_layers, time_parameter=1):
    assign_time_parameter(ps_layers, time_parameter)
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    nl = len(ps_layers)
    i = 0
    if nl >= 2:
        ptc = init_two_layer(qc, nq, ps_layers[0], ps_layers[1])
        i = 2
    psl_p = []
    while i < nl:
        # lookahead
        pl0 = ps_layers[i]
        psl_p = ps_layers[i-1]
        plp = []
        for i0 in pl0:
            ps0 = i0[0].ps
            for i1 in range(len(psl_p)):
                ps1 = psl_p[i1][0].ps
                pos = mutual_pos(ps0, ps1)
                psg = [[] for i2 in range(nq)]
                pcos = find_consecutive(pos)
                for i2 in pcos:
                    for i3 in range(len(i2) - 1):
                        psg[i2[i3]].append(i2[i3+1])
                for i2 in range(len(pcos)-1):
                    psg[pcos[i2][-1]].append(pcos[i2+1][-1])
            psn = ps2nodes(ps0)
            pch, cnotset, root = complement_tree1(psn, psg)
            syn_pauli_string(qc, nq, ps0, cnotset, root, i0[0].real)
        i += 1
    return qc

# assume no multi-padding
# assume singlet-block 
def simple_synthesis_single(ps_layers, time_parameter=1):
    assign_time_parameter(ps_layers, time_parameter)
    # sort inner block
    # sorted()
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    # psts = []
    nl = len(ps_layers)
    i = 0
    while i < nl:
        pl0 = ps_layers[i]
        if i == nl - 1:
            pl1 = [[pauliString('I'*nq)]]
        else:
            pl1 = ps_layers[i+1]
        init_two_layer(qc, nq, pl0, pl1)
        i += 2
    # qc = transpile(qc, optimization_level=1)
    return qc

# estimate CNOT
def cnot_estimate(ps_layers):
    c = 0
    for i in ps_layers:
        for j in i:
            for k in j:
                c += 2*max(len(k)-k.count('I')-1,0)
    return c


def uccsd_synthesis(ps_layers, time_parameter=1):    
    # ps_layers1 = reorder_layer1(ps_layers)
    print(ps_layers)
    qc = simple_synthesis_single(ps_layers, time_parameter=time_parameter)
    return qc

def uccsd_synthesis1(ps_layers, time_parameter=1):    
    ps_layers1 = reorder_layer1(ps_layers)
    # print(ps_layers1)
    # qc = simple_synthesis_single(ps_layers1, time_parameter=time_parameter)
    qc = prop_synthesis_single(ps_layers1, time_parameter=time_parameter)
    return qc
def uccsd_synthesis2(ps_layers, time_parameter=1):    
    ps_layers1 = reorder_layer1(ps_layers)
    # print(ps_layers1)
    # qc = simple_synthesis_single(ps_layers1, time_parameter=time_parameter)
    qc = simple_seq_synthesis1(ps_layers1, time_parameter=time_parameter)
    return qc

def singlet_mul_synthesis(ps_layers, time_parameter=1):
    ps_layers1 = reorder_layer1(ps_layers)
    qc = simple_synthesis_single(ps_layers1, time_parameter=time_parameter)
    return qc

def qiskit_synthesis(ps_layers, time_parameter=1):
    from qiskit.aqua.operators.legacy import evolution_instruction
    from qiskit.quantum_info import Pauli
    assign_time_parameter(ps_layers, time_parameter)
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    # psl = []
    for i in ps_layers:
        for j in i:
            for k in j:
                qc.append(evolution_instruction([[1, Pauli.from_label(k.ps[::-1])]], 1, 1), qc.qubits)
    return qc

def qiskit_synthesis1(ps_layers, time_parameter=1):
    from qiskit.aqua.operators.legacy import evolution_instruction
    from qiskit.quantum_info import Pauli
    assign_time_parameter(ps_layers, time_parameter)
    nq = len(ps_layers[0][0][0])
    qc = QuantumCircuit(nq)
    psl = []
    for i in ps_layers:
        for j in i:
            for k in j:
                psl.append([1, Pauli.from_label(k.ps[::-1])])
    qc.append(evolution_instruction(psl, 1, 1), qc.qubits)
    return qc

def print_qc(qc):
    c = qc.count_ops()
    t0 = sum(c.values())
    if 'cx' in c:
        t1 = c['cx']
    else:
        t1 = 0
    print('CNOT:', t1, ", Single:", t0-t1, ', Depth:', qc.depth())
