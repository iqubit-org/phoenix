"""
This script can be used/modified to create custom quantum hardware configuration file.
"""
import json
import random
import networkx as nx

#List of the single-qubit native gates supported by the hardware
single_q_gates = ['id','sx','x','u3']
#Two qubit native gate supported by the hardware,
#current compiler implementation assumes that the hardware
#supports only 1 two-qubit operation (cx)
two_q_gate = ['cx']

Q = 20
NEIGHBOR = 4
SEED = 0

coupling_graph = nx.random_regular_graph(NEIGHBOR,Q,seed=SEED)
dic = {}
dic['1Q'] = single_q_gates
dic['2Q'] = two_q_gate

for gate in single_q_gates:
    tdic = {}
    for q in range(Q):
        tdic['{}'.format(q)] = 1 #single-qubit gate success probability
    dic['{}'.format(gate)] = tdic

tdic = {}
for edge in coupling_graph.edges():
    tdic['({},{})'.format(edge[0],edge[1])] = random.uniform(0.96,0.99) #two-qubit gate success probability
dic['{}'.format(two_q_gate[0])] = tdic

with open('QC.json','w') as fp:
    json.dump(dic,fp,indent=4)
