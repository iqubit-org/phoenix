"""
This script can be used to construct QAOA-MaxCut
circuit for graphs defined as a networkx graph object.
"""
import json
import networkx as nx

N = 12 #number of nodes
P = 0.8 #edge probability to generate erdos-renyi-random graph
SEED = 0 #random seed value
g = nx.erdos_renyi_graph(N,P,seed=SEED) #networkx graph object

#creating a dictionary from the graph object.
dic = {}
for i, edge in enumerate(g.edges()):
    dic['{}'.format(i+1)] = "{}".format(edge)
#saving the graph as a json file
with open('QAOA_circ.json','w') as fp:
    json.dump(dic,fp,indent=4)
