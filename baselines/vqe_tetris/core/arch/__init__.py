__all__ = ['load_graph', 'load_coupling_map', 'pNode', 'pGraph', 'dijkstra', 'max_dist', 'graph_from_coupling']
import os
import numpy as np
max_dist = 1000000
max_size = 1000000
package_directory = os.path.dirname(os.path.abspath(__file__))
import csv
from utils.hardware import load_sycamore_graph

class pNode:
    def __init__(self, idx):
        # self.child = []
        self.idx = idx
        self.adj = []
        self.lqb = None # logical qubit
        # self.parent = []
    def add_adjacent(self, idx):
        self.adj.append(idx)

class pGraph:
    def __init__(self, G, C):
        n = G.shape[0]
        self.leng = n
        self.G = G # adj matrix
        self.C = C # cost matrix
        self.data = []
        self.coupling_map = []
        for i in range(n):
            nd = pNode(i)
            for j in range(n):
                if G[i, j] == 1:
                    nd.add_adjacent(j)
                    self.coupling_map.append([i, j])
            self.data.append(nd)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return self.leng
    def copy(self):
        pgh = pGraph(self.G, self.C)
        for i in range(len(self.data)):
            pgh.data[i].lqb = self.data[i].lqb
        return pgh


def minDistance(dist, sptSet):
    minv = max_size
    min_index = -1
    n = len(dist)
    for v in range(n):
        if dist[v] < minv and sptSet[v] == False:
            minv = dist[v]
            min_index = v
    return min_index
 
def dijkstra(dist_matrix, src):
    n = dist_matrix.shape[0]
    dist = [dist_matrix[src, i] for i in range(n)]
    sptSet = [False]*n
    for cout in range(n):
        u = minDistance(dist, sptSet)
        if u == -1:
            break
        sptSet[u] = True
        for v in range(n):
            if dist_matrix[u, v] > 0 and sptSet[v] == False and \
                dist[v] > dist[u] + dist_matrix[u, v]:
                dist[v] = dist[u] + dist_matrix[u, v]
    for i in range(n):
        dist_matrix[src, i] = dist[i]
        dist_matrix[i, src] = dist[i]
        
def is_code_reduced(code):
    if code in ['melbourne', 'mahattan']:
        reduced = True
    else:
        reduced = False
    return reduced

def load_graph(code, dist_comp=False, len_func=lambda x:x):
    if code == 'sycamore':
        return load_sycamore_graph(dist_comp)
    reduced = is_code_reduced(code)
    pth = os.path.join(package_directory, 'data', 'ibmq_'+code+'_calibrations.csv')
    cgs = []
    n = 0
    with open(pth, 'r') as cf:
        g = csv.DictReader(cf, delimiter=',', quotechar='\"')
        for i in g:
            cxval = ""
            for j in i.keys():
                if j.find('CNOT') != -1:
                    cxval = i[j]
            n += 1
            if ';' in cxval:
                dc = ';'
            else:
                dc = ','
            for j in cxval.split(dc):
                cgs.append(j.strip())
    if reduced:
        n -= 1 # qubit n is not used
    G = np.zeros((n, n))
    C = np.ones((n, n))*max_dist
    for i in range(n):
        C[i, i] = 0
    for i in cgs:
        si1 = i.find('_')
        si2 = i.find(':')
        offset = 0
        if i[:2] == 'cx':
            offset = 2
        iq1 = int(i[offset:si1])
        iq2 = int(i[si1+1:si2])
        acc = float(i[si2+2:])*1000
        if (iq1 < n and iq2 < n) or not reduced:
            G[iq1, iq2] = 1
            C[iq1, iq2] = len_func(acc/1000)
    if dist_comp == True:
        for i in range(n):
            dijkstra(C, i)
    return G, C

def graph_from_coupling(coup, dist_comp=True):
    n = max([max(i) for i in coup]) + 1
    G = np.zeros((n, n))
    C = np.ones((n, n))*max_dist
    for i in range(n):
        C[i, i] = 0
    for i in coup:
        G[i[0], i[1]] = 1
        C[i[0], i[1]] = 1
    if dist_comp == True:
        for i in range(n):
            dijkstra(C, i)
    return pGraph(G, C)

def load_coupling_map(code):
    reduced = is_code_reduced(code)
    pth = os.path.join(package_directory, 'data', 'ibmq_'+code+'_calibrations.csv')
    cgs = []
    n = 0
    with open(pth, 'r') as cf:
        g = csv.DictReader(cf, delimiter=',', quotechar='\"')
        for i in g:
            cxval = ""
            for j in i.keys():
                if j.find('CNOT') != -1:
                    cxval = i[j]
            n += 1
            if ';' in cxval:
                dc = ';'
            else:
                dc = ','
            for j in cxval.split(dc):
                cgs.append(j.strip())
    coupling = []
    if reduced:
        n -= 1
    for i in cgs:
        si1 = i.find('_')
        si2 = i.find(':')
        offset = 0
        if i[:2] == 'cx':
            offset = 2
        iq1 = int(i[offset:si1])
        iq2 = int(i[si1+1:si2])
        if (iq1 < n and iq2 < n) or not reduced:
            coupling.append([iq1, iq2])
    return coupling