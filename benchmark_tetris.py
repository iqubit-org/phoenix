import sys, time
import sys
import mypauli
sys.path.append('../core/')
sys.path.append('chem_json')
import argparse
from utils.parallel_bl import *
import qiskit
from qiskit import QuantumCircuit, transpile
import synthesis_SC
import synthesis_FT
from tools import *
from arch import *
import time, sys, os
from t_arch import *
import pdb
import random
from utils.synthesis_broccoli import synthesis
from utils.synthesis_max_cancel import synthesis_max_cancel
from utils.synthesis_lookahead import synthesis_lookahead
import pickle
import json
import numpy as np
from itertools import combinations, product
from copy import deepcopy
from typing import List, Tuple

random.seed(1926)

BENCHMARK_DPATH = 'chem_json'
HAMLIB_DPATH = ['hamlib100json/binaryoptimization', 'hamlib100json/chemistry', 'hamlib100json/condensedmatter', 'hamlib100json/discreteoptimization']

def group_paulis(paulis: List[str]):

    assert len(paulis) == len(np.unique(paulis)), 'Pauli strings must be unique'

    nontrivial = [tuple(np.where(np.array(list(pauli)) != 'I')[0]) for pauli in paulis]
    groups = {}
    for idx, pauli in zip(nontrivial, paulis):
        if idx not in groups:
            groups[idx] = [pauli]
        else:
            groups[idx].append(pauli)

    # sort "groups" according to the length of the keys and keys themselves
    groups = dict(sorted(groups.items(), key=lambda x: (-len(x[0]), x[0])))

    # reorder items to reduce overall length when organizing as circuit
    groups_on_length = {}  # {length: {idx: [paulis]}}
    for idx, paulis in groups.items():
        length = len(idx)
        if length not in groups_on_length:
            groups_on_length[length] = {idx: paulis}
        else:
            groups_on_length[length][idx] = paulis

    def least_overlap(indices, existing_indices):
        overlaps = []
        for idx in indices:
            overlap = 0
            for eidx in existing_indices:
                overlap += len(set(idx) & set(eidx))
            overlaps.append(overlap)
        return indices[np.argmin(overlaps)]

    groups.clear()
    for equal_len_groups in groups_on_length.values():
        selected_indices = []
        while equal_len_groups:
            idx = least_overlap(list(equal_len_groups.keys()), selected_indices)
            selected_indices.append(idx)
            groups[idx] = equal_len_groups.pop(idx)

    return groups

def json_load(filename):
    with open(filename, 'rb') as f:
        obj = json.load(f)
    return obj

def load_oplist(filename):
    data = json_load(filename)
    parr, coeff = data['paulis'], data['coeffs']
    # oplist = [mypauli.pauliString(ps=parr, coeff = coeff)]
    oplist = []
    for i in range(len(parr)):
        oplist.append([mypauli.pauliString(ps=parr[i], coeff = coeff[i])])
    return oplist

# oplist = load_oplist("chem_json/CH2_cmplt_BK_sto3g.json")

def tetris_benchmark(oplist):
    print("----------------tetris_benchmark pass------------------")
    # lnq = len(oplist[0][0])
    coup = load_coupling_map('manhattan')
    a2 = oplist
    qc, metrics = synthesis_lookahead(a2, arch='manhattan', use_bridge=False, swap_coefficient=3, k=10)
    pnq = qc.num_qubits
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)

    metrics.update({'CNOT': cnots,
                    'Single': singles,
                    'Total': cnots+singles,
                    'Depth': depth,
                    # 'qasm' : qiskit.qasm3.dump(qc2),
                    # 'latency1' : latency1,
                    # 'latency2' : latency2
                })
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return {'CNOT': cnots,
                    'Single': singles,
                    'Total': cnots+singles,
                    'Depth': depth,
                    # 'qasm' : qiskit.qasm3.dump(qc2),
                    # 'latency1' : latency1,
                    # 'latency2' : latency2
                }


def PH_benchmark(oplist):
    print("-----------------PH pass------------------")
    lnq = len(oplist[0][0])
    coup = load_coupling_map('manhattan')
    a2 = gate_count_oriented_scheduling(oplist)
    qc, total_swaps, total_cx = synthesis_SC.block_opt_SC(a2, arch='manhattan')
    pnq = qc.num_qubits
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)

    return {
        'n_qubits': lnq,
        'PH_swap_count': total_swaps,
        'PH_cx_count': total_cx,
        'CNOT': cnots,
        'Single': singles,
        'Total': cnots+singles,
        'Depth': depth,
        # 'qasm' : qc2.qasm(),
        # 'latency1' : latency1,
        # 'latency2' : latency2
    }

# for json_fname in [fname for fname in os.listdir(BENCHMARK_DPATH) if fname.endswith('.json')]:

## ham100 json file
# for each_dir in range(len(HAMLIB_DPATH)):
#     # print(fname)

#     for json_fname in [fname for fname in os.listdir(HAMLIB_DPATH[each_dir]) if fname.endswith('.json')]:
#         with open(os.path.join(HAMLIB_DPATH[each_dir], json_fname), 'r') as f:
#             data = json.load(f)

#         parr, coeff = data['paulis'], data['coeffs']
#         qubit_count = len(parr[0])
#         if 'I' * qubit_count in parr:
#             parr.remove('I' * qubit_count)

#         if qubit_count <= 65:
            
#             grouped = group_paulis(parr)
#             new_parr = []
#             for each in grouped.values():
#                 oplist = []
#                 for i in range(len(each)):
#                     oplist.append(mypauli.pauliString(ps=each[i]))
#                 new_parr.append(oplist)
            
#             # tetris_output = tetris_benchmark(new_parr)
#             # out_file = open('ham100_tetris_result/tetris_' + json_fname, "w")
#             # json.dump(tetris_output, out_file)

#             PH_output = PH_benchmark(new_parr)
#             out_file = open('ham100_PH_result/PH_' + json_fname, "w")
#             json.dump(PH_output, out_file)

    # "-------PH--------------"
    # parr, coeff = data['paulis'], data['coeffs']
    # oplist = []   
    # for i in range(len(parr)):
    #     oplist.append([mypauli.pauliString(ps=parr[i], coeff = coeff[i])])
    # PH_output = PH_benchmark(oplist)
    # out_file = open('uccsd_PH_result/PH_' + json_fname, "w")
    # json.dump(PH_output, out_file)


for json_fname in [fname for fname in os.listdir(BENCHMARK_DPATH) if fname.endswith('.json')]:
    with open(os.path.join(BENCHMARK_DPATH, json_fname), 'r') as f:
        data = json.load(f)

    parr, coeff = data['paulis'], data['coeffs']
    qubit_count = len(parr[0])
    if 'I' * qubit_count in parr:
        parr.remove('I' * qubit_count)

    
    grouped = group_paulis(parr)
    new_parr = []
    for each in grouped.values():
        oplist = []
        for i in range(len(each)):
            oplist.append(mypauli.pauliString(ps=each[i]))
        new_parr.append(oplist)

    PH_output = PH_benchmark(new_parr)
    out_file = open('uccsd_PH_result/PH_' + json_fname, "w")
    json.dump(PH_output, out_file)



##### test oplist #####
# def gene_dot_1d(w, seed=12):
#     random.seed(seed)
#     nq = w + 1
#     oplist = []
#     for i in range(nq - 1):
#         interaction = random.choice(['ZZ', 'XX', 'YY'])
#         ps = i*'I' + interaction + (nq-2-i)*'I'
#         oplist.append([mypauli.pauliString(ps, coeff=1.0)])
#     return oplist
# print(gene_dot_1d(4))