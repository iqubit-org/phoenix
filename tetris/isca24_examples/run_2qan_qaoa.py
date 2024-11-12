import sys

sys.path.append('../..')


import numpy as np
import time

from py2qan import BenchArch
from py2qan import HeuristicMapper
from py2qan import QuRouter
# Import qiskit 
import qiskit
import os
import pickle as pkl
from qiskit import transpile, QuantumCircuit
# from arch import load_coupling_map
from tetris.utils.hardware import pGraph, load_coupling_map
import pdb
from collections import defaultdict
import re

idx = 0

def qs_compiler(qasm, coupling_map, qaoa=True, layers=1, trials=1, mapper='qiskit', bgate='rzz', params=None):

    global idx

    print(f'coupling is: {coupling_map}')
    qs_circ = None
    qs_swap = (0, 0) # the number of swaps in the format (#swaps,#swaps merged with circuit gate)
    qs_g2 = 0 # the number of two-qubit gates without decomposition
    # Perform qubit mapping, routing, and scheduling only, without gate decomposition
    for trial in range(trials):
        # Both QAP and Qiskit mappers output inital qubit maps randomly, 
        # one can run the mapper several times to achieve better compilation results
        # Initial qubit mapping 
        start = time.time()
        hmapper = HeuristicMapper(qasm, coupling_map=coupling_map)
        if mapper == 'qap':
            # The default mapper based on Quadratic Assignment Problem
            init_map, cost = hmapper.run_qap(num_iter=10, lst_len=20)
        elif mapper == 'qiskit':
            # The mapper in Qiskit
            init_map = hmapper.run_qiskit(max_iterations=1)
            init_map = {i:i for i in range(len(init_map.items()))}
        
        # init_map = {circuit qubit index:device qubit index}
        print('The initial qubit map is \n', init_map)

        # Routing and scheduling, takes init_map as input
        router = QuRouter(qasm, init_map=init_map, coupling_map=coupling_map)
        if qaoa:
            # For QAOA, different layers have different gate parameters
            
            #qs_circ0, swaps1 = router.run_qaoa(layers=layers, gammas=params[layers-1][:layers], betas=params[layers-1][layers:], msmt=True) 
            qs_circ0, swaps1 = router.run_qaoa(layers=layers, gammas=[0.05], betas=[0.05], msmt=True) 
            # print(qs_circ0)
        else:
            # For quantum simulation circuits, we assume each layer has the same time steps
            qs_circ0, swaps1 = router.run(layers=layers, msmt='True')
        qs_circ0 = transpile(qs_circ0, basis_gates=None, optimization_level=3)

        qc_tmp = transpile(qs_circ0, basis_gates=['u1', 'u2', 'u3', 'cx', 'reset', 'measure'], optimization_level=3)

        qc_tmp.qasm(filename='qaoa_{}.qasm'.format(idx))
        idx += 1


        #
        #
        # qs_circ0 = transpile(qs_circ0, basis_gates=['u1', 'u2', 'u3', 'cx', 'reset', 'measure'], optimization_level=3)
        # print(qs_circ0)
        # print(type(qs_circ0))
        #
        # return qs_circ0.num_nonlocal_gates(), qs_circ0.depth(lambda instr: instr.operation.num_qubits > 1)
        #




        # qs_circ0 is the routed circuit without gate decomposition
        # swaps1 is a tuple=(#swaps,#swaps merged with circuit gate)
        print("drawing qs_circ0")
        #print(f"depth0 is {qs_circ0.depth()}")
        #print(qs_circ0)
        print(qs_circ0.qasm())
        end = time.time()
        print("Total run time: ", end - start)
        # print(qs_circ0)

        c = qs_circ0.qasm().split("\n")
        #print(c)

        depth_dict = defaultdict(int)
        gate_count = 0
        pattern = re.compile(r"\[(\d+)\]") #using the regular expression to find all numbers in []
        swap_count = 0
        for line in c:
            if line[0:3] == 'rzz' or line[0:4] == 'swap':
                if line[0:3] == 'rzz':
                    gate_count += 2
                if line[0:4] == 'swap':
                    gate_count += 3
                    swap_count += 1
                qubits = pattern.findall(line)
                q1 = int(qubits[0])
                q2 = int(qubits[1])
                depth_dict[q1] = depth_dict[q2] = max(depth_dict[q1], depth_dict[q2]) + 3
                #print('rzz',q1,"-",q2, ' ', depth_dict[q1])
            if line[0:3] == 'dZZ':
                gate_count += 3
                swap_count += 1
                qubits = pattern.findall(line)
                q1 = int(qubits[0])
                q2 = int(qubits[1])
                depth_dict[q1] = depth_dict[q2] = max(depth_dict[q1], depth_dict[q2]) + 4
                #print('dzz',q1,"-",q2, ' ', depth_dict[q1])
        depth = max(depth_dict.values())

        

        
    return gate_count, swap_count, depth, end - start

def qiskit_decompose(circ, basis_gates=['id', 'rz', 'u3', 'u2', 'cx', 'reset'], bgate='cx'):
    # Perform gate decomposition and optimization into cx gate set
    # For decomposition into other gate sets, e.g., the SYC, sqrt iSWAP, iSWAP, 
    # one can use Google Cirq for decomposition or the NuOp (https://github.com/prakashmurali/NuOp) decomposer
    decom_g2 = 0
    decom_circ = transpile(circ, basis_gates=basis_gates, optimization_level=3)
    if bgate in decom_circ.count_ops():
        decom_g2 += decom_circ.count_ops()[bgate] 
    if 'unitary' in decom_circ.count_ops().keys():
        decom_g2 += decom_circ.count_ops()['unitary']
    return decom_circ, decom_g2

def get_qasm(gate_list, q_count):
    #gate_list = [(0,3),(2,4),(0,5),(0,6),(1,3)]
    qasm_string = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q["+ str(q_count) +"];\ncreg c["+ str(q_count) +"];\n"
    for i in range(q_count):
        qasm_string = qasm_string + 'h q['+str(i)+'];\n'
    for g in gate_list:
        qasm_string += "rzz(0.15) q[" + str(g[0]) + "],q[" + str(g[1]) + "];\n"
    return qasm_string


# def runner(coupling, lq, pq, d ):
def runner(file_name):
    # Benchmarks
    qaoa = True
    param = None
    # if coupling == 'ibm':
    #     # coupling_path = "ibm_coupling/ibm_rings_" + str(pq) + ".txt"
    #     coupling_path = "ibm_coupling/heavyhex_distance7.txt"

    # else:
    #     coupling_path = "sycamore_coupling/sycamore_" + str(pq) + ".txt"
    # coupling_path = "ibm_coupling/ibm_rings_39.txt"
    coup = load_coupling_map('manhattan')

    #read input file:
    # graph_path = "benchmarks/random_graph/" + str(lq) + "_0." + str(d) + "_0.txt"
    # graph_path = "myBench/random_graph" + str(lq) + "_" + str(d) + "_0.txt"
    # graph_path = "../myBench/" + file_name
    graph_path = file_name

    # graph_path = "myBench/"+file_name
    print(f'graph_path is {graph_path}')
    #coupling_path = "ibm_coupling/ibm_rings_72.txt"
    gate_list = []
    
    #open text file in read mode
    graph_file = open(graph_path, "r")

    gates = graph_file.read().split('\n')

    info = gates[0].split(' ')
    print(info)
    logical_q_count = int(info[0])
    logical_e_count = int(info[1])
    for g in gates[1:]:
        g = g.split(' ')
        if len(g) == 2:
            gate_list.append((int(g[0]), int(g[1])))
    

    if logical_e_count != len(gate_list):
        assert(False)
    #close file
    graph_file.close()
       
    c_qasm = get_qasm(gate_list,logical_q_count)
    print(c_qasm)
    
    # test_circ = qiskit.QuantumCircuit.from_qasm_str(c_qasm)

    # Device information
    # gate set, assume cx as the native two-qubit gate
    basis_gates = ['id', 'rz', 'u3', 'u2', 'cx', 'reset']
    # reading coupling
    
    grid_topology = BenchArch(c_qasm, coupling_map=coup).topology
    #grid_topology = BenchArch(c_qasm, lattice_xy=lattice_xy).topology
    coupling_map = [list(edge) for edge in list(grid_topology.edges)]
    coupling_map += [[edge[1], edge[0]] for edge in list(grid_topology.edges)]

    total_gate_count, swap_count, depth, total_time = qs_compiler(c_qasm, coupling_map, qaoa=qaoa, layers=1, trials=1, bgate='rzz', params=param)
    
    print(total_gate_count, depth)
    return (total_gate_count, swap_count, depth, total_time)
# def test():
#     gate_list = [(0,3),(2,4),(0,5),(0,6),(1,3)]
#     logical_q_count = 7
#     # Benchmarks
#     qaoa = True
#     param = None
       
#     c_qasm = get_qasm(gate_list,logical_q_count)
#     print(c_qasm)
    
#     test_circ = qiskit.QuantumCircuit.from_qasm_str(c_qasm)

#     # Device information
#     # gate set, assume cx as the native two-qubit gate
#     basis_gates = ['id', 'rz', 'u3', 'u2', 'cx', 'reset']
#     # reading coupling

#     lattice_xy = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]

    
#     grid_topology = BenchArch(c_qasm, coupling_map=lattice_xy).topology
#     #grid_topology = BenchArch(c_qasm, lattice_xy=lattice_xy).topology
#     coupling_map = [list(edge) for edge in list(grid_topology.edges)]
#     coupling_map += [[edge[1], edge[0]] for edge in list(grid_topology.edges)]

#     total_gate_count, depth, total_time = qs_compiler(c_qasm, coupling_map, qaoa=qaoa, layers=1, trials=1, bgate='rzz', params=param)
    
#     print(total_gate_count, depth)
#     return (total_gate_count, depth, total_time)


# if __name__ == '__main__':
#     # print('random_graph10_0.1_1.txt')
#     # total_gate_count, swap_count, depth, total_time = runner('random_graph10_0.1_1.txt')


result = {}
swapCOUNT = []
cxCOUNT = []
depthCOUNT = []
# Specify the folder path
# folder_path = '../myBench'  # Replace with the actual path to your folder

folder_path = '../../benchmarks/qaoa_text'

# Get a list of all files in the folder
file_names = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]

# Print the list of file names
file_names.sort()
skip = ['.DS_Store', 'tetris',
        '2qan_results.txt', 'construct_qaoa_circ_json_regular.py',
        'random_graph10_0.1_0.txt', 'random_graph10_0.1_1.txt',
        'random_graph10_0.1_2.txt', 'random_graph10_0.1_4.txt',
        '__init__.py', 'construct_qaoa_circ_json_regular.py', 'tetris']

for file_name in file_names:
    if file_name in skip:
        continue

    print(file_name)

    print("scheduling: ", file_name)
    total_gate_count, swap_count, depth, total_time = runner(file_name)

    # num_2q, depth_2q = runner(file_name)
    # cxCOUNT.append(num_2q)
    # depthCOUNT.append(depth_2q)
    # result[file_name] = (num_2q, 0, depth_2q)


    result[file_name] = (total_gate_count, swap_count, depth, total_time)
    cxCOUNT.append(total_gate_count)
    swapCOUNT.append(swap_count)
    depthCOUNT.append(depth)
#save result
with open("../myBench/2qan_results.txt", "w") as file:
    for key, value in result.items():
        file.write(f"{key} CNOT gate count by 2QAN: {value[0]}\n")
# print(result)
print('depth:')
for i in depthCOUNT:
    print(i)
print('swap:')
for i in swapCOUNT:
    print(i)
print('cx:')
for i in cxCOUNT:
    print(i)


print(file_names)

###############################################################
    # coupling = ["ibm"]
    # # lq is logical qubit
    # lq = [10,12,14,16,18,20]
    # # pq is physical qubit
    # # ibm_pq = [39]
    # ibm_pq = 'heavyhex_distance7.txt'

    # google_pq = [64]
    # #d = [10, 20, 30]
    # #d = [[10, 20, 30], [20,40,60], [40, 80, 120], [80, 160, 240]]
    # d = [0.1,0.2,0.3]
    # result = {}
    # swapCOUNT = []
    # cxCOUNT = []
    # depthCOUNT = []
    # for it in range(len(lq)):
    #     for dens in d:
    #         print("scheduling: ",ibm_pq, lq[it], ibm_pq, dens)
    #         total_gate_count, swap_count, depth, total_time = runner(ibm_pq, lq[it], ibm_pq, dens)
    #         result[(ibm_pq,lq[it], dens,"random")] = (total_gate_count, swap_count, depth, total_time)
    #         cxCOUNT.append(total_gate_count)
    #         swapCOUNT.append(swap_count)
    #         depthCOUNT.append(depth)
    # #save result
    # with open("my_new_result.txt", "a") as file:
    #     for key, value in result.items():
    #         file.write(f"{key},{value}\n")
    # # print(result)
    # print('depth:')
    # for i in depthCOUNT:
    #     print(i)
    # print('swap:')
    # for i in swapCOUNT:
    #     print(i)
    # print('cx:')
    # for i in cxCOUNT:
    #     print(i)
