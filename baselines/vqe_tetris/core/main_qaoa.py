import sys, time
from benchmark.offline import *
from utils.parallel_bl import *
from qiskit import QuantumCircuit, transpile
import synthesis_SC
import synthesis_FT
from tools import *
from arch import *
import time, sys, os
from t_arch import *
import ipdb
import random
from utils.synthesis_broccoli import synthesis
from utils.bridge_friendly_block_scheduling import bridge_friendly_block_scheduling

random.seed(1926)

old_cwd = os.getcwd()
set_cwd()
ctime = time.time

def load_oplist(n_v):
    fth = os.path.join('data', 'qaoa', 'erdos_renyi_graph', str(n_v) + '.pickle')
    with open(fth, 'rb') as f:
        entry = pickle.load(f)
    return entry

# PH Mahattan device method
def PH_Mahattan(parr):
    print('PH passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [[block] for block in parr]
    qc, total_swaps = synthesis_SC.block_opt_SC(a2, arch='manhattan')
    pnq = qc.num_qubits
    print('PH, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx', 'swap'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    print_qc(qc2)
    qasm_str = qc2.qasm()
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    print('number of swaps than can be replaced by bridge:', analyze_bridge(qc2))
    return qasm_str

def analyze_bridge(qc):
    is_ancilla = [True for i in range(qc.num_qubits)]
    total_bridges = 0
    for ins in reversed(qc.data):
        if ins.operation.name == 'cx':
            q0 = ins.qubits[0].index
            q1 = ins.qubits[1].index
            is_ancilla[q0] = False
            is_ancilla[q1] = False
        elif ins.operation.name == 'swap':
            q0 = ins.qubits[0].index
            q1 = ins.qubits[1].index
            is_ancilla[q0], is_ancilla[q1] = is_ancilla[q1], is_ancilla[q0]
            if is_ancilla[q0] == False and is_ancilla[q1] == True:
                total_bridges = total_bridges + 1
                print((q0, q1))
        elif ins.operation.name == 'u3':
            q0 = ins.qubits[0].index
            is_ancilla[q0] = False
        else:
            raise Exception('Unexpected operation ' + ins.operation.name + '.Operation name can only be {cx, swap, u3}')
    return total_bridges

# Tetris Mahattan device method
def Tetris_Mahattan(parr, use_bridge):
    print('Tetris passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a2 = bridge_friendly_block_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [[block] for block in parr]
    qc, metrics = synthesis(a2, arch='manhattan', use_bridge=use_bridge)
    pnq = qc.num_qubits
    print('Tetris, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx', 'swap'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    print_qc(qc2)
    # qasm_str = qc2.qasm()
    # pdb.set_trace()
    # print(qasm_str)
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    print(metrics)
    print('number of swaps than can be replaced by bridge:', analyze_bridge(qc2))



############################
# QAOA Part
############################
# QAOA - 15,16,17,18,19,20

n_vertices = [15, 16, 17, 18, 19, 20]
k = 6

print("+++++++++PauliHedral+++++++++++")
for i in range(0,k):
    print('QAOA:', n_vertices[i])
    parr = load_oplist(n_vertices[i])
    print(len(parr))
    qasm_str = PH_Mahattan(parr)
    # with open(f'data/qaoa/{n_vertices[i]}.qasm', 'w') as file:
    #     file.write(qasm_str)

# print("+++++++++Our method+++++++++++")
# for i in range(0,k):
#     print('QAOA:', n_vertices[i])
#     parr = load_oplist(n_vertices[i])
#     Tetris_Mahattan(parr, use_bridge=False)
# print("+++++++++Our method(with bridge)+++++++++++")
# for i in range(0,k):
#     print('QAOA:', n_vertices[i])
#     parr = load_oplist(n_vertices[i])
#     Tetris_Mahattan(parr, use_bridge=True)
    
exit()
