import sys, time

from utils.parallel_bl import *
from qiskit import QuantumCircuit, transpile, qasm2
import synthesis_SC
import synthesis_FT
from tools import *
from arch import *
import time, sys, os
from t_arch import *
from config import test_scale
import pdb
import random
from utils.synthesis_broccoli import synthesis
from utils.synthesis_max_cancel import synthesis_max_cancel
from utils.synthesis_k_leaftrees import synthesis_k_leaftrees
from utils.synthesis_lookahead import synthesis_lookahead
from utils.bridge_friendly_block_scheduling import bridge_friendly_block_scheduling
import pickle

random.seed(1926)

old_cwd = os.getcwd()
ctime = time.time

def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_oplist(mapper_name, mole_name):
    if mapper_name == 'random':
        fth = os.path.join('data', 'random', f'random_{mole_name}.pickle')
    else:
        fth = os.path.join('data', mapper_name, mole_name + '_UCCSD.pickle')
    with open(fth, 'rb') as f:
        entry = pickle.load(f)
    return entry

def load_sycamore_coupling_map():
    reduced = True
    pth = os.path.join('arch', 'sycamore_64.txt')

    coupling = []
    n = 0
    with open(pth, 'r') as file:
        lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split()[:2])
        n = num_nodes
        
        # Add edges to the graph
        for edge in lines[1:]:
            node1, node2 = map(int, edge.split()[:2])
            coupling.append([node1, node2])
            coupling.append([node2, node1])

    return coupling

# PH Sycamore device method
def PH_Sycamore(parr):
    print('PH passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_sycamore_coupling_map()
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [[block] for block in parr]
    qc, total_swaps, total_cx = synthesis_SC.block_opt_SC(a2, arch='sycamore')
    pnq = qc.num_qubits
    latency1 = ctime() - t0
    print('PH, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)
    latency2 = ctime() - t0
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    print('Total swaps:', total_swaps)
    print('Total cx:', total_cx)
    return {
        'n_qubits': lnq,
        'PH_swap_count': total_swaps,
        'PH_cx_count': total_cx,
        'CNOT': cnots,
        'Single': singles,
        'Total': cnots+singles,
        'Depth': depth,
        'qasm' : qc2.qasm(),
        'latency1' : latency1,
        'latency2' : latency2
    }


# Tetris Sycamore device method
def Tetris_Sycamore(parr, use_bridge, swap_coefficient=3):
    print('Tetris passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_sycamore_coupling_map()
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [[block] for block in parr]
    qc, metrics = synthesis(a2, arch='sycamore', use_bridge=use_bridge, swap_coefficient=swap_coefficient)
    pnq = qc.num_qubits
    latency1 = ctime() - t0
    print('Tetris, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)
    latency2 = ctime() - t0
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    metrics.update({'CNOT': cnots,
                    'Single': singles,
                    'Total': cnots+singles,
                    'Depth': depth,
                    'qasm' : qc2.qasm(),
                    'latency1' : latency1,
                    'latency2' : latency2
                })
    print(metrics)
    return metrics

# Tetris Sycamore device method
def Tetris_max_cancel_Sycamore(parr, use_bridge):
    print('Tetris passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_sycamore_coupling_map()
    t0 = ctime()
    qc, metrics = synthesis_max_cancel(parr, arch='sycamore', use_bridge=use_bridge)
    pnq = qc.num_qubits
    latency1 = ctime() - t0
    print('Tetris, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)
    latency2 = ctime() - t0
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    metrics.update({'CNOT': cnots,
                    'Single': singles,
                    'Total': cnots+singles,
                    'Depth': depth,
                    'qasm' : qc2.qasm(),
                    'latency1' : latency1,
                    'latency2' : latency2
                })
    print(metrics)
    return metrics


# Tetris Sycamore device method
def Tetris_lookahead_Sycamore(parr, use_bridge, swap_coefficient=3, k=10):
    print('Tetris passes, Our schedule, Our synthesis, Sycamore', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_sycamore_coupling_map()
    t0 = ctime()
    # a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [block for blocks in a2 for block in blocks]
    a2 = parr
    qc, metrics = synthesis_lookahead(a2, arch='sycamore', use_bridge=use_bridge, swap_coefficient=swap_coefficient, k=k)
    pnq = qc.num_qubits
    latency1 = ctime() - t0
    print('Tetris, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)
    latency2 = ctime() - t0
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    metrics.update({'CNOT': cnots,
                    'Single': singles,
                    'Total': cnots+singles,
                    'Depth': depth,
                    'qasm' : qc2.qasm(),
                    'latency1' : latency1,
                    'latency2' : latency2
                })
    key_to_exclude = 'qasm'

    # Printing key-value pairs excluding a certain key
    for key, value in metrics.items():
        if key != key_to_exclude:
            print(f"{key}: {value}")
    return metrics


def run_random_benchmark(k=6):
    n_q = [10, 15, 20, 25, 30, 35]
    
    # metrics_list = []
    # print("+++++++++PauliHedral+++++++++++")
    # for i in range(0,k):
    #     print('Random:', n_q[i])
    #     parr = load_oplist('random', n_q[i])
    #     metrics = PH_Sycamore(parr)
    #     metrics_list.append((f'random_{n_q[i]}', metrics))
    # pickle_dump(metrics_list, f'runs_final/sycamore/random/PH_data.pickle')
    
    # metrics_list = []
    # print("+++++++++Our method+++++++++++")
    # for i in range(0,k):
    #     print('Random:', n_q[i])
    #     parr = load_oplist('random', n_q[i])
    #     metrics = Tetris_Sycamore(parr, use_bridge=False)
    #     metrics_list.append((f'random_{n_q[i]}', metrics))
    # pickle_dump(metrics_list, f'runs_final/sycamore/random/Tetris_data.pickle')
    
    # metrics_list = []
    # print("+++++++++Our method+++++++++++")
    # for i in range(0,k):
    #     print('Random:', n_q[i])
    #     parr = load_oplist('random', n_q[i])
    #     metrics = Tetris_max_cancel_Sycamore(parr, use_bridge=False)
    #     metrics_list.append((f'random_{n_q[i]}', metrics))
    # pickle_dump(metrics_list, f'runs_final/sycamore/random/Max_cancel_data.pickle')

    metrics_list = []
    print("+++++++++Our method+++++++++++")
    for i in range(0,k):
        print('Random:', n_q[i])
        parr = load_oplist('random', n_q[i])
        metrics = Tetris_lookahead_Sycamore(parr, use_bridge=False)
        metrics_list.append((f'random_{n_q[i]}', metrics))
    pickle_dump(metrics_list, f'runs_final/sycamore/random/Tetris_lookahead_data.pickle')
    
if __name__ == '__main__':
    
    ############################
    # UCCSD Part
    ############################
    # UCCSD- 8,12,16,20,24,28
    moles = ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']
    if test_scale == 'small':
        k = 6
    else:
        k = 6

    for mapper in ['jordan_wigner']:
        metrics_list = []
        
        # print("+++++++++PauliHedral+++++++++++")
        # for i in range(0,k):
        #     print('UCCSD:', moles[i])
        #     parr = load_oplist(mapper, moles[i])
        #     metrics = PH_Sycamore(parr)
        #     metrics_list.append((moles[i], metrics))

        # pickle_dump(metrics_list, f'runs_final/sycamore/{mapper}/PH_data.pickle')

        # metrics_list = []
        # print("+++++++++Our method+++++++++++")
        # for i in range(0,k):
        #     print('UCCSD:', moles[i])
        #     parr = load_oplist(mapper, moles[i])
        #     metrics = Tetris_Sycamore(parr, use_bridge=False)
        #     metrics_list.append((moles[i], metrics))

        # pickle_dump(metrics_list, f'runs_final/sycamore/{mapper}/Tetris_data.pickle')
        
        # metrics_list = []
        # print("+++++++++Our method+++++++++++")
        # for i in range(0,k):
        #     print('UCCSD:', moles[i])
        #     parr = load_oplist(mapper, moles[i])
        #     metrics = Tetris_max_cancel_Sycamore(parr, use_bridge=False)
        #     metrics_list.append((moles[i], metrics))

        # pickle_dump(metrics_list, f'runs_final/sycamore/{mapper}/Max_cancel_data.pickle')
        
        metrics_list = []
        print("+++++++++Our method+++++++++++")
        for i in range(0,k):
            print('UCCSD:', moles[i])
            parr = load_oplist(mapper, moles[i])
            metrics = Tetris_lookahead_Sycamore(parr, use_bridge=False)
            metrics_list.append((moles[i], metrics))

        pickle_dump(metrics_list, f'runs_final/sycamore/{mapper}/Tetris_lookahead_data.pickle')

    run_random_benchmark()
