import sys, time
import argparse
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
from utils.synthesis_lookahead_bfs import synthesis_lookahead_bfs
from utils.bridge_friendly_block_scheduling import bridge_friendly_block_scheduling
from utils.grey_code_scheduling import sort_paulistrings, extract_first_ps
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

# PH Mahattan device method
def PH_Mahattan(parr):
    print('PH passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [[block] for block in parr]
    qc, total_swaps, total_cx = synthesis_SC.block_opt_SC(a2, arch='manhattan')
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


# Tetris Mahattan device method
def Tetris_Mahattan(parr, use_bridge, swap_coefficient=3):
    print('Tetris passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [[block] for block in parr]
    qc, metrics = synthesis(a2, arch='manhattan', use_bridge=use_bridge, swap_coefficient=swap_coefficient)
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

# Tetris Mahattan device method
def Tetris_lookahead_Mahattan(parr, use_bridge, swap_coefficient=3, k=10):
    print('Tetris passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    # a2 = gate_count_oriented_scheduling(parr)#, length=length, maxiter=30)
    # a2 = [block for blocks in a2 for block in blocks]
    a2 = parr
    qc, metrics = synthesis_lookahead_bfs(a2, arch='manhattan', use_bridge=use_bridge, swap_coefficient=swap_coefficient, k=k)
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
    qc, metrics = synthesis_lookahead_bfs(a2, arch='sycamore', use_bridge=use_bridge, swap_coefficient=swap_coefficient, k=k)
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



def merge_block(parr, size):
    tmp = 0
    new_blocks = []
    while tmp < len(parr):
        new_block = []
        for i in range(size):
            new_block = new_block + parr[tmp]
            tmp = tmp + 1
            if tmp >= len(parr):
                break
        new_blocks.append(new_block)
    return new_blocks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('mole_id', type=int, help='id range from 0 to 5')
    parser.add_argument('w', type=float, help='weight in cost function')
    parser.add_argument('--optional_arg', type=float, default=0.0, help='Description of optional_arg')

    args = parser.parse_args()

    ############################
    # UCCSD Part
    ############################
    # UCCSD- 8,12,16,20,24,28
    moles = ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']
    if test_scale == 'small':
        k = 6
    else:
        k = 6

    mapper = 'jordan_wigner'#, 'parity', 'bravyi_kitaev']:
    
    overWrite = False
    start_path = "/common/home/zl606/vqe_project/core/"

    
    for i in range(args.mole_id, args.mole_id+1):
        full_path = os.path.join(start_path, f'runs_final/ablation/ithaca_mole{i}_different_w.pickle')
        if not overWrite and os.path.exists(full_path):
            print(f"File {full_path} already exist, skip.")
        else:
            print("Results will be stored in " + full_path)
            metrics_list = []
            print("+++++++++Our method+++++++++++")
            for swap_coefficient in [0.1, 0.5, 1, 2, 3, 4, 5, 10, 100]:
                print('UCCSD:', moles[i])
                print('swap_coeff:', swap_coefficient)
                parr = load_oplist(mapper, moles[i])
                metrics = Tetris_lookahead_Mahattan(parr, use_bridge=False, swap_coefficient=swap_coefficient)
                metrics_list.append((swap_coefficient, metrics))

            pickle_dump(metrics_list, f'runs_final/ablation/ithaca_mole{i}_different_w.pickle')
        
        full_path = os.path.join(start_path, f'runs_final/ablation/sycamore_mole{i}_different_w.pickle')
        if not overWrite and os.path.exists(full_path):
            print(f"File {full_path} already exist, skip.")
        else:
            print("Results will be stored in " + f'runs_final/ablation/sycamore_mole{i}_different_w_{args.w}.pickle')
            metrics_list = []
            print("+++++++++Our method+++++++++++")
            for swap_coefficient in [args.w]:#[0.1, 0.5, 1, 2, 3, 4, 5, 10, 100]:
                print('UCCSD:', moles[i])
                print('swap_coeff:', swap_coefficient)
                parr = load_oplist(mapper, moles[i])
                metrics = Tetris_lookahead_Sycamore(parr, use_bridge=False, swap_coefficient=swap_coefficient)
                metrics_list.append((swap_coefficient, metrics))

            pickle_dump(metrics_list, f'runs_final/ablation/sycamore_mole{i}_different_w_{args.w}.pickle')
        
