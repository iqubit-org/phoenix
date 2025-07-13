import sys, time
from utils.parallel_bl import *
from qiskit import transpile
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
from utils.synthesis_k_leaftrees import synthesis_k_leaftrees
from utils.bridge_friendly_block_scheduling import bridge_friendly_block_scheduling
import pickle
from benchmark.mypauli import pauliString


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
    a2 = depth_oriented_scheduling(parr, length=length, maxiter=30)
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

def load_data(folder_name):
    data = {}
    folder_path = 'data/qaoa/myBench/'

    files = os.listdir(folder_path)
    for file_name in files:
        if file_name.find('txt') == -1:
            continue
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split()[:2])
        n = num_nodes
        
        pauli_blocks = []
        # Add edges to the graph
        for edge in lines[1:]:
            node1, node2 = map(int, edge.split()[:2])
            ps = ['I' for i in range(num_nodes)]
            ps[node1] = 'Z'
            ps[node2] = 'Z'
            block = [pauliString(''.join(ps))]
            pauli_blocks.append(block)
        data[file_name] = pauli_blocks
        
    return data

def run_qaoa():
    # dataset = {**load_data('random_graph'), **load_data('regular_graph0')}
    dataset = load_data('')
    metrics_list = []
    print("+++++++++PauliHedral+++++++++++")
    for filename, blocks in dataset.items():
        print('Processing: ' + filename)
        parr = blocks
        metrics = PH_Mahattan(parr)
        metrics_list.append((filename, metrics))
    pickle_dump(metrics_list, f'runs_final/qaoa/PH_data.pickle')

if __name__ == '__main__':
    
    run_qaoa()