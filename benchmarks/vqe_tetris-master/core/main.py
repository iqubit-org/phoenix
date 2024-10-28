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
from config import test_scale
import ipdb
import random
from utils.synthesis_broccoli import synthesis
from utils.bridge_friendly_block_scheduling import bridge_friendly_block_scheduling

random.seed(1926)

old_cwd = os.getcwd()
set_cwd()
ctime = time.time

def load_oplist(mapper_name, mole_name):
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
    qc, total_swaps = synthesis_SC.block_opt_SC(a2, arch='manhattan')
    pnq = qc.num_qubits
    print('PH, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    print_qc(qc2)
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    print('Total swaps:', total_swaps)


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
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    print_qc(qc2)
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    print(metrics)

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

############################
# UCCSD Part
############################
# UCCSD- 8,12,16,20,24,28
moles = ['LiH', 'BeH2', 'CH4', 'MgH', 'LiCl', 'CO2']
if test_scale == 'small':
    k = 6
else:
    k = 6

mapper = 'jordan_wigner'
# mapper = 'parity'
# mapper = 'bravyikitaev'

print("+++++++++PauliHedral+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    # print(parr[-19:-18])
    # PH_Mahattan(parr[-19:-18])
    PH_Mahattan(parr)

print("+++++++++Our method+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    # Tetris_Mahattan(parr[-19:-18], use_bridge=False)
    Tetris_Mahattan(parr, use_bridge=False)
print("+++++++++Our method(with bridge)+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    # Tetris_Mahattan(parr[-19:-18], use_bridge=False)
    Tetris_Mahattan(parr, use_bridge=True)



print("+++++++++Our method(1 Tetris 2 blocks)+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    parr = merge_block(parr, 2)
    # Tetris_Mahattan(parr[-19:-18], use_bridge=False)
    Tetris_Mahattan(parr, use_bridge=False)
print("+++++++++Our method(1 Tetris 3 blocks)+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    parr = merge_block(parr, 3)
    # Tetris_Mahattan(parr[-19:-18], use_bridge=False)
    Tetris_Mahattan(parr, use_bridge=False)
print("+++++++++Our method with bridge(1 Tetris 2 blocks)+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    parr = merge_block(parr, 2)
    # Tetris_Mahattan(parr[-19:-18], use_bridge=False)
    Tetris_Mahattan(parr, use_bridge=True)
print("+++++++++Our method with bridge(1 Tetris 3 blocks)+++++++++++")
for i in range(0,k):
    print('UCCSD:', moles[i])
    parr = load_oplist(mapper, moles[i])
    parr = merge_block(parr, 3)
    # Tetris_Mahattan(parr[-19:-18], use_bridge=False)
    Tetris_Mahattan(parr, use_bridge=True)
exit()
