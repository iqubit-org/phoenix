import pickle
import sys, time

sys.path.append('../core/')
import argparse
from utils.parallel_bl import *
from qiskit import QuantumCircuit, transpile
import synthesis_SC
import synthesis_FT
from tools import *
from arch import *
import time, sys, os
from t_arch import *
import pdb
import random
from utils.synthesis_lookahead_bfs import synthesis_lookahead_bfs
ctime = time.time
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

import numpy as np
np.random.seed(1)
random.seed(1)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile, pulse
from qiskit.providers.fake_provider import FakeManhattan

def load_oplist(mapper_name, mole_name):
    if mapper_name == 'random':
        fth = os.path.join('../core/data', 'random', f'random_{mole_name}.pickle')
    else:
        fth = os.path.join('../core/data', mapper_name, mole_name + '_UCCSD.pickle')
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
    }, qc2


# Tetris Mahattan device method
def Tetris_lookahead_Mahattan(parr, use_bridge=False, swap_coefficient=3, k=10):
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
    return metrics, qc2

def test_fidelity(qc : QuantumCircuit):
    qc_inverse = qc.inverse()
    for ins in qc_inverse:
        qc.append(ins)
    qc.measure_all()
    noise_model = NoiseModel()
    backend = FakeManhattan()
    error_gate2 = depolarizing_error(0.001, 2)
    error_gate1 = depolarizing_error(0.0001, 1)
    noise_model.add_all_qubit_quantum_error(error_gate1, ['u3'])
    noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])
    print(noise_model)
    
    backend = AerSimulator(noise_model=noise_model)
    result = backend.run(qc).result()
    counts = result.get_counts(qc)
    # pdb.set_trace()
    # print(counts.get('0' * 65, 0) / 1024)
    return counts.get('0' * 65, 0) / 1024


if __name__ == '__main__':
    for i in [0]:
        if os.path.exists(f'experiment_results/fidelity_{i}.pickle'):
            continue
        moles = ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']
        mapper = 'jordan_wigner'
        print('UCCSD:', moles[i])
        parr = load_oplist(mapper, moles[i])
        
        num_samples = 100
        Y_ph_list = []
        Y_tetris_list = []
        for sample_once in range(num_samples):
            random.shuffle(parr)
            X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            Y_ph = []
            Y_tetris = []
            for size in X:
                metrics_ph, qc_ph = PH_Mahattan(parr[:size])
                metrics_tetris, qc_tetris = Tetris_lookahead_Mahattan(parr[:size])

                Y_ph.append(test_fidelity(qc_ph))
                Y_tetris.append(test_fidelity(qc_tetris))
                print(Y_ph)
                print(Y_tetris)
            Y_ph_list.append(Y_ph)
            Y_tetris_list.append(Y_tetris)
            print(f'Sample #{sample_once} finished')
            pickle_dump((X, np.array(Y_ph_list), np.array(Y_tetris_list)), f'experiment_results/fidelity_{i}.pickle')

        Y_ph_list = np.array(Y_ph_list)
        Y_tetris_list = np.array(Y_tetris_list)
    
    import matplotlib.pyplot as plt
    moles = ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']
    for i in [0]:
        X, Y_ph_list, Y_tetris_list = pickle_load(f'experiment_results/fidelity_{i}.pickle')
        Y_ph = Y_ph_list.mean(0)
        Y_tetris = Y_tetris_list.mean(0)
        Y_ph_min = Y_ph_list.min(0)
        Y_tetris_min = Y_tetris_list.min(0)
        Y_ph_max = Y_ph_list.max(0)
        Y_tetris_max = Y_tetris_list.max(0)
        
        fig, ax = plt.subplots()

        plt.errorbar(X, Y_ph, yerr=[Y_ph - Y_ph_min, Y_ph_max - Y_ph], label='PH', marker='*', alpha=0.9, ecolor='#1f77b4', capsize=5)
        plt.errorbar(X, Y_tetris, yerr=[Y_tetris - Y_tetris_min, Y_tetris_max - Y_tetris], label='Tetris', marker='s', alpha=0.9, ecolor='#ff7f0e', capsize=5)
        ytick_labels = ax.get_yticklabels()
        for label in ytick_labels:
            label.set_fontweight('bold')
            label.set_fontsize(24)
        xtick_labels = ax.get_xticklabels()
        for label in xtick_labels:
            label.set_fontweight('bold')
            label.set_fontsize(24)

        # Adding labels
        plt.xlabel('#Blocks', fontsize=24, fontweight='bold')
        plt.ylabel('fidelity', fontsize=24, fontweight='bold')
        # plt.title('Plot of fidelity_ and Y_tetris')
        legend = plt.legend(fontsize=24)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        plt.tight_layout()
        plt.savefig(f'figs/fidelity_{moles[i]}.pdf')

        # Display plot
