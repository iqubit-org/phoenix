import pickle
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
import pdb
import random
from utils.synthesis_broccoli import synthesis
from utils.synthesis_max_cancel import synthesis_max_cancel
from utils.synthesis_k_leaftrees import synthesis_k_leaftrees
from utils.synthesis_lookahead import synthesis_lookahead
from utils.synthesis_lookahead_bfs import synthesis_lookahead_bfs
from utils.bridge_friendly_block_scheduling import bridge_friendly_block_scheduling
from utils.grey_code_scheduling import sort_paulistrings, extract_first_ps
ctime = time.time
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)


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
    print(counts.get('0' * 65, 0) / 1024)
    return counts.get('0' * 65, 0) / 1024


if __name__ == '__main__':
    for i in [0, 1, 2, 3, 4, 5]:
        moles = ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']
        mapper = 'jordan_wigner'
        print('UCCSD:', moles[i])
        parr = load_oplist(mapper, moles[i])
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
        
        pickle_dump((X, Y_ph, Y_tetris), f'runs_final/noise_simulator/data_{i}.pickle')
    