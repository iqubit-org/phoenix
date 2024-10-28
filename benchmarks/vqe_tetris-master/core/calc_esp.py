import pickle
import pdb
import os
import numpy as np
import csv

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

mapper = 'jordan_wigner'
# mapper = 'bravyi_kitaev'
# mapper = 'parity'

PH_data_list = pickle_load(f'runs_final/PH_data.pickle')
Tetris_data_list = pickle_load(f'runs_final/jordan_wigner/Tetris_lookahead_data.pickle')
# Max_cancel_data_list = pickle_load(f'runs_final/Max_cancel_data.pickle')

from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile, pulse
from qiskit.providers.fake_provider import FakeManhattan
categories = []

def load_cx_error(filename):
    cx_error = {}
    t1_time = {}
    t2_time = {}
    x_error = {}
    reduced = True
    pth = os.path.join('arch', 'data', filename)
    cgs = []
    n = 0
    with open(pth, 'r') as cf:
        g = csv.DictReader(cf, delimiter=',', quotechar='\"')
        for i in g:
            cxval = ""
            t1val = ""
            t2val = ""
            xval = ""
            for j in i.keys():
                if j.find('CNOT') != -1:
                    cxval = i[j]
                if j.find('T1') != -1:
                    t1val = i[j]
                if j.find('T2') != -1:
                    t2val = i[j]
                if j.find('Pauli-X') != -1:
                    xval = i[j]
            t1_time[n] = float(t1val)
            t2_time[n] = float(t2val)
            if xval != "":
                x_error[n] = float(xval)
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
        cx_error[(iq1, iq2)] = float(i[si2+1:])
        if (iq1 < n and iq2 < n) or not reduced:
            coupling.append([iq1, iq2])
    return cx_error, t1_time, t2_time, x_error

backend = FakeManhattan()
cx_error, t1_time, t2_time, x_error = load_cx_error('ibm_ithaca_calibrations.csv')
cx_error, t1_time, t2_time, x_error_ = load_cx_error('ibmq_manhattan_calibrations.csv')

for i, (ph_data, tetris_data) in enumerate(zip(PH_data_list, Tetris_data_list)):
    mole, ph = ph_data
    mole, tetris = tetris_data
    categories.append(mole)
    print(mole)
    

    qc = QuantumCircuit.from_qasm_str(ph['qasm'])
    duration = ph['duration']
    total_success_rate = 0.
    involved_qubits = set()
    for ins in qc:
        if ins.operation.name == 'cx':
            lookup = (ins.qubits[0].index, ins.qubits[1].index)
            involved_qubits.add(lookup[0])
            involved_qubits.add(lookup[1])
            success_rate = 1 - cx_error[lookup]
            total_success_rate += np.log10(success_rate)
        elif ins.operation.name == 'u3':
            lookup = ins.qubits[0].index
            involved_qubits.add(lookup)
            success_rate = 1 - x_error[lookup]
            total_success_rate += 2 * np.log10(success_rate)
    for q in involved_qubits:
        s_r = 1. - 1. / t2_time[q] * (1 - np.exp(-0.00022 *duration / t1_time[q]))
        total_success_rate = total_success_rate + np.log10(s_r)
    ph['success_rate_decoherence'] = total_success_rate
    
    qc = QuantumCircuit.from_qasm_str(tetris['qasm'])
    duration = tetris['duration']
    total_success_rate = 0.
    involved_qubits = set()
    for ins in qc:
        if ins.operation.name == 'cx':
            lookup = (ins.qubits[0].index, ins.qubits[1].index)
            involved_qubits.add(lookup[0])
            involved_qubits.add(lookup[1])
            success_rate = 1 - cx_error[lookup]
            total_success_rate += np.log10(success_rate)
        elif ins.operation.name == 'u3':
            lookup = ins.qubits[0].index
            involved_qubits.add(lookup)
            success_rate = 1 - x_error[lookup]
            total_success_rate += 2 * np.log10(success_rate)
    for q in involved_qubits:
        s_r = 1. - 1. / t2_time[q] * (1 - np.exp(-0.00022 *duration / t1_time[q]))
        total_success_rate = total_success_rate + np.log10(s_r)
    tetris['success_rate_decoherence'] = total_success_rate
    tetris['ESP_imp'] = tetris['success_rate_decoherence'] - ph['success_rate_decoherence']
    print(tetris['ESP_imp'])

pickle_dump(PH_data_list, 'runs_final/PH_data.pickle')
pickle_dump(Tetris_data_list, 'runs_final/jordan_wigner/Tetris_lookahead_data.pickle')
