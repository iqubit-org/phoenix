from qiskit.quantum_info.operators import Operator
from qiskit import QuantumCircuit, transpile
from load_coupling_map import load_coupling_map, print_qc, pickle_dump, pickle_load
import numpy as np
import pdb
import argparse
import re
import os
import time
ctime = time.time

def rotxy(theta, phi):
    return [[np.cos(theta/2), complex(-np.sin(theta/2)*np.sin(phi), -np.sin(theta/2)*np.cos(phi))],
            [complex(np.sin(theta/2)*np.sin(phi), -np.sin(theta/2)*np.cos(phi)), np.cos(theta/2)]]

def extract_phi_gamma(line):
    # Regular expression pattern to match phi and gamma values
    pattern = r"phi = (.*?), gamma = (.*?)\)"

    # Search for phi and gamma values using the pattern
    match = re.search(pattern, line)
    if match:
        phi = float(match.group(1))
        gamma = float(match.group(2))
        return phi, gamma
    else:
        return None, None

def extract_gamma(line):
    # Regular expression pattern to match phi value
    pattern = r"gamma = (.*?)\)"

    # Search for phi value using the pattern
    match = re.search(pattern, line)
    if match:
        gamma = float(match.group(1))
        return gamma
    else:
        return None

def extract_1q(line):
    # Regular expression pattern to match phi and gamma values
    pattern = r"on phys Q (.*?)\n"

    # Search for phi and gamma values using the pattern
    match = re.search(pattern, line)
    if match:
        q0 = int(match.group(1))
        return q0
    else:
        return None

def extract_2q(line):
    # Regular expression pattern to match phi and gamma values
    pattern = r"on phys Q(.*?) and phys Q(.*?)\n"

    # Search for phi and gamma values using the pattern
    match = re.search(pattern, line)
    if match:
        q0 = int(match.group(1))
        q1 = int(match.group(2))
        return q0, q1
    else:
        return None, None

def classify_instruction(qc, line):
    num_rotxy = 0
    num_cphase = 0
    if "ROTXY" in line:
        phi, gamma = extract_phi_gamma(line)
        q0 = extract_1q(line)
        qc.unitary(Operator(rotxy(phi, gamma)), [q0])
        num_rotxy = num_rotxy + 1
    elif "CPHASE" in line:
        gamma = extract_gamma(line)
        q0, q1 = extract_2q(line)
        qc.cp(gamma, q0, q1)
        num_cphase = num_cphase + 1
    else:
        assert "SWAP" in line
    return num_rotxy, num_cphase

def main():
    parser = argparse.ArgumentParser(description="Classify instructions in a file")
    parser.add_argument("n_qubits", help="number of qubits")
    args = parser.parse_args()
    filename = f'pcoast_{args.n_qubits}.txt'
    n_qubits = int(args.n_qubits)
        
    qc = QuantumCircuit(n_qubits)
    
    try:
        num_rotxy = 0
        num_cphase = 0
        with open(filename, "r") as file:
            # Read lines from the file
            input_lines = file.readlines()
            # Iterate over each line
            for line in input_lines[2:]:
                # Classify the instruction type
                delta_rotxy, delta_cphase = classify_instruction(qc, line)
                num_rotxy += delta_rotxy
                num_cphase += delta_cphase
                
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        return
    
    if os.path.exists(f'experiments/pcoast_{n_qubits}.pickle'):
        print(f'experiments/pcoast_{n_qubits}.pickle already exists!')
        metric = pickle_load(f'experiments/pcoast_{n_qubits}.pickle')
        metric.update({'num_rotxy': num_rotxy, 'num_cphase': num_cphase})
        pickle_dump(metric, f'experiments/pcoast_{n_qubits}.pickle')
        return
    
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(n_qubits)), optimization_level=3)
    cnots, singles, depth = print_qc(qc2)
    latency2 = ctime() - t0
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    metric = {
        'n_qubits': n_qubits,
        'CNOT': cnots,
        'Single': singles,
        'Total': cnots+singles,
        'Depth': depth,
        'qasm' : qc2.qasm(),
        'latency2' : latency2
    }
    pickle_dump(metric, f'experiments/pcoast_{n_qubits}.pickle')

if __name__ == "__main__":
    main()