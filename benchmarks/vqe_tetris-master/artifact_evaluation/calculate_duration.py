import pickle
import os

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


prefix = 'experiment_results'
for k in range(1, 7):
    for mapper in ['jordan_wigner', 'bravyi_kitaev', 'random']:
        for filename in [f'{prefix}/{mapper}/PH_data_{k}.pickle', f'{prefix}/{mapper}/Tetris_data_{k}.pickle']:
            if not os.path.exists(filename):
                continue
            
            data = pickle_load(filename)

            backend = FakeManhattan()
            if not 'duration' in data[1]:
                print(f'calculating duration in {filename}')
                qc = QuantumCircuit.from_qasm_str(data[1]['qasm'])
                with pulse.build(FakeManhattan()) as my_program:
                    pulse.call(qc)

                print(my_program.duration)
                data[1]['duration'] = my_program.duration
                print(f'The duration is {my_program.duration} dt')
                pickle_dump(data, filename)
            else:
                print(f'duration already calculated in {filename}')
