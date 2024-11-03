import pickle

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

# mapper = 'jordan_wigner'
mapper = 'bravyi_kitaev'
# mapper = 'parity'
for mapper in ['jordan_wigner', 'bravyi_kitaev', 'random']:
    PH_data_list = pickle_load(f'runs_final/{mapper}/PH_data.pickle')
    Tetris_data_list = pickle_load(f'runs_final/{mapper}/Tetris_data.pickle')
    Max_cancel_data_list = pickle_load(f'runs_final/{mapper}/Max_cancel_data.pickle')
    Tetris_lookahead_data_list = pickle_load(f'runs_final/{mapper}/Tetris_lookahead_data.pickle')

    categories = []
    ph_cx_cancel_ratio = []
    tetris_cx_cancel_ratio = []
    max_cancel_cx_cancel_ratio = []

    ph_swaps = []
    tetris_swaps = []
    max_cancel_swaps = []

    ph_cnots = []
    tetris_cnots = []
    max_cancel_cnots = []

    ph_depth = []
    tetris_depth = []
    max_cancel_depth = []

    backend = FakeManhattan()

    for i, (ph_data, tetris_data, max_cancel_data, tetris_lookahead_data) in enumerate(zip(PH_data_list, Tetris_data_list, Max_cancel_data_list, Tetris_lookahead_data_list)):
        mole, ph = ph_data
        mole, tetris = tetris_data
        mole, max_cancel = max_cancel_data
        mole, tetris_lookahead = tetris_lookahead_data
        categories.append(mole)
        print(mole)
        # bravyi_kitaev
        # if mole == 'LiH':
        #     ph['duration'], tetris['duration'], max_cancel['duration'] = 16290080, 14569280, 21678752
        # elif mole == 'BeH2':
        #     ph['duration'], tetris['duration'], max_cancel['duration'] = 34032704, 27529184, 43391616
        # elif mole == 'CH4':
        #     ph['duration'], tetris['duration'], max_cancel['duration'] = 85000224, 79392864, 151582592
        # elif mole == 'MgH2':
        #     ph['duration'], tetris['duration'], max_cancel['duration'] = 204157376, 143235552, 329790496
        # elif mole == 'LiCl':
        #     ph['duration'], tetris['duration'] = 406154912, 350287808
        # elif mole == 'CO2':
        #     ph['duration'], tetris['duration'] = 567852224, 366858784
            
        
        original_cx_count = tetris['original CNOT count']
        
        # ph['CX_cancel_ratio'] = 1.0 - 1.0 * (ph['CNOT'] - ph['PH_swap_count'] * 3) / original_cx_count
        # tetris['CX_cancel_ratio'] = 1.0 - 1.0 * (tetris['CNOT'] - tetris['tetris_swap_count'] * 3) / original_cx_count
        # max_cancel['CX_cancel_ratio'] = 1.0 - 1.0 * (max_cancel['CNOT'] - max_cancel['tetris_swap_count'] * 3) / original_cx_count
        tetris_lookahead['CX_cancel_ratio'] = 1.0 - 1.0 * (tetris_lookahead['CNOT'] - tetris_lookahead['tetris_swap_count'] * 3) / original_cx_count
        
        # ph['Swap_insertion'] = ph['PH_swap_count']
        # tetris['Swap_insertion'] = tetris['tetris_swap_count']
        # max_cancel['Swap_insertion'] = max_cancel['tetris_swap_count']
        tetris_lookahead['Swap_insertion'] = tetris_lookahead['tetris_swap_count']

        # if not 'duration' in ph:
        #     print('PH')
        #     qc = QuantumCircuit.from_qasm_str(ph['qasm'])
        #     with pulse.build(FakeManhattan()) as my_program:
        #         pulse.call(qc)

        #     print(my_program.duration)
        #     ph['duration'] = my_program.duration
        
        # if not 'duration' in tetris:
        #     print('tetris')
        #     qc = QuantumCircuit.from_qasm_str(tetris['qasm'])
        #     with pulse.build(FakeManhattan()) as my_program:
        #         pulse.call(qc)

        #     print(my_program.duration)
        #     tetris['duration'] = my_program.duration
        
        # if not 'duration' in max_cancel:
        #     qc = QuantumCircuit.from_qasm_str(max_cancel['qasm'])
        #     with pulse.build(FakeManhattan()) as my_program:
        #         pulse.call(qc)

        #     print(my_program.duration)
        #     max_cancel['duration'] = my_program.duration
 
        if not 'duration' in tetris_lookahead:
            print('tetris_lookahead')
            qc = QuantumCircuit.from_qasm_str(tetris_lookahead['qasm'])
            with pulse.build(FakeManhattan()) as my_program:
                pulse.call(qc)

            print(my_program.duration)
            tetris_lookahead['duration'] = my_program.duration

    # pickle_dump(PH_data_list, f'runs_final/{mapper}/PH_data.pickle')
    # pickle_dump(Tetris_data_list, f'runs_final/{mapper}/Tetris_data.pickle')
    # pickle_dump(Max_cancel_data_list, f'runs_final/{mapper}/Max_cancel_data.pickle')
    pickle_dump(Tetris_lookahead_data_list, f'runs_final/{mapper}/Tetris_lookahead_data.pickle')