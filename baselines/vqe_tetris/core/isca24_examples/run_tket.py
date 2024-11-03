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
from utils.grey_code_scheduling import sort_paulistrings, extract_first_ps
import pickle
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit, Node
from pytket import Circuit, OpType
from pytket.circuit import PauliExpBox
from pytket.passes import DecomposeBoxes, PauliSimp, DecomposeSingleQubitsTK1, FullPeepholeOptimise, RoutingPass, DecomposeSwapsToCXs
from pytket.qasm import circuit_from_qasm, circuit_from_qasm_str, circuit_from_qasm_io, circuit_to_qasm_str, circuit_to_qasm
from pytket.placement import place_with_map
from pytket.placement import Placement, LinePlacement, GraphPlacement, NoiseAwarePlacement

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

def TK_Mahattan(parr):
    print('TK passes, mahattan', flush=True)
    n = len(parr[0][0])
    q = [Qubit(i) for i in range(n)]
    oplist = {}
    def to_pauli_list(ps):
        r = []
        for i in ps:
            if i == 'I':
                r.append(Pauli.I)
            elif i == 'X':
                r.append(Pauli.X)
            elif i == 'Y':
                r.append(Pauli.Y)
            elif i == 'Z':
                r.append(Pauli.Z)
        return r
    for i in parr:
        for j in i:
            op = QubitPauliString(q, to_pauli_list(j.ps))
            oplist[op] = 1/3.14
    def add_excitation(circ, term_dict, param=1.0):
        for term, coeff in term_dict.items():
            qubits, paulis = zip(*term.to_list())
            pauli_list = []
            for pauli in paulis:
                if pauli == 'I':
                    pauli_list.append(Pauli.I)
                elif pauli == 'X':
                    pauli_list.append(Pauli.X)
                elif pauli == 'Y':
                    pauli_list.append(Pauli.Y)
                elif pauli == 'Z':
                    pauli_list.append(Pauli.Z)
                else:
                    EOFError()
            pbox = PauliExpBox(pauli_list, coeff * param)
            circ.add_pauliexpbox(pbox, [x[1][0] for x in qubits])
    ansatz = Circuit(n)
    t0 = ctime()
    add_excitation(ansatz, oplist)
    PauliSimp().apply(ansatz)
    print(f"TK Pauli Simp: {ctime()-t0}")
    t0 = ctime()
    FullPeepholeOptimise().apply(ansatz)
    print(f"TK O2: {ctime()-t0}")
    t0 = ctime()
    initial_map = {Qubit(i): Node(i) for i in range(n)}
    if n == 28:
        initial_map[Qubit(27)] = Node(29) # fix tket bug when handling CO2
    place_with_map(ansatz, initial_map)
    RoutingPass(mahattan_arch).apply(ansatz)
    DecomposeSwapsToCXs(mahattan_arch).apply(ansatz)
    DecomposeSingleQubitsTK1().apply(ansatz)
    print(f"CNOT: {ansatz.n_gates_of_type(OpType.CX)}, Single: {ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX)}, Total: {ansatz.n_gates}, Depth: {ansatz.depth()}")
    metric_tketO2 = {'CNOT_tketO2': ansatz.n_gates_of_type(OpType.CX), 'Single_tketO2': ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX), 'Total_tketO2': ansatz.n_gates, 'Depth_tketO2': ansatz.depth()}
    print("TK Route:", ctime()-t0)
    latency1 = ctime()-t0
    ansatz = Circuit(n)
    add_excitation(ansatz, oplist)
    PauliSimp().apply(ansatz)
    DecomposeSingleQubitsTK1().apply(ansatz)
    qstr = circuit_to_qasm_str(ansatz)
    qc = QuantumCircuit.from_qasm_str(qstr)
    t0 = ctime()
    qc = transpile(qc, basis_gates=['cx', 'u3'], coupling_map=mahattan_coupling, initial_layout=list(range(n)), optimization_level=3)
    cnots, singles, depth = print_qc(qc)
    print("Qiskit L3:", ctime()-t0)
    latency2 = ctime()-t0
    metric_tketO2.update({
        'n_qubits': n,
        'CNOT': cnots,
        'Single': singles,
        'Total': cnots+singles,
        'Depth': depth,
        'qasm' : qstr,
        'latency1' : latency1,
        'latency2' : latency2
    })
    return metric_tketO2



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

    for mapper in ['jordan_wigner']:#, 'parity', 'bravyi_kitaev']:
        metrics_list = []
        print("+++++++++Our method+++++++++++")
        for i in range(0,k):
            print('UCCSD:', moles[i])
            parr = load_oplist(mapper, moles[i])
            metrics = TK_Mahattan(parr)
            metrics_list.append((moles[i], metrics))

            pickle_dump(metrics_list, f'runs_final/{mapper}/tket_data.pickle')
            print(metrics_list)
