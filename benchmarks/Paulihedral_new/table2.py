import sys, time
from benchmark.offline import *
from parallel_bl import *
from qiskit import QuantumCircuit, transpile
import synthesis_SC
import synthesis_FT
from tools import *
from arch import *
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit, Node
from pytket import Circuit, OpType
from pytket.circuit import PauliExpBox
from pytket.passes import DecomposeBoxes, PauliSimp, DecomposeSingleQubitsIBM, O2Pass, RoutingPass, DecomposeSwapsToCXs
from pytket.qasm import circuit_from_qasm, circuit_from_qasm_str, circuit_from_qasm_io, circuit_to_qasm_str, circuit_to_qasm
import time, sys, os
from t_arch import *
from pytket.routing import place_with_map
from pytket.routing import Placement, LinePlacement, GraphPlacement, NoiseAwarePlacement
from config import test_scale

old_cwd = os.getcwd()
set_cwd()
ctime = time.time

# PH Mahattan device method
def PH_Mahattan(parr):
    print('PH passes, Our schedule, Our synthesis, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a2 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = synthesis_SC.block_opt_SC(a2, arch='manhattan')
    pnq = qc.num_qubits
    print('PH, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    print_qc(qc2)
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    ansatz = circuit_from_qasm_str(qc1.qasm())
    n = ansatz.n_qubits
    t0 = ctime()
    O2Pass().apply(ansatz)
    print("TKet O2:", ctime()-t0)
    t0 = ctime()
    initial_map = {Qubit(i): Node(i) for i in range(n)}
    if n == 28:
        initial_map[Qubit(27)] = Node(29) # fix tket bug when handling CO2
    place_with_map(ansatz, initial_map)
    RoutingPass(mahattan_device).apply(ansatz)
    DecomposeSwapsToCXs(mahattan_device).apply(ansatz)
    DecomposeSingleQubitsIBM().apply(ansatz)
    print(f"CNOT: {ansatz.n_gates_of_type(OpType.CX)}, Single: {ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX)}, Total: {ansatz.n_gates}, Depth: {ansatz.depth()}")
    print("TKet Route:", ctime()-t0)
# TKet Mahattan device method
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
            qubits, paulis = zip(*term.to_dict().items())
            pbox = PauliExpBox(paulis, coeff * param)
            circ.add_pauliexpbox(pbox, qubits)
    ansatz = Circuit(n)
    t0 = ctime()
    add_excitation(ansatz, oplist)
    PauliSimp().apply(ansatz)
    print(f"TK Pauli Simp: {ctime()-t0}")
    t0 = ctime()
    O2Pass().apply(ansatz)
    print(f"TK O2: {ctime()-t0}")
    t0 = ctime()
    initial_map = {Qubit(i): Node(i) for i in range(n)}
    if n == 28:
        initial_map[Qubit(27)] = Node(29) # fix tket bug when handling CO2
    place_with_map(ansatz, initial_map)
    RoutingPass(mahattan_device).apply(ansatz)
    DecomposeSwapsToCXs(mahattan_device).apply(ansatz)
    DecomposeSingleQubitsIBM().apply(ansatz)
    print(f"CNOT: {ansatz.n_gates_of_type(OpType.CX)}, Single: {ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX)}, Total: {ansatz.n_gates}, Depth: {ansatz.depth()}")
    print("TK Route:", ctime()-t0)
    ansatz = Circuit(n)
    add_excitation(ansatz, oplist)
    PauliSimp().apply(ansatz)
    DecomposeSingleQubitsIBM().apply(ansatz)
    qstr = circuit_to_qasm_str(ansatz)
    qc = QuantumCircuit.from_qasm_str(qstr)
    t0 = ctime()
    qc = transpile(qc, basis_gates=['cx', 'u3'], coupling_map=mahattan_coupling, initial_layout=list(range(n)), optimization_level=3)
    print_qc(qc)
    print("Qiskit L3:", ctime()-t0)

############################
# UCCSD Part
############################
# UCCSD- 8,12,16,20,24,28
moles = ['LiH', 'BeH2', 'CH4', 'MgH', 'LiCl', 'CO2']
orbital = [8, 12, 16, 20, 24, 28]
if test_scale == 'small':
    k = 2
else:
    k = 6
for i in range(0,k):
    print('UCCSD:', orbital[i])
    parr = load_oplist(moles[i], benchmark='uccsd')
    PH_Mahattan(parr)
    TK_Mahattan(parr)

############################
# QAOA Part
############################
from benchmark.qaoa import *
if test_scale == 'small':
    seeds = [83,193]
else:
    seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
print("QAOA, seeds=", seeds)
nseed = len(seeds)
for cfg in [[4,20],[8,20],[12, 20]]:
    for seed in seeds:
        G = rand_reg(cfg[0], cfg[1], seed=seed)
        parr = gene_qaoa_oplist(G)
        print('Regular graph:', cfg, 'Seed:', seed)
        PH_Mahattan(parr)
        TK_Mahattan(parr)

for cfg in [[20,0.1],[20,0.3],[20, 0.5]]:
    for seed in seeds:
        G = rand_er(cfg[0], cfg[1], seed=seed)
        parr = gene_qaoa_oplist(G)
        print('ER graph:', cfg, 'Seed:', seed)
        PH_Mahattan(parr)
        TK_Mahattan(parr)
for n in [4,5]:
    for seed in seeds:
        parr = tsp_oplist(n, seed=seed)
        print('TSP:', n, 'Seed:', seed)
        PH_Mahattan(parr)
        TK_Mahattan(parr)

# PH no device method
def PH(parr):
    print('PH passes, Our schedule, Our synthesis', flush=True)
    nq = len(parr[0][0])
    length = nq//2 # `length' is a hyperparameter, and can be adjusted for best performance
    t0 = ctime()
    a1 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = synthesis_FT.block_opt_FT(a1)
    print('PH, Time costed:', ctime()-t0, flush=True)
    qc1 = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=0)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    print_qc(qc2)
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    ansatz = circuit_from_qasm_str(qc1.qasm())
    t0 = ctime()
    O2Pass().apply(ansatz)
    DecomposeSingleQubitsIBM().apply(ansatz)
    print(f"CNOT: {ansatz.n_gates_of_type(OpType.CX)}, Single: {ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX)}, Total: {ansatz.n_gates}, Depth: {ansatz.depth()}")
    print("TKet O2:", ctime()-t0)
# TKet no device method
def TK(parr):
    print('TK passes', flush=True)
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
            qubits, paulis = zip(*term.to_dict().items())
            pbox = PauliExpBox(paulis, coeff * param)
            circ.add_pauliexpbox(pbox, qubits)
    ansatz = Circuit(n)
    t0 = ctime()
    add_excitation(ansatz, oplist)
    PauliSimp().apply(ansatz)
    print(f"TK Pauli Simp: {ctime()-t0}")
    t0 = ctime()
    O2Pass().apply(ansatz)
    print(f"CNOT: {ansatz.n_gates_of_type(OpType.CX)}, Single: {ansatz.n_gates-ansatz.n_gates_of_type(OpType.CX)}, Total: {ansatz.n_gates}, Depth: {ansatz.depth()}")
    print("TK O2:", ctime()-t0)
    ansatz = Circuit(n)
    add_excitation(ansatz, oplist)
    PauliSimp().apply(ansatz)
    DecomposeSingleQubitsIBM().apply(ansatz)
    qstr = circuit_to_qasm_str(ansatz)
    qc = QuantumCircuit.from_qasm_str(qstr)
    t0 = ctime()
    qc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=3)
    print_qc(qc)
    print("Qiskit L3:", ctime()-t0)

############################
# Molecule Hamiltonian Part
############################
moles = ['N2', 'H2S', 'MgO', 'CO2', 'NaCl']
if test_scale == 'small':
    k = 2
else:
    k = 5
for i in range(0,k):
    print('Molecule:', moles[i])
    parr = load_oplist(moles[i], benchmark='molecule')
    PH(parr)
    TK(parr)

############################
# Ising Part
############################
from benchmark.ising import *
ising_parr = [gene_dot_1d(29), gene_dot_2d(4,5, interaction='Z'), gene_dot_3d(1,2,4, interaction='Z')]
heisen_parr = [gene_dot_1d(29, interaction='Z')+gene_dot_1d(29, interaction='X')+gene_dot_1d(29, interaction='Y'), gene_dot_2d(4,5, interaction='Z')+gene_dot_2d(4,5, interaction='Y')+gene_dot_2d(4,5, interaction='X'), gene_dot_3d(1,2,4, interaction='Z')+gene_dot_3d(1,2,4, interaction='Y')+gene_dot_3d(1,2,4, interaction='X')]

dim = 3
for i in range(dim):
    parr = ising_parr[i]
    print(f"Ising {i+1}D")
    PH(parr)
    TK(parr)
for i in range(dim):
    parr = heisen_parr[i]
    print(f"Heisenberg {i+1}D")
    PH(parr)
    TK(parr)

############################
# Random Part
############################
from benchmark.hami import *
if test_scale == 'small':
    seeds = [193]
    k = 1
else:
    seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
    k = 6
print("Random, seeds=", seeds)
psize = [30, 40, 50, 60, 70, 80]

for seed in seeds:
    for nq in psize[0:k]:
        parr = gene_cond_random_oplist(nq,5*nq*nq, seed=seed)
        print(f"Random-{nq}, seed:{seed}")
        PH(parr)
        TK(parr)

os.chdir(old_cwd)