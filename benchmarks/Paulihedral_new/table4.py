import sys, time, os
from benchmark.offline import *
from parallel_bl import *
from qiskit import QuantumCircuit, transpile
import synthesis_SC
import synthesis_FT
from tools import *
from arch import *
from config import test_scale

old_cwd = os.getcwd()
set_cwd()
ctime = time.time

# PH Mahattan device method
def PH_Mahattan(parr):
    # print('PH passes, mahattan', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a1 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = synthesis_SC.block_opt_SC(a1, arch='manhattan')
    # print('DO + Block, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(qc.num_qubits)), optimization_level=3)
    c0, s0, d0 = print_qc(qc2, None)
    print('DO + Block:', end=" ")
    print(f"CNOT: {c0}, Single: {s0}, Total: {c0+s0}, Depth: {d0}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)
    qc = synthesis_SC.block_opt_SC(a2, arch='manhattan')
    # print('GCO + Block, Time costed:', ctime()-t0, flush=True)
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(qc.num_qubits)), optimization_level=3)
    c1, s1, d1 = print_qc(qc2, None)
    print('GCO + Block:', end=" ")
    print(f"CNOT: {c1}, Single: {s1}, Total: {c1+s1}, Depth: {d1}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    a1 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = synthesis_SC.qiskit_synthesis(a1, initial_layout=list(range(lnq)), arch='manhattan')
    # print('DO + Qiskit, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(qc.num_qubits)), optimization_level=3)
    c2, s2, d2 = print_qc(qc2, None)
    print('DO + Qiskit:', end=" ")
    print(f"CNOT: {c2}, Single: {s2}, Total: {c2+s2}, Depth: {d2}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)
    qc = synthesis_SC.qiskit_synthesis(a2, initial_layout=list(range(lnq)), arch='manhattan')
    # print('GCO + Qiskit, Time costed:', ctime()-t0, flush=True)
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(qc.num_qubits)), optimization_level=3)
    c3, s3, d3 = print_qc(qc2, None)
    print('GCO + Qiskit:', end=" ")
    print(f"CNOT: {c3}, Single: {s3}, Total: {c3+s3}, Depth: {d3}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    # compute percentage
    print("DO vs GCO", end=" ")
    print(f"CNOT: {(c0/c1-1):.2%}, Single: {(s0/s1-1):.2%}, Total: {((c0+s0)/(c1+s1)-1):.2%}, Depth: {(d0/d1 -1):.2%}")
    print("Block-Wise Compilation vs Qiskit")
    print(f"CNOT: {(c0/c2-1):.2%}, Single: {(s0/s2-1):.2%}, Total: {((c0+s0)/(c2+s2)-1):.2%}, Depth: {(d0/d2 -1):.2%}")

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

############################
# QAOA Part
############################
from benchmark.qaoa import *
if test_scale == 'small':
    seeds = [83,193]
else:
    seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
print("QAOA, Seeds=", seeds)
nseed = len(seeds)
for cfg in [[4,20],[8,20],[12, 20]]:
    for seed in seeds:
        G = rand_reg(cfg[0], cfg[1], seed=seed)
        parr = gene_qaoa_oplist(G)
        print('Regular graph:', cfg, 'Seed:', seed)
        PH_Mahattan(parr)
for cfg in [[20,0.1],[20,0.3],[20, 0.5]]:
    for seed in seeds:
        G = rand_er(cfg[0], cfg[1], seed=seed)
        parr = gene_qaoa_oplist(G)
        print('ER graph:', cfg, 'Seed:', seed)
        PH_Mahattan(parr)
for n in [4, 5]:
    for seed in seeds:
        parr = tsp_oplist(n, seed=seed)
        print('TSP:', n, 'Seed:', seed)
        PH_Mahattan(parr)

# PH no device method
def PH(parr):
    # print('PH passes', flush=True)
    lnq = len(parr[0][0])
    length = lnq // 2
    t0 = ctime()
    a1 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = synthesis_FT.block_opt_FT(a1)
    # print('DO + Block, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    c0, s0, d0 = print_qc(qc2, None)
    print('DO + Block:', end=" ")
    print(f"CNOT: {c0}, Single: {s0}, Total: {c0+s0}, Depth: {d0}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)
    qc = synthesis_FT.block_opt_FT(a2)
    # print('GCO + Block, Time costed:', ctime()-t0, flush=True)
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    c1, s1, d1 = print_qc(qc2, None)
    print('GCO + Block:', end=" ")
    print(f"CNOT: {c1}, Single: {s1}, Total: {c1+s1}, Depth: {d1}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    a1 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = synthesis_FT.qiskit_synthesis(a1)
    # print('DO + Qiskit, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    c2, s2, d2 = print_qc(qc2, None)
    print('DO + Qiskit:', end=" ")
    print(f"CNOT: {c2}, Single: {s2}, Total: {c2+s2}, Depth: {d2}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    a2 = gate_count_oriented_scheduling(parr)
    qc = synthesis_FT.qiskit_synthesis(a2)
    # print('GCO + Qiskit, Time costed:', ctime()-t0, flush=True)
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3)
    c3, s3, d3 = print_qc(qc2, None)
    print('GCO + Qiskit:', end=" ")
    print(f"CNOT: {c3}, Single: {s3}, Total: {c3+s3}, Depth: {d3}")
    # print('Qiskit L3, Time costed:', ctime()-t0, flush=True)
    # compute percentage
    print("DO vs GCO", end=" ")
    print(f"CNOT: {(c0/c1-1):.2%}, Single: {(s0/s1-1):.2%}, Total: {((c0+s0)/(c1+s1)-1):.2%}, Depth: {(d0/d1 -1):.2%}")
    print("Block-Wise Compilation vs Qiskit")
    print(f"CNOT: {(c0/c2-1):.2%}, Single: {(s0/s2-1):.2%}, Total: {((c0+s0)/(c2+s2)-1):.2%}, Depth: {(d0/d2 -1):.2%}")

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

for i in range(dim):
    parr = heisen_parr[i]
    print(f"Heisenberg {i+1}D")
    PH(parr)

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
        print(f"Random-{nq}")
        PH(parr)

os.chdir(old_cwd)