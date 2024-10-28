import io, os, sys, time
import subprocess
from subprocess import Popen, PIPE
from benchmark.offline import *
from parallel_bl import *
from qiskit import QuantumCircuit, transpile
from synthesis_SC import *
from synthesis_FT import *
from tools import *
from arch import *
from config import test_scale

old_cwd = os.getcwd()
set_cwd()
ctime = time.time

# PH Mahattan device method
def PH_Mahattan(parr):
    lnq = len(parr[0][0])
    length = lnq // 2 # `length' is a hyperparameter, and can be adjusted for best performance. Here we keep `length' fixed for simplicity.
    coup = load_coupling_map('manhattan')
    t0 = ctime()
    a2 = depth_oriented_scheduling(parr, length=length, maxiter=30)
    qc = block_opt_SC(a2, arch='manhattan')
    pnq = qc.num_qubits
    print('PH, Time costed:', ctime()-t0, flush=True)
    t0 = ctime()
    qc2 = transpile(qc, basis_gates=['u3', 'cx'], coupling_map=coup, initial_layout=list(range(pnq)), optimization_level=3)
    print_qc(qc2)
    print('Qiskit L3, Time costed:', ctime()-t0, flush=True)

############################
# QAOA Part
############################
print('PH passes, Our schedule, Our synthesis, mahattan', flush=True)
from benchmark.qaoa import *
if test_scale == 'small':
    seeds = [83,193]
else:
    seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
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

############################
# QAOA Compiler by Alam et al.
############################
print("QAOA-Compiler by Alam et al.")
os.chdir("./QAOA-Compiler")
root = 'circ'

def console(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE)
    out, err = p.communicate()
    return (p.returncode, out, err)

def run_qaoa_compiler(root, files):
    f = sys.stdout
    for i in files:
        cmd = f"python main.py -d Mahattan.json -ci {root}/{i} -co examples/Config.json  -p VIC"
        t0 = time.time()
        _, out, _ = console(cmd)
        name = i[:-5]
        c = out.decode().split("\n")[-2]
        r = c.split(',')
        depth = int(r[0].split(':')[-1].strip())
        cnot = int(r[1].split(':')[-1].strip())
        single = int(r[2].split(':')[-1].strip())
        total = int(r[3].split(':')[-1].strip())
        esp = float(r[4].split(':')[-1].strip())
        print(f"{name}, CNOT: {cnot}, Single: {single}, Total: {total}, Depth: {depth}, TIME: {time.time()-t0}", file=f, flush=True)
# run regular graph
files = []
for cfg in [[4,20],[8,20],[12, 20]]:
    for seed in seeds:
        files.append(f"reg-{cfg[0]}-{cfg[1]}-{seed}.json")
run_qaoa_compiler('circ', files)
# run ER graph
files = []
for cfg in [[20,0.1],[20,0.3],[20, 0.5]]:
    for seed in seeds:
        files.append(f"er-{cfg[0]}-0-{int(cfg[1]*10)}-{seed}.json")
run_qaoa_compiler('circ1', files)

os.chdir(old_cwd)