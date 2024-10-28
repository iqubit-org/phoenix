from qiskit import QuantumCircuit, transpile, execute
import qiskit
from qubit_place import synth_qaoa1, qiskit_synthesis
from benchmark.qaoa import *
from tools import *
from qiskit.test.mock import FakeMelbourne
from qiskit.tools.monitor import job_monitor
from arch import *
import numpy as np

set_cwd()

# qiskit.IBMQ.save_account('b5f84fb0f26117e037cfab1d4a026a03165d687c635825e18740447d4a45d9300a4e13d1054f27dcf61b4acd54fb6a4fdf7c9f56e96bf48abd51f1ef593120bc', overwrite=True)
# qiskit.IBMQ.load_account()
# provider = qiskit.IBMQ.get_provider(hub='ibm-q-research', group='santabarbara-1', project='main')

device_name = 'ibmq_16_melbourne'
backend = FakeMelbourne()

def get_device_graph(backend):
    coup = []
    config = backend.configuration()
    for i in config.coupling_map:
        if i not in coup:
            coup.append(i)
            coup.append(list(reversed(i)))
    graph = graph_from_coupling(coup)
    return coup, graph

def gene_reg(deg, nodes, seed, gamma, beta):
    a = []
    G = rand_reg(deg, nodes, seed=seed)
    parr = gene_qaoa_oplist(G)
    coup, graph = get_device_graph(backend)
    a2 = [[[parr[i][0]]] for i in range(len(parr))]
    # print('Our qaoa synthesis, l3, melbourne')
    qc1 = synth_qaoa1(a2, graph=graph, gamma=gamma, beta=beta)
    qc1 = transpile(qc1, basis_gates=['u3', 'cx'], backend=backend, coupling_map=coup, optimization_level=3)
    qc1.measure_all()
    print_qc(qc1)
    # print('Qiskit synthesis, l3, melbourne')
    qc3 = qiskit_synthesis(a2, graph=graph, gamma=gamma, beta=beta)
    qc3 = transpile(qc3, basis_gates=['u3', 'cx'], backend=backend, coupling_map=coup, optimization_level=3)
    qc3.measure_all()
    print_qc(qc3)
    a += [qc1]*5+[qc3]*5
    return a
    
def gene_er(nodes, prob, seed, gamma, beta):
    a = []
    G = rand_er(nodes, prob, seed=seed)
    parr = gene_qaoa_oplist(G)
    coup, graph = get_device_graph(backend)
    a2 = [[[parr[i][0]]] for i in range(len(parr))]
    # print('Our qaoa synthesis, l3, melbourne')
    qc1 = synth_qaoa1(a2, graph=graph, gamma=gamma, beta=beta)
    qc1 = transpile(qc1, basis_gates=['u3', 'cx'], backend=backend, coupling_map=coup, optimization_level=3)
    qc1.measure_all()
    # print_qc(qc1)
    # print('Qiskit synthesis, l3, melbourne')
    qc3 = qiskit_synthesis(a2, graph=graph, gamma=gamma, beta=beta)
    qc3 = transpile(qc3, basis_gates=['u3', 'cx'], backend=backend, coupling_map=coup, optimization_level=3)
    qc3.measure_all()
    # print_qc(qc3)
    a += [qc1]*5+[qc3]*5
    return a

# backend = provider.get_backend(device_name) # uncomment this for real system test
# deg, nodes, seed, gamma, beta
reg_config = [[4,7,12,-0.4896551724137931,0.7409909909909911],[4,8,12,-0.6,1.0500000000000003],[4,9,12,-0.6,0.7725225225225225],[4,10,12,-0.593103448275862,-2.3554054054054054]]
er_config = [[7,0.5,12,-1.5908026755852842,-0.600501672240803],[8,0.5,12,-1.6750836120401338,0.8322742474916387],[9,0.5,12,1.6329431438127089,1.6118729096989965],[10,0.5,12,0.49515050167224084,3.0657190635451506]]
n = len(reg_config)
import time
for i in [0]: # range(n):
    a = []
    rc = reg_config[i]
    ec = er_config[i]
    a += gene_reg(rc[0], rc[1], rc[2], rc[3], rc[4])
    a += gene_er(ec[0], ec[1], ec[2], ec[3], ec[4])
    # job = execute(a, backend, shots=8192)
    # print(job.job_id())
    # time.sleep(20)