import sys
import warnings

import matplotlib.pyplot as plt

import qiskit
from qiskit_ibm_provider import IBMProvider
from qiskit import QuantumCircuit, execute,  Aer
from qiskit.result import marginal_counts
from qiskit.providers.ibmq.job import job_monitor
from qiskit.tools.visualization import plot_histogram
IBMProvider()

# Fill in your hub/group/provider
provider = IBMProvider(instance="ibm-q/open/main") 
# choose a system that supports mid-circuit measurements
backend = provider.get_backend('simulator_statevector')
# simulator = Aer.get_backend('qasm_simulator')

config = backend.configuration()
n_qubits = config.n_qubits

qc_simp = QuantumCircuit(2,4)
qc_simp.h(0)
qc_simp.cx(0, 1)
# qc_simp.measure([0,1], [0,1])
qc_simp.measure([1], [1])
qc_simp.barrier([0, 1])
qc_simp.h(0)

qc_simp.measure([0, 1], [0, 1])

# qc_simp.barrier([0, 1])
# qc_simp.x(1)
# qc_simp.measure([0,1], [2,3])

simp_job = execute(qc_simp, backend=backend)
print(simp_job.job_id())
job_monitor(simp_job)
simp_counts1 = marginal_counts(simp_job.result(), indices=[0, 1]).get_counts()
# simp_counts2 = marginal_counts(simp_job.result(), indices=[2,3]).get_counts()
print("Meas 1 result: ", simp_counts1)
# print("Meas 2 result: ", simp_counts2)
