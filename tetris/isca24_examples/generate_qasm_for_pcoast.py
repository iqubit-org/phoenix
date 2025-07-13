from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevSuperFastMapper, BravyiKitaevMapper
from qiskit import QuantumCircuit, transpile, qasm2
import pickle
import ast
import pdb

mapper = JordanWignerMapper()
# mapper = ParityMapper()
# mapper = BravyiKitaevMapper()

drivers = {'LiH': PySCFDriver(atom="Li .0 .0 .0; H .0 .0 1.3", basis='sto3g'),
            'BeH2': PySCFDriver(atom="Be .0 .0 .0; H .0 .0 -0.76; H .0 .0 0.76", basis='sto3g'),
            'CH4': PySCFDriver(atom="C .0 .0 .0; H .0 .0 1.0; H .0 .0 -1.0; H .0 1.0 .0; H .0 -1.0 .0", basis='sto3g'),
            'MgH2': PySCFDriver(atom="Mg .0 .0 .0; H .0 .0 -1.3; H .0 .0 1.3", basis='sto3g'),
            'LiCl': PySCFDriver(atom="Li .0 .0 .0; Cl .0 .0 -1.5", basis='sto3g'),
            'CO2': PySCFDriver(atom="C .0 .0 .0; O .0 .0 1.0; O .0 .0 -1.0", basis='sto3g')}

for mole in ['LiH', 'BeH2', 'CH4', 'MgH2']:
    driver = drivers[mole]

    problem = driver.run()


    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )
    ansatz.assign_parameters({param: 1. for param in ansatz.parameters})
    qc = transpile(ansatz, basis_gates=['cx', 'u3'], optimization_level=0)

    pdb.set_trace()

    print(qc.qasm())
