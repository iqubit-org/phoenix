#############################
# Generate UCCSD and Molecule Hamitonian
#############################
import pickle

import benchmark.uccsd as uccsd 
# UCCSD-8
parr = uccsd.lih_oplist()
with open('LiH_UCCSD.pickle', 'wb') as f:
    pickle.dump(parr, f)
# UCCSD-12
parr = uccsd.beh2_oplist()
with open('BeH2_UCCSD.pickle', 'wb') as f:
    pickle.dump(parr, f)
# UCCSD-16
parr = uccsd.ch4_oplist()
with open('CH4_UCCSD.pickle', 'wb') as f:
    pickle.dump(parr, f)
# UCCSD-20
parr = uccsd.mgh_oplist()
with open('MgH_UCCSD.pickle', 'wb') as f:
    pickle.dump(parr, f)
# UCCSD-24
parr = uccsd.licl_oplist()
with open('LiCl_UCCSD.pickle', 'wb') as f:
    pickle.dump(parr, f)
# UCCSD-28
parr = uccsd.co2_oplist()
with open('CO2_UCCSD.pickle', 'wb') as f:
    pickle.dump(parr, f)

import benchmark.molecule as molecule
# N2
parr = molecule.gene_molecule_oplist(molecule.N2)
with open('N2.pickle', 'wb') as f:
    pickle.dump(parr, f)
# H2S
parr = molecule.gene_molecule_oplist(molecule.H2S)
with open('H2S.pickle', 'wb') as f:
    pickle.dump(parr, f)
# MgO
parr = molecule.gene_molecule_oplist(molecule.MgO)
with open('MgO.pickle', 'wb') as f:
    pickle.dump(parr, f)
# CO2
parr = molecule.gene_molecule_oplist(molecule.CO2)
with open('CO2.pickle', 'wb') as f:
    pickle.dump(parr, f)
# NaCl
parr = molecule.gene_molecule_oplist(molecule.NaCl)
with open('NaCl.pickle', 'wb') as f:
    pickle.dump(parr, f)