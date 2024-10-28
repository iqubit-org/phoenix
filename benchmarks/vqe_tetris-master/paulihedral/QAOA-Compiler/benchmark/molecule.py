from qiskit.chemistry.transformations import (FermionicTransformation, FermionicTransformationType, FermionicQubitMappingType)
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from .mypauli import pauliString
# qubit: 30. num_pauli: 16122
CO2 = [['O', [1.4, 0., 0.]],['C', [0., 0., 0.]],['O', [-1.4, 0., 0.]]]
# qubit: 36, num_pauli: 67859
NaCl = [['Na', [0., -1.5, -1.5]],['Cl', [1.5, -1.5, -1.5]]]
# qubit: 18, num_pauli_string: 2740
CH4 = [['C', [0.,1.0,1.0]],['H', [0.051054399,0.051054399,0.051054399]],
    ['H', [1.948945601,1.948945601,0.051054399]],
    ['H', [0.051054399,1.948945601,1.948945601]],
    ['H', [1.948945601,0.051054399,1.948945601]]]
LiH = [['Li', [0,0,0]],['H', [0,0,0.76]]]
# 14, 1086
H2O = [['O', [0.,0.,0.]],['H', [0.95,-0.55,0.]],['H', [-0.95,-0.55,0.]]]
def get_qubit_op(geo):
    molecule = Molecule(geometry=geo, charge=0, multiplicity=1)
    driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
    fermionic_transformation = FermionicTransformation(transformation=FermionicTransformationType.FULL, qubit_mapping=FermionicQubitMappingType.BRAVYI_KITAEV, two_qubit_reduction=False, freeze_core=False)
    qubit_op, _ = fermionic_transformation.transform(driver)
    # print(qubit_op.num_qubits, ",", len(qubit_op))
    # print(qubit_op.primitive_strings())
    # for i in qubit_op:
    #     print(i.primitive)
    # print(fermionic_transformation.molecule_info)
    return qubit_op

def gene_molecule_oplist(atom_geo):
    qubit_op = get_qubit_op(atom_geo)
    oplist = []
    for i in qubit_op:
        oplist.append([pauliString(i.primitive.to_label(), coeff=i.coeff)])
    return oplist

def gene_h20_oplist():
    return gene_molecule_oplist(H2O)

if __name__ == "__main__":
    from myutil import count, test_func    
    # parr = gene_molecule_oplist(H2O)
    # test_func(parr, "H2O")
    # count(parr)
    # parr = gene_molecule_oplist(CH4)
    # test_func(parr, "CH4")
    # count(parr)
    import time
    t0 = time.time()
    parr = gene_molecule_oplist(NaCl)
    print('Gene time:', time.time()-t0)
    test_func(parr, "NaCl")
    count(parr)
    
if __name__ == "__main1__":
    gene_molecule_oplist(H2O)
