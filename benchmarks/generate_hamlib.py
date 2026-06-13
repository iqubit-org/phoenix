"""
This script is to generate .json and .qasm files from the original 100_representative.json file which is 100 selected benchmarks from HamLib.
"""
import os
import json


JSON_DIR = "hamlib"


with open("100_representative.json", "r") as f:
    data = json.load(f)


for ham in data:
    program_name = "{}-{}".format(ham["ham_problem"], ham["ham_instance"].strip("/"))
    program_name = program_name.split(",")[0]
    program_name = program_name.replace("ham_", "")
    program_name = program_name.replace("ham-", "")
    print(program_name)

    json_fname = os.path.join(JSON_DIR, ham["ham_category"], program_name + ".json")

    # save to json
    with open(json_fname, "w") as f:
        json_body = {
            "num_qubits": ham["ham_qubits"],
            "num_terms": ham["ham_terms"],
            "paulis": ham["ham_hamlib_hamiltonian_terms"],
            "coeffs": ham["ham_hamlib_hamiltonian_coefficients"],
        }
        json.dump(json_body, f, indent=4)
