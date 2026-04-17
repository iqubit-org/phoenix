import sys

sys.path.append("../..")

import json
import os
import phoenix
from phoenix.primitive import simplification

with open("../demo.json", "r") as f:
    data = json.load(f)

bsf = phoenix.models.BSF(data["paulis"])
print(bsf.paulis)
print(bsf.mat)
print("cost:", simplification.heuristic_bsf_cost(bsf))

n = bsf.num_qubits

print(n)
for i in range(n):
    bsf_ = bsf.apply_s(i)
    print("cost after applying S({}): {}".format(i, simplification.heuristic_bsf_cost(bsf_)))

print(n)
for i in range(n):
    bsf_ = bsf.apply_h(i)
    print("cost after applying S({}): {}".format(i, simplification.heuristic_bsf_cost(bsf_)))
