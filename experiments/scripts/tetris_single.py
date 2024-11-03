from bench_utils import *
import json

json_fname = '../../benchmarks/uccsd_json/CH2_cmplt_BK_sto3g.json'
# json_fname = '../../benchmarks/uccsd_json/LiH_frz_P_sto3g.json'


with open(json_fname, 'r') as f:
    data = json.load(f)

circ = tetris_pass(data['paulis'], data['coeffs'],
                   coupling_map=Manhattan)

circ = tetris_pass(data['paulis'], data['coeffs'],
                   coupling_map=Sycamore)

circ = tetris_pass(data['paulis'], data['coeffs'],
                   coupling_map=All2all)
