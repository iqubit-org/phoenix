import pdb
import pickle

def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

ws = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0]

ls = []

for w in ws:
    filename = f'runs_final/ablation/sycamore_mole{5}_different_w_{w}.pickle'
    ls = ls + pickle_load(filename)
    
filename = f'runs_final/ablation/sycamore_mole{5}_different_w.pickle'
pickle_dump(ls, filename)
