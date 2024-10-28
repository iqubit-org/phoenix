import os
import csv
import sys
import pickle
package_directory = os.path.dirname(os.path.abspath(__file__))

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def is_code_reduced(code):
    if code in ['melbourne', 'mahattan']:
        reduced = True
    else:
        reduced = False
    return reduced

def load_coupling_map(code):
    reduced = is_code_reduced(code)
    pth = os.path.join(package_directory, '../core/arch/data', 'ibmq_'+code+'_calibrations.csv')
    cgs = []
    n = 0
    with open(pth, 'r') as cf:
        g = csv.DictReader(cf, delimiter=',', quotechar='\"')
        for i in g:
            cxval = ""
            for j in i.keys():
                if j.find('CNOT') != -1:
                    cxval = i[j]
            n += 1
            if ';' in cxval:
                dc = ';'
            else:
                dc = ','
            for j in cxval.split(dc):
                cgs.append(j.strip())
    coupling = []
    if reduced:
        n -= 1
    for i in cgs:
        si1 = i.find('_')
        si2 = i.find(':')
        offset = 0
        if i[:2] == 'cx':
            offset = 2
        iq1 = int(i[offset:si1])
        iq2 = int(i[si1+1:si2])
        if (iq1 < n and iq2 < n) or not reduced:
            coupling.append([iq1, iq2])
    return coupling

def print_qc(qc, f=sys.stdout):
    from qiskit import transpile
    qc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=0)
    c = qc.count_ops()
    t0 = sum(c.values())
    if 'cx' in c:
        t1 = c['cx']
    else:
        t1 = 0
    if f != None:
        print('CNOT: ' + str(t1) + ", Single: " + str(t0-t1) + ', Total: ' + str(t0) + ', Depth:', qc.depth(), file=f, flush=True)
    return t1, t0-t1, qc.depth()
    
