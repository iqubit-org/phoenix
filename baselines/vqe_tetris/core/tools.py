def calc_qc(qc):
    from qiskit import transpile
    # qc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=0)
    c = qc.count_ops()
    t0 = sum(c.values())
    # print(c)
    t2 = 0
    t1 = 0
    if 'cx' in c:
        t1 = c['cx']
    if 'swap' in c:
        t2 = c['swap']
    return t1+t2*3, t0-t1-t2, qc.depth()
    
def count_sched(sched):
    c = 0
    s = 0
    nq = len(sched[0][0][0]) # sched : pauli layers
    ns = 0
    for i in sched:
        for k in i:
            for j in k:
                c += 2*max(nq - 1- j.count('I'), 0)
                s += 2*(nq - j.count('I') - j.count('Z')) + 1 # accurate
                ns += 1
    return nq, ns, c, s
def count_oplist(parr):
    sched = [[i] for i in parr]
    return count_sched(sched)

import sys
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
    
import os    
def set_cwd():
    def get_script_path():
        return os.path.dirname(os.path.realpath(sys.argv[0]))
    os.chdir(get_script_path())