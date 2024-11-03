import numpy as np
import random
from .mypauli import pauliString

def gene_random_oplist(qubit, order=3, seed=10):
    s = min(np.math.factorial(qubit), int(qubit**order), int(4**qubit))
    random.seed(seed)
    oplist = []
    ps = ['I', 'X', 'Y', 'Z']
    while len(oplist) < s:
        pss = ""
        for i in range(qubit):
            pss += random.choice(ps)
        if pss not in oplist:
            oplist.append(pss)
    res = []
    for i in oplist:
        res.append([pauliString(i, coeff=1.0)])
    return res

def gene_cond_random_oplist(qubit, num, seed=10):
    s = min(np.math.factorial(qubit), num, int(4**qubit))
    random.seed(seed)
    oplist = []
    ps = "XYZ"
    while len(oplist) < s:
        t = random.randint(1, qubit//2)
        t0 = random.randint(0, t)
        pss = ""
        for i in range(qubit-2*t):
            pss += random.choice(ps)
        pss = 2*t0*'I' + pss + 2*(t-t0)*'I'
        if pss not in oplist:
            oplist.append(pss)
    res = []
    for i in oplist:
        res.append([pauliString(i, coeff=1.0)])
    return res

def gene_oplist_strlist(sl):
    return [[pauliString(i, coeff=1.0)] for i in sl] 

if __name__ == "__main__":
    from myutil import count
    count(gene_cond_random_oplist(40, 1600))
    count(gene_cond_random_oplist(60, 3600))
    count(gene_cond_random_oplist(80, 6400))
    

if __name__ == "__main1__":
    #measure time
    import time
    start = time.time()
    #gene_random_oplist1(70) # Time consumed: 117.7322154045105
    #gene_random_oplist(80) # Time consumed: 1942.9519095420837
    #gene_random_oplist(70) # Time consumed: 859.5379593372345
    #gene_random_oplist1(80) # Time consumed: 212.92153787612915
    #gene_random_oplist1(100) # Time consumed: 522.8441543579102  
    print("Time consumed:", time.time()-start)
