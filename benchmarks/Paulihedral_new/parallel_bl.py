def pXOR(ps1, ps2):
    s = ""
    for i in range(len(ps1)):
        if ps1[i] == "I":
            s += ps2[i]
        elif ps2[i] == "I":
            s += ps1[i]
    return s
def pDiff(ps1, ps2):
    for i in range(len(ps1)):
        if ps1[i] != "I" and ps2[i] != "I":
            return False
    return True
def pLen(ps1):
    return len(ps1) - ps1.count('I')
def pINC(ps1, ps2): # ps1 includes ps2
    for i in range(len(ps2)):
        if ps2[i] != "I" and ps1[i] == "I":
            return False
    return True
def pLatency(ps, mode=0):
    return pLen(ps)
def pOR(ps1, ps2):
    s = ""
    for i in range(len(ps1)):
        if ps1[i] == "I":
            s += ps2[i]
        else:
            s += ps1[i]
    return s

class pauli_block: 
    def __init__(self, pb, idx=0):
        self.block = pb
        # if len(pb) > 0: # assume no empty
        self.nq = len(pb[0]) # num_qubit
        s = "I"*self.nq
        for i in pb:
            s = pOR(s, i.ps)
        self.mstr = s
        self.latency = 0
        for i in pb:
            self.latency += max(2*(self.nq - i.count('I')) - 1, 0) # it is an upper bound
            if self.nq - i.count('I') - i.count('Z') > 0:
                # self.latency += 2
                self.latency += 1
        self.len = pLen(self.mstr)
        self.id = idx
    def __repr__(self):
        s = "["
        s += self.block[0].ps
        for i in self.block[1:]:
            s += ", " + i.ps
        s += "]"
        return s
            
def generate_templates(pb, lt=2):
    ps = pb.mstr
    nq = pb.nq
    l = pb.latency
    idx0 = 0
    idx1 = 0
    t = []
    while idx0 < nq:
        while idx1 < len(ps) and ps[idx1] == "I":
            idx1 += 1
        if (idx1 - idx0) >= 2:
            ts = (idx0-0)*'I'+(idx1-idx0)*'X'+(nq-idx1)*'I'
            t.append([ts, l])
        while idx1 < len(ps) and ps[idx1] != "I":
            idx1 += 1
        idx0 = idx1
    return t

def generate_templates1(ps, nq, latency, lt=2):
    l = latency
    idx0 = 0
    idx1 = 0
    t = []
    while idx0 < nq:
        while idx1 < len(ps) and ps[idx1] == "I":
            idx1 += 1
        if (idx1 - idx0) >= 2:
            ts = (idx0-0)*'I'+(idx1-idx0)*'X'+(nq-idx1)*'I'
            t.append([ts, l])
        while idx1 < len(ps) and ps[idx1] != "I":
            idx1 += 1
        idx0 = idx1
    return t

def size_order_bl(pbarr):
    d = {}
    for i in pbarr:
        if i.len in d.keys():
            d[i.len].append(i)
        else:
            d[i.len] = [i]
    return d
def mul_lexi_order_bl(pd):
    def _key(pb):
        s = 0
        ps = pb.mstr
        for i in ps:
            s *= 4
            if i == 'I':
                s += 0
            elif i == 'X':
                s += 1
            elif i == 'Y':
                s += 2
            elif i == 'Z':
                s += 3
        return s
    for i in pd.keys():
        pd[i] = sorted(pd[i], key=_key)
    return pd

# def lexi_order_bl(pbl):
#     def _key(pb):
#         s = 0
#         ps = pb.mstr
#         for i in ps:
#             s *= 4
#             if i == 'I':
#                 s += 0
#             elif i == 'X':
#                 s += 1
#             elif i == 'Y':
#                 s += 2
#             elif i == 'Z':
#                 s += 3
#         return s
#     pbl = sorted(pbl, key=_key)
#     return pbl
    
def mutual(ps1, ps2):
    s = 0
    for i in range(len(ps1)):
        if ps1[i] == ps2[i] and ps1[i] != 'I':
            s += 1
    return s

def gate_count_oriented_scheduling(parr):
    nq = len(parr[0][0]) # num_qubits
    parr_flat = []
    for i in parr:
        parr_flat.append(pauli_block(i))
    def _key(pb):
        s = 0
        ps = pb.mstr
        for i in ps:
            s *= 4
            if i == 'I':
                s += 0
            elif i == 'X':
                s += 1
            elif i == 'Y':
                s += 2
            elif i == 'Z':
                s += 3
        return s
    psl = sorted(parr_flat, key=_key)
    # layers --> blocks --> strings: 3 level
    return [[i.block] for i in psl]
    
lexi_order_bl = gate_count_oriented_scheduling

def parallel_order_size_bl(parr, maxiter=1):
    nq = len(parr[0][0]) # num_qubits
    parr_flat = []
    for i in parr:
        parr_flat.append(pauli_block(i))
    pd = size_order_bl(parr_flat)
    pd = mul_lexi_order_bl(pd)
    pdl = sorted(pd.keys(), key=lambda x:-x)
    pdix = []
    for i in range(0,nq+1):
        ix = []
        for j in range(len(pdl)):
            if i >= pdl[j]:
                ix.append(j)
        pdix.append(ix)
    pb_sort = []    
    ps_layers = []
    ps_occupy = []
    pb_remain = []
    for i in pdl:
        pb_sort += pd[i]
        pb_remain.append(pd[i])
    np = len(pb_sort)
    for i in range(np):
        pb_sort[i].id = i
    next_pb = 0
    pb_remain[0] = pb_remain[0][1:]
    # print(pb_sort)
    # print(pb_remain)
    while next_pb < np:
        pl = [pb_sort[next_pb].block]
        pso = pb_sort[next_pb].mstr
        latency = pb_sort[next_pb].latency
        pts = generate_templates1(pso, nq, latency)
        cnt = 0
        # start padding        
        # print('pl:', pl)
        # print('l0', pb_remain)
        while len(pts) > 0 and cnt < maxiter:
            cnt += 1
            tmp1 = []
            for i in pts:
                pi1 = pdix[pLen(i[0])]
                for j in pi1:
                    pi2 = pb_remain[j]
                    # print('l1', pi2)
                    tmp2 = []
                    for k in range(len(pi2)):
                        if pi2[k].latency <= i[1] and pINC(i[0], pi2[k].mstr) == True:
                            tmp1.append(pi2[k].mstr)
                            pl.append(pi2[k].block)
                            i[1] -= pi2[k].latency
                        else:
                            tmp2.append(pi2[k])
                    pb_remain[j] = tmp2
                    # print('l', tmp2)
            # print('l2', pb_remain)
            for i in tmp1:
                # print(pso)
                # print(i)
                pso = pOR(pso, i) # use pOR instead pXOR, multi-padding
            pts = generate_templates1(pso, nq, latency)
            if len(tmp1) == 0:
                break
        # print('l3', pb_remain)
        ps_layers.append(pl)
        ps_occupy.append(pso)
        # decide next_pb
        maxidx = np
        maxmu = -1
        maxbag = -1
        maxbid = -1
        for i in range(len(pdl)):
            for j in range(len(pb_remain[i])):
                if pb_remain[i][j].len >= pb_sort[next_pb].len:
                    m = mutual(pso, pb_remain[i][j].mstr)
                    if m > maxmu:
                        maxidx = pb_remain[i][j].id
                        maxmu = m
                        maxbag = i
                        maxbid = j
                else:
                    break
        # print('l4', pb_remain)
        if maxidx == np:
            for i in range(len(pdl)):
                if len(pb_remain[i]) > 0:
                    maxidx = pb_remain[i][0].id                    
                    pb_remain[i] = pb_remain[i][1:]
                    break
        else:
            pb_remain[maxbag] = pb_remain[maxbag][0:maxbid] + pb_remain[maxbag][maxbid+1:]
        # print('l5', pb_remain)
        next_pb = maxidx
    return ps_layers, ps_occupy

def depth_oriented_scheduling(parr, length=0, maxiter=1):
    nq = len(parr[0][0]) # num_qubits
    parr_flat = []
    for i in parr:
        parr_flat.append(pauli_block(i))    
    parr_big = []
    parr_small = []
    for i in parr_flat:
        if i.len > length:
            parr_big.append(i)
        else:
            parr_small.append(i)
    def _key(pb):
        s = 0
        ps = pb.mstr
        for i in ps:
            s *= 4
            if i == 'I':
                s += 0
            elif i == 'X':
                s += 1
            elif i == 'Y':
                s += 2
            elif i == 'Z':
                s += 3
        return -s
    parr_big = sorted(parr_big, key=_key)
    parr_small = sorted(parr_small, key=_key)
    # print(parr_small)
    ne = len(parr_flat)
    ps_layers = []
    while ne > 0:
        if len(parr_big) > 0:
            cpb = parr_big[0]
            parr_big = parr_big[1:]
        elif len(parr_small) > 0:
            cpb = parr_small[0]
            parr_small = parr_small[1:]
        pl = [cpb.block]
        ne -= 1
        pso = cpb.mstr
        latency = cpb.latency
        pts = generate_templates1(pso, nq, latency)
        cnt = 0
        while len(pts) > 0 and cnt < maxiter:
            cnt += 1
            tmp1 = []
            for i in pts:
                tmp2 = []
                for j in range(len(parr_small)):
                    if parr_small[j].latency <= i[1] and pINC(i[0], parr_small[j].mstr):
                        pl.append(parr_small[j].block)
                        tmp1.append(parr_small[j].mstr)
                        i[1] -= parr_small[j].latency
                        ne -= 1
                    else:
                        tmp2.append(parr_small[j])
                parr_small = tmp2
            for i in tmp1:
                pso = pOR(pso, i)
            pts = generate_templates1(pso, nq, latency)
            if len(tmp1) == 0:
                break
        ps_layers.append(pl)
    return ps_layers
