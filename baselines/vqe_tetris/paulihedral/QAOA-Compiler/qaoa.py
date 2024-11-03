from benchmark.qaoa import *
import json
def to_json(parr):
    zzs = []
    for i in parr:
        ip = []
        ps = i[0].ps
        for j in range(len(ps)):
            if ps[j] == 'Z':
                ip.append(j)
        zzs.append(ip)
    zd = {}
    for i in range(len(zzs)):
        if len(zzs[i]) == 2:
            zd[str(i+1)] = f"({zzs[i][0]}, {zzs[i][1]})"
    return zd
seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
tp = "reg"
data = "circ"
for cfg in [[4,20],[8,20],[12, 20]]:
    for seed in seeds: 
        G = rand_reg(cfg[0], cfg[1], seed=seed)
        parr = gene_qaoa_oplist(G)
        zd = to_json(parr)
        with open(f"{data}/{tp}-{cfg[0]}-{cfg[1]}-{seed}.json", "w") as write_file:
            json.dump(zd, write_file)
seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
tp = "er"
for cfg in [[20,0.1],[20,0.3],[20, 0.5]]:
    for seed in seeds:
        G = rand_er(cfg[0], cfg[1], seed=seed)
        parr = gene_qaoa_oplist(G)
        zd = to_json(parr)
        with open(f"{data}/{tp}-{cfg[0]}-{cfg[1]}-{seed}.json", "w") as write_file:
            json.dump(zd, write_file)
seeds = [83, 193, 239, 367, 439, 571, 661, 743, 881, 967, 1049, 1153, 1201, 1367, 1489, 1543, 1621, 1787, 1889, 1949]
tp = "tsp"
for n in [4,5]:
    for seed in seeds:
        parr = tsp_oplist(n, seed=seed)
        zd = to_json(parr)
        with open(f"{data}/{tp}-{n}-{seed}.json", "w") as write_file:
            json.dump(zd, write_file)