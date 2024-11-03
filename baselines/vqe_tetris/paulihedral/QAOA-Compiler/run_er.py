import os
from subprocess import Popen, PIPE
import time
import sys
root = "circ1"
def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(get_script_path())
files = sorted(os.listdir(root))
seeds = [83,193]
files1 = []
for i in files:
    for seed in seeds:
        if f"-{seed}" in i:
            files1.append(i)
files = files1
def console(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE)
    out, err = p.communicate()
    return (p.returncode, out, err)
f = sys.stdout
for i in files:
    cmd = f"python main.py -d Mahattan.json -ci {root}/{i} -co examples/Config.json  -p VIC"
    # print(cmd)
    t0 = time.time()
    _, out, _ = console(cmd)
    # print(out)
    name = i[:-5]
    c = out.decode().split("\n")[-2]
    r = c.split(',')
    depth = int(r[0].split(':')[-1].strip())
    cnot = int(r[1].split(':')[-1].strip())
    single = int(r[2].split(':')[-1].strip())
    total = int(r[3].split(':')[-1].strip())
    esp = float(r[4].split(':')[-1].strip())
    print(f"{name},{cnot},{single},{total},{depth},{time.time()-t0}", flush=True)
    print(f"{name},{cnot},{single},{total},{depth},{esp},{time.time()-t0}", file=f, flush=True)
f.close()