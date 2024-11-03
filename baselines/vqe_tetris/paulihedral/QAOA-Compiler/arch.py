import sys
c = open('mahattan.txt').readlines()
a = []
for i in c:
    a += i.split(',')
a = [i.strip() for i in a]
b = []
for i in a:
    t = i.split(':')
    # print(t[1].strip())
    p = 1 - float(t[1].strip())
    qs = t[0][2:].split('_')
    # print(t[0])
    b.append((int(qs[0]), int(qs[1]), p))
for i in b:
    print(f"\"({i[0]},{i[1]})\": {i[2]},")
for i in b:
    print(f"({i[0]},{i[1]}),",end="")