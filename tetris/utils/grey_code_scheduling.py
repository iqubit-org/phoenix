#!/usr/bin/env python

import sys
sys.setrecursionlimit(4*sys.getrecursionlimit())

import nnf
import pydot

def extract_first_ps(blocks):
    map = {}
    with open('tmp.txt', 'w') as f:
        for block in blocks:
            f.write(block[0].__str__() + '\n')
            map[block[0].__str__()] = block
    return 'tmp.txt', map

def sort_paulistrings(filename, map):
    qubit_count = -1
    paulis = {'I','X','Y','Z'}
    qubit_pauli_to_var = {}
    names = set()

    # parse once and check basic properties
    with open(filename, 'r') as file:
        for num, line in enumerate(file):
            if qubit_count==-1:
                qubit_count=len(line)-1
            assert qubit_count+1==len(line)

    # build data structures
    for qubit in range(qubit_count):
        for pauli in paulis:
            qubit_pauli_to_var[(qubit,pauli)] = nnf.Var((qubit,pauli))
            names.add((qubit,pauli))

    # parse again to build full logic statement
    terms = set()
    with open(filename, 'r') as file:
        for line in file:
            children = set()
            for num, char in enumerate(line):
                if char=='\n':
                    continue
                for pauli in paulis:
                    var = qubit_pauli_to_var[(num,pauli)]
                    children.add(var if pauli==char else ~var)
            term = nnf.And(children)
            assert (term.term())
            terms.add(term)

    dnf = nnf.Or(terms)
    assert(dnf.is_MODS())
    assert(dnf.simply_conjunct())
    assert(dnf.decomposable())
    assert(dnf.smooth())

    cnf = dnf.to_CNF()
    assert(cnf.consistent())
    assert(cnf.is_CNF())

    dDNNF = nnf.dsharp.compile(cnf, executable = '/common/home/zl606/dsharp/dsharp', smooth = False)
    # assert(dDNNF.marked_deterministic())
    assert(dDNNF.decomposable())
    # print(dDNNF.model_count())

    # pydot.graph_from_dot_data(dDNNF.to_DOT(color=True))[0].write_png("dNNF.png")
    ret = []
    def recurse (dDNNF, string, depth):

        string = list(string)
        # print(string)
        if depth == 0:
            # print("".join(string))
            ret.append(map["".join(string)])
            return()

        pauli_index = -1

        seen = {dDNNF}
        queue = [dDNNF]

        while queue:

            node = queue.pop(0)

            if isinstance(node,nnf.Var):
                if isinstance(node.name,tuple) and node.true:

                    if pauli_index==-1:
                        pauli_index=node.name[0]

                    if pauli_index==node.name[0]:
                        pauli = node.name[1]
                        string[pauli_index]=pauli
                        recurse(dDNNF.condition({(pauli_index,pauli):True}).simplify(),string,depth-1)

            else:
                for child in node.children:
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)


    recurse(dDNNF,qubit_count*[" "],qubit_count)

    return ret
    for model in dDNNF.models():
        string = qubit_count * [""]
        for key, val in model.items():
            if val and not isinstance(key,nnf.Aux):
                string[key[0]] = key[1]
        print("".join(string))
