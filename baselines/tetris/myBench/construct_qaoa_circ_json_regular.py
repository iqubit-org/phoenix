"""
This script can be used to construct QAOA-MaxCut
circuit json (an input to the compiler)
from graphs defined as networkx graph objects.

also can be used to generate circuit in txt format 
"""
#to run this script
#python construc_qaoa_circ_json.py N K

import json
import networkx as nx
import sys
import os
'''
if len(sys.argv) < 3:
	try:
		print("constructing a qaoa circuit wrong")
	except:
		sys.exit(1)
'''

#Max_N = int(sys.argv[1]) #max number of nodes
#K = int(sys.argv[2]) #regular k graph

prob = [6, 7]
Node_count = [10, 12]

for case in range(2):
	for N in Node_count:
		for K in prob:
			if((N*K %2) == 0):
				#for regular K, N*K has to be an even number
				#random seed value
				g = nx.random_regular_graph(K,N,seed=None) #networkx graph object
				#print(sorted(map(sorted, g.edges())))
				
				'''
				#################################
				#saving the graph as a json file
				#creating a dictionary from the graph object.
				dic = {}
				print(len(g.edges()))
				for i, edge in enumerate(g.edges()):
				    dic['{}'.format(i+1)] = "{}".format(edge)
				    print(edge[0],edge[1])
				
				#saving the graph as a json file
				path1 = 'testcases/regularK_json/node_'+str(N)
				file_name = 'Node'+ str(N)+'_regular_'+str(K)+'.json'
				path = path1 + '/'+file_name
				print(path)
				#os.mkdir(path1)
				with open(path,'w') as fp:
				    json.dump(dic,fp,indent=4)
				fp.close()
				'''
				#################################
				#saving the graph as a txt file

				path = str(N)+'_'+ str(K)+'_'+str(case)+'.txt'
				#print(path)
				with open(path,'w') as fp:
					str0 = ""+str(N) + " " + str(len(g.edges()))
					print("%s\n" %str0)
					fp.write("%s\n" %str0)
					for edge in enumerate(g.edges()):
						str1 = ""
						str1 = str1+str(edge[1][0])+" "+str(edge[1][1])
						fp.write("%s\n" %str1)

				
