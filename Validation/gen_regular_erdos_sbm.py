##
# Create 3 types of graphs:
# Random Regular 
# Erdos Reyni
# Stochastic block model
##


import networkx as nx
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
import math
import time

N=4000
D_regular = 3
D_erdos = math.log(N)*1.02/N
epoch_time = int(time.time())

#G=nx.random_regular_graph(D_regular, N, seed=2)

#G=nx.erdos_renyi_graph(N, p=D_erdos, directed=False,seed= epoch_time)
G=nx.stochastic_block_model([2000,1000,1000], [[0.01, 0.003, 0.003], [0.003, 0.01, 0.003], [0.003, 0.003, 0.01]], nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)
del G.graph['partition']

print("Num Vertices:",nx.number_of_nodes(G))
print("Num Edges:", nx.number_of_edges(G))

L=nx.laplacian_matrix(G)
B=np.eye(N)-1.0/N*np.ones(N)
H=-1.0/2*B@L@B
Lambda,U= sc.linalg.eigh(H, subset_by_index=[N-20,N-1])
DD=np.diag(np.sqrt(abs(Lambda)))@U.transpose()
s=0
for u,v in G.edges():
    G[u][v]["distance"]=1.0
    G[u][v]["ot"] = 0.0
    G[u][v]["curv"] = 0.0
    G[u][v]["edist"]=np.linalg.norm(DD[:,u]-DD[:,v])
    s += G[u][v]["edist"]

s=s/G.number_of_edges()
for u,v in G.edges():
    G[u][v]["edist"]=G[u][v]["edist"]/s

#nx.write_graphml(G,"regular_"+"{:.2f}".format(D_regular)+"_"+str(N)+".graphml")
#nx.write_graphml(G,"erdos_"+"{:.2f}".format(D_erdos)+"_"+str(N)+".graphml")
nx.write_graphml(G,"sbm.graphml")

