################################################
# Create 2 types of graphs:
# Cylinder Random Geometric Graph
# Swiss Roll Random Geometric graph
################################################
import math
import random
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np

import scipy as sc
import time

X=[]
N=10000
D=5
#for i in range(0,N):
#    phi=2*math.pi*random.random()
#    r=2
#    x=r*math.sin(phi)
#    y=r*math.cos(phi)
#    z=10*random.random()-5
#    X.append((x,y,z))
[X,t]=make_swiss_roll(n_samples=N,noise=0.05, random_state=None)
nbrs = NearestNeighbors(n_neighbors=D, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
G=nx.from_numpy_array(nbrs.kneighbors_graph(X))
G.remove_edges_from(nx.selfloop_edges(G))



L=nx.laplacian_matrix(G)
B=np.eye(N)-1.0/N*np.ones(N)
H=-1.0/2*B@L@B
Lambda,U= sc.linalg.eigh(H, subset_by_index=[N-20,N-1])
DD=np.diag(np.sqrt(abs(Lambda)))@U.transpose()


for u in G.nodes():
    G.nodes[u]["x"] = X[u][0]
    G.nodes[u]["y"] = X[u][1]
    G.nodes[u]["z"] = X[u][2]

s=0
for u,v,d in G.edges(data=True):
    #G[u][v]["distance"]=np.linalg.norm(np.array(X[u]) - np.array(X[v]))
    G[u][v]["ot"] = 0.0
    G[u][v]["distance"]=1.0
    G[u][v]["curv"] = 0.0
    G[u][v]["edist"]=np.linalg.norm(DD[:,u]-DD[:,v])
    d.pop("weight", None)
    s += G[u][v]["edist"]


s=s/G.number_of_edges()
ss=0
for u,v in G.edges():
    G[u][v]["edist"]=G[u][v]["edist"]/s
    ss+=G[u][v]["edist"]

#nx.write_graphml(G,"cylinder_"+str(D)+"_"+str(N)+".graphml")
nx.write_graphml(G,"SwissRoll.graphml")



print("Num Vertices:",nx.number_of_nodes(G))
print("Num Edges:", nx.number_of_edges(G))
