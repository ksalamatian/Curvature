import math
import random
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
points=[]
N=3000
D=6
for i in range(0,N):
    phi=2*math.pi*random.random()
    r=2
    x=r*math.sin(phi)
    y=r*math.cos(phi)
    z=10*random.random()-5
    points.append((x,y,z))
nbrs = NearestNeighbors(n_neighbors=D, algorithm='ball_tree').fit(points)
distances, indices = nbrs.kneighbors(points)
G=nx.from_numpy_array(nbrs.kneighbors_graph(points))
G.remove_edges_from(nx.selfloop_edges(G))
L=nx.laplacian_matrix(G).todense()
B=np.eye(N)-1.0/N*np.ones(N)
H=1.0/2*B*L*B
Lambda,U= np.linalg.eig(H)
DD=np.diag(np.sqrt(abs(Lambda)))*U
for u in G.nodes():
    G.nodes[u]["x"] = points[u][0]
    G.nodes[u]["y"] = points[u][1]
    G.nodes[u]["z"] = points[u][2]
for u,v,d in G.edges(data=True):
    G[u][v]["dist"]=1.0
    G[u][v]["ot"] = 0.0
    G[u][v]["curv"] = 0.0
    G[u][v]["edist"]=np.linalg.norm(DD[:,u]-DD[:,v])
    d.pop("weight", None)
#print(G.edges(data=True))
nx.write_graphml(G,"../cylinder_"+str(D)+"_"+str(N)+".graphml")
