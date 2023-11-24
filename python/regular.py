import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import time

N=10000
D=math.log(N)*1.05/N
epoch_time = int(time.time())
#G=nx.erdos_renyi_graph(N, p=D, directed=False,seed= epoch_time)
G=nx.connected_watts_strogatz_graph(3000, 8, 0.01, tries=100, seed=None)
#G=nx.barabasi_albert_graph(N, 9, seed=None, initial_graph=None)
p0=np.log(N)/N
a=15
b=0.05
#G=nx.stochastic_block_model([1000,1000, 1000], [[a/N,b/N, b/N], [b/N,a/N,b/N], [b/N,b/N, a/N]], nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)
#del G.graph['partition']


#G=nx.random_regular_graph(D, N, seed=2)
#G=nx.grid_2d_graph(50, 50)
#G=nx.newman_watts_strogatz_graph(2000, 3, 0.1)
#G.edges.data("dist", default=1.0)
#G.edges.data("ot",default=0.0)
#G.edges.data("curv",default=0.0)
L=nx.laplacian_matrix(G)
B=np.eye(N)-1.0/N*np.ones(N)
H=1.0/2*B*L*B
Lambda,U= np.linalg.eigh(H)
HH=U.transpose()*np.diag(Lambda)*U
DD=np.diag(np.sqrt(abs(Lambda)))*U
for u,v in G.edges():
    G[u][v]["dist"]=1.0
    G[u][v]["ot"] = 0.0
    G[u][v]["curv"] = 0.0
    G[u][v]["edist"]=np.linalg.norm(DD[:,u]-DD[:,v])
#print(G.edges(data=True))
nx.write_graphml(G,"../small_"+"{:.2f}".format(D)+"_"+str(N)+".graphml")
#nx.draw(G)
#plt.show()