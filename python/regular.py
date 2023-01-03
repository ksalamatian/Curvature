import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

N=3000
D=4
G=nx.random_regular_graph(D, N, seed=2)
#G=nx.grid_2d_graph(50, 50)
#G=nx.newman_watts_strogatz_graph(2000, 3, 0.1)
#G.edges.data("dist", default=1.0)
#G.edges.data("ot",default=0.0)
#G.edges.data("curv",default=0.0)
L=nx.laplacian_matrix(G).todense()
B=np.eye(N)-1.0/N*np.ones(N)
H=1.0/2*B*L*B
Lambda,U= np.linalg.eig(H)
DD=np.diag(np.sqrt(abs(Lambda)))*U
for u,v in G.edges():
    G[u][v]["dist"]=1.0
    G[u][v]["ot"] = 0.0
    G[u][v]["curv"] = 0.0
    G[u][v]["edist"]=np.linalg.norm(DD[:,u]-DD[:,v])
print(G.edges(data=True))
nx.write_graphml(G,"regular_"+str(D)+"_"+str(N)+".graphml")
nx.draw(G)
plt.show()