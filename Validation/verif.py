import networkx as nx
import numpy as np
import ot
G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/processed/processed.1.graphml")

u='n1'
v='n2073'
sources=[u]
dests=[v]
for n in G[u]:
    sources.append(n)
dists=dict()
for n in G[v]:
    dests.append(n)
sources=sorted(sources)
dests=sorted(dests)
dists=np.zeros((len(sources),len(dests)))
for n in range(0,len(sources)):
    for m in range(0,len(dests)):
        dists[n][m]=nx.shortest_path_length(G,sources[n],dests[m], weight='dist')
a=np.ones(len(sources))*1/(len(sources)-1)
a[0]=0.0
b=np.ones(len(dests))*1/(len(dests)-1)
b[1]=0.0
opt=ot.emd2(a,b,dists)
print(opt)

