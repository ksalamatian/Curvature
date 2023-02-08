import networkx as nx
import numpy as np

def twoNeighborhood(G,n,k):
    neighborhood=set()
    for m in nx.neighbors(G,n):
       for p in nx.neighbors(G,m):
        neighborhood.add(p)
    for m in nx.neighbors(G,n):
        if m in neighborhood:
            neighborhood.remove(m)
    if n in neighborhood:
        neighborhood.remove(n)
    return list(neighborhood)
#G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/processed/processed.50.graphml")
#G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/processed/processed.3.graphml")
G=nx.read_graphml("/Users/ksalamatian/processed.50.graphml")

N=nx.number_of_nodes(G)
print("Num Vertices:",N)
print("Num Edges:", nx.number_of_edges(G))
edist=[]
rdist=[]
for u,v in G.edges:
    edist.append(G[u][v]["edist"])
    rdist.append(G[u][v]["dist"])

avgEdist=np.mean(edist)
stdEdist=np.std(edist/avgEdist)
avgRdist=np.mean(rdist)
stdRdist=np.std(rdist/avgRdist)
edist=np.array(edist)/avgEdist
rdist=np.array(rdist)
ratio=np.divide(edist,rdist)
A=nx.adjacency_matrix(G)
AEdist=1/avgEdist*nx.adj_matrix(G,weight='edist')
ARdist=nx.adj_matrix(G,weight="dist")
#L=nx.laplacian_matrix(G)
#B=np.eye(N)-1.0/N*np.ones(N)
#H=-1.0/2*B*L*B
#Lambda,U= np.linalg.eig(H)
#DD=np.diag(np.sqrt(abs(Lambda)))*U
#nodesList=list(G.nodes)
A2=A*A
A2Edist=AEdist*AEdist
A2Rdist=ARdist*ARdist
#print(np.mean(A2Edist))
#print(np.std(A2Edist))
#print(np.mean(A2Rdist))
#print(np.std(A2Rdist))

dist1=[]
dist2=[]
dist3=[]
for node in nodesList:
    nodes=twoNeighborhood(G,node,2)
    indexNode=nodesList.index(node)
    for n in nodes:
        indexN=nodesList.index(n)
        dist1.append(np.linalg.norm(DD[:,indexN]-DD[:,indexNode]))
        dist2.append(nx.shortest_path_length(G,source=node, target=n, weight="dist"))
        dist3.append(nx.shortest_path_length(G,source=node, target=n, weight="edist"))
for u,v,d in G.edges(data=True):
    edist.append(d["edist"])
    rdist.append(d["dist"])

print('edistavg:',np.mean(edist)," edistStd: ", np.std(edist))
print('rdistavg:',np.mean(rdist)," rdistStd: ", np.std(rdist))
print('edistavg 2 Hop:',np.mean(dist3)," edistStd 2 Hop: ", np.std(dist3))
print('rdistavg 2 Hop:',np.mean(dist2)," rdistStd 2 Hop: ", np.std(dist2))
