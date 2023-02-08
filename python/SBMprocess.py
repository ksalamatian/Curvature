import networkx as nx
import numpy as np
G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/processed/processed.16.graphml")

N=nx.number_of_nodes(G)
print("Num Vertices:",N)
print("Num Edges:", nx.number_of_edges(G))
edist=[]
rdist=[]
for u,v in G.edges:
    edist.append(G[u][v]["edist"])
    rdist.append(G[u][v]["dist"])
edist=np.array(edist)
rdist=np.array(rdist)
avgEdist=np.mean(edist)
edist=edist/avgEdist
stdEdist=np.std(edist)
avgRdist=np.mean(rdist)
rdist=rdist/avgRdist
stdRdist=np.std(rdist)
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
#print(np.mean(A2Edist))
#print(np.std(A2Edist))
#print(np.mean(A2Rdist))
#print(np.std(A2Rdist))

dist1=[]
dist2=[]
dist3=[]
rdistMat=dict()
edistMat=dict()
curvMat=dict()
for i in range(0,3):
    for j in range(0,3):
        rdistMat[(i,j)]=[]
        edistMat[(i,j)]=[]
        curvMat[(i,j)]=[]
for u,v,d in G.edges(data=True):
    rdistMat[(G.nodes[u]["block"],G.nodes[v]["block"])].append(d["dist"])
    edistMat[(G.nodes[u]["block"],G.nodes[v]["block"])].append(d["edist"])
    curvMat[(G.nodes[u]["block"],G.nodes[v]["block"])].append(d["curv"])

avgrdistMat=np.zeros((3,3))
avgedistMat=np.zeros((3,3))
stdrdistMap=np.zeros((3,3))
stdedistMap=np.zeros((3,3))
for  i in range(0,3):
    for j in range(0,3):
        avgrdistMat[i,j]=np.mean(rdistMat[(i,j)])
        avgedistMat[i,j]=np.mean(edistMat[(i,j)])/avgEdist
        stdrdistMap[i,j]=np.std(rdistMat[(i,j)])
        stdedistMap[i,j]=np.std(edistMat[(i,j)])/avgEdist




print('edistavg:',np.mean(edist)," edistStd: ", np.std(edist))
print('rdistavg:',np.mean(rdist)," rdistStd: ", np.std(rdist))
print('edistavg 2 Hop:',np.mean(dist3)," edistStd 2 Hop: ", np.std(dist3))
print('rdistavg 2 Hop:',np.mean(dist2)," rdistStd 2 Hop: ", np.std(dist2))
