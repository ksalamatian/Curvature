import networkx as nx
import numpy as np

def toCyCoord(x,y,z):
    if x>0:
        phi=np.arctan(x/y)
    elif (x<0) and (y >=0):
        phi=np.arctan(x/y)+np.pi
    else:
        phi=np.arctan(x/y)-np.pi
    r=np.sqrt(x*x+y*y)

    return [r, phi,z]


G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/processed/processed.50.graphml")
N=nx.number_of_nodes(G)
M=nx.number_of_edges(G)
print("Num Vertices:",N)
print("Num Edges:", M)
edist=[]
rdist=[]
circdist=[]
zdist=[]
x=dict()
y=dict()
z=dict()
x=nx.get_node_attributes(G,"x")
y=nx.get_node_attributes(G,"y")
z=nx.get_node_attributes(G,"z")
for u,v in G.edges:
    edist.append(G[u][v]["edist"])
    rdist.append(G[u][v]["dist"])
    [r1,phi1, z1]=toCyCoord(x[u],y[u],z[u])
    [r2,phi2, z2]=toCyCoord(x[v],y[v],z[v])
    circdist.append(r2*np.abs(phi1-phi2))
    zdist.append(np.abs(z2-z1))
data=np.stack((np.array(circdist), np.array(zdist),np.array(rdist),np.array(edist)/np.mean(edist)))
print(np.cov(data))

