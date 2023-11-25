import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def toCyCoord(x,y,z):
#    if x>0:
    phi=np.arctan(x/y)
#    elif (x<0) and (y >=0):
#        phi=np.arctan(x/y)+np.pi
#    else:
#        phi=np.arctan(x/y)-np.pi
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
fulldist=[]
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
#    [r1,phi1, z1]=toCyCoord(x[u],y[u],z[u])
##    [r2,phi2, z2]=toCyCoord(x[v],y[v],z[v])
#   dphi=np.abs(phi1-phi2)
#   if(dphi>np.pi):
#        dphi=2*np.pi-dphi
#    circdist.append(r2*dphi)
    circdist.append(np.sqrt((x[u]-x[v])*(x[u]-x[v])+(y[u]-y[v])*(y[u]-y[v])))
    zdist.append(np.abs(z[u]-z[v]))
    fulldist.append(np.sqrt((x[u]-x[v])*(x[u]-x[v])+(y[u]-y[v])*(y[u]-y[v])+(z[u]-z[v])*(z[u]-z[v])))
circdist=np.array(circdist)
zdist=np.array(zdist)
fulldist=np.array(fulldist)
edist=np.array(edist)
rdist=np.array(rdist)
rescale=len(fulldist)/np.sum((fulldist))
fulldist=fulldist*rescale
#zdist=zdist*rescale
#circdist=circdist*rescale
circdist[np.where(circdist==0.0)]=1e-6
ratio0 = zdist/circdist
data=np.stack((ratio0, circdist, zdist,fulldist, rdist,edist/np.mean(edist)))
ratio1 = rdist/circdist
ratio2 = rdist/zdist
ratio3 = edist/circdist
ratio4 = edist/zdist
ratio5=  rdist/fulldist
ratio6=  edist/fulldist
ratio7= rdist/ratio0
#plt.scatter(ratio1, ratio3)
#plt.show()
#plt.scatter(ratio2, ratio4)
#plt.show()
#plt.scatter(ratio5, ratio6)
#plt.show()
#plt.scatter(ratio1,ratio2)
#plt.show()
#plt.scatter(ratio3,ratio4)
#plt.show()
#plt.scatter(ratio3,ratio4)
#plt.show()
#plt.xscale('log')
#plt.yscale('log')
plt.scatter(ratio1,ratio3)
plt.show()
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(0, 10)
plt.scatter(rdist,ratio0)
plt.show()
plt.ylim(0, 10)
plt.scatter(edist,ratio0)
plt.show()



print(np.corrcoef(data))
print(np.mean(rdist), np.std(rdist), np.mean(edist), np.std(edist))
print(np.mean(ratio1),np.mean(ratio2),np.mean(ratio3),np.mean(ratio4),np.mean(ratio5),np.mean(ratio6),)