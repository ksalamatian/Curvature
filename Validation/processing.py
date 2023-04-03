import networkx as nx
import numpy as np

#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/V2/erdos_0.00_10000.graphml")
#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/V2/regular_3.00_10000.graphml")
#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/V2/cylinder_surgery1.graphml")

#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/V2/sbm.graphml")
#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/V2/SwissRoll.graphml")

#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/Regular/sbm_ricci_5.graphml")
#G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/Regular/cylinder_ricci_10.graphml")
G=nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/Regular/SwissRoll_Ricci_10.graphml")



#for u,v in G.edges:
#    if G[u][v]["edist"] < 0.001:
#        G[u][v]["edist"] = 0.001

N=nx.number_of_nodes(G)
print("Num Vertices:",N)
print("Num Edges:", nx.number_of_edges(G))
edist=[]


for u,v in G.edges:
    edist.append(G[u][v]["distance"])

avgEdist=np.mean(edist)
stdEdist=np.std(edist/avgEdist)



print('edistavg:',avgEdist," edistStd: ", stdEdist)
#nx.write_graphml(G,"SwissRoll_surgery1.graphml")


