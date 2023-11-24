import networkx as nx
import numpy as np

N=5000
#G1=nx.random_regular_graph(3,5000)
G1=nx.erdos_renyi_graph(N,np.log(N)/N)
d1={i:"1-"+str(i) for i in range(0,5000)}
nx.set_node_attributes(G1,d1, "name")
#G2=nx.random_regular_graph(6,5000)
G2=nx.erdos_renyi_graph(N,2*np.log(N)/N)
d2={i:"2-"+str(i) for i in range(0,5000)}
nx.set_node_attributes(G2, d2, "name")
G3=nx.union(G1,G2, rename=("1-", "2-"))
G3.add_edge('1-0','2-0')
print(G3.number_of_nodes())
print(G1.number_of_edges())
print(G2.number_of_edges())
print(G3.number_of_edges())
nx.write_graphml(G1,"../validation1.graphml")