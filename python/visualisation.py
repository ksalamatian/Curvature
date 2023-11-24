from pyvis.network import Network
import networkx as nx
G = nx.read_graphml("../roadnetwork7.graphml")
nt = Network('1000px', '1000px', notebook=True, select_menu=True)
nt.from_nx(G)
nt.show('nx.html')


maxX=0;
minX=10000000;
maxY=0;
minY=10000000;


for n in G.nodes:
    G.nodes[n]["x"], G.nodes[n]["y"] , zone, type= utm.from_latlon(G.nodes[n]["lat"], G.nodes[n]["long"])
    maxX=max(G.nodes[n]["x"],maxX)
    maxY=max(G.nodes[n]["y"],maxY)
    minX=min(G.nodes[n]["x"],minX)
    minY=min(G.nodes[n]["y"],minY)
coeffX=10000/(maxX-minX)
coeffY=10000/(maxY-minY)
for n in G.nodes:
    G.nodes[n]["x"]=(G.nodes[n]["x"]-minX)*coeffX
    G.nodes[n]["y"]=(G.nodes[n]["y"]-minY)*coeffY
nx.write_graphml(G,"../roadnetwork7.graphml")

