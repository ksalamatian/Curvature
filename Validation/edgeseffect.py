import networkx as nx
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
import math
import time
import random
import subprocess
import os


#################################################################################################################################

G = nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/roadnetwork3.graphml")

for u,v in G.edges():

   

   G[u][v]["distance_avant"]=0.0

   G[u][v]["ratio1"] = 10.0
   G[u][v]["ratio2"] = 10.0


nx.write_graphml(G, "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/graph_impact1.graphml")

####################################################################################################################################

bottleneck_edges = []
subprocess.run(["cmake", ".."])
subprocess.run(["cmake", "--build", "."])
subprocess.run(["./Road", "-P", "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs", "-F", "graph_impact1.graphml", "-A", "MSMD"])
G = nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/results/processed.30.graphml")

max_dist = 0
max_edge = (1,2)
for u,v in G.edges:

    if G[u][v]["distance"]> max_dist:
        max_dist = G[u][v]["distance"]
        max_edge = (u, v)

    G[u][v]["distance_avant"] = G[u][v]["distance"]
    G[u][v]["distance"] = 1.0
    G[u][v]["curv"] = 99999.0
    G[u][v]["ot"] = 99999.0

    print("#Iter1 distance avant = ", G[u][v]["distance_avant"], "   distance = ", G[u][v]["distance"])



bottleneck_edges.append(max_edge)
G.remove_edge(*max_edge)



nx.write_graphml(G, "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/graph_impact_iter1.graphml")

print("distance max: ", max_dist)
print("edge max: ", max_edge)

subprocess.run(["cmake", ".."])
subprocess.run(["cmake", "--build", "."])
subprocess.run(["./Road", "-P", "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs", "-F", "graph_impact_iter1.graphml", "-A", "MSMD"])
G = nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/results/processed.30.graphml")

max_dist = 0
max_edge = (1,2)
for u,v in G.edges:

    if G[u][v]["distance"]> max_dist:
        max_dist = G[u][v]["distance"]
        max_edge = (u, v)


    G[u][v]["ratio1"] = G[u][v]["distance"]/G[u][v]["distance_avant"]


    G[u][v]["distance_avant"] = G[u][v]["distance"]



bottleneck_edges.append(max_edge)
G.remove_edge(*max_edge)

nx.write_graphml(G, "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/graph_impact_iter2.graphml")

print("distance max: ", max_dist)
print("edge max: ", max_edge)




subprocess.run(["cmake", ".."])
subprocess.run(["cmake", "--build", "."])
subprocess.run(["./Road", "-P", "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs", "-F", "graph_impact_iter2.graphml", "-A", "MSMD"])
G = nx.read_graphml("/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/results/processed.30.graphml")

max_dist = 0
max_edge = (1,2)
for u,v in G.edges:

    if G[u][v]["distance"]> max_dist:
        max_dist = G[u][v]["distance"]
        max_edge = (u, v)


    G[u][v]["ratio2"] = G[u][v]["distance"]/G[u][v]["distance_avant"]


    G[u][v]["distance_avant"] = G[u][v]["distance"]



bottleneck_edges.append(max_edge)
G.remove_edge(*max_edge)

nx.write_graphml(G, "/home/nsaloua/Research/Topologie/Graph-Embedding/Graphs/graph_impact_iter3.graphml")

print("distance max: ", max_dist)
print("edge max: ", max_edge)




print(bottleneck_edges)

















