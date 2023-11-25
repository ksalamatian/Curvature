import networkx as nx
import numpy as np
import pandas as pd
#G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/processed/processed.49.graphml")
G=nx.read_graphml("/Users/ksalamatian/processed.8.graphml")

country2countrydict={}
for u,v,data in G.edges(data=True):
    if G.nodes[u]["Country"] < G.nodes[v]["Country"] :
        if G.nodes[u]["Country"] not in country2countrydict:
            country2countrydict[G.nodes[u]["Country"]]={}
        if G.nodes[v]["Country"] not in country2countrydict[G.nodes[u]["Country"]]:
            country2countrydict[G.nodes[u]["Country"]][G.nodes[v]["Country"]]={"details":[], "mean":0.0, "std":0.0}
        country2countrydict[G.nodes[u]["Country"]][G.nodes[v]["Country"]]["details"].append(G[u][v]["distance"])
    elif G.nodes[u]["Country"] >= G.nodes[v]["Country"]:
        if G.nodes[v]["Country"] not in country2countrydict:
            country2countrydict[G.nodes[v]["Country"]]={}
        if G.nodes[u]["Country"] not in country2countrydict[G.nodes[v]["Country"]]:
            country2countrydict[G.nodes[v]["Country"]][G.nodes[u]["Country"]]={"details":[], "mean":0.0, "std":0.0}
        country2countrydict[G.nodes[v]["Country"]][G.nodes[u]["Country"]]["details"].append(G[u][v]["distance"])
s=set()
for c1 in country2countrydict:
    s.add(c1)
    for c2 in country2countrydict[c1]:
        s.add(c2)
        country2countrydict[c1][c2]["mean"]=np.mean(country2countrydict[c1][c2]["details"])
        country2countrydict[c1][c2]["std"]=np.std(country2countrydict[c1][c2]["details"])
        country2countrydict[c1][c2]["sum"]=np.sum(country2countrydict[c1][c2]["details"])
inout=dict()
for c1 in country2countrydict:
    arr=[]
    inout[c1]={"insum":0.0,"inavg":0.0,"outsum":0.0,"outavg":0.0,
               "ratiosum":0.0, "ratioavg":0.0}
    for c2 in country2countrydict[c1]:
        if c1 == c2:
           inout[c1]["insum"]=np.sum(country2countrydict[c1][c2]["details"])
           inout[c1]["inavg"]=np.mean(country2countrydict[c1][c2]["details"])
        else:
            arr+=country2countrydict[c1][c2]["details"]
    inout[c1]["outsum"]=np.sum(arr)
    inout[c1]["outavg"]=np.mean(arr)
    inout[c1]["ratiosum"]=inout[c1]["insum"]/inout[c1]["outsum"]
    inout[c1]["ratioavg"]=inout[c1]["inavg"]/inout[c1]["outavg"]


ddict={i: [country2countrydict[i][j]["mean"] for j in country2countrydict[i].keys() ] for i in country2countrydict.keys()}

df=pd.DataFrame(columns=['insum','outsum','inavg','outavg','ratiosum','ratioavg'])
for c in inout:
    df.loc[c]=[inout[c]["insum"],inout[c]["outsum"],inout[c]["inavg"],inout[c]["outavg"], inout[c]["insum"]/inout[c]["outsum"], inout[c]["inavg"]/inout[c]["outavg"]]
df.to_csv('out.csv')

