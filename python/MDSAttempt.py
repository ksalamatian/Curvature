import networkx as nx
import pickle
import numpy as np
with open('mypicklefile', 'rb') as f1:
    mat=pickle.load(f1)
data=np.zeros((4000,4000))
i=0
j=0
for key1 in mat:
    j=0
    for key2 in mat[key1]:
        data[i][j]=mat[key1][key2]
        j +=1
    i +=1
print(data)
