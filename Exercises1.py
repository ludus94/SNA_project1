"Exercises 1"
"Exercises 1"
import networkx as nx
from priorityq import PriorityQueue
import random
import numpy as np
from scipy.sparse import linalg
import itertools as it
from joblib import Parallel, delayed
import math

def load_graph():
    Data = open('musae_facebook_edges.csv', "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()
    G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=str)
    return G

##VERSIONE SPECTRAL
def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}


def spectral(G,sample=None):

    if sample is None:
        sample = G.nodes()

    #n=G.number_of_nodes()
    n=len(sample)
    #nodes=sorted(G.nodes())
    nodes=sorted(sample)
    L = nx.laplacian_matrix(G, nodes).asfptype()
    #print(L) #To see the laplacian of G uncomment this line

    w,v = linalg.eigsh(L,n-1)
    #print(w) #Print the list of eigenvalues
    #print(v) #Print the matrix of eigenvectors
    #print(v[:,0]) #Print the eigenvector corresponding to the first returned eigenvalue
    r,c=np.shape(v)

    c1=set()
    c2=set()
    c3=set()
    c4=set()
    for i in range(n):
        if c>=2:
            if v[i,0] < 0 and v[i,1]<0 :
                c1.add(nodes[i])
            elif v[i,0]<0 and v[i,1]>0:
                c2.add(nodes[i])
            elif v[i,0]>0 and v[i,1]<0:
                c3.add(nodes[i])
            elif v[i,0]>0 and v[i,1]>0:
                c4.add(nodes[i])

    return c1,c2,c3,c4

def parallel_spectral(G,j):#j è il numero di jobs

    c1_ov=set()
    c2_ov=set()
    c3_ov=set()
    c4_ov=set()
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        list=parallel(delayed(spectral)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        for j in range(len(list)):
            c1_ov=c1_ov.union(list[j][0])
            c2_ov=c2_ov.union(list[j][1])
            c3_ov=c3_ov.union(list[j][2])
            c4_ov=c4_ov.union(list[j][3])
    return c1_ov,c2_ov,c3_ov,c4_ov




G=load_graph()
#hierarchical(G)
'''G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'C')
G.add_edge('B', 'D')
G.add_edge('D', 'E')
G.add_edge('D', 'F')
G.add_edge('D', 'G')
G.add_edge('E', 'F')
G.add_edge('F', 'G')
G.add_edge('F', 'fox')'''

c1,c2,c3,c4=parallel_spectral(G,60)
print("cluster 1:"+str(c1))
print("cluster 2:"+str(c2))
print("cluster 3:"+str(c3))
print("cluster 4:"+str(c4))
