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
#K-MEANS

def parallel_4means(G,j):#j è il numero di jobs

    c1_ov=set()
    c2_ov=set()
    c3_ov=set()
    c4_ov=set()
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        list=parallel(delayed(four_means)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        for j in range(len(list)):
            c1_ov=c1_ov.union(list[j][0])
            c2_ov=c2_ov.union(list[j][1])
            c3_ov=c3_ov.union(list[j][2])
            c4_ov=c4_ov.union(list[j][3])
    #print(list)
    print("cluster1:",c1_ov,"\n")
    print("cluster2:",c2_ov,"\n")
    print("cluster3:",c3_ov,"\n")
    print("cluster4:",c4_ov,"\n")

def four_means(G,sample=None):

    if sample is None:
        sample = G.nodes()

    n=len(sample)
    # Choose four clusters represented by vertices that are not neighbors
    u = random.choice(list(nx.subgraph(G,sample)))
    v = random.choice(list(nx.non_neighbors(nx.subgraph(G,sample), u)))
    #intersect_u_v=list(set(list(nx.non_neighbors(nx.subgraph(G,sample), u))) & set(list(nx.non_neighbors(nx.subgraph(G,sample), v))))
    w= random.choice(list(nx.non_neighbors(nx.subgraph(G,sample), v)))
    #intesect_u_v_w=list(set(intersect_u_v) & set(list(nx.non_neighbors(nx.subgraph(G,sample), w))) )
    z=random.choice(list(nx.non_neighbors(nx.subgraph(G,sample), w)))

    cluster0=set()
    cluster1=set()
    cluster2=set()
    cluster3=set()
    cluster4=set()
    cluster0.add(u)
    cluster1.add(v)
    cluster2.add(w)
    cluster3.add(z)
    added = 4
    while added < n:
        #print(added)
        lista=[]
        for el in sample:
            if el not in cluster0|cluster1|cluster2|cluster3 and (len(set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0 or len(set(G.neighbors(el)).intersection(cluster2)) != 0 or len(set(G.neighbors(el)).intersection(cluster3)) != 0 ):
                lista.append(el)
            else:
                if el not in cluster0|cluster1|cluster2|cluster3:
                    cluster4.add(el)
                    added+=1

            if len(lista)!=0:
                x = random.choice(lista)
                if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
                    cluster0.add(x)
                    added+=1
                    lista.remove(x)
                elif len(set(G.neighbors(x)).intersection(cluster1)) != 0:
                    cluster1.add(x)
                    added+=1
                    lista.remove(x)
                elif len(set(G.neighbors(x)).intersection(cluster2)) != 0:
                    cluster2.add(x)
                    added+=1
                    lista.remove(x)
                elif len(set(G.neighbors(x)).intersection(cluster3)) != 0:
                    cluster3.add(x)
                    added+=1
                    lista.remove(x)

        cluster1=cluster1.union(cluster4)

    return cluster0, cluster1,cluster2,cluster3




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
G.add_edge('F', 'fox')
G.add_edge('F', 'fix')
G.add_edge('F', 'fux')
G.add_edge('fux', 'fox')
G.add_edge('fux', 'fex')
G.add_edge('fex', 'GRANMAESTRO')
G.add_edge('GRANMAESTRO', 'masto_Del_fox')'''

parallel_4means(G,40)
'''c1,c2,c3,c4=parallel_spectral(G,60)
print("Metodo Spectral\n")
print("cluster 1:"+str(c1))
print("cluster 2:"+str(c2))
print("cluster 3:"+str(c3))
print("cluster 4:"+str(c4))'''



