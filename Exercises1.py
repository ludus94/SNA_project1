"Exercises 1"
"Exercises 1"
import networkx as nx
from priorityq import PriorityQueue
import random
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

def hierarchical(G):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
            print("v")
            if u != v:#se sono 2 nodi distinti
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 0)#intendo come nodi vicini quelli che hanno un arco tra loro
                else:
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 1)
    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())
    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())#prelevo prima quelli a priorità più alta
        clusters.remove(s[0])
        clusters.remove(s[1])#ipotizziamo sia un nuovo cluster
        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)#or bit a bit. è come se inserissi un unico nodo che rappresenta un cluster
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)#w sono gli altri cluster di cui rappresento la distanza
        clusters.add(s[0] | s[1])
        print(clusters)
        a = input("Do you want to continue? (y/n) ")
        if a == "n":
            done = True
    # da notare che pq rappresenta la distanza tra cluster mentre "cluster" rapp. gli stessi

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
    print(v) #Print the matrix of eigenvectors
    #print(v[:,0]) #Print the eigenvector corresponding to the first returned eigenvalue


    c1=set()
    c2=set()
    c3=set()
    c4=set()
    for i in range(n):
        if v[i,0] < 0 and v[i,1]<0:
            c1.add(nodes[i])
        if v[i,0]<0 and v[i,1]>0:
            c2.add(nodes[i])
        if v[i,0]>0 and v[i,1]<0:
            c3.add(nodes[i])
        if v[i,0]>0 and v[i,1]>0:
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
        c1,c2,c3,c4=parallel(delayed(spectral)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        '''for r1 in c1:
            for el in r1:
                c1_ov.add(el)
        for r2 in c2:
            for el1 in r2:
                c2_ov.add(el1)
        for r3 in c3:
            for el2 in r3:
                c3_ov.add(el2)
        for r4 in c4:
            for el3 in r4:
                c4_ov.add(el3)'''
        c1_ov=c1_ov.union(c1)
        c2_ov=c2_ov.union(c2)
        c3_ov=c3_ov.union(c3)
        c4_ov=c4_ov.union(c4)
        return c1_ov,c2_ov,c3_ov,c4_ov




#G=load_graph()
#hierarchical(G)
G = nx.Graph()
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

c1,c2,c3,c4=parallel_spectral(G,4)
print(c1,c2,c3,c4)


