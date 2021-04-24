import networkx as nx
import numpy as np
from scipy.sparse import linalg
import itertools as it
from joblib import Parallel, delayed
import math
from priorityq import PriorityQueue

def load_graph():
    Data = open('musae_facebook_edges.csv', "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()
    G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=str)
    return G

#CENTRALITY MEASURES
#Returns the top k nodes of G according to the centrality measure "measure"
def top(G,measure,k):
    pq = PriorityQueue()
    cen=measure(G)
    for u in G.nodes():
        pq.add(u, -cen[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out


#The measure associated to each node is exactly its degree
#Un esempio di questo è la degree-centrality quindi “quanti vicini” ha quel nodo , se ne ha tanti potrebbe essere un mio influencer.
def degree(G):
    cen=dict()
    for u in G.nodes():
        cen[u] = G.degree(u)
    return cen

#The measure associated to each node is the sum of the (shortest) distances of this node from each remaining node
def closeness(G):
    cen=dict()

    for u in G.nodes():
        visited=set()
        visited.add(u)
        queue=[u]
        dist=dict()
        dist[u]=0

        while len(queue) > 0:
            v = queue.pop(0)
            for w in G[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    dist[w] = dist[v]+1#contiene per ogni nodo la lunghezza del path minimo da esso alla radice

        cen[u]=sum(dist.values())

    return cen

# Computes edge and vertex betweenness of the graph in input
def betweenness(G):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree=[] #it lists the nodes in the order in which they are visited
        spnum={i:0 for i in G.nodes()} #it saves the number of shortest paths from s to i
        parents={i:[] for i in G.nodes()} #it saves the parents of i in each of the shortest paths from s to i
        distance={i:-1 for i in G.nodes()} #the number of shortest paths starting from s that use the edge e
        eflow={frozenset(e):0 for e in G.edges()} #the number of shortest paths starting from s that use the edge e
        vflow={i:1 for i in G.nodes()} #the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue=[s]
        spnum[s]=1
        distance[s]=0
        while queue != []:
            c=queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1: #if vertex i has not been visited
                    queue.append(i)
                    distance[i]=distance[c]+1
                if distance[i] == distance[c]+1: #if we have just found another shortest path from s to i
                    spnum[i]+=spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c=tree.pop()
            for i in parents[c]:
                eflow[frozenset({c,i})]+=vflow[c] * (spnum[i]/spnum[c]) #the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i]+=eflow[frozenset({c,i})] #each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c,i})]+=eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c]+=vflow[c] #betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw,node_btw
#PARALLEL IMPLEMENTATION
def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

def top_parallel(G,k,j):
    pq = PriorityQueue()
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        result=parallel(delayed(closeness_par)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

    for u in result:#u is a dict
        for el in u.keys():
            pq.add(el, -u[el])
      # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out

def closeness_par(G,sample=None):
    cen=dict()
    if sample is None:
        sample = G.nodes()
    for u in sample:
        visited=set()
        visited.add(u)
        queue=[u]
        dist=dict()
        dist[u]=0

        while len(queue) > 0:
            v = queue.pop(0)
            for w in G[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    dist[w] = dist[v]+1#contiene per ogni nodo la lunghezza del path minimo da esso alla radice

        cen[u]=sum(dist.values())

    return cen


#The measure associated to each node is its betweenness value
def btw(G):
    return betweenness(G)[1]
G=load_graph()
'''
print("Centrality measures:")
print("degree")
print(top(G,degree,500))
'''
print("closeness")
print(top_parallel(G,500,33))
'''
print("betweenness")
print(top(G,btw,500))
'''
