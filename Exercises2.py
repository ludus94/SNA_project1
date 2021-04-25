import networkx as nx
import numpy as np
from scipy.sparse import linalg
import itertools as it
from joblib import Parallel, delayed
import math
from priorityq import PriorityQueue
import operator


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
#-------------------------------------------------PAGE RANKING OF UNDIRECTED GRAPHS
def rank(graph,d=0.85,n_iterations=50):

    V = graph.number_of_nodes()  #is the number of nodes of the graph
    ranks = dict()#dict of ranks
    for node in graph.nodes():
        ranks[node] = 1/V

    for _ in range(n_iterations):
        for el in graph.nodes():
            rank_sum = 0
            curr_rank = ranks[el]

            for n in graph.neighbors(el):
                if ranks[n] is not None:
                    outlinks = len(list(graph.neighbors(n)))
                    rank_sum += (1 / float(outlinks)) * ranks[n]#contributo al rank del nodo "el" da parte dei nodi adiacenti

            # computazione del rank di el
            ranks[el] = ((1 - float(d)) * (1/float(V))) + d*rank_sum

    return ranks

def top_rank(k,rank):
    pq = PriorityQueue()
    for u in rank.keys():
        pq.add(u, -rank[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out
#----------------VERSIONE PARALLELIZZATA DI PAGE RANK----------
def rank_parallel(graph,sample=None,d=0.85,n_iterations=50):
    if sample is None:
        sample=graph

    V = graph.number_of_nodes()  #is the number of nodes of the graph
    ranks = dict()#dict of ranks
    for node in graph.nodes():
        ranks[node] = 1/V

    for _ in range(n_iterations):
        for el in sample:
            rank_sum = 0
            curr_rank = ranks[el]

            for n in graph.neighbors(el):
                if ranks[n] is not None:
                    outlinks = len(list(graph.neighbors(n)))
                    rank_sum += (1 / float(outlinks)) * ranks[n]#contributo al rank del nodo "el" da parte dei nodi adiacenti

            # computazione del rank di el
            ranks[el] = ((1 - float(d)) * (1/float(V))) + d*rank_sum

    return ranks

def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

def parallel_rank(G,s,number_of_iteration,j):#j è il numero di jobs, s è il parametro della page-ranking, number_of_iteration è il numero di iterazioni del page-ranking

    total_rank=dict()
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        list=parallel(delayed(rank_parallel)(G,X,s,number_of_iteration) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        for j in list:
            for el in j.keys():
                total_rank[el]=j[el]
    return total_rank

#---------------------HITS NAIVE ITERATION--------------------------------
def hits(graph,k):

    auth = dict()
    hubs= dict()
    for node in graph.nodes():
        auth[node] = 1
        hubs[node] = 1
    for i in range(k):#We perform a sequence of k hub-authority updates
        for node in graph.nodes():
            auth[node] =sum(hubs[el] for el in graph[node])#First apply the Authority Update Rule to the current set of scores.
        for node in graph.nodes():
            hubs[node] =sum(auth[el] for el in graph[node])#Then apply the Hub Update Rule to the resulting set of scores.

    auth_n,hubs_n=normalize(G,auth,hubs)
    return auth_n,hubs_n


def normalize(G,auth,hubs):
    auth_sum = sum(auth[node] for node in G.nodes())
    hub_sum = sum(hubs[node] for node in G.nodes())

    for node in G.nodes():
        auth[node] =auth[node]/auth_sum
        hubs[node] =hubs[node]/hub_sum
    return auth,hubs


def top_hits(G,k,num_node):
    pq = PriorityQueue()
    pq2=PriorityQueue()
    auth_n,hubs_n=hits(G,k)
    for u in G.nodes():
        pq.add(u, -auth_n[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    for u in G.nodes():
        pq2.add(u, -hubs_n[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    out2=[]
    for i in range(num_node):
        out.append(pq.pop())
        out2.append(pq2.pop())
    return out,out2

#---------------------HITS Pararell ITERATION--------------------------------
def parallel_hits(G,k,j):#j è il numero di jobs
    auth_total=dict()
    hubs_total=dict()
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        list=parallel(delayed(hits_sample)(G,X,k) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        for i in list:
            for el in i[0].keys():
                auth_total[el]=i[0][el]
            for el in i[1].keys():
                hubs_total[el]=i[1][el]
    return auth_total,hubs_total

def hits_sample(graph,sample,k):
    if sample is None:
        sample=graph
    auth = dict()
    hubs= dict()
    subgraph=nx.subgraph(G,sample)
    for node in sample:
        auth[node] = 1
        hubs[node] = 1
    for i in range(k):#We perform a sequence of k hub-authority updates
        for node in sample:
            auth[node] =sum(hubs[el] for el in subgraph[node] )#First apply the Authority Update Rule to the current set of scores.
        for node in sample:
            hubs[node] =sum(auth[el] for el in subgraph[node])#Then apply the Hub Update Rule to the resulting set of scores.

    auth_n,hubs_n=normalize(subgraph,sample,auth,hubs)
    return auth_n,hubs_n


def normalize(G,sample,auth,hubs):
    auth_sum = sum(auth[node] for node in G.nodes())
    hub_sum = sum(hubs[node] for node in G.nodes())

    for node in G.nodes():
        auth[node] =auth[node]/auth_sum
        hubs[node] =hubs[node]/hub_sum
    return auth,hubs



def top_hits_parall(G,k,num_node,j):
    pq = PriorityQueue()
    pq2=PriorityQueue()
    auth_n,hubs_n=parallel_hits(G,k,j)
    for u in G.nodes():
        pq.add(u, -auth_n[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    for u in G.nodes():
        pq2.add(u, -hubs_n[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    out2=[]
    for i in range(num_node):
        out.append(pq.pop())
        out2.append(pq2.pop())
    return out,out2

G=load_graph()
'''
G = nx.Graph()
G.add_edge('1', '2')
G.add_edge('1', '3')
G.add_edge('2', '3')
G.add_edge('2', '4')
G.add_edge('4', '5')
G.add_edge('4', '6')
G.add_edge('4', '7')
G.add_edge('5', '6')
G.add_edge('6', '7')

print("Centrality measures:")
print("degree")
print(top(G,degree,500))

print("closeness")
print(top_parallel(G,500,33))

print("betweenness")
print(top(G,btw,500))

print("Page ranking Naive")
rank=rank(G,0.85,50)
print(top_rank(500,rank))
print("\n Page ranking Parallelo")
res=parallel_rank(G,0.85,50,10)#G,s,number_of_iteration,j
print(top_rank(500,res))
'''
"""print("Hits Naive")
a,h=top_f(G,30,500)
print(a)"""
print("Parallel Hits")
a,h=top_hits_parall(G,30,500,20)
print(a)