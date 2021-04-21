"Exercises 1"
"Exercises 1"
import networkx as nx
from priorityq import PriorityQueue
import random
from scipy.sparse import linalg
Data = open('musae_facebook_edges.csv', "r")
next(Data, None)  # skip the first line in the input file
Graphtype = nx.Graph()
G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=int)
print(G)
def hierarchical(G):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
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