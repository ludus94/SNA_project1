import csv, sys
import networkx as nx
from priorityq import PriorityQueue
import random
from scipy.sparse import linalg

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
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 0)
                else:
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())

    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])

        print(clusters)
        """Bisogna lavorare nel seguente modom fermare il ciclo quando il numero di cluster è uguale a 4"""
        if len(clusters)==4:
            done = True
    return clusters
# da notare che pq rappresenta la distanza tra cluster mentre "cluster" rapp. gli stessi


#-----------------------------------
def load_label():
    import csv

    filename = 'musae_facebook_target.csv'
    label=dict()
    with open(filename,'r', newline='',encoding='UTF-8') as f:
        reader = csv.DictReader(f)
        try:
            for row in reader:
                label[row['id']]=row['page_type']
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
    return label


def count_occ_labels(cl1):

    diz=load_label()

    count1=0#tvshow
    count2=0#company
    count3=0#politician
    count4=0#government
    for el in cl1:
        if diz[el]=='tvshow':
            count1+=1
        elif diz[el]=='company':
            count2+=1
        elif diz[el]=='politician':
            count3+=1
        elif diz[el]=='government':
            count4+=1

    return count1,count2,count3,count4


load_label()
G=load_graph()
"""G=nx.Graph()
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
G.add_edge('F', 'fix')
G.add_edge('F', 'fux')
G.add_edge('fux', 'fox')
G = nx.Graph()
G.add_edge('1', '2')
G.add_edge('1', '3')
G.add_edge('2', '3')
G.add_edge('2', '4')
G.add_edge('4', '5')
G.add_edge('4', '6')
G.add_edge('4', '7')
G.add_edge('5', '6')
G.add_edge('6', '7')"""
clusters=hierarchical(G)

"""cl1=clusters[0]
cl2=clusters[1]
cl3=clusters[2]
cl4=clusters[3]
count1,count2,count3,count4=count_occ_labels(cl1)
print("Cluster1:\n tvshow: "+str(count1)+
   " company: "+ str(count2)+
   " politician:"+str(count3)+
    " government:"+str(count4)+"\n")
count1,count2,count3,count4=count_occ_labels(cl2)
print("Cluster2:\n tvshow: "+str(count1)+
   " company: "+ str(count2)+
   " politician:"+str(count3)+
    " government:"+str(count4)+"\n")
count1,count2,count3,count4=count_occ_labels(cl3)
print("Cluster3:\n tvshow: "+str(count1)+
   " company: "+ str(count2)+
   " politician:"+str(count3)+
    " government:"+str(count4)+"\n")
count1,count2,count3,count4=count_occ_labels(cl4)
print("Cluster4:\n tvshow: "+str(count1)+
   " company: "+ str(count2)+
   " politician:"+str(count3)+
    " government:"+str(count4)+"\n")
    """
