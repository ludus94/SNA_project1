"Exercises 1"
import sys
import networkx as nx
from priorityq import PriorityQueue
import random
import numpy as np
from scipy.sparse import linalg
import itertools as it
from joblib import Parallel, delayed
import math
import csv

#Function to load a graph muse_facebook_edges
def load_graph():
    Data = open('musae_facebook_edges.csv', "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()
    G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=str)
    return G

#Function to load a label muse_facebook_target for real cluster
def load_label():

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

#Function to load the real cluster
def load_real_clusters():
    diz=load_label()
    cl_politician = set()
    cl_company = set()
    cl_government = set()
    cl_tvshow = set()
    for key_diz in diz.keys():
        if diz[key_diz] == 'company':
            cl_company.add(key_diz)
        elif diz[key_diz] == 'government':
            cl_government.add(key_diz)
        elif diz[key_diz] == 'politician':
            cl_politician.add(key_diz)
        elif diz[key_diz] == 'tvshow':
            cl_tvshow.add(key_diz)
    print("Cluster reali acquisiti")
    return cl_company,cl_government,cl_politician,cl_tvshow

#Function to compare a sperimental cluster to a real cluster
def compute_cluster_accuracy(cl_reale,cl_sperimentale):
    lung=len(cl_reale)
    intersect=cl_reale.intersection(cl_sperimentale)
    len_int=len(intersect)
    return float(len_int/lung)*100

#Function to sample the graph
def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

# hierarchical parallel
def hierarchical(G,sample=None):

    if sample is None:
        sample=G.nodes()

    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in sample:
        for v in sample:
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 0)
                else:
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset([u]) for u in sample)

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

        clusters.add((s[0] | s[1]))

        if len(clusters) ==4:
            done = True

    return clusters
# Function to parallel hierarchical
def parallel_hier(G,j):#j è il numero di jobs
    c1_ov=frozenset()
    c2_ov=frozenset()
    c3_ov=frozenset()
    c4_ov=frozenset()
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        list=parallel(delayed(hierarchical)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results

        for j in range(len(list)): #list è una lista con j elementi
            count=0
            for x in list[j]:
                if count == 0:
                    c1_ov=c1_ov.union(x)
                elif count==1:
                    c2_ov=c2_ov.union(x)
                elif count==2:
                    c3_ov=c3_ov.union(x)
                elif count==3:
                    c4_ov=c4_ov.union(x)
                count+=1
    return c1_ov,c2_ov,c3_ov,c4_ov

##VERSIONE SPECTRAL Parallel
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

#K-MEANS Parallel
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
    return c1_ov,c2_ov,c3_ov,c4_ov

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

        lista_cl4 = list(cluster4)
        cl_len = len(lista_cl4)
        it=int(math.ceil(cl_len/4))
        for el in range(cl_len):
            if(el >=0 and el < it):
                item=lista_cl4[el]
                cluster0.add(item)
            elif (el >= it and el < 2*it):
                item = lista_cl4[el]
                cluster1.add(item)
            elif (el >= 2*it and el < 3*it):
                item = lista_cl4[el]
                cluster2.add(item)
            elif (el >= 3*it and el < 4*it):
                item = lista_cl4[el]
                cluster3.add(item)


    return cluster0, cluster1,cluster2,cluster3

def betweenness_par(G,sample=None):

    if sample is None:
        sample=G.nodes()
    subgraph=nx.subgraph(G,sample)
    edge_btw={frozenset(e):0 for e in subgraph.edges()}
    node_btw={i:0 for i in subgraph.nodes()}

    for s in subgraph.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in subgraph.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in subgraph.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in subgraph.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in subgraph.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in subgraph.nodes()} #the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue=[s]
        spnum[s]=1
        distance[s]=0
        while queue != []:
            c=queue.pop(0)
            tree.append(c)
            for i in subgraph[c]:
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
def bwt_cluster_parallel(G,j):
    #PARALLELIZZAZIONE
    pq=PriorityQueue()
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        lista=parallel(delayed(betweenness_par)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results

    for j in lista:
        for i in j[0].keys():
            pq.add(i,-j[0][i])#Fase di assemblaggio

    graph=G.copy()
    done=False
    while not done:

        edge=tuple(sorted(pq.pop()))
        graph.remove_edges_from([edge])

        list_connected_comp=list(nx.connected_components(graph))
        if len(list(nx.connected_components(graph))) == 4:
             done = True
    return list_connected_comp


def betweenness(G):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in
                 G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i] / spnum[
                    c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c,
                                             i})]  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c,
                                                                i})]  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[
                    c]  # betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw, node_btw


def bwt_cluster_naive(G):
    eb, nb = betweenness(G)
    pq = PriorityQueue()
    for i in eb.keys():
        pq.add(i, -eb[i])
    graph = G.copy()

    done = False
    while not done:
        edge = tuple(sorted(pq.pop()))
        graph.remove_edges_from([edge])
        list_connected_comp = list(nx.connected_components(graph))
        if len(list(nx.connected_components(graph))) == 4:
            done = True

    return list_connected_comp

def print_accuracy(i,cl_sperimentale):
    cl_company,cl_government,cl_politician,cl_tvshow=load_real_clusters()
    print("----Accuracy of real cluster on sperimental---\n")
    print("Cluster"+str(i)+" company:",compute_cluster_accuracy(cl_company,cl_sperimentale))
    print("Cluster"+str(i)+" government:",compute_cluster_accuracy(cl_government,cl_sperimentale))
    print("Cluster"+str(i)+" politician:",compute_cluster_accuracy(cl_politician,cl_sperimentale))
    print("Cluster"+str(i)+" tv show:",compute_cluster_accuracy(cl_tvshow,cl_sperimentale))
def print_clusters(method,c1,c2,c3,c4):
    print("Metodo "+method+" parallel\n")
    print("Grandezza del cluster 1: "+str(len(c1))+"\n")
    print("Grandezza del cluster 2: "+str(len(c2))+"\n")
    print("Grandezza del cluster 3: "+str(len(c3))+"\n")
    print("Grandezza del cluster 4: "+str(len(c4))+"\n")
    print("cluster 1:"+str(c1)+"\n")
    print("cluster 2:"+str(c2)+"\n")
    print("cluster 3:"+str(c3)+"\n")
    print("cluster 4:"+str(c4)+"\n")

G=load_graph()

inp=0

while(inp!=int(8)):
    print("Insert 0=hier_naive 1=parallel_hier 2=4-means_naive 3=parallel_4means 4=spectral_naive 5=parallel_spectral 6=Girvan Newman_naive 7=Girvan Newman 8=exit:")
    inp=int(input())
    if(inp==0):
        clusters=hierarchical(G,None)
        count=1
        c1=frozenset()
        c2=frozenset()
        c3=frozenset()
        c4=frozenset()
        for i in clusters:
            if count == 1:
                c1=i
            elif count == 2:
                c2=i
            elif count == 3:
                c3=i
            elif count == 4:
                c4=i
            count+=1

        print_clusters("Hierarchical Naive",c1,c2,c3,c4)
        #Cluster 1
        print_accuracy(1,c1)
        #Cluster 2
        print_accuracy(2,c2)
        #Cluster 3
        print_accuracy(3,c3)
        #Cluster 4
        print_accuracy(4,c4)
    elif(inp==1):
        c1,c2,c3,c4=parallel_hier(G,40)
        print_clusters("Hierarchical Parallel",c1,c2,c3,c4)
        #Cluster 1
        print_accuracy(1,c1)
        #Cluster 2
        print_accuracy(2,c2)
        #Cluster 3
        print_accuracy(3,c3)
        #Cluster 4
        print_accuracy(4,c4)
    elif(inp==2):
        c1,c2,c3,c4=four_means(G,None)
        print_clusters("4-means naive",c1,c2,c3,c4)
        #Cluster 1
        print_accuracy(1,c1)
        #Cluster 2
        print_accuracy(2,c2)
        #Cluster 3
        print_accuracy(3,c3)
        #Cluster 4
        print_accuracy(4,c4)
    elif(inp==3):
        c1,c2,c3,c4=parallel_4means(G,40)
        print_clusters("4-means parallelo",c1,c2,c3,c4)
        #Cluster 1
        print_accuracy(1,c1)
        #Cluster 2
        print_accuracy(2,c2)
        #Cluster 3
        print_accuracy(3,c3)
        #Cluster 4
        print_accuracy(4,c4)
    elif(inp==4):
        c1,c2,c3,c4=spectral(G,None)
        print_clusters("Spectral Naive",c1,c2,c3,c4)
        #Cluster 1
        print_accuracy(1,c1)
        #Cluster 2
        print_accuracy(2,c2)
        #Cluster 3
        print_accuracy(3,c3)
        #Cluster 4
        print_accuracy(4,c4)
    elif(inp==5):
        c1,c2,c3,c4=parallel_spectral(G,40)
        print_clusters("Spectral Parallel",c1,c2,c3,c4)
        #Cluster 1
        print_accuracy(1,c1)
        #Cluster 2
        print_accuracy(2,c2)
        #Cluster 3
        print_accuracy(3,c3)
        #Cluster 4
        print_accuracy(4,c4)
    elif(inp==6):
        list=bwt_cluster_naive(G)
        print_clusters("Betweenness Naive",list[0],list[1],list[2],list[3])
        #Cluster 1
        print_accuracy(1,list[0])
        #Cluster 2
        print_accuracy(2,list[1])
        #Cluster 3
        print_accuracy(3,list[2])
        #Cluster 4
        print_accuracy(4,list[3])
    elif(inp==7):
        list=bwt_cluster_parallel(G,40)
        print_clusters("Betweenness Parallel",list[0],list[1],list[2],list[3])
        #Cluster 1
        print_accuracy(1,list[0])
        #Cluster 2
        print_accuracy(2,list[1])
        #Cluster 3
        print_accuracy(3,list[2])
        #Cluster 4
        print_accuracy(4,list[3])


