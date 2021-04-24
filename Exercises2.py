import networkx as nx
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