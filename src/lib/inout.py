import networkx as nx
import numpy as np


def test():
    return "LOL"


def g_input_read():
    # generiamo un grafo completo un po' a caso
    G = nx.complete_graph(100)

    for (u,v) in G.edges():
        G[u][v]['weight'] = np.random.random() # ]0,1[

    return G

def p_input_read():
    p = {}
    genes = [i for i in range(100)]
    for i in range(100):
        id = "Patient#"+str(i+1)
        patient_genes = np.random.choice(genes, size=int(np.random.random()*100), replace=False)
        p[id] = set(patient_genes)

    return p