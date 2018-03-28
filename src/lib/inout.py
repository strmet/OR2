import networkx as nx
import random


def test():
    return "LOL"


def input_read():
    # generiamo un grafo completo un po' a caso
    G = nx.complete_graph(100)

    for (u,v) in G.edges():
        G[u][v]['weight'] = random.random()

    return G

