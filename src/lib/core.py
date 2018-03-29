import networkx as nx
import random


def delta_removal(G, delta):
    """

    :param G: input graph
    :param delta: threshold
    :return: the graph G_I obtained from G by removing any edge with weight < delta
    """

    removes = []

    for (v,u) in G.edges():
        if G[v][u]['weight'] < delta:
            removes.append((v,u))

    G.remove_edges_from(removes)

    return G


def cover(patients, l_v):

    patients_covered =[]

    for p in patients:
        genes_list = patients[p]
        for v in l_v:
            if v in genes_list:
                patients_covered.append(p)
                break # da sistemare!!!

    return set(patients_covered)


def combinatorial_algorithm(G, delta, k, patients):

    G = delta_removal(G, delta)

    C = []
    for v in G.nodes():
        C_v = set([v])
        p_v = [{u: nx.shortest_path(G,v,u)} for u in G.nodes() if u is not v]
        while len(C_v) < k:
            max = -1
            for u in set(G.nodes()).difference(C_v):
                l_v = set(p_v[u-1]) # This "-1" may look ugly, but it is necessary due to list indexing
                if len(l_v.union(C_v)) <= k:
                    P_v = cover(patients, l_v)
                    P_c = cover(patients, C_v)



