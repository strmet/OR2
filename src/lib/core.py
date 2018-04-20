import networkx as nx
import random


def delta_removal(G, delta):
    """

    :param G: input graph
    :param delta: threshold
    :return: the graph G_I obtained from G by removing any edge with weight < delta
    """

    removes = []

    for (v, u) in G.edges():
        if G[v][u]['weight'] < delta:
            removes.append((v, u))

    G.remove_edges_from(removes)

    return G


def cover_score(patients, l_v):
    return len([
        p for p in patients
        if genes_in_sample(l_v, patients[p])
    ])


def genes_in_sample(l_v, genes_list):
    for v in l_v:
        if v in genes_list:
            return True
    return False


def prob_score(patients,S):
    som=0.0

    for j in patients:
        prod=1.0

        for i in S:
            if i in patients[j]:
                prod*=1.0-patients[i][j]
        som+=1-prod

    return som


def combinatorial_algorithm(G, k, patients, delta=0.8, prob=False):
    G = delta_removal(G, delta)

    C = {}
    P_C_score = 0

    if prob:
        scoring = prob_score
    else:
        scoring = cover_score

    for v in G.nodes():
        C_v = {v}
        P_C_v_score = scoring(patients, C_v)  # no need to compute this \foreach u

        p_v = {u: nx.shortest_path(G, v, u) for u in G.nodes() if u is not v}

        while len(C_v) < k:
            maximum = -1
            l_v_max = set()

            for u in G.nodes() - C_v:  # "-" is an overloaded operator, it means 'difference'

                l_v = set(p_v[u])

                if len(l_v | C_v) <= k:  # "|" is an overloaded operator, it means 'union'
                    P_v_score = scoring(patients, l_v)
                    s = P_v_score - P_C_v_score / len(l_v - C_v)
                    if maximum < s:
                        maximum = s
                        l_v_max = l_v
            C_v = C_v | l_v_max
            P_C_v_score = scoring(patients, C_v)  # no need to compute this \foreach u

        if P_C_v_score > P_C_score:
            C = C_v
            P_C_score = scoring(patients, C)
            print("_________________")
            print()
            print("Best solution updated!")
            print("Current C (ids): ", C)
            print("Current P_C (cardinality):", len(P_C))

    return C, P_C_score


def obj_func(patients,S):
    return cover_score(patients, S)

"""
suppongo patients sia un dizionario con:
chiave:id paziente -->  dizionario di geni
Dizionario dei geni è :
chiave:numero gene --> prob del gene j per il paziente i
"""

