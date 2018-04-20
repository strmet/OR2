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


def cover(patients, l_v):
    patients_covered = []

    for p in patients:
        genes_list = patients[p]
        for v in l_v:
            if v in genes_list:
                patients_covered.append(p)
                break  # da sistemare!!!

    return set(patients_covered)


def combinatorial_algorithm(G, delta, k, patients):
    G = delta_removal(G, delta)

    C = {}
    P_C = {}

    for v in G.nodes():
        C_v = {v}
        P_C_v = cover(patients, C_v)  # no need to compute this \foreach u

        p_v = {u: nx.shortest_path(G, v, u) for u in G.nodes() if u is not v}

        while len(C_v) < k:
            maximum = -1

            l_v_max = set()

            for u in set(G.nodes()).difference(C_v):

                l_v = set(p_v[u])

                if len(l_v.union(C_v)) <= k:
                    s = score_old(patients,l_v,P_C_v,C_v)
                    if maximum < s:
                        maximum = s
                        l_v_max = l_v
            C_v = C_v.union(l_v_max)
            P_C_v = cover(patients, C_v)  # no need to compute this \foreach u

        if len(P_C_v) > len(P_C):
            C = C_v
            P_C = cover(patients, C)
            print("_________________")
            print()
            print("Best solution updated!")
            print("Current C (ids): ", C)
            print("Current P_C (cardinality):", len(P_C))

    return C, P_C


def score_old(patients,l_v,P_C_v,C_v):
    P_v = cover(patients, l_v)
    ratio = len(P_v.difference(P_C_v)) / len(l_v.difference(C_v))
    return ratio
def obj_func(patients,S):
    return len(cover(patients, l_v))

"""
suppongo patients sia un dizionario con:
chiave:id paziente -->  dizionario di geni
Dizionario dei geni è :
chiave:numero gene --> prob del gene j per il paziente i
"""

def score(patients,S):
    som=0
    for j in patients:
        #def scoreSingolo(patients,S):
        prod=1
        #dif=S.difference(set(patients[j])  potrei anche farlo ma è costoso
        for i in S:
            if( i in patients[j] ):
                prod*=1-patients[i][j]
            """
            non serve
            else:
                prod*=1-0
            """
        membro=1-prod
    som+=membro
            
    
    
