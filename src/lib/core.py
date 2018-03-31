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

    C = {}
    P_C = {}
    #print("Our graph:")
    #print(G.nodes)
    #print(G.edges)
    #input()
    #print("Our samples:")
    #print(patients)
    #input()
    for v in G.nodes():
        C_v = {v}
        P_C_v = cover(patients, C_v) # no need to compute this \foreach u
        #print("C_v: ", C_v)
        #input()
        p_v = {u: nx.shortest_path(G,v,u) for u in G.nodes() if u is not v}
        #print("p_v: ", p_v)
        #input()
        while len(C_v) < k:
            maximum = -1
            #print("P_C: ", P_C)
            #input()
            #print("P_C_v", P_C_v)
            #input()
            l_v_max = set()
            #print("l_v_max: ", l_v_max)
            #input()
            #print("V - C_v: ", set(G.nodes()).difference(C_v))
            #input()
            #print("current v: ", v)
            #input()
            for u in set(G.nodes()).difference(C_v):
                #print("current u: ", u)
                #input()
                l_v = set(p_v[u])
                #print("l_v: ", l_v)
                #input()
                #print("l_v U C_v: ", l_v.union(C_v))
                #input()
                if len(l_v.union(C_v)) <= k:
                    P_v = cover(patients, l_v)
                    #print("P_v: ", P_v)
                    #input()
                    #print("num: ", len(P_v.difference(P_C_v)))
                    #print("den: ", len(l_v.difference(C_v)))
                    #input()
                    ratio = len(P_v.difference(P_C_v))/len(l_v.difference(C_v))
                    if maximum < ratio:
                        maximum = ratio
                        l_v_max = l_v
            C_v = C_v.union(l_v_max)
            P_C_v = cover(patients, C_v) # no need to compute this \foreach u

        #print("C_v: ", len(C_v), C_v)
        #print("C: ", len(C), C)
        #print("P_C_v: ", len(P_C_v), P_C_v)
        #print("P_C: ", len(P_C), P_C)
        if len(P_C_v) > len(P_C):
            C = C_v
            P_C = cover(patients, C)
            print("_________________")
            print()
            print("Best solution updated!")
            print("Current C (ids): ", C)
            print("Current P_C (cardinality, no samples_list):", len(P_C))


    return C

