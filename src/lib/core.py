import networkx as nx


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


def set_cover(patients, l_v):
    return set([
        p for p in patients
        if genes_in_sample(l_v, patients[p])
    ])


def genes_in_sample(l_v, genes_list):
    for v in l_v:
        if v in genes_list:
            return True
    return False


def prob_cover(patients, l_v):
    som=0.0

    for j in patients:
        prod=1.0

        for i in l_v:
            if i in patients[j]:
                prod*=1.0-patients[i][j]
        som+=prod

    return len(patients) - som


def combinatorial_algorithm(G, k, patients, delta=0.8, prob=False):
    G = delta_removal(G, delta)

    C = {}
    P_C = set()

    if prob:
        cover = prob_cover
        ratio = lambda elem_diff, nodes_diff: elem_diff / len(nodes_diff)
        better = lambda current, best: current>best
    else:
        cover = set_cover
        ratio = lambda elem_diff, nodes_diff: len(elem_diff) / len(nodes_diff)
        better = lambda current, best: len(current)>len(best)

    for v in G.nodes():
        C_v = {v}
        P_C_v = cover(patients, C_v)  # no need to compute this \foreach u

        p_v = {u: set(nx.shortest_path(G, v, u)) for u in G.nodes() if u is not v}

        while len(C_v) < k:
            maximum = -1
            l_v_max = set()

            for u in G.nodes() - C_v:  # "-" is an overloaded operator, it means 'difference'
                l_v = p_v[u]

                if len(l_v | C_v) <= k:  # "|" is an overloaded operator, it means 'union'
                    P_v = cover(patients, l_v)

                    s = ratio(P_v - P_C_v, l_v - C_v)
                    if maximum < s:
                        maximum = s
                        l_v_max = l_v
            C_v = C_v | l_v_max
            P_C_v = cover(patients, C_v)

        if better(P_C_v,P_C):  # if we've found a better solution, update it and let us know
            C = C_v
            P_C = cover(patients, C)
            print("_________________")
            print()
            print("Best solution updated!")
            print("Current C (ids): ", C)
            if prob:
                print("Current P_C (cardinality):", P_C)
            else:
                print("Current P_C (cardinality):", len(P_C))

    return C, P_C


def score_cover(patients, l_v):
    len(set_cover(patients, l_v))


def enumerating_algorithm(G, k, patients, delta=0.8, prob=False):
    G = delta_removal(G, delta)

    if prob:
        obj_func = prob_cover
    else:
        obj_func = score_cover


    parametri={}
    parametri['k']=k
    parametri['pazienti']=patients
    parametri['funzioneLimite']=calcoloFunzioneLimiteSimple
    parametri['scoringFunction']=obj_func
    
    bestSolution,bestScore=BDDE(G, parametri)
    print(bestSolution)
    print(bestScore)