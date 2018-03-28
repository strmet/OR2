import networkx as nx
import random


def delta_removal(G, delta):
    """

    :param G: input graph
    :param delta: ... (per ora interpretiamolo come la probabilità di rimozione)
    :return: the graph G_I obtained from G by removing any edge with weight < delta (per ora rimuovo a caso)
    """

    for v in G.nodes():
        nodes_connected_to_v = [to for (fr, to) in G.edges(v)]

        # if there is any node connected to this one, operate
        if len(nodes_connected_to_v) > 0:
            for u in nodes_connected_to_v:
                if G[v][u]['weight'] < delta:
                    G.remove_edge(v, u)

    return G


def combinatorial_algorithm(G, delta, k):

    G = delta_removal(G, delta)
    print(G.edges)
    c = []
    for v in G.nodes:
        c_v = [v]
        not_v_nodes = [u for u in G.nodes if u is not v]
        p_v = [{u: nx.shortest_path(G,v,u)} for u in not_v_nodes]
        print(p_v)