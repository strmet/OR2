import networkx as nx


class BDDE:
    def __init__(self, G, samples, k=2, delta=0.8, prob=False, starting_score=-1, starting_solution=[]):
        self.G = G
        self.k = k
        self.delta = delta
        self.tree = None
        self.root = None
        self.samples = samples
        self.sample_size = len(self.samples)
        self.prob = prob
        if self.prob:
            qs = {
                s: {gene: 1 - self.samples[s][gene]
                    for gene in self.samples[s]}
                for s in self.samples
            }
            self.min_qs = {
                s: qs[s][min(qs[s], key=qs[s].get)]
                for s in qs
            }
            self.scoring_function = lambda subgraph: prob_cover(self.samples,subgraph)
        else:
            self.scoring_function = lambda subgraph: score_cover(self.samples,set(subgraph))

        self.best_score = starting_score
        self.leaves_number = 0
        self.best_subgraph = starting_solution

    def fb(self, C):
        summation = 0.0
        c_size = len(C)
        for j in self.samples:
            prod = 1.0
            for i in C:
                prod *= (1.0-self.samples[j][i]) if i in self.samples[j] else 1

            prod *= self.min_qs[j]**(self.k-c_size)
            summation += prod

        return self.sample_size - summation

    def fb_greaterthan_f(self, C):
        """
        Returns True if the branch has to be pruned inside the BDDE algorithm.
        Furthermore, it evaluates leaves.
        :param C: The current vertexes set to be evalued
        :return: True if pruned, False otherwise
        """
        if len(C)>self.k:
            # Prune it if it's too big
            return True
        elif len(C)<self.k:
            # Prune it if the bounding function can't reach the current best,
            # but do it only when using the probability version.
            return self.prob and self.fb(C)<self.best_score
        else:
            # This means we've reached a leaf.
            # We should evaluate it, as it wasn't previously pruned!
            score = self.scoring_function(C)
            self.leaves_number+=1
            if score>self.best_score:
                self.best_score = score
                self.best_subgraph = C
                print("_________________")
                print()
                print("Best solution updated!")
                print("Current C (ids): ", self.best_subgraph)
                print("Current P_C (cardinality):", self.best_score)
            # No need to go further.
            return True

    def enumeration_algorithm(self):

        sorted_by_degree_vertices = [v[0] for v in sorted(self.G.degree, key=lambda x: x[1])]
        self.leaves_number=0

        for idx,v in enumerate(sorted_by_degree_vertices):
            self.root=v
            self.tree=nx.DiGraph()
            self.DEPTH([],v,[])
            self.G.remove_node(v)

            del self.tree
            del self.root

        print("Quante foglie?")
        print(self.leaves_number)

    def BREADTH(self, S,n,U):
        vn=n.data
        if vn in U:
            return None

        S1=S+[vn]
        if self.fb_greaterthan_f(S1):
            return None

        n1=n
        for nxx in self.getNodesFromBranch(n):
            n2=self.BREADTH(S1,nxx,U)
            if n2 is not None:
                self.tree.add_edge(n1,n2)
                del n2

        del S1
        return n1

    def DEPTH(self, S,v,beta):
        S1=S+[v]
        if self.fb_greaterthan_f(S1):
            return None

        n=Nodo(v)
        beta1=[]

        xn=self.getxn(S,v)
        xn.sort(reverse=True)
        for i in range(0,len(beta),1):
            n1=self.BREADTH(S1,beta[i],xn)
            if n1 is not None:
                self.tree.add_edge(n,n1)
                beta1.append(n1)
                del n1
        for v in xn:
            n1=self.DEPTH(S1,v,beta1)
            if n1 is not None:
                self.tree.add_edge(n,n1)
                beta1.append(n1)
                del n1
        del xn
        del beta1
        del beta
        del S1
        return n

    def getNodesFromBranch(self, n):
        W=[]
        neighbors=self.tree[n]
        for v in neighbors:
            if self.tree.has_edge(n,v):
                W.append(v)
        del neighbors
        return W

    def getxn(self, S,v):
        L=nx.Graph()
        L.add_nodes_from(self.G.neighbors(v))

        if self.root in L:
            L.remove_node(self.root)
        for n in S:
            L.remove_nodes_from(self.G.neighbors(n))
        return list(L.nodes())


class Nodo(object):
    def __init__(self,data):
        self.data=data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self)


def delta_removal(G, delta):
    """
    DEPRECATED. We're not even using this version of the problem.
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


def prob_cover(patients, l_v):
    som=0.0

    for j in patients:
        prod=1.0

        for i in l_v:
            if i in patients[j]:
                prod*=1.0-patients[j][i]
        som+=prod

    return len(patients) - som


def combinatorial_algorithm(G, k, patients, delta=0.8):
    G = delta_removal(G, delta)

    C = {}
    P_C = set()

    for v in G.nodes():
        C_v = {v}
        P_C_v = set_cover(patients, C_v)  # no need to compute this \foreach u

        p_v = {u: set(nx.shortest_path(G, v, u)) for u in G.nodes() if u is not v}

        while len(C_v) < k:
            maximum = -1
            l_v_max = set()

            for u in G.nodes() - C_v:  # "-" is an overloaded operator, it means 'difference'
                l_v = p_v[u]

                if len(l_v | C_v) <= k:  # "|" is an overloaded operator, it means 'union'
                    P_v = set_cover(patients, l_v)

                    s = len(P_v - P_C_v)/ len(l_v - C_v)
                    if maximum < s:
                        maximum = s
                        l_v_max = l_v
            C_v = C_v | l_v_max
            P_C_v = set_cover(patients, C_v)

        if len(P_C_v) > len(P_C):  # if we've found a better solution, update it and let us know
            C = C_v
            P_C = P_C_v
            print("_________________")
            print()
            print("Best solution updated!")
            print("Current C (ids): ", C)
            print("Current P_C (cardinality):", len(P_C))

    return C, P_C


def prob_combinatorial_algorithm(G, k, patients, delta=0.8):
    G = delta_removal(G, delta)

    C = {}
    P_C = -1

    for v in G.nodes():
        C_v = {v}
        P_C_v = prob_cover(patients, C_v)  # no need to compute this \foreach u

        p_v = {u: set(nx.shortest_path(G, v, u)) for u in G.nodes() if u is not v}

        while len(C_v) < k:
            maximum = -1
            l_v_max = set()

            for u in G.nodes() - C_v:  # "-" is an overloaded operator, it means 'difference'
                l_v = p_v[u]

                if len(l_v | C_v) <= k:  # "|" is an overloaded operator, it means 'union'
                    P_v = prob_cover(patients, l_v | C_v)

                    s = (P_v - P_C_v)/ len(l_v - C_v)
                    if maximum < s:
                        maximum = s
                        l_v_max = l_v
            C_v = C_v | l_v_max
            P_C_v = prob_cover(patients, C_v)

        if P_C_v > P_C:  # if we've found a better solution, update it and let us know
            C = C_v
            P_C = P_C_v
            print("_________________")
            print()
            print("Best solution updated!")
            print("Current C (ids): ", C)
            print("Current P_C (cardinality):", P_C)

    return C, P_C


def score_cover(patients, l_v):
    return len([
        p for p in patients
        if len(l_v & patients[p])>0
    ])


def set_cover(patients, l_v):
    return set([
        p for p in patients
        if len(l_v & patients[p])>0
    ])

