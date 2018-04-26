import networkx as nx


class BDDE:
    def __init__(self, G, samples, k=2, delta=0.8, f_bound=None, prob=False):
        self.G = G
        self.k = k
        self.delta = delta
        self.tree = nx.DiGraph()
        self.root = 0
        self.samples = samples
        if f_bound is None:  # default = no boundings
            self.bounding_function = lambda S: False
        elif f_bound is "furbo":
            self.bounding_function = lambda S: self.our_bound(S,samples)
        else:
            self.bounding_function = lambda S: f_bound(S, self.k, self.samples)
        if prob:
            self.scoring_function = lambda subgraph: prob_cover(self.samples,subgraph)
        else:
            self.scoring_function = lambda subgraph: score_cover(self.samples,subgraph)
        self.best_score = -1
        self.best_min = -1
        self.leaves_number = 0
        self.best_subgraph = []

    def our_bound(self, S, samples):
        return cardinality_bound(S,self.k) or prob_cover(samples,S,min=True) >= prob_cover(self.best_subgraph)

    def enumeration_algorithm(self):

        ListaNodi=list(self.G.nodes)
        self.best_score = -1
        self.best_subgraph = []
        self.leaves_number=0
        for v in ListaNodi:
            radice=v
            self.root=radice
            B=nx.DiGraph()
            self.tree=B
            self.DEPTH([],v,[])
            self.G.remove_node(v)

            if len(B)>0:
                leafs=[]
                root=None
                for n in B:
                    if B.out_degree(n)==0:  # allora è una foglia
                        leafs.append(n)
                    if B.in_degree(n)==0:  # allora è la radice
                        root=n
                self.leaves_number+=len(leafs)
                for n in leafs:
                    path_to_leaf=list(nx.shortest_path(B,root,n))
                    path_to_leaf=[node.data for node in path_to_leaf ]
                    score=self.scoring_function(path_to_leaf)

                    if score>self.best_score and len(path_to_leaf)==self.k:
                        self.best_score=score
                        self.best_subgraph=path_to_leaf
                        self.best_min = prob_cover(self.samples,self.best_subgraph,min=True)
                        print("_________________")
                        print()
                        print("Best solution updated!")
                        print("Current C (ids): ", self.best_subgraph)
                        print("Current P_C (cardinality):", self.best_score)

        print("Quante foglie?")
        print(self.leaves_number)

    def BREADTH(self, S,n,U):
        vn=n.data
        if vn in U:
            return None

        S1=S+[vn]
        if self.bounding_function(S1):
            return None

        B=self.tree
        n1=Nodo(vn)
        for nxx in self.getNodesFromBranch(n):
            n2=self.BREADTH(S1,nxx,U)
            if n2 is not None:
                B.add_edge(n1,n2)
        return n1

    def DEPTH(self, S,v,beta):
        S1=S+[v]
        if self.bounding_function(S1):
            return None

        n=Nodo(v)
        beta1=[]

        xn=self.getxn(S,v)
        xn.sort(reverse=True)
        B=self.tree
        for i in range(0,len(beta),1):
            n1=self.BREADTH(S1,beta[i],xn)
            if n1 is not None:
                B.add_edge(n,n1)
                beta1.append(n1)
        for v in xn:
            n1=self.DEPTH(S1,v,beta1)
            if n1 is not None:
                B.add_edge(n,n1)
                beta1.append(n1)
        return n

    def getNodesFromBranch(self, n):
        B=self.tree
        W=[]
        neighbors=B.neighbors(n)
        for v in neighbors:
            if B.has_edge(n,v):
                W.append(v)
        return W

    def getxn(self, S,v):
        neighbors=self.G.neighbors(v)
        L=nx.Graph()
        L.add_nodes_from(neighbors)

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


def cardinality_bound(S,k,samples=None):
        return len(S)>k


def prob_cover(patients, l_v, min=False):
    som=0.0

    for j in patients:
        prod=1.0

        for i in l_v:
            if i in patients[j]:
                prod*=1.0-patients[j][i]
        som+=prod

    if min:
        return som
    else:
        return len(patients) - som


def combinatorial_algorithm(G, k, patients, delta=0.8, prob=False):
    G = delta_removal(G, delta)

    C = {}
    if prob:
        P_C = -1
    else:
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
    return len([
        p for p in patients
        if genes_in_sample(l_v, patients[p])
    ])