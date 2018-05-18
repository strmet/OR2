import argparse
import os.path
from collections import namedtuple
import networkx as nx
from .WindFarm import WindFarm
import random
import matplotlib.pyplot as plt
import sys
import heapq
import time
import pprint as pp

class Heuristics:

    """

    py:class:: Heuristics()

    Everything related to the last part of the course

    """

    def __init__(self):
        self.__points = []
        self.__cables = []
        self.__n_substations = 0
        self.__out_dir_name = 'test'
        self.__data_select = 1
        self.__cluster = False
        self.__time_limit = 60
        self.__c = 0

        # "Named tuples", very useful for describing input data
        # without creating new classes
        self.__EdgeSol = namedtuple("EdgeSol", ["s", "d"])
        self.__Edge = namedtuple("Edge", ["s", "d", "cost"])
        self.__CableSol = namedtuple("CableSol", ["s", "d", "power"])
        self.__Point = namedtuple("Point", ["x", "y", "power"])
        self.__Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])
        
    def parse_command_line(self):

        """

        py:function:: parse_command_line(self)

        Parses the command line.

        :return: None

        """

        parser = argparse.ArgumentParser(description='Process details about instance and interface.')

        parser.add_argument('--dataset', type=int,
                            help="dataset selection; datasets available: [1,29]. " +
                                                        "You can use '30' for debug purposes")
        parser.add_argument('--cluster', action="store_true",
                            help='type --cluster if you want to use the cluster')
        parser.add_argument('--timeout', type=int,
                            help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--outfolder', type=str,
                            help='name of the folder to be created inside the /out' +
                                                          ' directory, which contains everything related to this run')

        args = parser.parse_args()

        if args.outfolder:
            self.__out_dir_name = args.outfolder

        if args.dataset:
            self.__data_select = args.dataset

        if args.cluster:
            self.__cluster = True

        if args.timeout:
            self.__time_limit = args.timeout

        self.__build_input_files()
        self.__build_custom_parameters()

    def __build_custom_parameters(self):

        """

        py:function:: __build_custom_parameters(self)

        Set the name and some constant parameters of the wind farm correctly,
        based on the dataset selection

        """

        if not type(self.__data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(self.__data_select))
        if self.__data_select <= 0 or self.__data_select >= 32:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-31]. Given: " + str(self.__data_select))

        # We assume that, in this context, we'll never have a WF >=10
        wf_number = 0
        if 0 <= self.__data_select <= 6:
            wf_number = 1
            self.__c = 10
        elif 7 <= self.__data_select <= 15:
            wf_number = 2
            self.__c = 100
        elif 16 <= self.__data_select <= 19:
            wf_number = 3
            self.__c = 4
        elif 20 <= self.__data_select <= 21:
            wf_number = 4
            self.__c = 10
        elif 26 <= self.__data_select <= 29:
            wf_number = 5
            self.__c = 10
        elif 30 <= self.__data_select <= 31:
            wf_number = 6
            self.__c = 12

        if wf_number == 0:
            raise ValueError("Something went wrong with the Wind Farm number;\n" +
                             "check the dataset selection parameter: " + str(self.__data_select))

        self.__name = "Wind Farm 0" + str(wf_number)

    def __build_input_files(self):

        """

        py:function:: __build_input_files(self)

        Sets the input file correctly, based on the dataset selection

        """

        if not type(self.__data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(self.__data_select))
        if self.__data_select <= 0 or self.__data_select >= 32:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-31]. Given: " + str(self.__data_select))

        data_tostring = str(self.__data_select)
        if 1 <= self.__data_select <= 9:
            data_tostring = "0" + data_tostring

        abspath = os.path.abspath(os.path.dirname(__file__)).strip()

        path_dirs = abspath.split('/')
        path_dirs = [str(el) for el in path_dirs]
        path_dirs.remove('')

        self.__project_path = ''
        or2_found = False

        i = 0
        while not or2_found:
            if path_dirs[i] == 'OR2':
                or2_found = True
            self.__project_path += '/' + path_dirs[i]
            i += 1

        self.__turb_file = self.__project_path + "/data/data_" + data_tostring + ".turb"
        self.__cbl_file = self.__project_path + "/data/data_" + data_tostring + ".cbl"

    def __read_turbines_file(self):

        """

        py:function:: read_turbines_file(self)

        Read the turbines file

        """

        # the following opens and closes the file within the block
        with open(self.__turb_file, "r") as fp:
            for line in fp:
                words = list(map(int, line.split()))
                self.__points.append(self.__Point(words[0], words[1], words[2]))
                if int(words[2]) < 0.5:
                    self.__n_substations += 1

        self.__n_nodes = len(self.__points)
        self.__n_turbines = self.__n_nodes - self.__n_substations

    def __read_cables_file(self):

        """

        py:function:: read_cables_file(self)

        Read the cables file

        """

        # the following opens and closes the file within the block
        with open(self.__cbl_file, "r") as fp:
            for line in fp:
                words = line.split()
                self.__cables.append(self.__Cable(int(words[0]), float(words[1]), int(words[2])))

        self.__n_cables = len(self.__cables)

    def read_input(self):

        """

        This function reads the input files by invoking the private methods which read
        both the turbines and the cables files.

        :return: None

        """

        self.__read_turbines_file()
        self.__read_cables_file()

    def MST_randomized_costs(self, delta_interval=0.1):

        """
            Generate a graph using random costs
        :param delta_interval: The interval 1 +/- delta interval is used to sample the cost
        :return: a graph represented as a list of edges

        """

        g = nx.Graph()
        edges = []

        for i in range(self.__n_nodes):
            for j in range(i + 1, self.__n_nodes):
                # Get the true distance
                dist = WindFarm.get_distance(self.__points[i], self.__points[j])

                # Shake the graph
                random_dist = random.uniform(dist * (1 - delta_interval), dist * (1 + delta_interval))

                edges.append((i, j, random_dist))

        g.add_weighted_edges_from(edges)

        MST = nx.minimum_spanning_tree(g)

        return self.__convert_graph(MST)

    def direct_mst(self, edges, substation=0):

        """

            Given a EdgeSol list and the index of the substation direct the graph

        :param edges: EdgeSol list
        :param substation: index of the substation (range [0, ..., n - 1])
        :return:

        """

        graph = [[] for i in range(self.__n_nodes)]
        succ = [[] for i in range(self.__n_nodes)]
        prec = [[] for i in range(self.__n_nodes)]

        for edge in edges:  # Add edges in both directions inside our data structure
            graph[edge.s].append(edge.d)
            graph[edge.d].append(edge.s)

        ''' 
            We need to use an additional data structure storing the visited nodes since, we must cover the most general 
            case where the graph is not still a tree.
            In fact in the previous step we have stored both directions
        '''
        visited = [False] * self.__n_nodes  # Visited nodes
        queue = [substation]  # The node where I start to search

        visited[substation] = True
        while queue:  # Using a BFS fill the data structures
            vertex = queue.pop(0)

            for successor in graph[vertex]:
                if visited[successor] is False:  # Avoid to get back and be stuck in loops

                    # New node
                    visited[successor] = True
                    queue.append(successor)
                    succ[successor] = vertex
                    prec[vertex].append(successor)

        # Correction to avoid []
        succ[substation] = substation
        graph = nx.DiGraph()

        for predecessor, successor in enumerate(succ):
            graph.add_edge(predecessor, successor)

        return prec, succ, graph

    def cost_solution(self, prec, succ, M1=10**8, M2=10**9, M3=10**10, substation=0):

        """

            Compute the cost of the solution given as input

        :param prec: List of lists. Each element's index represents one vertex, the values are the children
        :param succ: List of integers. The index is the vertex and the value is its parent
        :param M1: penalty for each crossing
        :param M2: penalty for number of cables > C
        :param M3: penalty for not feasible solutions due to cable overloading
        :param substation: ...?

        :return: Cost of the solution

        """
        num_cables_to_substation = len(prec[substation])

        queue = []
        for i in range(self.__n_nodes):  # Extract leaves
            if len(prec[i]) == 0:
                queue.append(i)

        # Initialize each turbine power with the power produced by itself
        turbine_power = [point.power for i, point in enumerate(self.__points)]
        # What we really need at the end
        edges_with_power = []

        while queue:  # Start the BFS from the leaves

            # Take a new edge
            predecessor = queue.pop(0)
            successor = succ[predecessor]

            # Since all the childs in the tree have been visited it is possible to set the proper cable
            edges_with_power.append(self.__CableSol(predecessor, successor, turbine_power[predecessor]))
            # Update the power applied with the new turbine
            turbine_power[successor] += turbine_power[predecessor]

            # Remove from the list the child and leave the other siblings
            prec[successor] = [x for x in prec[successor] if x != predecessor]

            # Add to the queue only if all the childs have been visited i.e. all the siblings have been removed
            # Best way to avoid recursion
            if len(prec[successor]) == 0 and successor != 0:
                queue.append(successor)

        cost = 0
        num_crossings = 0

        # Compute number of crossings
        for ab in edges_with_power:
            edges_violating_ab = [e2 for e2 in edges_with_power
                                  # Filter out anything that goes to/comes from a and b.
                                  if not (e2.s == ab.s or e2.d == ab.s or e2.s == ab.d or e2.d == ab.d)
                                  # Extract the violated edges only.
                                  and WindFarm.are_crossing(self.__points[ab.s],
                                                            self.__points[ab.d],
                                                            self.__points[e2.s],
                                                            self.__points[e2.d])]
            num_crossings += len(edges_violating_ab)

        num_crossings /= 2  # Since they are counted in both directions
        num_crossings = int(num_crossings)
        cost += num_crossings * M1

        for edge in edges_with_power:
            found = False
            for cable in self.__cables:
                if edge.power <= cable.capacity:  # Da correggere per il caso di soluzioni infeasible
                    found = True
                    cost += WindFarm.get_distance(self.__points[edge.s], self.__points[edge.d]) * cable.price
                    break

            if found is False:
                cost += M3

        if num_cables_to_substation > self.__c:
            cost += (num_cables_to_substation - self.__c) * M2

        return cost

    def __convert_graph(self, graph):

        """

        Take the graph from the networkx's library and transform it in a list of edges

        :param graph: Networkx's graph
        :return: EdgeSol list

        """

        edges = graph.edges(data=True)

        sol = []
        for edge in edges:
            sol.append(self.__EdgeSol(edge[0], edge[1]))

        return sol

    def grasp(self, num_edges=5, substation=0, num_fixed_edges=5):

        """

            Apply the GRASP methodology to Prim-Dijkstra's algorithm
            At each iteration, when a new edge must be chosen, it is sampled from a pool of
            closest vertexes.

            As reference, the implementation is the one in the Fischetti's book (fig 7.13)

        :param num_edges: the size of the pool
        :param substation: The index of the substation (range [0, ..., n - 1])
        :param num_fixed_edges: the number of edges fixed to the subsation
        :return: the list of oriented edges

        """

        # Build the matrix
        matrix = [
            [
                # Fill the matrix with all the distances
                WindFarm.get_distance(self.__points[i], self.__points[j])
                for i in range(self.__n_nodes)
            ]
            for j in range(self.__n_nodes)
        ]

        # Step 1
        pred = [0] * self.__n_nodes
        selected = [False] * self.__n_nodes

        selected[substation] = True
        L = matrix[substation]

        # Select the indexes of closest edges
        indexes = sorted(range(len(L)),
                         key=lambda idx: L[idx])
        indexes.remove(substation)
        indexes = indexes[:num_fixed_edges]

        for index in indexes:  # Mark the edges as in the MST
            selected[index] = True

        for index in indexes:  # Update the L data structure
            for j in range(1, self.__n_nodes):
                if not selected[j] and matrix[index][j] < L[j]:
                    L[j] = matrix[index][j]
                    pred[j] = index

        # Step 2
        for k in range(0, self.__n_nodes - 6):

            # Step 3
            mins = [sys.float_info.max] * num_edges  # Set each element to +infinite
            hs = [None] * num_edges  # The indexes

            for j in range(1, self.__n_nodes):
                if not selected[j] and L[j] < max(mins):
                    # Substitute the new value instead of the highest one in the list
                    max_value, max_index = max((x, i) for i, x in enumerate(mins))
                    mins[max_index] = L[j]
                    hs[max_index] = j

            # Now that the pool is ready choose one vertex
            hs = [x for x in hs if x is not None]
            h = random.choice(hs)  # Select randomly one of the edges

            # From now on the algorithm is the same as in the book
            # Step 4
            selected[h] = True

            # Step 5
            for j in range(1, self.__n_nodes):
                if not selected[j] and matrix[h][j] < L[j]:
                    L[j] = matrix[h][j]
                    pred[j] = h

        # Return the tree extracting the information from the pred data structure
        edges = []

        for i, j in enumerate(pred):
            edges.append(self.__EdgeSol(i, j))

        return edges

    def genetic_algorithm(self, pop_number=100, twins_per_breed=1):
        """
        This is the main structure of the genetic algorithm,
        as it is explained in the Zhou, Gen article (1997).
        Below you can find its description.
        :param pop_number: how many starting solution we're provided with
        :param twins_per_breed: how many twins will be generated for each chromosome couple
        :return: the best solution found after the timeout.
        """

        """
            ------------ THIS ALGORITHM IN SHORT ------------
            
            0: initiate and evaluate the first generation; we get:
                - pop[t]: list of tuples (cost, chromosome), ordered by their cost;
                - BEST_chromosome <-- pop[t][0][0]: the encoding of the best tree found;
                - BEST_obj_value <-- pop[t][0][1]: the lowest cost found until now;

            // Note how the encoding of the Tree is masked in this algorithm.
            while (not timeout):
                current_children <-- REPRODUCTION(pop[t]) // cross-over phase.
                children[t] <-- EVALUATION(current_children) // creates the list of tuples for such children.
                pop[t+1] <-- SELECTION(pop[t], children[t]) // applies the selection criteria.
                if pop[t+1][0][1] < BEST_obj_value:
                    BEST_chromosome <-- pop[t+1][0][0]
                    BEST_obj_value <-- pop[t+1][0][1]
                MUTATE(pop[t+1]) // gamma-ray on the chromosome; will change the costs.
                pop[t+1] <-- EVALUATION(pop[t+1]) // we have to restore the order by costs.
                t <-- t+1
            
            return DECODING(BEST_chromosome), BEST_obj_value
        """

        # We try to move the random seed a bit, before engaging the algorithm
        for i in range(1000):
            random.randint(0,1)

        print("Generating",pop_number,"solutions. This may take a while.")
        pop = []  # t=1: 'first generation'
        for i in range(pop_number):
            prec, succ, tree = self.direct_mst(self.grasp(num_edges=12))
            pop.append((self.cost_solution(prec,succ), self.__encode(tree)))

        BEST_idx = min(range(len(pop)),
                       key=lambda idx: pop[idx][0])
        BEST_chromosome = pop[BEST_idx][1]
        BEST_obj_value = pop[BEST_idx][0]

        now = time.clock()

        while True:  # time.clock() - now < self.__time_limit:
            current_children = self.__reproduction(pop, twins_per_breed)

            children = self.__evaluation(current_children)

            # From now on we have a new pop, t=t+1 ('next generation'):
            best_from_pop, pop = self.__selection(pop, children)

            if best_from_pop[0] < BEST_obj_value:
                BEST_chromosome = best_from_pop[1]
                BEST_obj_value = best_from_pop[0]
            pop = self.__mutate(pop)

    def __mutate(self, pop, select_prob=0.15, single_mut_prob=0.15):
        """
        This method take the whole population and mutates some of the chromosomes, randomly.

        :param pop: The population to be mutated
        :param select_prob: Probability that a chromosome will suffer at least one mutation
        :param single_mut_prob: Probability that, given that a chromosome is selected, a gene will mutate.
        :return: the mutated population.
        """
        return [
            self.__gamma_ray(t[1], single_mut_prob) for t in pop
            if random.random() < select_prob
        ]

    def __gamma_ray(self, chromosome, prob=0.15):
        """
        This method's name comes from Fischetti's class: sometimes, some genes are hit by some gamma ray.
        When this happens (i.e. randomly), that single gene will mutate into something else, which is random, too.
        :param chromosome: The chromosome struck by the gamma ray
        :param prob: the probability that a single gene will mutate because of the gamma ray.
        :return: the (eventually) mutated chromosome.
        """
        return [
            c if random.random() < prob else random.randint(0, self.__n_nodes-1)
            for c in chromosome
        ]

    def __selection(self, oldpop, newchild, expected_lucky_few=0.15):
        """
        This method takes the parents and their children and returns the same number of chromosomes,
        such that the population number won't change.
        The population returned will have a majority of 'good' genes,
        while some lucky few 'bad' genes are added anyways.

        :param oldpop: The parents.
        :param newchild: Their children.
        :param expected_lucky_few: The *expected proportion* of the 'lucky few' that will be spared, even if 'bad'.
        :return: The best solution within the new population and the new population itself.
        """
        pop_length = len(oldpop)

        pop_union = sorted(oldpop+newchild, key=lambda x:x[0])
        # I want to save the best value this population has to offer;
        # I have no guarantees that this value will survive the selection.
        # (probably, yes, but still.)
        best_one = pop_union[0]

        lucky_few = random.gauss(mu=expected_lucky_few, sigma=1e-6)
        bad_genes = int(pop_length*lucky_few)
        good_genes = pop_length-bad_genes

        newpop = []
        for i in range(good_genes):
            newpop.append(random.choice(pop_union[:good_genes]))
        for i in range(bad_genes):
            newpop.append(random.choice(pop_union[:-bad_genes]))

        return best_one, newpop

    def __evaluation(self, new_children):
        """
        This method takes a list of raw chromosomes and it associates, for each c., its cost.
        :param new_children: The new children to be evaluated after their creation
        :return: The children, but with their cost with the form: list of (chromosome, cost)
        """
        tuples_children = []
        for c in new_children:
            prec, succ, tree = self.direct_mst(self.__extract_EdgeSols(self.__decode(c)))
            tuples_children.append((self.cost_solution(prec,succ), c))

        return tuples_children

    def __extract_EdgeSols(self, graph):
        """
        Useful for the evaluation process, since evaluating a solution requires a list of EdgeSol.
        :param graph: the de-coded chromosome
        :return: a list of EdgeSol representing the tree
        """
        return [
            self.__EdgeSol(e[0], e[1])
            for e in graph.edges()
        ]

    def __reproduction(self, pop, twins_per_breed=1):
        """
        This method receives the population and creates the new children to be added to the new population.

        :param pop: The population that will give birth to the new children
        :param twins_per_breed: fixed a couple of chromosomes, this number is how many twins they're generating.
        :return: A list of chromosomes, representing the new children.
        """
        pop_number = len(pop)

        return [
            child
            for i in range(int(pop_number/2))
            for j in range(twins_per_breed)
            for child in self.__breed(pop[i][1], pop[pop_number-1-i][1])
        ]

    def __breed(self, parent_1, parent_2):
        """
        This method gets two chromosomes (we don't need their associated cost) and generates two twins (children).

        :param parent_1: Father
        :param parent_2: Mather
        :return: A couple of twins
        """
        chr_length = len(parent_1)
        if chr_length != len(parent_2):
            raise ValueError("Chromosomes must have the same lengths; lengths given:",
                             chr_length,
                             len(parent_2))

        child_1 = [
            parent_1[i] if random.randint(0,1) == 1 else parent_2[i]
            for i in range(chr_length)
        ]

        child_2 = [
            parent_2[i] if random.randint(0,1) == 1 else parent_1[i]
            for i in range(chr_length)
        ]

        return child_1, child_2

    def __encode(self, tree):
        """
        Takes a tree and it encodes it into a chromosome,
        which is a data structure that can be used by a genetic algorithm
        and, at the same time, be de-coded into a tree again.

        :param tree: the directed graph, representing the tree.
        :return: the chromosome, i.e. the encoding, i.e. a list of numbers representing the tree.
        """

        # We want the leaves
        leaves = [v for v in tree.nodes() if tree.in_degree(v) == 0]
        heapq.heapify(leaves)
        encoding = []
        n_nodes = len(tree.nodes())

        while len(encoding) < n_nodes-1:
            current_leaf = heapq.heappop(leaves)
            leaf_parent = list(tree[current_leaf])[0]
            encoding.append(leaf_parent)
            tree.remove_node(current_leaf)
            if tree.in_degree(leaf_parent) == 0:
                heapq.heappush(leaves, leaf_parent)

        return encoding

    def __decode(self, chromosome):
        """
        Takes a chromosome and it decodes it into a tree,
        so that we can evaluate its cost or use it outside the genetic algorithm.
        :param chromosome: the encoding, i.e. a list of numbers representing the tree.
        :return: a tree, which will be directed graph, representing the tree.
        """
        tree = nx.DiGraph()

        nodes_in_tree = [el+1 for el in range(self.__n_nodes)]
        p_hat = [
            n for n in nodes_in_tree
            if n not in chromosome
        ]
        heapq.heapify(p_hat)

        for idx,current_digit in enumerate(chromosome):
            i = heapq.heappop(p_hat)

            tree.add_edge(i,current_digit)

            if current_digit not in chromosome[idx+1:]:
                heapq.heappush(p_hat,current_digit)

        return tree

    def plot(self, graph):

        """

            Plot the Wind farm

        :param graph: The graph represented with the Networkx's data structure
        :return: Nothing

        """

        mapping = {}

        for i in range(self.__n_nodes):
            if self.__points[i].power < -0.5:
                mapping[i] = 'S{0}'.format(i)  # Now the graph is plotted with the pshysical index in memory
            else:
                mapping[i] = 'T{0}'.format(i)

        pos = {i: (point.x, point.y) for i, point in enumerate(self.__points)}

        # Avoid re scaling of axes
        plt.gca().set_aspect('equal', adjustable='box')

        # draw graph
        nx.draw(graph, pos, with_labels=True, node_size=1300, alpha=0.3, arrows=True, labels=mapping, node_color='g', linewidth=10)

        plt.show()

