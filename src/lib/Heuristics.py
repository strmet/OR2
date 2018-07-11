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
import math
import pprint as pp
import numpy as np

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
        self.__iterations = 1000
        self.__c = 0

        # Useful data structures:
        self.__EdgeSol = namedtuple("EdgeSol", ["s", "d"])
        self.__Edge = namedtuple("Edge", ["s", "d", "cost"])
        self.__CableSol = namedtuple("CableSol", ["s", "d", "power"])
        self.__Point = namedtuple("Point", ["x", "y", "power"])
        self.__Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])
        self.__sorted_distances = None

        # Parameters
        self.__encode = self.__prufer_encode
        self.__decode = self.__prufer_decode
        self.__chromosome_len = 0
        self.__prufer = True

        
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
        parser.add_argument('--strategy', type=str, choices=['prufer', 'succ'],
                            help="Choose the heuristic strategy to apply")
        parser.add_argument('--cluster', action="store_true",
                            help='type --cluster if you want to use the cluster')
        parser.add_argument('--timeout', type=int,
                            help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--outfolder', type=str,
                            help='name of the folder to be created inside the /out' +
                                 ' directory, which contains everything related to this run')
        parser.add_argument('--iterations', type=int,
                            help='How many iterations for the genetic algorithm')

        args = parser.parse_args()

        if args.outfolder:
            self.__out_dir_name = args.outfolder

        if args.dataset:
            self.__data_select = args.dataset

        if args.cluster:
            self.__cluster = True

        if args.timeout:
            self.__time_limit = args.timeout

        if args.iterations:
            self.__iterations = args.iterations

        if args.strategy == 'prufer':
            self.__encode = self.__prufer_encode
            self.__decode = self.__prufer_decode
            self.__prufer = True
        elif args.strategy == 'succ':
            self.__encode = lambda x,y: y
            self.__decode = self.__succ_decode
            self.__prufer = False
        else:
            print("No strategy given; the standard one is Prufer.")

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

        self.__sorted_distances = [
            sorted(
                [
                    (WindFarm.get_distance(p1,p2), i,j)
                    for j,p2 in enumerate(self.__points)
                ],
                key=lambda x:x[0])
            for i,p1 in enumerate(self.__points)
        ]

        self.__chromosome_len = self.__n_nodes-1 if self.__prufer else self.__n_nodes

        # self.__build_polygon()

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
        self.__cables = sorted(self.__cables,
                               key=lambda x:x.capacity)

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

    def get_graph(self, succ):
        """

        :param prec:
        :param succ:
        :return: nx.DiGraph() tree
        """
        tree = nx.DiGraph()
        for idx,s in enumerate(succ):
            tree.add_edge(idx,s)

        return tree

    def solution_cost(self, prec, succ,
                      M1=10**12, M2=10**11, M3=10**10, substation=0):
        """
        Given a tree (with both data structures, tree/edgesol) returns the cost of such tree,
        which root is in the parameter substation
        :param prec: list of lists; each list represents the incoming edges of that node (<-> position)
        :param succ: list representing, for each element, the outgoing edge of that node (i.e. parent)
        :param M1: cable overloading
        :param M2: incoming cables into the substation > C
        :param M3: crossing
        :param substation: the substation we're considering
        :return: the total costs
        """

        # Phase 0: init
        cost = 0
        edgesols = []

        # Phase 1: cable costs
        total_power, cable_costs = self.__rec_inc_pwrcosts(prec, substation, edgesols,
                                                           bigM=M1)
        cost += cable_costs

        # Phase 2: incoming cables into the substation
        if len(prec[substation]) > self.__c:
            cost += M2 * (len(prec[substation]) - self.__c)

        # Phase 3: crossings
        # TODO: re-introduce the crossings inside the cost
        '''num_crossings = 0
        for e1 in edgesols:
            violating_edges = [
                e2 for e2 in edgesols
                # Filter out anything that goes to/comes from a and b.
                if not (e2.s == e1.s or e2.d == e1.s or e2.s == e1.d or e2.d == e1.d)
                # Extract the violated edges only.
                and WindFarm.are_crossing(self.__points[e1.s],
                                          self.__points[e1.d],
                                          self.__points[e2.s],
                                          self.__points[e2.d])
            ]
            num_crossings += len(violating_edges)
        num_crossings /= 2  # Since crossings are counted in both directions.
        num_crossings = int(num_crossings)  # Since we want a int number, even if num_crossing is always even.
        cost += num_crossings * M3'''

        return cost

    def __rec_inc_pwrcosts(self, prec, current_node, edgesols, bigM=10**12):
        """
        Given a tree (represented by the the data structure prec), this function will recursively compute
        the costs given by the cables' displacement, by setting the right cable (based on the incoming power).
        Also, this method will add a big-M cost for each unfeasible cable in the solution.
        Finally, this method will add every edge visited on the edgesols list (by adding a EdgeSol namedtuple).

        :param prec: List of lists; each list represent the incoming edges to that node.
        :param current_node: The node we're visiting at the moment.
        :param edgesols: The list of EdgeSols visited until this moment
        :param bigM: the cost of to the unfeasibility due the cable overloadings.
        :return: the total power incoming to current_node and the total costs computed by this call.
        """
        costs = 0
        cumulative_power = self.__points[current_node].power

        for inc in prec[current_node]:
            # Recursive step:
            prev_power, prev_costs = self.__rec_inc_pwrcosts(prec, inc, edgesols,
                                                             bigM=bigM)

            # Accumulate the power onto this point
            cumulative_power += prev_power
            # Accumulate the costs
            costs += prev_costs
            feasible, cable = self.__get_cable(prev_power)
            price = cable.price if feasible else 0
            costs += price * WindFarm.get_distance(self.__points[current_node],
                                                   self.__points[inc])
            if not feasible:
                costs += bigM*(prev_power-cable.capacity)

            edgesols.append(self.__EdgeSol(inc,current_node))

        return cumulative_power, costs

    def __get_cable(self, power):
        """
        This method assumes that self.__cables is ordered by capacity (as it should).
        Given a node and its outgoing power,
        this method returns the correct cable to support its power.
        If the power's too much, this method will return the tuple (False, MAXCABLE),
        to ensure that the caller will know that he's going to add an unfeasible solution to the pool.
        If everything's O.K., the function will return (True, CORRECTCABLE), where CORRECTCABLE means that
        it chooses the cable that minimizes the capacity needed (and therefore, its cost).

        :param power: the outgoing power to be endured by the cable
        :return: True/False and the correct cable that minimizes the costs,
                or the biggest cable if it returns False.
        """
        for cable in self.__cables:
            if power <= cable.capacity:
                return True, cable

        return False, self.__cables[-1]

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

    def genetic_algorithm(self,
                          pop_number=300,
                          diversification_rate=7,
                          memory=True):
        """
        This is the main structure of the genetic algorithm,
        as its paradigm has been presented in class.

        :param pop_number: the number of solutions per generation
        :param diversification_rate: how many iterations before we diversify the population
        :return: the best solution found after the timeout and its cost.
        """

        # We try to move the random seed 'a bit', before engaging the algorithm
        for i in range(1000):
            random.randint(0,1000)

        print("Generating",pop_number,"solutions. This may take a while.")
        pop = []  # t=1: 'first generation'

        grasp_pop = int(pop_number*(1/5))  # TODO we're experimenting proportions, here
        for i in range(grasp_pop):
            prec, succ, tree = self.direct_mst(self.grasp(num_edges=5))
            pop.append((self.solution_cost(prec, succ), self.__encode(prec, succ)))

        bfs_pop = int(pop_number*(1/5))  # TODO we're experimenting proportions, here
        for i in range(bfs_pop):
            prec, succ = self.bfs_build(substation=0, nearest_per_node=7, selected_per_node=3)
            pop.append((self.solution_cost(prec, succ), self.__encode(prec, succ)))

        rnd_number = pop_number-bfs_pop-grasp_pop
        for i in range(rnd_number):    # TODO we're experimenting proportions, here
            random_chromosome = [random.randint(0,self.__n_nodes-1) for i in range(self.__chromosome_len)]
            prec, succ = self.__decode(random_chromosome)
            pop.append((self.solution_cost(prec, succ), random_chromosome))

        BEST_idx = min(range(len(pop)),
                       key=lambda idx: pop[idx][0])
        BEST_chromosome = pop[BEST_idx][1]
        BEST_obj_value = pop[BEST_idx][0]

        # Initializing parameters
        fitness = float(sum(el[0] for el in pop)) / len(pop)
        print("First fitness:", '%.6e' % fitness)
        BEST_fitness = fitness

        intensification_phase = True

        # remove after debugging (visualizating purposes)
        prec, succ = self.__decode(BEST_chromosome)
        self.plot(self.get_graph(succ))

        # Iterations counter(s)
        j = 0
        int_counter = 0

        print("Genetic Algorithm starts now...")
        starting_time = time.clock()
        current_time = time.clock()

        while j < self.__iterations and current_time - starting_time < self.__time_limit:
            if not intensification_phase:
                intensification_phase = True
            elif int_counter >= diversification_rate:
                intensification_phase = False
                int_counter = 0
                BEST_fitness = 10**12
                print("--- GAMMA RAYS INCOMING ---")
            else:
                intensification_phase = True
            int_counter += 1

            current_children = self.__reproduction(pop, BEST_chromosome)

            # Duplicates check
            self.__check_duplicates(pop, current_children)

            new_children = self.__evaluation(current_children)

            # From now on we have a new pop, t=t+1 ('next generation'):
            e_l_f = 0.00 if intensification_phase else 0.95
            fitness, best_from_pop, pop = self.__selection(pop, new_children,
                                                           expected_lucky_few=e_l_f)
            # debugging; to be deleted
            print("New pop, new fitness",'%.6e' % fitness)

            # Updating the "best" values
            if fitness < BEST_fitness:
                BEST_fitness = fitness
                int_counter = 0
            if best_from_pop[0] < BEST_obj_value:
                BEST_chromosome = best_from_pop[1]
                BEST_obj_value = best_from_pop[0]

            # Gamma ray incoming
            if not intensification_phase:
                s_p = 1  # We should mutate everything
                s_m_p = 0.25
                pop = self.__mutate(pop,
                                    select_prob=s_p,
                                    single_mut_prob=s_m_p)

            current_time = time.clock()
            j += 1

        print("Elapsed time:")
        print(current_time-starting_time)
        print("Generation passed:", j)

        return self.__decode(BEST_chromosome), BEST_obj_value

    def __mutate(self, pop, select_prob=0.20, single_mut_prob=0.20):
        """
        This method take the whole population and mutates some of the chromosomes, randomly.

        :param pop: The population to be mutated
        :param select_prob: Probability that a chromosome will suffer at least one mutation (i.e. he will be selected).
        :param single_mut_prob: Probability that, given that a chromosome is selected, a gene will mutate.
        :return: the mutated population.
        """

        mutated_pop = []

        for t in pop:
            if random.random() < select_prob:
                mutated_chromosome = self.__gamma_ray(t[1], single_mut_prob)
                prec, succ = self.__decode(mutated_chromosome)
                new_cost = self.solution_cost(prec, succ)
                mutated_pop.append((new_cost, mutated_chromosome))
            else:
                mutated_pop.append(t)

        return mutated_pop

    def __gamma_ray(self, chromosome, prob=0.20):
        """
        This method's name comes from Fischetti's class: sometimes, SOME genes are hit by some gamma ray.
        When this happens (i.e. RANDOMLY), a single gene will mutate into something else, which is random, too.
        :param chromosome: The chromosome struck by the gamma ray
        :param prob: the probability that a SINGLE GENE will mutate because of the gamma ray.
        :return: the (EVENTUALLY) mutated chromosome.
        """
        return [
            random.randint(0, self.__n_nodes-1) if random.random() < prob else g
            for g in chromosome
        ]

    def __selection(self, oldpop, newchild,
                    expected_lucky_few=0.20):
        """
        This method takes the parents and their children and returns the same number of chromosomes,
        such that the population number won't change.
        The population returned will have a majority of 'good' genes,
        while some lucky few 'bad' genes are added anyways.
        The expected value of 'bad' genes is a parameter of this function.

        :param oldpop: The parents. List of tuples (cost, chromosome)
        :param newchild: Their children. List of tuples (cost, chromosome)
        :param expected_lucky_few: The EXPECTED PROPORTION of the 'lucky few' that will be spared, even if 'bad'.
        :return: The fitness of the new population,
                the new best solution within the new population and
                the new population itself.
        """

        pop_length = len(oldpop)

        pop_union = sorted(oldpop+newchild, key=lambda x:x[0])
        # I want to save the best value this population has to offer;
        # I have no guarantees that this value will survive the selection.
        # (probably, yes, but still.)
        best_one = pop_union[0]

        # If the expectation is lesser or equal than zero,
        # or greater or equal than one, we interpret it as
        # there is no aleatory process behind the "lucky_few" selections.
        if expected_lucky_few <= 0:
            lucky_few = 0
        elif expected_lucky_few >= 1:
            lucky_few = 1
        else:
            lucky_few = random.gauss(
                mu=expected_lucky_few,
                sigma=5*expected_lucky_few/100  # This is a totally a arbitrary choice (sigma = +/- 5% mu)
            )
            # We want to guarantee that we've obtained a probability which makes sense
            lucky_few = max(0,lucky_few)
            lucky_few = min(lucky_few,1)

        bad_genes = int(pop_length*lucky_few)
        good_genes = pop_length-bad_genes

        newpop = []
        for i in range(good_genes):
            good_gene = random.choice(pop_union[:good_genes])
            newpop.append(good_gene)
            pop_union.remove(good_gene)
        for i in range(bad_genes):
            bad_gene = random.choice(pop_union)
            newpop.append(bad_gene)
            pop_union.remove(bad_gene)

        fitness = float(sum(el[0] for el in newpop)) / len(newpop)
        return fitness, best_one, newpop

    def __check_duplicates(self, oldpop, newchild):
        """
        This method assumes oldpop has no duplicates (even if so, this shouldn't be a problem);
        by remembering the chromosomes in oldpop, it checks for duplicates in newchild.
        If any chromosome \in newchild appears twice or more, or it has an occurence in newpop,
        it will suffer a certain mutation by sending a single gamma ray onto them.

        :param oldpop: the previous population; list of (COST, CHROMOSOME) tuples
        :param newchild: the newborns; list of CHROMOSOMES **ONLY**!!
        :return: Nothing. This method directly modifies the children, if necessary.
        """

        # Time to remove the duplicates.
        seen = set([str(x[1]) for x in oldpop])
        for idx,x in enumerate(newchild):
            string = str(x)
            if string in seen:
                # Warning; implementation note:
                # Parameters by reference should be passed like this.
                # (notice how passing it x, directly, would pass this parameter by value).
                self.__single_gamma_ray(newchild[idx])
            else:
                seen.add(string)

        # are there STILL some duplicates? Let's see.
        seen = set([str(x[1]) for x in oldpop])

        for idx,x in enumerate(newchild):  # search again for duplicates
            string = str(x)
            if string in seen:  # if so, just randomize this single chromosome...
                self.__gamma_ray(newchild[idx], prob=0.20)

        # at this point, the population may have duplicates with low probability,
        # but at least we modified the most we can
        return

    def __single_gamma_ray(self, chromosome):
        """
        This method takes a chromosome and strucks it with a CERTAIN, SINGLE gamma ray.
        The gene to be mutated is randomly chosen.

        :param chromosome: The chromosome struck by the gamma ray
        :return: Nothing. This method will directly modify the chromosome.
        """

        gene_idx = random.randint(0, len(chromosome)-1)
        chromosome[gene_idx] = random.randint(0, self.__n_nodes-1)
        return

    def __evaluation(self, new_children):
        """
        This method takes a list of raw chromosomes and it associates, for each c., its cost.
        :param new_children: The new children to be evaluated after their creation
        :return: The children, but with their cost with the form: list of (cost, chromosome)
        """

        new_eval_children = []
        for c in new_children:
            prec, succ = self.__decode(c)
            new_eval_children.append((self.solution_cost(prec, succ),c))

        return new_eval_children

    def __reproduction(self, pop, alpha):
        """
        This method receives the population and creates the new children to be added to the new population.

        :param pop: The population that will give birth to the new children
        :return: A list of chromosomes, representing the new children.
        """

        pop_number = len(pop)

        # I need to make a deepcopy of this list, or pop will be deleted.
        pop_temp = list(pop)

        children = []

        # If the pop number is even, mating processes will be fine (so that no one will be left alone).
        if pop_number % 2 != 0:
            # If the pop number is odd, instead,
            # we choose one chromosome to reproduce by meiosis (later):
            alone_chromosome = random.choice(pop_temp)
            pop_temp.remove(alone_chromosome)
            self.__gamma_ray(alone_chromosome, prob=0.5)
            children.append(alone_chromosome)

        # Let the mating process begin.
        while len(pop_temp) > 0:
            parent_1 = random.choice(pop_temp)
            pop_temp.remove(parent_1)
            parent_2 = random.choice(pop_temp)
            pop_temp.remove(parent_2)
            new = self.__breed(parent_1[1], parent_2[1])
            for child in new:
                children.append(child)

        # Let the alpha mate with everybody.
        for mate in pop:
            if str(mate[1]) is not str(alpha):
                new = self.__breed(alpha, mate[1])
                for child in new:
                    children.append(child)

        return children

    def __breed(self, parent_1, parent_2):
        """
        This method gets two chromosomes (we don't need their associated cost),
        and generates two children.
            -   The first one will be the random cross-over between the two parents.
            -   The second one will be complementary of the first one.

        This policy allows to deeply explore (by intensificating) the solutions,
        while still creating some diversity in this search.

        :param parent_1: Father (chromosome)
        :param parent_2: Mather (chromosome)
        :return: A list of two children.
        """

        chr_length = len(parent_1)
        child_1 = []
        child_2 = []

        for i in range(chr_length):
            coin_flip = random.randint(0,1)

            child_1.append(parent_1[i] if coin_flip == 1 else parent_2[i])
            child_2.append(parent_2[i] if coin_flip == 1 else parent_1[i])

        return [child_1, child_2]

    def __prufer_encode(self, prec, succ):
        """
        Here we follow the prufer encoding.
        Given the data structure representing the tree, we return the encoding,
        which is the sequence of numbers (1 <= - <= self.__n_nodes-1) we want.
        :param prec: list of lists. Each list represent the children of the node
        :param succ: list of numbers representing the parent of the node
        :return: a list of numbers, i.e. the encoded tree, which is a chromosome.
        """

        leaves = [
            idx
            for idx,p_list in enumerate(prec)
            if len(p_list) == 0
        ]
        heapq.heapify(leaves)
        encoding = []
        n_nodes = len(prec)
        while len(encoding) < n_nodes - 1:
            current_leaf = heapq.heappop(leaves)
            leaf_parent = succ[current_leaf]

            encoding.append(leaf_parent)

            prec[leaf_parent].remove(current_leaf)
            if len(prec[leaf_parent]) == 0:
                heapq.heappush(leaves, leaf_parent)

        return encoding

    def __prufer_decode(self, chromosome, substation=0):
        """

        :param chromosome:
        :return:
        """

        prec = [[] for i in range(self.__n_nodes)]
        succ = [0 for i in range(self.__n_nodes)]
        nodes_in_tree = [i for i in range(self.__n_nodes)]
        p_hat = [
            n for n in nodes_in_tree
            if n not in chromosome
        ]

        heapq.heapify(p_hat)
        for idx,current_digit in enumerate(chromosome):
            i = heapq.heappop(p_hat)
            prec[current_digit].append(i)
            succ[i] = current_digit

            if current_digit not in chromosome[idx+1:]:
                heapq.heappush(p_hat, current_digit)

        last = heapq.heappop(p_hat)
        succ[last] = last

        self.__treefix(prec, succ, substation)

        return prec, succ

    def __treefix(self, prec, succ, node):
        """
        This recursive method check if the node passed is the root of the anti-arborescence.
        A node is the root <=> succ[node] = node.
        If the current node isn't the root, we fix such situation, with a recursive call
        made onto the next node, which indeed shouldn't be the next one of the current node.

        :param prec: list of lists, data structure of our tree
        :param succ: list of numbers, data structure of our tree
        :param node: the current node to be examined
        :return: Nothing. The data structures will be directly modified if necessary.
        """

        next_node = succ[node]
        if next_node != node:  # This means that the root isn't in the current node
            # Therefore, we should find it recursively.
            self.__treefix(prec, succ, next_node)

            # At this point we've found the root: it's the next_node.
            # Node will become the new root.
            prec[next_node].remove(node)
            succ[next_node] = node
            prec[node].append(next_node)
            succ[node] = node  # Root condition in our data structures.
        else:
            succ[node] = node

        # At this point, node is the new root.
        # Other recursive calls will fix the situation, if necessary.

        return

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

    def bfs_build(self, substation=0, nearest_per_node=5, selected_per_node=2):
        """
        This method builds a solution with a BFS-like search.

        :param substation: which node is our substation?
        :param nearest_per_node: how many nodes are considered for the linking?
        :param selected_per_node: how many nodes are actually linked per node?
        :return: prec, succ (data structure representing a tree)
        """
        if selected_per_node > nearest_per_node:
            raise ValueError("These parameters make no sense. Given:",
                             nearest_per_node,
                             selected_per_node)
        # We move the random seed, before proceeding.
        for i in range(1000):
            random.random()

        # 12 is the maximum of every dataset, besides the infinite-cable one.
        c = min(self.__c, 12)

        # It takes at least the following number to have a sustainable solution
        # (we want to endure the power --> feasible solution)
        # x = ceiling (sum_{p \in points} {p.pwr} / max(cable_capacity))
        min_cables_to_subst = int(
            sum(p.power for p in self.__points)
            /
            max(self.__cables, key=lambda cab:cab.capacity).capacity
        ) + 1  # This "+1" works as the ceiling function, while int() is a well-known floor function

        # We choose a random integer value between these two
        p = random.randint(min_cables_to_subst, c)

        starting_nodes = set([self.__sorted_distances[substation][j][2] for j in range(p+1)])
        starting_nodes.remove(substation)

        prec = [
            []
            for i in range(self.__n_nodes)
        ]
        succ = [
            i
            for i in range(self.__n_nodes)
        ]
        visited = set()

        visited.add(substation)
        queue = []
        for node in starting_nodes:
            prec[substation].append(node)
            succ[node] = substation
            queue.append(node)
            visited.add(node)

        # We randomize the order in which we explore the first elements of the queue
        for i in range(100):
            random.shuffle(queue)

        # After this warm-up, the algorithm may begin:
        while len(queue) > 0:
            node = queue[0]
            queue.remove(node)
            has_prec = False

            closests = [self.__sorted_distances[node][j][2] for j in range(nearest_per_node)]
            selected = random.sample(closests,selected_per_node)
            for c in selected:
                if c not in visited:
                    has_prec = True
                    visited.add(c)
                    if c not in starting_nodes:
                        queue.append(c)
                    prec[node].append(c)
                    succ[c] = node

            # If we didn't manage to find some proper neighbours, we at least try again with a wider range
            if not has_prec:
                closests = [self.__sorted_distances[node][j][2] for j in range(int(nearest_per_node*2)+1)]
                selected = random.sample(closests,int(selected_per_node*1.5)+1)
                for c in selected:
                    if c not in visited:
                        visited.add(c)
                        if c not in starting_nodes:
                            queue.append(c)
                        prec[node].append(c)
                        succ[c] = node

        # This algorithm doesn't guarantee to add every node.
        if len(visited) < self.__n_nodes:  # If any is left behind, randomly choose its successor.
            not_visited = set([i for i in range(self.__n_nodes)]) - visited

            for node in not_visited:
                # But be smart about it:
                closests = [self.__sorted_distances[node][j][2] for j in visited][:nearest_per_node]
                closests.remove(node)
                selected = random.choice(closests)  # And at the same time, randomize this smart choice.
                prec[selected].append(node)
                succ[node] = selected

        return prec, succ

    def __succ_decode(self, chromosome, substation=0):
        """
        Decoding the succ data structure with our own personal method (DFS_loopfix)

        :param chromosome: Remember, here chromosome === succ
        :param substation: Where is the substation, in this data-set?
        :return: prec, succ
        """
        prec = [[] for i in range(self.__n_nodes)]
        for idx, g in enumerate(chromosome):
            if g != idx:  # The root can't have itself as predecessor.
                prec[g].append(idx)

        # We now fix every loop in this chromosome.
        not_visited = set([x for x in range(self.__n_nodes)])
        connected_components = dict()
        while len(not_visited) > 0:  # If we still have to visit some nodes
            roots = [
                x
                for idx, x in enumerate(chromosome)
                if x == idx and x in not_visited
            ]  # Extract the roots of every connected component
            #print("We have those roots:",roots)
            #input()
            for root in roots:
                visited = set()
                self.__treevisit(prec, chromosome, root, visited, not_visited)  # Visit them
                connected_components[root] = visited
            if len(not_visited) > 0:  # If we've left behind some nodes
                # This means, because of the theorem, that we have a loop, somewhere.
                rnd_node = random.sample(not_visited, 1)[0]  # Pick a random node to start the loop-search
                visited_in_this_cc = set()  # Elements visited in this connected components
                self.__DFS_loopfix(rnd_node, prec, chromosome, visited_in_this_cc)

        # At this point, we may have several disconnected components.
        roots = [
            x
            for idx, x in enumerate(chromosome)
            if x == idx
        ]  # extract the roots of every connected component

        #print(connected_components)
        if substation in connected_components:  # This may be useful for efficacy/efficiency purposes
            roots.remove(substation)

        while len(connected_components) > 1:  # are there multiple connected components?
            root = random.choice(roots)  # select any
            #self.plot(self.get_graph(chromosome))
            #print("root:",root)
            #print("prec[root]:",prec[root])
            #print("succ[root]:",chromosome[root])

            # Find the closest point to this root:
            # such point can't be in the connected component of the root.
            closest = self.__find_closest(root, connected_components[root])
            #print("closest:",closest)
            #print("prec[closest]:", prec[closest])
            #print("succ[closest]:", chromosome[closest])
            chromosome[root] = closest
            prec[closest].append(root)

            # Update the connected components dict
            closest_root = self.__find_corresponding_connected_component(closest, connected_components)
            connected_components[closest_root] = connected_components[root] | connected_components[closest_root]
            del connected_components[root]
            #print("Updated connected components:",connected_components)
            #input()
            # We have one less connected component, now.
            roots.remove(root)

        # Sets the root in the substation, if the root isn't in the substation.
        self.__treefix(prec, chromosome,
                       node=substation)

        return prec, chromosome

    def __find_corresponding_connected_component(self, node, connected_components):
        """
        Given a node, it finds its root in his connected component.
        :param node: a node (integer)
        :param connected_components: dictionary of sets, representing connected components
        :return: The root of its connected component
        """
        for c in connected_components:
            if node in connected_components[c]:
                return c

        return None

    def __find_closest(self, root, connected_component):
        """
        Finds the closest element from a root to another connected component and returns it
        :param root: a node (integer)
        :param connected_component: dictionary of sets, representing connected components
        :return: The nearest node to the root, outside its connected component
        """

        # The following is a for cycle onto a list of (dist, i,j) tuples,
        # where i is our root, j is whatever we're interested in

        for element in self.__sorted_distances[root]:
            if element[2] not in connected_component:  # If the outgoing component is not in the connected component...
                return element[2]  # Once we've found the closest, just return it

        # If this function has correctly been called,
        # there is at least one node not in the connected component of this root.
        raise ValueError("Something's wrong. This shouldn't happen at all")

    def __treevisit(self, prec, succ, node, visited, not_visited):
        """
        This method visits a tree by recursively observing the prec(s) of 'node':
        the visits cause 'not_visited' to remove 'node'.
        This method assumes there are no loops and that this connected component has never been visited.

        :param prec: list of lists, data structure of our tree.
        :param succ: list of integers, data structure of our tree.
        :param node: the current node to be examined.
        :param not_visited: the nodes that we still have to visit.
        :return: Nothing. The connected component visited with this method and
                the 'not_visited' data structures will be directly modified.
        """
        #print("node:", node)
        not_visited.remove(node)
        visited.add(node)
        #print("adding it to visited:",visited)
        precs = prec[node]
        #print("precs:", precs)
        #input()
        for p in precs:
            self.__treevisit(prec, succ, p, visited, not_visited)

        return

    def __DFS_loopfix(self, node, prec, succ, visited):
        """
        This method takes a node and checks if whether or not
        its predecessors have been visited in its connected component.
        If they weren't visited, we keep visiting his predecessors, recursively.
        If this node was visited, instead, we delete the node's outgoing edge,
        and we flip the direction of the inquired edge (our former predecessor).
        
        One could prove the following theorem:
        
            HP:
                - let G = (V,E) be a directed graph
                - let S <= V be a full and connected subgraph
                - let out_degree(v) <= 1 \forall v \in V
            TH:
                - # loops in the subgraph induced by S <= 1
        
        This means that in the moment we've found a loop and we've fixed it,
        we can terminate our search.

        :param node: current node to be visited
        :param prec: list of lists, representing the tree
        :param succ: list of nodes, representing the tree
        :param visited: whatever we've visited, already.
        :return: nothing. The tree wil be modified, directly.
        """

        visited.add(node)
        n = succ[node]
        if n in visited:
            # This is a loop and we must fix the situation:
            prec[n].remove(node)  # We choose to disconnect the looping edge
            succ[node] = node  # and that the new root is the current node

            return  # We've found a loop and we've fixed it: horray!

        self.__DFS_loopfix(succ[node], prec, succ, visited)  # No loops found, yet: search recursively
        return

    # --- OLD IDEAS ---

    '''def __build_polygon(self):
        """
        Takes the points, scans them and builds the smallest polygon that contains them.
        Here, "smallest polygon" = the hull containing them.

        This D&C algorithm runs in O(n log(n)) expected time, but it may have O(n^2) runtime.
        (just like quicksort)

        The implemented algorithm is described here: https://en.wikipedia.org/wiki/Quickhull
        If you wish to understand how it works, we truly reccomend to check that wiki page.

        :return: Nothing; the hull will be built inside the class
        """

        self.__polygon = []

        # Remembering the point idx is necessary.
        labeled_points = [(idx, point) for idx, point in enumerate(self.__points)]
        sx = min(labeled_points, key=lambda t:t[1].x)  # Get the leftmost point. Call this point A.
        dx = max(labeled_points, key=lambda t:t[1].x)  # Get the rightmost point. Call this point B.

        # Add A, B to the hull
        self.__polygon.append(sx)
        self.__polygon.append(dx)

        # Separate the two group of points
        over_ab = []
        under_ab = []
        for t in labeled_points:
            if Heuristics.is_ccw(t[1], sx[1],dx[1]):
                over_ab.append(t)
            else:
                under_ab.append(t)

        # Call the core method, quickhull
        self.__quickhull(over_ab, sx, dx)
        self.__quickhull(under_ab, sx, dx)

        mean_min_distance = self.__mean_min_dist()
        hull_len = len(self.__polygon)
        to_be_added = []
        for i in range(hull_len):
            j = (i+1) % hull_len
            for idx, point in enumerate(self.__points):
                d = self.__projected_distance(point, self.__polygon[i][1], self.__polygon[j][1])

                if d < 1.25*mean_min_distance:
                    to_be_added.append((idx, point))

        substation = [p for p in self.__points if p.power < -0.5][0]
        maximum_dist_from_sub = max(WindFarm.get_distance(p, substation) for p in self.__points)
        self.__polygon = [
            t
            for t in self.__polygon
            if WindFarm.get_distance(t[1],substation) >= maximum_dist_from_sub/2
        ]
        self.__polygon = set([t[0] for t in self.__polygon])
        to_be_added = set([t[0] for t in to_be_added])

        self.__polygon = self.__polygon | to_be_added


    def __mean_min_dist(self):

        mean = 0.0
        cnt = 0
        for idx,p in enumerate(self.__points):
            if p.power > 0.5:
                minimum = min(WindFarm.get_distance(p, p2) for p2 in self.__points if p2.power > 0.5)
                cnt += 1
                mean += minimum

        mean /= cnt

        return int(mean)


    def __projected_distance(self, p, a,b):
        return (
            abs((b.x-a.x)*(a.y-p.y) - (a.x-p.x)*(b.y-a.y))
            /
            math.sqrt((b.x-a.x)**2 + (b.y-a.y)**2)
        )

    def __quickhull(self, points, a, b):
        """
        Famous recursive algorithm to find the hull of the points given.

        This recursive method accepts points in some region w.r.t. the segment ab:
        this means that ab doesn't cut points.

        Then, it takes the points and recursively searches for the farthest point from ab,
        which is in the hull we want to compute. If any point is outside the the abc triangle,
        we recursively repeat the process until no points are left outside of the triangle.

        "Being inside the triangle" means that those points can't be the hull.

        :param points: the points we're investigating
        :param a: leftmost point of the segment
        :param b: rightmost point of the segment
        :return: Nothing. This method will add the points to the hull.
        """

        # Find the farthest point, say C, from segment AB
        c = max(points, key=lambda t:Heuristics.distance_point_to_line(t[1], a[1],b[1]))

        # Add it to the shell
        self.__polygon.append(c)

        # Check whatever is outstide the triangle ABC
        ac = []
        cb = []
        for t in points:
            if Heuristics.is_ccw(t[1], a[1],c[1]):
                ac.append(t)
            elif Heuristics.is_ccw(t[1], c[1],b[1]):
                cb.append(t)
            else:  # This means this point is inside the triangle; we're fine.
                pass

        # Recursive step
        if len(ac) > 0:
            self.__quickhull(ac, a,c)
        if len(cb) > 0:
            self.__quickhull(cb, c,b)
    

    @staticmethod
    def distance_point_to_line(p, a,b):
        """
        This method simply computes the distance between a point p and a segment ab.

        :param p: point
        :param a: point
        :param b: point
        :return: distance(p, ab)
        """
        return abs((p.y - a.y) * (b.x - a.x) - (b.y - a.y) * (p.x - a.x))

    @staticmethod
    def is_ccw(p, a,b):
        """
        This metod takes vector 0p and the vector ab, and returns true if the vector 0p is counter-clock-wise
        w.r.t. the segment (vector) ab given.

        :param p: point
        :param a: point
        :param b: point
        :return: True if the vector 0p is counter-clock-wise w.r.t. the segment (vector) ab, False otherwise
        """
        return (p.y-a.y)*(b.x-a.x) - (b.y-a.y)*(p.x-a.x) > 0
'''


