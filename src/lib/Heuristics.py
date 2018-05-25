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
        self.__c = 0

        # "Named tuples", very useful for describing input data
        # without creating new classes
        self.__EdgeSol = namedtuple("EdgeSol", ["s", "d"])
        self.__Edge = namedtuple("Edge", ["s", "d", "cost"])
        self.__CableSol = namedtuple("CableSol", ["s", "d", "power"])
        self.__Point = namedtuple("Point", ["x", "y", "power"])
        self.__Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])

        # Useful for our implementation
        self.__maxpricecable = None
        self.__distances = None

        # Parameters
        self.__encode = self.__prufer_encode
        self.__decode = self.__prufer_decode
        
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
        # debug
        parser.add_argument('--strategy', type=str, choices=['prufer', 'succ'],
                            help="Choose the heuristic strategy to apply")
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
        if args.strategy == 'prufer':
            self.__encode = self.__prufer_encode
            self.__decode = self.__prufer_decode
        '''elif args.strategy == 'succ':
            self.__encode = lambda x,y: y
            self.__decode = self.__succ_decode'''

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
        self.__distances = np.array([
            sorted([
                (WindFarm.get_distance(self.__points[i], self.__points[j]), i, j)
                for j in range(self.__n_nodes)
            ], key=lambda x:x[0])
            for i in range(self.__n_nodes)
        ])

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
        self.__maxpricecable = max(self.__cables, key=lambda c:c.price)

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
                      M1=10**9, M2=10**12, M3=10**10, substation=0):
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

        :param prec:
        :param current_node:
        :param edgesols:
        :param bigM:
        :return:
        """
        costs = 0
        cumulative_power = self.__points[current_node].power

        """
            Note:
                it is not necessary to check whether or not current_node is a leaf.
                If it's a leaf, prec[current_node] is empty,
                and therefore the following for/in is ignored.
        """

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
        for cable in self.__cables:
            if power <= cable.capacity:
                return True, cable

        return False, max(self.__cables, key=lambda c:c.capacity)

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
                          pop_number=100,
                          max_iter=5000,
                          diversificating_rate=5,
                          memory = True,
                          prufer=True):
        """
        This is the main structure of the genetic algorithm,
        as it is explained in the Zhou, Gen article (1997).
        Below you can find its description.
        :param pop_number: how many starting solution we're provided with
        :param max_iter:
        :param memory:
        :param prufer:
        :return: the best solution found after the timeout and its cost.
        """

        """
            ------------ THIS ALGORITHM IN SHORT ------------
            
            0: initiate and evaluate the first generation; we get:
                - pop[t]: list of tuples (cost, chromosome);
                - BEST_chromosome <-- pop[t][0][1]: the encoding of the best tree found;
                - BEST_obj_value <-- pop[t][0][0]: the lowest cost found until now;

            // Note how the encoding of the Tree is masked in this algorithm.
            while (not timeout):
                current_children <-- REPRODUCTION(pop[t]) // cross-over phase.
                children[t] <-- EVALUATION(current_children) // creates the list of tuples for such children.
                pop[t+1] <-- SELECTION(pop[t], children[t]) // applies the selection criteria.
                if pop[t+1][0][0] < BEST_obj_value:
                    BEST_chromosome <-- pop[t+1][0][1]
                    BEST_obj_value <-- pop[t+1][0][0]
                MUTATE(pop[t+1]) // gamma-ray on the chromosome; will change the costs.
                pop[t+1] <-- EVALUATION(pop[t+1]) // we have to restore the order by costs.
                t <-- t+1 // next generation
            
            return DECODING(BEST_chromosome), BEST_obj_value
        """

        # We try to move the random seed 'a bit', before engaging the algorithm
        for i in range(1000):
            random.randint(0,1000)

        print("Generating",pop_number,"solutions. This may take a while.")
        grasp_pop = int(pop_number/2)
        pop = []  # t=1: 'first generation'
        for i in range(grasp_pop):
            prec, succ, tree = self.direct_mst(self.grasp(num_edges=5))
            pop.append((self.solution_cost(prec, succ), self.__encode(prec, succ)))
        rnd_number = pop_number-grasp_pop

        for i in range(rnd_number):
            random_chromosome = [random.randint(0,self.__n_nodes-1) for i in range(self.__n_nodes-1)]
            prec, succ = self.__decode(random_chromosome, debug=True)
            pop.append((self.solution_cost(prec, succ), random_chromosome))

        BEST_idx = min(range(len(pop)),
                       key=lambda idx: pop[idx][0])
        BEST_chromosome = pop[BEST_idx][1]
        BEST_obj_value = pop[BEST_idx][0]

        # Initializing parameters
        fitness = float(sum(el[0] for el in pop)) / len(pop)
        print("First fitness:", fitness, math.log(fitness,10))

        intensification_phase = True
        max_intensification_iterations = max_iter/diversificating_rate

        # remove after debugging (visualizating purposes)
        prec, succ = self.__decode(BEST_chromosome)
        self.plot(self.get_graph(succ))

        # Iterations counter(s)
        j = 0
        int_counter = 0

        print("Genetic Algorithm starts now...")
        starting_time = time.clock()
        current_time = time.clock()
        while j < max_iter and current_time - starting_time < self.__time_limit:
            if not intensification_phase:
                intensification_phase = True
            elif int_counter >= max_intensification_iterations:
                intensification_phase = False
                int_counter = 0
                print("--- GAMMA RAYS INCOMING ---")
            else:
                intensification_phase = True
            int_counter += 1

            current_children = self.__reproduction(pop)

            # Duplicates check
            self.__check_duplicates(pop, current_children)

            new_children = self.__evaluation(current_children)

            # From now on we have a new pop, t=t+1 ('next generation'):
            e_l_f = 0.00 if intensification_phase else 0.95
            fitness, best_from_pop, pop = self.__selection(pop, new_children,
                                                           expected_lucky_few=e_l_f)

            # debugging; to be deleted
            print("New pop, new fitness",fitness, math.log(fitness, 10))

            if best_from_pop[0] < BEST_obj_value:
                BEST_chromosome = best_from_pop[1]
                BEST_obj_value = best_from_pop[0]

            # Gamma ray incoming
            s_p = 0.00 if intensification_phase else 0.95
            s_m_p = 0.00 if intensification_phase else 0.50
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
        seen = set([str(x[1]) for x in oldpop])

        # Time to remove the duplicates.
        for idx,x in enumerate(newchild):
            string = str(x)
            if string in seen:
                # Attenzione: il passaggio per riferimento funziona solo così.
                # passargli 'x' direttamente fa interpretare a Python un passaggio per valore.
                self.__single_gamma_ray(newchild[idx])
            else:
                seen.add(string)

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

    def __reproduction(self, pop):
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
            # we choose one chromosome to reproduce by meiosis:
            alone_chromosome = random.choice(pop_temp)
            pop_temp.remove(alone_chromosome)
            self.__single_gamma_ray(alone_chromosome)
            children.append(alone_chromosome)

        # Let the mating process begin.
        while len(pop_temp) > 0:
            parent_1 = random.choice(pop_temp)
            pop_temp.remove(parent_1)
            parent_2 = random.choice(pop_temp)
            pop_temp.remove(parent_2)
            twins = self.__breed(parent_1[1], parent_2[1])
            for child in twins:
                children.append(child)

        return children

    def __breed(self, parent_1, parent_2):
        """
        This method gets two chromosomes (we don't need their associated cost),
        and generates four children.
            -   The first one will be the random cross-over between the two parents.
            -   The second one will be complementary of the first one.
            -   The third one will have everything mother and father have in common;
                if they don't have a gene in common, we choose randomly such gene from them.
            -   The fourth one will be like the third one, but with the complementary choices
                onto the random choice we've made.
        This policy allows to deeply explore (by intensificating) the solutions, while still creating
        some diversity in this search.

        :param parent_1: Father (chromosome)
        :param parent_2: Mather (chromosome)
        :return: A list of four children.
        """

        chr_length = len(parent_1)
        child_1 = []
        child_2 = []
        child_3 = []
        child_4 = []

        for i in range(chr_length):
            coin_flip = random.randint(0,1)

            child_1.append(parent_1[i] if coin_flip == 1 else parent_2[i])
            child_2.append(parent_2[i] if coin_flip == 1 else parent_1[i])
            if parent_1[i] == parent_2[i]:
                child_3.append(parent_1[i])
                child_4.append(parent_1[i])
            else:
                child_3.append(parent_1[i] if coin_flip == 1 else parent_2[i])
                child_4.append(parent_2[i] if coin_flip == 1 else parent_1[i])

        return [child_1, child_2, child_3, child_4]

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

    def __prufer_decode(self, chromosome, substation=0, debug=False):
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

        self.__treefix(prec, succ, substation, debug=debug)

        return prec, succ

    def __treefix(self, prec, succ, node, debug=False):
        """
        This recursive method check if the node passed is the root of the anti-arborescence.
        A node is the root <=> succ[node] = node.
        If the current node isn't the root, we fix such situation, with a recursive call
        made onto the next node, which indeed shouldn't be the next one of the current node.

        :param prec: list of lists, data structure of our tree
        :param succ: list of lists, data structure of our tree
        :param node: the current node to be examined
        :return: Nothing. The data structures will be directly modified if necessary.
        """

        next_node = succ[node]
        if next_node != node:
            # This means that the root isn't in the current node
            # We should find it recursively.
            self.__treefix(prec, succ, next_node)
            # At this point we've found the root: it's the next_node.
            # I (node) will become the new root.
            prec[next_node].remove(node)
            succ[next_node] = node
            prec[node].append(next_node)
            succ[node] = node  # Root condition in our data structures.
        else:
            succ[node] = node

        # At this point, node is the new root.
        # Other recursive calls will fix the situation if necessary.

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

    """def __succ_decode(self, chromosome, substation=0):
        prec = [[] for i in range(self.__n_nodes)]

        for idx, g in enumerate(chromosome):
            prec[g].append(idx)

        root = [x for idx, x in enumerate(chromosome) if x == idx]
        root = root[0] if len(root) == 1 else 0

        # Check for loops or disconnections
        flag = self.__check_connected_components(prec, chromosome, substation=root)

        if flag == 1:
            # LOOP DETECTED
            pass
        elif flag == -1:
            # DISCONNECTED
            self.__disconnectfix(chromosome)
        elif flag == 0:
            # LOOP&DISCONNECTED
            pass
        else:
            # ALL OK
            pass

        # Sets the root in the substation
        self.__treefix(prec, chromosome,
                       node=substation)
        return prec, chromosome

    def __check_connected_components(self, prec, succ,
                                     substation=0):

        visited = set()
        self.__DFS_loopfix(substation, prec, succ, visited, substation=substation)
        not_visited = set([x for x in range(self.__n_nodes)])-visited
        if len(not_visited) > 0:
            # This means we've got some disconnected components
            while len(not_visited) > 0:
                # Extract the minimum in this matrix
                rows = list(not_visited)
                cols = list(visited)
                extracted_matrix = self.__distances[rows[:, np.newaxis], cols]
                min = extracted_matrix.min()
                i = min[1]
                j = min[2]

                # Extract the root of the next connected component, if any
                i = [x for x in not_visited if x == succ[x]]
                i = i[0] if len(i) > 0 else random.choice(i)
                # Extract the closest node to the connected component
                j = random.choice(visited)
                if succ[i] != i:
                    prec[succ[i]].remove(i)
                succ[i] = j

                self.__DFS_loopfix(i, prec, succ, visited, substation=substation)
                not_visited = set([x for x in range(self.__n_nodes)])-visited

    def __new_treefix(self, prec, succ, node, visited):
        visited.add(node)
        successore = succ[node]

        if node == successore:
            # We've found the root of this connected component
            for n in prec[node]:
                if n in visited:
                    # This is a loop and we must fix the situation
                    prec[succ[node]].remove(node)
                    succ[node] = n
                    prec[node].remove(n)
                    prec[succ[node]].append(node)
                self.__treefix(prec, succ, n, visited)
        else:
            # For sure, this graph is not radicated in the substation.
            # Also, this graph may have loops.
            # (That's why we don't call it anti-arborescence, yet)

            if successore not in visited:
                self.__treefix(prec, succ, successore, visited)

                succ[successore] = node
                prec[node].append(successore)
                prec[successore].remove(node)
            else:
                # Loop detected
                for n in prec[node]:
                    if n in visited:
                        # This is a loop and we must fix the situation
                        prec[succ[node]].remove(node)
                        succ[node] = n
                        prec[node].remove(n)
                        prec[succ[node]].append(node)

                self.__DFS_loopfix(n, prec, succ, visited, substation=substation)

        return

    def __DFS_loopfix(self, node, prec, succ, visited, substation=0):

        visited.add(node)

        for n in prec[node]:
            if n in visited:
                # This is a loop and we must fix the situation
                prec[succ[node]].remove(node)
                succ[node] = n
                prec[node].remove(n)
                prec[succ[node]].append(node)

            self.__DFS_loopfix(n, prec, succ, visited, substation=substation)

        return
"""


