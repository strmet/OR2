import argparse
import os.path
from collections import namedtuple
import networkx as nx
from .WindFarm import WindFarm
import random
import matplotlib.pyplot as plt
import sys


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

    def cost_solution(self, prec, succ):

        """

            Compute the cost of the solution given as input

        :param prec: List of lists. Each element's index represents one vertex, the values are the childs
        :param succ: List of integers. The index is the vertex and the value is its parent
        :return: Cost of the solution

        """

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

        for edge in edges_with_power:
            for cable in self.__cables:
                if edge.power <= cable.capacity: # Da correggere per il caso di soluzioni infeasible
                    cost += WindFarm.get_distance(self.__points[edge.s], self.__points[edge.d]) * cable.price
                    break  # TODO Ask the professor and rearrange the loop

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

    def grasp(self, num_edges=5, substation=0):

        """

            Apply the GRASP methodology to Prim-Dijkstra's algorithm
            At each iteration, when a new edge must be chosen, it is sampled from a pool of
            closest vertexes.

            As reference, the implementation is the one in the Fischetti's book (fig 7.13)

        :param num_edges: the size of the pool
        :param substation: The index of the substation (range [0, ..., n - 1])
        :return: the list of oriented edges

        """

        # Build the matrix
        matrix = [[0 for i in range(self.__n_nodes)] for j in range(self.__n_nodes)]

        # Fill the matrix with all the distances
        for i in range(self.__n_nodes):
            for j in range(self.__n_nodes):
                matrix[i][j] = WindFarm.get_distance(self.__points[i], self.__points[j])

        # Step 1
        pred = [0] * self.__n_nodes
        flag = [0] * self.__n_nodes
        L = [0] * self.__n_nodes

        flag[substation] = 1
        L = matrix[substation]

        # Step 2
        for k in range(0, self.__n_nodes - 1):

            # Step 3
            mins = [sys.float_info.max] * num_edges  # Set each element to +infinite
            hs = [None] * num_edges  # The indexes

            for j in range(1, self.__n_nodes):
                if flag[j] == 0 and L[j] < max(mins):
                    # Substitute the new value instead of the highest one in the list
                    max_value, max_index = max((x, (i)) for i, x in enumerate(mins))
                    mins[max_index] = L[j]
                    hs[max_index] = j

            # Now that the pool is ready choose one vertex
            hs = [x for x in hs if x is not None]
            h = random.choice(hs)  # Select randomly one of the edges

            # From now on the algorithm is the same as in the book
            # Step 4
            flag[h] = 1

            # Step 5
            for j in range(1, self.__n_nodes):
                if flag[j] == 0 and matrix[h][j] < L[j]:
                    L[j] = matrix[h][j]
                    pred[j] = h

        # Return the tree extracting the information from the pred data structure
        edges = []

        for i, j in enumerate(pred):
            edges.append(self.__EdgeSol(i, j))

        return edges

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

