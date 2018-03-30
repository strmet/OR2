"""
Everything related to the instance of our problem is written in here.
"""

from collections import namedtuple
from docplex.mp.model import Model
import os.path
import warnings
import argparse
import cplex
import math
import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx
import matplotlib.pyplot as plt


class ValueWarning(UserWarning):
    pass


class ParseWarning(UserWarning):
    pass


class Instance:
    """
    py:class:: instance()

    This class stores all the useful information about input data and parameters

    """

    def __init__(self, dataset_selection=1):
        # Model variables
        self.name = ''
        self.y_start = 0
        self.f_start = 0
        self.x_start = 0

        # Input data variables
        self.n_nodes = 0
        self.num_cables = 0
        self.points = []
        self.cables = []
        self.c = 10

        # Dataset selection and consequent input files building
        self.data_select = dataset_selection
        self.__build_input_files()
        self.__build_name()

        # Named tuples, describing input data
        self.Edge = namedtuple("Edge", ["source", "destination"])
        self.Point = namedtuple("Point", ["x", "y", "power"])
        self.Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])
        self.CableSol = namedtuple("CableSol", ["source", "destination", "capacity"])

        # Parameters, commenting out whatever we don't use
        # self.model_type
        # self.num_threads
        self.cluster = False
        self.time_limit = 1200
        self.rins = 7
        self.polishtime = 900
        self.debug_mode = False
        self.verbosity = 0
        self.interface = 'cplex'
        # available_memory
        # self.max_nodes
        # self.cutoff
        # self.integer_costs

    def __build_input_files(self):
        """
        py:function:: __build_input_files(self)

        Sets the input file correctly, based on the dataset selection

        """
        if not type(self.data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(d))
        if self.data_select <= 0 or self.data_select >= 31:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-30]. Given: " + str(d))

        data_tostring = str(self.data_select)
        if 1 <= self.data_select <= 9:
            data_tostring = "0" + data_tostring

        self.turb_file = "../data/data_" + data_tostring + ".turb"
        self.cbl_file = "../data/data_" + data_tostring + ".cbl"

    def __build_name(self):
        """
        py:function:: __build_name(self)

        Sets the name of the wind farm correctly, based on the dataset selection

        """
        if not type(self.data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(d))
        if self.data_select <= 0 or self.data_select >= 31:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-30]. Given: " + str(d))

        # We assume that, in this context, we'll never have a WF >=10
        wf_number = 0
        if 1 <= self.data_select <= 6:
            wf_number = 1
        elif 7 <= self.data_select <= 15:
            wf_number = 2
        elif 16 <= self.data_select <= 19:
            wf_number = 3
        elif 20 <= self.data_select <= 21:
            wf_number = 4
        elif 26 <= self.data_select <= 29:
            wf_number = 5
        elif 30 <= self.data_select <= 31:
            wf_number = 6

        if wf_number == 0:
            raise ValueError("Something went wrong with the Wind Farm number;\n" +
                             "check the dataset selection parameter: " + str(self.data_select))

        self.name = "Wind Farm 0"+str(wf_number)

    @property
    def data_select(self):
        return self.__data_select

    @data_select.setter
    def data_select(self, d):
        if not type(d) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(d))
        if d <= 0 or d >= 31:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-30]. Given: " + str(d))
        self.__data_select = d

    @property
    def time_limit(self):
        return self.__time_limit

    @time_limit.setter
    def time_limit(self, t):
        if t <= 3:
            raise TimeoutError("It doesn't make sense to run this program for less than 3 seconds; given: "
                               + str(t))
        self.__time_limit = t

    @property
    def rins(self):
        return self.__rins

    @rins.setter
    def rins(self, freq):
        if not type(freq) == int:
            raise TypeError("The frequency RINS will be applied should be an integer; given: " + str(freq))
        if freq < -1:
            raise ValueError("Invalid RINS parameter. Integer values above -1 are OK; given: " + str(freq))
        self.__rins = freq

    @property
    def polishtime(self):
        return self.__polishtime

    @polishtime.setter
    def polishtime(self, t):
        if t<=3:
            raise TimeoutError("Can't start polishing within just 3 seconds;  given: " + str(t))
        if t>=self.time_limit:
            raise TimeoutError("Polishing can't start after the timeout limit (" +
                               str(self.time_limit) + "); given: " + str(t))
        self.__polishtime = t

    @property
    def num_cables(self):
        return self.__num_cables

    @num_cables.setter
    def num_cables(self, nc):
        if not type(nc) == int:
            raise TypeError("The number of cables must be a positive integer number; given:" + str(nc))
        if not nc >= 0:
            raise AttributeError("The number of cables must be a positive integer number; given:" + str(nc))
        self.__num_cables = nc

    @property
    def cluster(self):
        return self.__cluster

    @cluster.setter
    def cluster(self, c):
        if not type(c) == bool:
            raise TypeError("Expecting 'cluster' to be either set True or False; given:" + str(c))
        self.__cluster = c

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, c):
        if not type(c) == int:
            raise TypeError("The parameter 'c' must be a positive integer number; given: " + str(c))
        if c <= 0:
            warnings.warn("Substations must accept at least one cable; setting 'c' to its default value (10)",
                          ValueWarning)
            self.__c = 10
        else:
            self.__c = c

    @property
    def interface(self):
        return self.__interface

    @interface.setter
    def interface(self, choice):
        if not type(choice) == str:
            warnings.warn("Choice not given as a string. Trying a conversion.", ValueWarning)
            choice = str(choice)
        if choice != "cplex" and choice != "docplex":
            raise NameError(
                "It is possible to choose either 'cplex' or 'docplex' as opt libraries; given: " + choice)
        self.__interface = choice

    @property
    def turb_file(self):
        return self.__turb_file

    @turb_file.setter
    def turb_file(self, fname):
        if not type(fname) == str:
            warnings.warn("Turbines filename not given as string. Trying a conversion.", ValueWarning)
            fname = str(fname)
        if not os.path.isfile(fname):
            raise FileNotFoundError("Can't find the '.trb' file; filename given: " + fname)
        self.__turb_file = fname

    @property
    def cbl_file(self):
        return self.__cbl_file

    @cbl_file.setter
    def cbl_file(self, fname):
        if not type(fname) == str:
            warnings.warn("Cables filename not given as string. Trying a conversion.", ValueWarning)
            fname = str(fname)
        if not os.path.isfile(fname):
            raise FileNotFoundError("Can't find the '.cbl' file; filename given: " + fname)
        self.__cbl_file = fname

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, s):
        # it doesn't make sense to raise an Exception,
        # if the name isn't a string: just convert it!
        self.__name = str(s)

    def ypos(self, offset):

        """
        py:function:: ypos(self, offset)

        .. description missing ..

        :param offset: Offset w.r.t ystart and the edge indexed by (i, j)

        """

        return self.y_start + offset

    def parse_command_line(self):

        """
        py:function:: parse_command_line(self)

        Parses the command line

        :return: None
        """

        parser = argparse.ArgumentParser(description='Process details about instance and interface.')

        parser.add_argument('--dataset', type=int, help='dataset selection; datasets available: [1,29]')
        parser.add_argument('--cluster', action="store_true", help='type --cluster if you want to use the cluster')
        parser.add_argument('--interface', choices=['docplex', 'cplex'], help='Choose interface ')
        parser.add_argument('--C', type=int, help='the maximum number of cables linked to a substation')
        parser.add_argument('--rins', type=int, help='the frequency with which the RINS Heuristic will be applied')
        parser.add_argument('--timeout', type=int, help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--polishtime', type=int, help='the time to wait before applying polishing')

        args = parser.parse_args()

        if args.dataset:
            self.data_select = args.dataset

        if args.cluster:
            self.cluster = True

        if args.interface == 'docplex' or args.interface == 'cplex':
            self.interface = args.interface
        else:
            warnings.warn("Invalid interface; '" + str(args.interface) + "' given. "
                          + "Using the default value: " + self.interface,
                          ParseWarning)
            self.interface = 'cplex'

        if args.C is not None:
            self.c = args.C
        else:
            warnings.warn("Invalid 'C' value; '" + str(args.C) + "' given. Using the default value: " + str(self.c),
                          ParseWarning)

        if args.rins:
            self.rins = args.rins

        if args.timeout:
            self.time_limit = args.timeout

        if args.polishtime:
            self.polishtime = args.polishtime


    def read_turbines_file(self):

        """
        py:function:: read_turbines_file(self)

        Read the turbines file

        """

        points = []

        # the following opens and closes the file within the block
        with open(self.turb_file, "r") as fp:
            for line in fp:
                words = list(map(int, line.split()))
                points.append(self.Point(words[0], words[1], words[2]))

        self.n_nodes = len(points)
        self.points = points

    def read_cables_file(self):

        """
        py:function:: read_cables_file(self)

        Read the cables file

        """

        cables = []

        # the following opens and closes the file within the block
        with open(self.turb_file, "r") as fp:
            for line in fp:
                #if len(cables) > 3: break
                words = line.split()
                cables.append(self.Cable(int(words[0]), float(words[1]), int(words[2])))

        self.num_cables = len(cables)
        self.cables = cables

    def fpos(self, offset):

        """
        py:function:: fpos(offset, self)

        .. description missing ..

        :param offset: Offset w.r.t fstart and the edge indexed by (i, j)

        """
        return self.f_start + offset

    def xpos(self, offset, k):

        """
        py:function:: xpos(offset, k, inst)

        .. description missing ..

        :param offset: Offset w.r.t xstart and the edge indexed by (i, j)
        :param k: index of the cable considered

        """
        return self.x_start + offset * self.num_cables + k

    @staticmethod
    def get_distance(point1, point2):

        """
        py:function:: get_distance(point1, point2)
        Get the distance between two points

        :param point1: First point
        :param point2: Second point

        """

        return math.sqrt(
            (point1.x - point2.x) ** 2
            +
            (point1.y - point2.y) ** 2
        )

    def build_model_classical_cplex(self):

        """
        Build the model using classical cplex API

        :return: The model filled with variables and constraints
        """
        if not self.interface == 'cplex':
            raise NameError("For some reason the classical model has been called when " +
                            "the 'interface' variable has been set to: " + self.interface)

        model = cplex.Cplex()

        model.set_problem_name(self.name)
        model.objective.set_sense(model.objective.sense.minimize)

        edges = [self.Edge(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes)]

        # Add y(i,j) variables
        self.y_start = model.variables.get_num()
        name_y_edges = ["y({0},{1})".format(i + 1, j + 1) for i in range(self.n_nodes) for j in range(self.n_nodes)]

        for edge in name_y_edges:
            model.variables.add(
                types=[model.variables.type.binary],
                names=[edge]
            )
            if self.debug_mode:
                if self.ypos(index) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

        # Add f(i,j) variables
        self.f_start = model.variables.get_num()
        name_f_edges = ["f({0},{1})".format(i + 1, j + 1) for i in range(self.n_nodes) for j in range(self.n_nodes)]

        for edge in name_f_edges:
            model.variables.add(
                types=[model.variables.type.continuous],
                names=[edge]
            )
            if self.debug_mode:
                if self.fpos(index) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

        # Add x(i,j,k) variables
        self.x_start = model.variables.get_num()

        for edge in edges:
            for k in range(self.num_cables):
                model.variables.add(
                    types=[model.variables.type.binary],
                    names=["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)],
                    obj=[self.cables[k].price * Instance.get_distance(self.points[edge.source], self.points[edge.destination])]
                )
                if self.debug_mode:
                    if self.xpos(index, k) != model.variables.get_num() - 1:
                        raise NameError('Number of variables and index do not match')

        # No self-loop constraint (on x,y,f)
        for i in range(self.n_nodes):
            model.variables.set_upper_bounds([("y({0},{1})".format(i+1, i+1), 0)])
        for i in range(self.n_nodes):
            model.variables.set_upper_bounds([("f({0},{1})".format(i+1, i+1), 0)])
        for i in range(self.n_nodes):
            for k in range(self.num_cables):
                model.variables.set_upper_bounds([("x({0},{1},{2})".format(i+1, i+1, k+1), 0)])

        # Energy flow must be positive
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                model.variables.set_lower_bounds([("f({0},{1})".format(i + 1, j + 1), 0)])

        # Out-degree constraints
        for h, point in enumerate(self.points):
            if point.power < -0.5:  # if it's a substation
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(h + 1, j + 1) for j in range(self.n_nodes)],
                        val=[1.0] * self.n_nodes
                    )],
                    senses=["E"],
                    rhs=[0]
                )
            else:                   # if it's a turbine
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(h + 1, j + 1) for j in range(self.n_nodes)],
                        val=[1.0] * self.n_nodes
                    )],
                    senses=["E"],
                    rhs=[1]
                )

        # Flow balancing constraint
        for h, point in enumerate(self.points):
            if point.power > 0.5:  # if it's a substation
                summation = ["f({0},{1})".format(h + 1, j + 1) for j in range(self.n_nodes) if h != j] \
                      + \
                      ["f({0},{1})".format(j + 1, h + 1) for j in range(self.n_nodes) if h != j]
                coefficients = [1] * (self.n_nodes - 1) + [-1] * (self.n_nodes - 1)
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=summation,
                        val=coefficients,
                    )],
                    senses=["E"],
                    rhs=[point.power]
                )

        # Maximum number of cables linked to a substation
        for h, point in enumerate(self.points):
            if point.power < -0.5:
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(i + 1, h + 1) for i in range(self.n_nodes)],
                        val=[1] * self.n_nodes,
                    )],
                    senses=["L"],
                    rhs=[self.c]
                )

        # Avoid double cable between two points
        for edge in edges:
            summation = ["y({0},{1})".format(edge.source + 1, edge.destination + 1)] \
                  + \
                  ["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1) for k in
                   range(self.num_cables)]
            coefficients = [1] + [-1] * self.num_cables
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=summation,
                    val=coefficients,
                )],
                senses=["E"],
                rhs=[0]
            )

        # Guarantee that the cable is enough for the connection
        for edge in edges:
            summation = ["f({0},{1})".format(edge.source + 1, edge.destination + 1)] \
                        + \
                        ["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1) for k in
                         range(self.num_cables)]
            coefficients = [-1] + [cable.capacity for cable in self.cables]

            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=summation,
                    val=coefficients,
                )],
                senses=["G"],
                rhs=[0]
            )

        # Adding the parameters to the model
        model.parameters.mip.strategy.rinsheur.set(self.rins)
        model.parameters.mip.polishafter.time.set(self.polishtime)
        model.parameters.timelimit.set(self.time_limit)

        # Writing the model to a proper location
        model.write("../out/lpmodel.lp")

        return model

    def build_model_docplex(self):

        """
        Build the model using docplex API

        :return: The model filled with variables and constraints
        """
        if not self.interface == 'docplex':
            raise NameError("For some reason the docplex model has been called when " +
                            "the 'interface' variable has been set to: " + self.interface)

        edges = [self.Edge(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes)]

        model = Model(name=self.name)
        model.set_time_limit(self.time_limit)

        # Add y_i.j variables
        self.y_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            model.binary_var(name="y({0},{1})".format(edge.source + 1, edge.destination + 1))
            if self.debug_mode:
                if self.ypos(index) != model.get_statistics().number_of_variables - 1:
                    raise NameError('Number of variables and index do not match')

        # Add f_i.j variables
                self.f_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            model.continuous_var(name="f({0},{1})".format(edge.source + 1, edge.destination + 1))
            if self.debug_mode:
                if self.fpos(index) != model.get_statistics().number_of_variables - 1:
                    raise NameError('Number of variables and index do not match')

        # Add x_i.j.k variables
        self.x_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            for k in range(self.num_cables):
                model.binary_var(name="x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))

                if self.debug_mode:
                    if self.xpos(index, k) != model.get_statistics().number_of_variables - 1:
                        raise NameError('Number of variables and index do not match')

        # No self-loops constraints on y_i.i variables
        for i in range(self.n_nodes):
            var = model.get_var_by_name("y({0},{1})".format(i + 1, i + 1))
            model.add_constraint(var == 0)

        # No self-loops constraints on f_i.i variables
        for i in range(self.n_nodes):
            var = model.get_var_by_name("f({0},{1})".format(i + 1, i + 1))
            model.add_constraint(var == 0)

        # Out-degree constraints
        for h in range(len(self.points)):
            if self.points[h].power < -0.5:
                model.add_constraint(
                    model.sum(model.get_var_by_name("y({0},{1})".format(h + 1, j + 1)) for j in range(self.n_nodes))
                    ==
                    0
                )
            else:
                model.add_constraint(
                    model.sum(model.get_var_by_name("y({0},{1})".format(h + 1, j + 1)) for j in range(self.n_nodes))
                    ==
                    1
                )

        # Maximum number of cables linked to a substation
        for h in range(len(self.points)):
            if self.points[h].power < -0.5:
                model.add_constraint(
                    model.sum(model.get_var_by_name("y({0},{1})".format(i + 1, h + 1)) for i in range(self.n_nodes))
                    <=
                    self.c
                )
        # Flow balancing constraint
        for h in range(len(self.points)):
            if self.points[h].power > 0.5:
                model.add_constraint(
                    model.sum(model.get_var_by_name("f({0},{1})".format(h + 1, j + 1)) for j in range(self.n_nodes))
                    ==
                    model.sum(model.get_var_by_name("f({0},{1})".format(j + 1, h + 1)) for j in range(self.n_nodes))
                    + self.points[h].power
                )

        # Avoid double cable between two points
        for edge in edges:
            model.add_constraint(
                model.get_var_by_name("y({0},{1})".format(edge.source + 1, edge.destination + 1))
                ==
                model.sum(
                    model.get_var_by_name("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)) for k
                    in range(self.num_cables))
            )

        # Guarantee that the cable is enough for the connection
        for edge in edges:
            model.add_constraint(
                model.sum(
                    model.get_var_by_name("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)) *
                    self.cables[k].capacity
                    for k in range(self.num_cables)
                )
                >=
                model.get_var_by_name("f({0},{1})".format(edge.source + 1, edge.destination + 1))
            )

        # Objective function
        model.minimize(
            model.sum(
                self.cables[k].price * Instance.get_distance(self.points[i], self.points[j]) * model.get_var_by_name(
                    "x({0},{1},{2})".format(i + 1, j + 1, k + 1))
                for k in range(self.num_cables) for i in range(self.n_nodes) for j in range(self.n_nodes)
            )
        )

        # Adding the parameters to the model
        model.parameters.mip.strategy.rinsheur.set(self.rins)
        model.parameters.mip.polishafter.time.set(self.polishtime)
        model.parameters.timelimit.set(self.time_limit)

        # Writing the model to a proper location
        model.write("../out/lpmodel.lp")

        return model

    def get_solution(self, model):

        """
        Read all x(i, j, k) variables and store the ones with value '1' into the solution

        :param model: The CPLEX' model
        :return: List of CableSol named tuples
        """

        sol = []
        edges = [self.Edge(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes)]
        if self.interface == 'cplex':
            for edge in edges:
                for k in range(self.num_cables):
                    val = model.solution.get_values("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))
                    if val > 0.5:
                        sol.append(self.CableSol(edge.source + 1, edge.destination + 1, k + 1))
        else:
            for edge in edges:
                for k in range(self.num_cables):
                    val = model.solution.get_value("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))
                    if val > 0.5:
                        sol.append(self.CableSol(edge.source + 1, edge.destination + 1, k + 1))

        return sol

    def plot_solution(self, model):

        """
        py:function:: plot_solution(inst, edges)

        Plot the solution using the plot.ly library

        :param model: CPLEX internal model

        """
        edges = self.get_solution(model)
        G = nx.DiGraph()

        for index, node in enumerate(self.points):
            G.add_node(index, pos=(node.x, node.y))

        for edge in edges:
            G.add_edge(edge.source - 1, edge.destination - 1, weight=edge.capacity)

        edge_trace = Scatter(
            x=[],
            y=[],
            line=Line(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        for edge in G.edges():
            x0, y0 = G.node[edge[0]]['pos']
            x1, y1 = G.node[edge[1]]['pos']
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]

        node_trace = Scatter(
            x=[],
            y=[],
            text=["Substation #{0}".format(i + 1) if self.points[i].power < -0.5 else "Turbine #{0}".format(i + 1) for i in range(self.n_nodes)],
            mode='markers',
            hoverinfo='text',
            marker=Marker(
                showscale=False,
                colorscale='Greens',
                reversescale=True,
                color=[],
                size=10,
                line=dict(width=2))
        )

        # Prepare data structure for plotting (x, y, color)
        for node in G.nodes():
            x, y = G.node[node]['pos']
            node_trace['x'].append(x)
            node_trace['y'].append(y)
            node_trace['marker']['color'].append("#32CD32")


        # Create figure
        fig = Figure(data=Data(
            [edge_trace, node_trace]),
            layout=Layout(
                title='<br><b style="font-size:20px>'+self.name+'</b>',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(scaleanchor="x", scaleratio=1,showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        py.plot(fig, filename='wind_farm.html')


    def plot_high_quality(self, model, export=False):

        """
        py:function:: plot_high_quality(inst, edges)

        Plot the solution using standard libraries

        :param inst: First point
        :param edges: List of edges given back by CPLEX
        :type edges: List of CableSol

        """
        edges = self.get_solution(model)
        G = nx.DiGraph()

        mapping = {}

        for i in range(self.n_nodes):
            if (self.points[i].power < -0.5):
                mapping[i] = 'S{0}'.format(i + 1)
            else:
                mapping[i] = 'T{0}'.format(i + 1)

        for index, node in enumerate(self.points):
            G.add_node(index)

        for edge in edges:
            G.add_edge(edge.source - 1, edge.destination - 1)

        pos = {i: (point.x, point.y) for i, point in enumerate(self.points)}

        # Avoid re scaling of axes
        plt.gca().set_aspect('equal', adjustable='box')

        # draw graph
        nx.draw(G, pos, with_labels=True, node_size=1300, alpha=0.3, arrows=True, labels=mapping, node_color='g', linewidth=10)

        if (export == True):
            plt.savefig('../out/img/foo.svg')

        # show graph
        plt.show()

