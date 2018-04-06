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


class WindFarm:
    """
    py:class:: instance()

    This class stores all the useful information about input data and parameters

    """

    # Class constructor
    def __init__(self, dataset_selection=1):
        # The model itself
        self.__model = None

        # Model variables
        self.__name = ''
        self.__y_start = 0
        self.__f_start = 0
        self.__x_start = 0
        self.__slack_start = 0

        # Input data variables
        self.__n_nodes = 0
        self.__num_cables = 0
        self.__points = []
        self.__cables = []
        self.c = 10

        # Dataset selection and consequent input files building, and output parameters
        self.data_select = dataset_selection
        self.__build_input_files()
        self.__build_name()
        self.out_dir_name = 'test'

        # Named tuples, describing input data
        self.__Edge = namedtuple("Edge", ["source", "destination"])
        self.__Point = namedtuple("Point", ["x", "y", "power"])
        self.__Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])
        self.__CableSol = namedtuple("CableSol", ["source", "destination", "capacity"])

        # Parameters, commenting out whatever we don't use
        # self.model_type (???)
        # self.num_threads
        self.cluster = False
        self.time_limit = 60
        self.rins = 7
        self.polishtime = 45
        self.__debug_mode = False
        self.verbosity = 0
        self.__interface = 'cplex'
        # available_memory
        # self.max_nodes
        # self.cutoff
        # self.integer_costs

    # Private methods, internal to our class:

    def __read_turbines_file(self):

        """
        py:function:: read_turbines_file(self)

        Read the turbines file

        """

        points = []

        # the following opens and closes the file within the block
        with open(self.turb_file, "r") as fp:
            for line in fp:
                words = list(map(int, line.split()))
                points.append(self.__Point(words[0], words[1], words[2]))

        self.__n_nodes = len(points)
        self.__points = points

    def __read_cables_file(self):

        """
        py:function:: read_cables_file(self)

        Read the cables file

        """

        cables = []

        # the following opens and closes the file within the block
        with open(self.cbl_file, "r") as fp:
            for line in fp:
                # if len(cables) > 3: break
                words = line.split()
                cables.append(self.__Cable(int(words[0]), float(words[1]), int(words[2])))

        self.__num_cables = len(cables)
        self.__cables = cables

    def __fpos(self, offset):

        """
        py:function:: __fpos(offset, self)

        .. description missing ..

        :param offset: Offset w.r.t fstart and the edge indexed by (i, j)

        """
        return self.__f_start + offset

    def __xpos(self, offset, k):

        """
        py:function:: __xpos(offset, k, inst)

        .. description missing ..

        :param offset: Offset w.r.t xstart and the edge indexed by (i, j)
        :param k: index of the cable considered

        """
        return self.__x_start + offset * self.__num_cables + k

    def __slackpos(self, offset):

        """
        py:function:: __slackpos(offset, k, inst)

        .. description missing ..

        :param offset: Offset w.r.t slackstart and the substation indexed by h

        """
        return self.__slack_start + offset

    def __build_model_cplex(self):

        """
        Build the model using classical cplex API

        :return: The model filled with variables and constraints
        """
        if not self.__interface == 'cplex':
            raise NameError("For some reason the classical model has been called when " +
                            "the 'interface' variable has been set to: " + self.__interface)

        model = cplex.Cplex()

        model.set_problem_name(self.__name)
        model.objective.set_sense(model.objective.sense.minimize)

        edges = [self.__Edge(i, j) for i in range(self.__n_nodes) for j in range(self.__n_nodes)]

        # Add y(i,j) variables
        self.__y_start = model.variables.get_num()
        name_y_edges = ["y({0},{1})".format(i + 1, j + 1) for i in range(self.__n_nodes) for j in range(self.__n_nodes)]

        for edge in name_y_edges:
            model.variables.add(
                types=[model.variables.type.binary],
                names=[edge]
            )
            if self.__debug_mode:
                if self.__ypos(index) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

        # Add f(i,j) variables
        self.__f_start = model.variables.get_num()
        name_f_edges = ["f({0},{1})".format(i + 1, j + 1) for i in range(self.__n_nodes) for j in range(self.__n_nodes)]

        for edge in name_f_edges:
            model.variables.add(
                types=[model.variables.type.continuous],
                names=[edge]
            )
            if self.__debug_mode:
                if self.__fpos(index) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

        # Add x(i,j,k) variables
        self.__x_start = model.variables.get_num()

        for edge in edges:
            for k in range(self.__num_cables):
                model.variables.add(
                    types=[model.variables.type.binary],
                    names=["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)],
                    obj=[self.__cables[k].price *
                         WindFarm.get_distance(self.__points[edge.source], self.__points[edge.destination])]
                )
                if self.__debug_mode:
                    if self.__xpos(index, k) != model.variables.get_num() - 1:
                        raise NameError('Number of variables and index do not match')

        # Add s(h) (slack) variables
        self.__slack_start = model.variables.get_num()

        for h, point in enumerate(self.__points):
            if point.power < -0.5:
                model.variables.add(
                    types=[model.variables.type.continuous],
                    names=["s({0})".format(h+1)],
                    obj=[1e9]
                )
                if self.__debug_mode:
                    if self.__slackpos(h) != model.variables.get_num() - 1:
                        raise NameError('Number of variables and index do not match')

        # No self-loop constraint (on x,y,f)
        for i in range(self.__n_nodes):
            model.variables.set_upper_bounds([("y({0},{1})".format(i + 1, i + 1), 0)])
        for i in range(self.__n_nodes):
            model.variables.set_upper_bounds([("f({0},{1})".format(i + 1, i + 1), 0)])
        for i in range(self.__n_nodes):
            for k in range(self.__num_cables):
                model.variables.set_upper_bounds([("x({0},{1},{2})".format(i + 1, i + 1, k + 1), 0)])

        # Energy flow must be positive
        for i in range(self.__n_nodes):
            for j in range(self.__n_nodes):
                model.variables.set_lower_bounds([("f({0},{1})".format(i + 1, j + 1), 0)])

        # Out-degree constraints
        for h, point in enumerate(self.__points):
            if point.power < -0.5:  # if it's a substation
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(h + 1, j + 1) for j in range(self.__n_nodes)],
                        val=[1.0] * self.__n_nodes
                    )],
                    senses=["E"],
                    rhs=[0]
                )
            else:  # if it's a turbine
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(h + 1, j + 1) for j in range(self.__n_nodes)],
                        val=[1.0] * self.__n_nodes
                    )],
                    senses=["E"],
                    rhs=[1]
                )

        # Flow balancing constraint
        for h, point in enumerate(self.__points):
            if point.power > 0.5:  # if it's a substation
                summation = ["f({0},{1})".format(h + 1, j + 1) for j in range(self.__n_nodes) if h != j] \
                            + \
                            ["f({0},{1})".format(j + 1, h + 1) for j in range(self.__n_nodes) if h != j]
                coefficients = [1] * (self.__n_nodes - 1) + [-1] * (self.__n_nodes - 1)
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=summation,
                        val=coefficients,
                    )],
                    senses=["E"],
                    rhs=[point.power]
                )

        # Maximum number of cables linked to a substation
        for h, point in enumerate(self.__points):
            if point.power < -0.5:
                constraint = ["y({0},{1})".format(i+1, h+1) for i in range(self.__n_nodes)] \
                             + \
                             ["s({0})".format(h+1)]
                coefficients = [1] * self.__n_nodes + [-1]
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=constraint,
                        val=coefficients,
                    )],
                    senses=["L"],
                    rhs=[self.c]
                )

        # Avoid double cable between two points
        for edge in edges:
            summation = ["y({0},{1})".format(edge.source + 1, edge.destination + 1)] \
                        + \
                        ["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1) for k in
                         range(self.__num_cables)]
            coefficients = [1] + [-1] * self.__num_cables
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
                         range(self.__num_cables)]
            coefficients = [-1] + [cable.capacity for cable in self.__cables]

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
        model.write(self.__project_path + "/out/" + self.out_dir_name + "/lpmodel.lp")

        return model

    def __build_model_docplex(self):

        """
        Build the model using docplex API

        :return: The model filled with variables and constraints
        """
        if not self.__interface == 'docplex':
            raise NameError("For some reason the docplex model has been called when " +
                            "the 'interface' variable has been set to: " + self.__interface)

        edges = [self.__Edge(i, j) for i in range(self.__n_nodes) for j in range(self.__n_nodes)]

        model = Model(name=self.__name)
        model.set_time_limit(self.time_limit)

        # Add y_i.j variables
        self.__y_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            model.binary_var(name="y({0},{1})".format(edge.source + 1, edge.destination + 1))
            if self.__debug_mode:
                if self.__ypos(index) != model.get_statistics().number_of_variables - 1:
                    raise NameError('Number of variables and index do not match')

                # Add f_i.j variables
                self.__f_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            model.continuous_var(name="f({0},{1})".format(edge.source + 1, edge.destination + 1))
            if self.__debug_mode:
                if self.__fpos(index) != model.get_statistics().number_of_variables - 1:
                    raise NameError('Number of variables and index do not match')

        # Add x_i.j.k variables
        self.__x_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            for k in range(self.__num_cables):
                model.binary_var(name="x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))

                if self.__debug_mode:
                    if self.__xpos(index, k) != model.get_statistics().number_of_variables - 1:
                        raise NameError('Number of variables and index do not match')

        # No self-loops constraints on y_i.i variables
        for i in range(self.__n_nodes):
            var = model.get_var_by_name("y({0},{1})".format(i + 1, i + 1))
            model.add_constraint(var == 0)

        # No self-loops constraints on f_i.i variables
        for i in range(self.__n_nodes):
            var = model.get_var_by_name("f({0},{1})".format(i + 1, i + 1))
            model.add_constraint(var == 0)

        # Out-degree constraints
        for h in range(len(self.__points)):
            if self.__points[h].power < -0.5:
                model.add_constraint(
                    model.sum(model.get_var_by_name("y({0},{1})".format(h + 1, j + 1)) for j in range(self.__n_nodes))
                    ==
                    0
                )
            else:
                model.add_constraint(
                    model.sum(model.get_var_by_name("y({0},{1})".format(h + 1, j + 1)) for j in range(self.__n_nodes))
                    ==
                    1
                )

        # Maximum number of cables linked to a substation
        for h in range(len(self.__points)):
            if self.__points[h].power < -0.5:
                model.add_constraint(
                    model.sum(model.get_var_by_name("y({0},{1})".format(i + 1, h + 1)) for i in range(self.__n_nodes))
                    <=
                    self.c
                )
        # Flow balancing constraint
        for h in range(len(self.__points)):
            if self.__points[h].power > 0.5:
                model.add_constraint(
                    model.sum(model.get_var_by_name("f({0},{1})".format(h + 1, j + 1)) for j in range(self.__n_nodes))
                    ==
                    model.sum(model.get_var_by_name("f({0},{1})".format(j + 1, h + 1)) for j in range(self.__n_nodes))
                    + self.__points[h].power
                )

        # Avoid double cable between two points
        for edge in edges:
            model.add_constraint(
                model.get_var_by_name("y({0},{1})".format(edge.source + 1, edge.destination + 1))
                ==
                model.sum(
                    model.get_var_by_name("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)) for k
                    in range(self.__num_cables))
            )

        # Guarantee that the cable is enough for the connection
        for edge in edges:
            model.add_constraint(
                model.sum(
                    model.get_var_by_name("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)) *
                    self.__cables[k].capacity
                    for k in range(self.__num_cables)
                )
                >=
                model.get_var_by_name("f({0},{1})".format(edge.source + 1, edge.destination + 1))
            )

        # Objective function
        model.minimize(
            model.sum(
                self.__cables[k].price * Instance.get_distance(self.__points[i],
                                                               self.__points[j]) * model.get_var_by_name(
                    "x({0},{1},{2})".format(i + 1, j + 1, k + 1))
                for k in range(self.__num_cables) for i in range(self.__n_nodes) for j in range(self.__n_nodes)
            )
        )

        # Adding the parameters to the model
        model.parameters.mip.strategy.rinsheur.set(self.rins)
        model.parameters.mip.polishafter.time.set(self.polishtime)
        model.parameters.timelimit.set(self.time_limit)

        # Writing the model to a proper location
        model.write(self.__project_path + "/out/lpmodel.lp")

        return model

    def __get_solution(self):

        """
        Read all x(i, j, k) variables and store the ones with value '1' into the solution

        :return: List of CableSol named tuples
        """

        sol = []
        edges = [self.__Edge(i, j) for i in range(self.__n_nodes) for j in range(self.__n_nodes)]
        if self.__interface == 'cplex':
            for edge in edges:
                for k in range(self.__num_cables):
                    val = self.__model.solution.get_values(
                        "x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))
                    if val > 0.5:
                        sol.append(self.__CableSol(edge.source + 1, edge.destination + 1, k + 1))
        elif self.__interface == 'docplex':
            for edge in edges:
                for k in range(self.__num_cables):
                    val = self.__model.solution.get_value(
                        "x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))
                    if val > 0.5:
                        sol.append(self.__CableSol(edge.source + 1, edge.destination + 1, k + 1))
        else:
            raise ValueError("The given interface isn't a valid option. " +
                             "Either 'docplex' or 'cplex' are valid options; given: " +
                             str(self.__interface))

        return sol

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

        abspath = os.path.abspath(os.path.dirname(__file__)).strip()
        path_dirs = abspath.split('/')
        path_dirs.remove('')

        self.__project_path = ''
        or2_found = False
        i = 0
        while not or2_found:
            if path_dirs[i] == 'OR2':
                or2_found = True
            self.__project_path += '/' + path_dirs[i]
            i += 1

        self.turb_file = self.__project_path + "/data/data_" + data_tostring + ".turb"
        self.cbl_file = self.__project_path + "/data/data_" + data_tostring + ".cbl"

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

        self.__name = "Wind Farm 0" + str(wf_number)

    def __ypos(self, offset):

        """
        py:function:: __ypos(self, offset)

        Debug-purpose defined function.

        :param offset: Offset w.r.t ystart and the edge indexed by (i, j)

        """

        return self.__y_start + offset

    def __plot_high_quality(self, show=False, export=False):

        """
        py:function:: plot_high_quality(inst, edges)

        Plot the solution using standard libraries

        :param export: whatever this means

        """
        edges = self.__get_solution()
        G = nx.DiGraph()

        mapping = {}

        for i in range(self.__n_nodes):
            if self.__points[i].power < -0.5:
                mapping[i] = 'S{0}'.format(i + 1)
            else:
                mapping[i] = 'T{0}'.format(i + 1)

        for index, node in enumerate(self.__points):
            G.add_node(index)

        for edge in edges:
            G.add_edge(edge.source - 1, edge.destination - 1)

        pos = {i: (point.x, point.y) for i, point in enumerate(self.__points)}

        # Avoid re scaling of axes
        plt.gca().set_aspect('equal', adjustable='box')

        # draw graph
        nx.draw(G, pos, with_labels=True, node_size=1300, alpha=0.3, arrows=True, labels=mapping, node_color='g',
                linewidth=10)

        if export:
            plt.savefig(self.__project_path + '/out/' + self.out_dir_name + '/img/foo.svg')

        # show graph
        if show:
            plt.show()

    # Get and set methods, in the Pythonic way

    @property
    def out_dir_name(self):
        return self.__out_dir_name

    @out_dir_name.setter
    def out_dir_name(self, d):
        if not type(d) == str:
            warnings.warn("Out path not given as string. Trying a conversion.", ValueWarning)
            d = str(d)
        if not os.path.exists(self.__project_path + '/out/' + d):
            os.makedirs(self.__project_path + '/out/' + d)
        if not os.path.exists(self.__project_path + '/out/' + d + '/img'):
            os.makedirs(self.__project_path + '/out/' + d + '/img')
        self.__out_dir_name = d

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
        if not type(t) == int:
            raise TypeError("Timeout time should be given as an integer; given: " + str(t))
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
            raise TypeError("The frequency at which RINS will be applied should be an integer; given: " + str(freq))
        if freq < -1:
            raise ValueError("Invalid RINS parameter. Integer values above -1 are OK; given: " + str(freq))
        self.__rins = freq

    @property
    def polishtime(self):
        return self.__polishtime

    @polishtime.setter
    def polishtime(self, t):
        if not type(t) == int:
            raise TypeError("Polish time should be an integer; given: " + str(t))
        if t <= 3:
            raise TimeoutError("Can't start polishing within just 3 seconds;  given: " + str(t))
        if t >= self.time_limit:
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
            raise TypeError("Expecting 'cluster' to be a boolean, either set True or False; given:" + str(c))
        self.__cluster = c

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, c):
        if not type(c) == int:
            raise TypeError("The parameter 'c' must be a positive, INTEGER number; given: " + str(c))
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

    def build_model(self):
        """
        Builds the model with the right interface.
        Raise an error if the specified interface doesn't exist.
        """

        if self.__interface == 'cplex':
            self.__model = self.__build_model_cplex()
        elif self.__interface == 'docplex':
            self.__model = self.__build_model_docplex()
        else:
            raise ValueError("The given interface isn't a valid option." +
                             "Either 'docplex' or 'cplex' are valid options; given: " +
                             str(self.__interface))

    def solve(self):
        """
        Simply solves the problem by invoking the .solve() method within the model selected.
        """
        self.__model.solve()

    def write_solutions(self):
        """
        Writes the solutions obtained by first invoking the built-in function of the model,
        and then by returning our private __get_solution() method, which returns the list of the
        x_{ij}^k variables set to one.
        :return: the list of x_{ij}^k variables set to one from the solution
        """

        self.__model.solution.write(self.__project_path + "/out/" + self.out_dir_name + "/mysol.sol")
        return self.__get_solution()

    def parse_command_line(self):

        """
        py:function:: parse_command_line(self)

        Parses the command line.

        :return: None
        """

        parser = argparse.ArgumentParser(description='Process details about instance and interface.')

        parser.add_argument('--dataset', type=int, help="dataset selection; datasets available: [1,29]. " +
                                                        "You can use '30' for debug purposes")
        parser.add_argument('--cluster', action="store_true", help='type --cluster if you want to use the cluster')
        parser.add_argument('--interface', choices=['docplex', 'cplex'], help='Choose interface ')
        parser.add_argument('--C', type=int, help='the maximum number of cables linked to a substation')
        parser.add_argument('--rins', type=int, help='the frequency with which the RINS Heuristic will be applied')
        parser.add_argument('--timeout', type=int, help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--polishtime', type=int, help='the time to wait before applying polishing')
        parser.add_argument('--outfolder', type=str, help='name of the folder to be created inside the /out' +
                                                          ' directory, which contains everything related to this run')

        args = parser.parse_args()

        if args.outfolder:
            self.out_dir_name = args.outfolder

        if args.dataset:
            self.data_select = args.dataset

        if args.cluster:
            self.cluster = True

        if args.interface == 'docplex' or args.interface == 'cplex':
            self.__interface = args.interface
        else:
            warnings.warn("Invalid interface; '" + str(args.interface) + "' given. "
                          + "Using the default value: " + self.__interface,
                          ParseWarning)
            self.__interface = 'cplex'

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

    def read_input(self):
        """
        This function reads the input files by invoking the private methods which read
        both the turbines and the cables files.

        :return: None
        """

        self.__read_turbines_file()
        self.__read_cables_file()

    def plot_solution(self, show=False, high=False, export=False):
        """
        py:function:: plot_solution(inst, edges)
        Plots the solution using the plot.ly library

        :param show: if =True, the exported plot will be shown right away.
        :param high: if =True, an high-quality img will be plotted, also
        :param export: if =True, such high-quality img will be exported
        :return:
        """

        edges = self.__get_solution()
        G = nx.DiGraph()

        for index, node in enumerate(self.__points):
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
            text=["Substation #{0}".format(i + 1) if self.__points[i].power < -0.5 else "Turbine #{0}".format(i + 1) for
                  i in range(self.__n_nodes)],
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
                title='<br><b style="font-size:20px>' + self.__name + '</b>',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        py.plot(fig, filename=self.__project_path + '/out/' + self.out_dir_name + '/img/wind_farm.html')

        if high:
            self.__plot_high_quality(show=show, export=export)

    @staticmethod
    def get_distance(point1, point2):

        """
        py:function:: get_distance(point1, point2)
        Get the distance between two given points

        :param point1: First point
        :param point2: Second point

        """

        return math.sqrt(
            (point1.x - point2.x) ** 2
            +
            (point1.y - point2.y) ** 2
        )
