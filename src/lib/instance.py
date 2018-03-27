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


class ValueWarning(UserWarning):
    pass


class ParseWarning(UserWarning):
    pass


class Instance:
    """
    py:class:: instance()

    This class stores all the useful information about input data and parameters

    """

    def __init__(self, trb_f='', cbl_f='', name="MyOptimizer"):
        # Model variables
        self.name = name
        self.y_start = 0
        self.f_start = 0
        self.x_start = 0

        # Input data variables
        self.cbl_file = cbl_f
        self.turb_file = trb_f
        self.n_nodes = 0
        self.num_cables = 0
        self.points = []
        self.cables = []
        self.c = 10

        # Named tuples, describing input data
        self.Edge = namedtuple("Edge", ["source", "destination"])
        self.Point = namedtuple("Point", ["x", "y", "power"])
        self.Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])
        self.CableSol = namedtuple("CableSol", ["source", "destination", "capacity"])

        # Parameters, commenting out whatever we don't use
        # self.model_type
        # self.num_threads
        self.cluster = False
        self.time_limit = 300
        self.rins = 7
        self.polishtime = 400
        self.debug_mode = False
        self.verbosity = 0
        self.interface = 'cplex'
        # available_memory
        # self.max_nodes
        # self.cutoff
        # self.integer_costs

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
        py:function:: ypos(offset, inst)

        Plot the solution using standard libraries

        :param offset: Offset w.r.t ystart and the edge indexed by (i, j)

        """

        return self.y_start + offset

    def parse_command_line(self):

        """
        Parse the command line

        :return: None
        """

        parser = argparse.ArgumentParser(description='Process details about instance and interface.')

        parser.add_argument('--cf', type=str, nargs=1,
                            help='cable relative file path, starting from folder \'data\'')
        parser.add_argument('--tf', type=str, nargs=1,
                            help='turbine relative file path, starting from folder \'data\'')
        parser.add_argument('--cluster', action="store_true", help='type --cluster if you want to use the cluster')
        parser.add_argument('--interface', choices=['docplex', 'cplex'], help='Choose interface ')
        parser.add_argument('--C', type=int, help='the maximum number of cables linked to a substation')
        parser.add_argument('--rins', type=int, help='the frequency with which the RINS Heuristic will be applied')
        parser.add_argument('--timeout', type=int, help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--polishtime', type=int, help='the time to wait before applying polishing')

        args = parser.parse_args()

        if args.cf and args.tf:
            # Modifying one of these two files implies to modify the other one, too.
            self.cbl_file = args.cf[0]
            self.turb_file = args.tf[0]
        elif (not args.cf and args.tf) or (args.cf and not args.tf):
            # Therefore, modifying just one and one only raises this error.
            raise NameError("Both --cf and --tf must be set")
        else:
            # Leaving both parameters alone means to set the default values,
            # which is allowed but we want to be warned about it.
            warnings.warn("No '.cbl'/'.trb' input files are set; using the default values: "
                          + self.cbl_file
                          + ", " + self.turb_file, ParseWarning)

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

        file = open("../data/" + self.turb_file, "r")

        points = []
        for index, line in enumerate(file):
            #if index >= 3:break
            words = list(map(int, line.split()))
            points.append(
                self.Point(words[0], words[1], words[2])
            )

        file.close()
        self.n_nodes = len(points)
        self.points = points

    def read_cables_file(self):

        """
        py:function:: read_cables_file(self)
        Read the cables file

        """

        file = open("../data/" + self.cbl_file, "r")

        cables = []
        for index, line in enumerate(file):
            #if index >= 3: break
            words = line.split()
            cables.append(
                self.Cable(int(words[0]), float(words[1]), int(words[2]))
            )

        file.close()
        self.num_cables = len(cables)
        self.cables = cables

    def fpos(self, offset):

        """
        py:function:: fpos(offset, self)

        Plot the solution using standard libraries

        :param offset: Offset w.r.t fstart and the edge indexed by (i, j)

        """
        return self.f_start + offset

    def xpos(self, offset, k):

        """
        py:function:: xpos(offset, k, inst)

        Plot the solution using standard libraries

        :param offset: Offset w.r.t xstart and the edge indexed by (i, j)
        :param k: index of the cable considered

        """
        return self.x_start + offset * self.num_cables + k

    @staticmethod
    def get_distance(point1, point2):

        """
        py:function:: get_distance(point1, point2)
        Get the distance between two poins

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

        for index, edge in enumerate(name_y_edges):
            model.variables.add(
                types=[model.variables.type.binary],
                names=[name_y_edges[index]],
            )
            if self.debug_mode:
                if self.ypos(index) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

        # Add f(i,j) variables
        self.f_start = model.variables.get_num()
        name_f_edges = ["f({0},{1})".format(i + 1, j + 1) for i in range(self.n_nodes) for j in range(self.n_nodes)]

        for index, edge in enumerate(name_f_edges):
            model.variables.add(
                types=[model.variables.type.continuous],
                names=[name_f_edges[index]]
            )
            if self.debug_mode:
                if self.fpos(index) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

        # Add x(i,j,k) variables
        self.x_start = model.variables.get_num()

        for index, edge in enumerate(edges):
            for k in range(self.num_cables):
                model.variables.add(
                    types=[model.variables.type.binary],
                    names=["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)],
                    obj=[self.cables[k].price * Instance.get_distance(self.points[edge.source], self.points[edge.destination])]
                )
                if self.debug_mode:
                    if self.xpos(index, k) != model.variables.get_num() - 1:
                        raise NameError('Number of variables and index do not match')

        # No self-loop constraint
        for i in range(self.n_nodes):
            model.variables.set_upper_bounds([("y({0},{1})".format(i + 1, i + 1), 0)])

        for i in range(self.n_nodes):
            model.variables.set_upper_bounds([("f({0},{1})".format(i + 1, i + 1), 0)])

        # Out-degree constraints
        for h in range(len(self.points)):
            if self.points[h].power < -0.5:
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(h + 1, j + 1) for j in range(self.n_nodes)],
                        val=[1.0] * self.n_nodes
                    )],
                    senses=["E"],
                    rhs=[0]
                )
            else:
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=["y({0},{1})".format(h + 1, j + 1) for j in range(self.n_nodes)],
                        val=[1.0] * self.n_nodes
                    )],
                    senses=["E"],
                    rhs=[1]
                )

        # Flow balancing constraint
        for h in range(len(self.points)):
            if self.points[h].power > 0.5:
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
                    rhs=[self.points[h].power]
                )

        # Maximum number of cables linked to a substation
        for h in range(len(self.points)):
            if self.points[h].power < -0.5:
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
            coefficients = [-1] + [self.cables[k].capacity for k in range(self.num_cables)]
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
