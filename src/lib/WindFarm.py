"""
Everything related to the instance of our problem is written in here.
"""
try:
    from docplex.mp.model import Model
    import plotly.offline as py
    from plotly.graph_objs import *
    import networkx as nx
    import matplotlib.pyplot as plt
except:
    pass

import os.path
import warnings
import time
import argparse
import cplex
import math
from collections import namedtuple

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
        self.__n_substations = 0
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
        self.__Point = namedtuple("Point", ["id", "x", "y", "power"])
        self.__Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])
        self.__CableSol = namedtuple("CableSol", ["source", "destination", "capacity"])

        # Parameters, commenting out whatever we don't use
        # self.model_type (???)
        # self.num_threads
        self.cluster = False
        self.time_limit = 60
        self.rins = 7
        self.polishtime = 45
        self.__slack = True
        self.cross_mode = 'no'
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
        i=1
        with open(self.turb_file, "r") as fp:
            for line in fp:
                words = list(map(int, line.split()))
                points.append(self.__Point(i, words[0], words[1], words[2]))
                if int(words[2]) < 0.5:
                    self.__n_substations += 1
                i+=1

        self.__n_nodes = len(points)
        self.__n_turbines = self.__n_nodes - self.__n_substations
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

    def __fpos(self, i_off, j_off):
        """

        :param i_off:
        :param j_off:
        :return:
        """

        return self.__f_start + i_off*self.__n_nodes + j_off

    def __xpos(self, i_off, j_off, k_off):
        """
        .. description missing ..
        :param i_off:
        :param j_off:
        :param k_off:
        :return:
        """

        return self.__x_start + i_off*self.__n_nodes*self.__num_cables + j_off*self.__num_cables + k_off

    def __ypos(self, i_off, j_off):
        """

        :param i_off:
        :param j_off:
        :return:
        """

        return self.__y_start + i_off*self.__n_nodes + j_off

    def __slackpos(self, slack_off):

        """
        py:function:: __slackpos(offset, k, inst)

        .. description missing ..

        :param offset: Offset w.r.t slackstart and the substation indexed by h

        """
        return self.__slack_start + slack_off

    def __build_model_cplex(self):

        """
        Build the model using classical cplex API

        :return: The model filled with variables and constraints
        """
        if not self.__interface == 'cplex':
            raise NameError("For some reason the classical model has been called when " +
                            "the 'interface' variable has been set to: " + self.__interface)

        self.__model = cplex.Cplex()

        self.__model.set_problem_name(self.__name)
        self.__model.objective.set_sense(self.__model.objective.sense.minimize)

        # Add y(i,j) variables
        self.__y_start = self.__model.variables.get_num()
        self.__model.variables.add(
            types=[self.__model.variables.type.binary]
                  * (self.__n_nodes**2),
            names=["y({0},{1})".format(i+1, j+1)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)]
        )

        # Add f(i,j) variables
        self.__f_start = self.__model.variables.get_num()
        self.__model.variables.add(
            types=[self.__model.variables.type.continuous]
                  * (self.__n_nodes**2),
            names=["f({0},{1})".format(i+1, j+1)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)]
        )

        # Add x(i,j,k) variables
        self.__x_start = self.__model.variables.get_num()
        self.__model.variables.add(
            types=[self.__model.variables.type.binary]
                  * (self.__n_nodes**2)
                  * self.__num_cables,
            names=["x({0},{1},{2})".format(i+1, j+1, k+1)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)
                   for k in range(self.__num_cables)],
            obj=[cable.price * WindFarm.get_distance(v,u)
                 for v in self.__points
                 for u in self.__points
                 for cable in self.__cables]
        )

        if self.__slack:
            # Add s(h) (slack) variables
            self.__slack_start = self.__model.variables.get_num()
            self.__model.variables.add(
                types=[self.__model.variables.type.continuous]
                      * self.__n_substations,
                names=["s({0})".format(h+1)
                       for h, point in enumerate(self.__points)
                       if point.power < -0.5],
                obj=[1e9] * self.__n_substations
            )
        else:
            # No variables should be added, then.
            self.__slack_start = -1

        # No self-loop constraints on y(i,i) = 0 \forall i
        self.__model.variables.set_upper_bounds([
            (self.__ypos(i,i), 0)
            for i in range(self.__n_nodes)
        ])

        # No self-loop constraints on f(i,i) = 0 \forall i
        self.__model.variables.set_upper_bounds([
            (self.__fpos(i,i), 0)
            for i in range(self.__n_nodes)
        ])

        # No self-loop constraints on x(i,i,k) = 0 \forall i,k
        self.__model.variables.set_upper_bounds([
            (self.__xpos(i,i,k), 0)
            for i in range(self.__n_nodes)
            for k in range(self.__num_cables)
        ])

        # Energy flow must be positive
        self.__model.variables.set_lower_bounds([
            (self.__fpos(i,j), 0)
            for i in range(self.__n_nodes)
            for j in range(self.__n_nodes)
        ])

        # Out-degree constraints (substations)
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__ypos(h,j) for j in range(self.__n_nodes)],
                val=[1.0] * self.__n_nodes
            )
                for h,point in enumerate(self.__points)
                if point.power<-0.5
            ],
            senses=["E"] * self.__n_substations,
            rhs=[0] * self.__n_substations
        )

        # Out-degree constraints (turbines)
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__ypos(h,j) for j in range(self.__n_nodes)],
                val=[1.0] * self.__n_nodes
            )
                for h,point in enumerate(self.__points)
                if point.power>=0.5
            ],
            senses=["E"] * self.__n_turbines,
            rhs=[1] * self.__n_turbines
        )

        # Flow balancing constraint
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__fpos(h,j) for j in range(self.__n_nodes) if h!=j] +
                    [self.__fpos(j,h) for j in range(self.__n_nodes) if j!=h],
                val=[1] * (self.__n_nodes - 1) + [-1] * (self.__n_nodes - 1)
            )
                for h,point in enumerate(self.__points)
                if point.power >= 0.5
            ],
            senses=["E"] * self.__n_turbines,
            rhs=[point.power for point in self.__points if point.power > 0.5]
        )

        # Maximum number of cables linked to a substation
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__ypos(i,h) for i in range(self.__n_nodes)] +
                    [self.__slackpos(h)] if self.__slack else [],
                val=[1] * self.__n_nodes +
                    [-1] if self.__slack else []
            )
                for h,point in enumerate(self.__points)
                if point.power < -0.5
            ],
            senses=["L"] * self.__n_substations,
            rhs=[self.c] * self.__n_substations
        )

        # Avoid double cables between two points
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__ypos(i,j)] + [self.__xpos(i,j,k) for k in range(self.__num_cables)],
                val=[1] + [-1] * self.__num_cables
            )
                for i in range(self.__n_nodes)
                for j in range(self.__n_nodes)
            ],
            senses=["E"] * (self.__n_nodes**2),
            rhs=[0] * (self.__n_nodes**2)
        )

        # We want to guarantee that the cable is enough for the connection
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__fpos(i,j)] + [self.__xpos(i,j,k) for k in range(self.__num_cables)],
                val=[-1] + [cable.capacity for cable in self.__cables]
            )
                for i in range(self.__n_nodes)
                for j in range(self.__n_nodes)
            ],
            senses=["G"] * (self.__n_nodes**2),
            rhs=[0] * (self.__n_nodes**2)
        )

        # (in case) No-crossing constraints
        if self.cross_mode == 'lazy':
            for i, a in enumerate(self.__points):
                # We want to lighten these constraints as much as possible; for this reason, j=i+1
                for j, b in enumerate(self.__points[i+1:], start=i+1):
                    for k, c in enumerate(self.__points):
                        self.__add_violating_constraint(i,j,k)

        # Adding the parameters to the self.__model
        self.__model.parameters.mip.strategy.rinsheur.set(self.rins)
        if self.polishtime < self.time_limit:
            self.__model.parameters.mip.polishafter.time.set(self.polishtime)
        self.__model.parameters.timelimit.set(self.time_limit)

        # Writing the model to a proper location
        self.__model.write(self.__project_path + "/out/" + self.out_dir_name + "/lpmodel.lp")

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

        # Add y(i,j) variables
        self.__y_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            model.binary_var(name="y({0},{1})".format(edge.source + 1, edge.destination + 1))
            if self.__debug_mode:
                if self.__ypos(index) != model.get_statistics().number_of_variables - 1:
                    raise NameError('Number of variables and index do not match')

        # Add f(i,j) variables
        self.__f_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            model.continuous_var(name="f({0},{1})".format(edge.source + 1, edge.destination + 1))
            if self.__debug_mode:
                if self.__fpos(index) != model.get_statistics().number_of_variables - 1:
                    raise NameError('Number of variables and index do not match')

        # Add x(i,j,k) variables
        self.__x_start = model.get_statistics().number_of_variables
        for index, edge in enumerate(edges):
            for k in range(self.__num_cables):
                model.binary_var(name="x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))

                if self.__debug_mode:
                    if self.__xpos(index, k) != model.get_statistics().number_of_variables - 1:
                        raise NameError('Number of variables and index do not match')

        # No self-loops constraints on y(i,i) variables
        for i in range(self.__n_nodes):
            var = model.get_var_by_name("y({0},{1})".format(i + 1, i + 1))
            model.add_constraint(var == 0)

        # No self-loops constraints on f(i,i) variables
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
                self.__cables[k].price * WindFarm.get_distance(self.__points[i],
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

        if not (self.__interface=='cplex' or self.__interface=='docplex'):
            raise ValueError("The given interface isn't a valid option. " +
                             "Either 'docplex' or 'cplex' are valid options; given: " +
                             str(self.__interface))

        sol = [self.__CableSol(i+1,j+1,k+1)
               for i in range(self.__n_nodes)
               for j in range(self.__n_nodes)
               for k in range(self.__num_cables)
               if self.__model.solution.get_values(self.__xpos(i,j,k))>0.5]
        
        return sol

    def __build_input_files(self):
        """
        py:function:: __build_input_files(self)

        Sets the input file correctly, based on the dataset selection

        """
        if not type(self.data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(self.data_select))
        if self.data_select <= 0 or self.data_select >= 31:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-30]. Given: " + str(self.data_select))

        data_tostring = str(self.data_select)
        if 1 <= self.data_select <= 9:
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

        self.turb_file = self.__project_path + "/data/data_" + data_tostring + ".turb"
        self.cbl_file = self.__project_path + "/data/data_" + data_tostring + ".cbl"

    def __build_name(self):
        """
        py:function:: __build_name(self)

        Sets the name of the wind farm correctly, based on the dataset selection

        """
        if not type(self.data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(self.data_select))
        if self.data_select <= 0 or self.data_select >= 31:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-30]. Given: " + str(self.data_select))

        # We assume that, in this context, we'll never have a WF >=10
        wf_number = 0
        if 0 <= self.data_select <= 6:
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

    def __are_crossing(self, pt1, pt2, pt3, pt4):
        """
        Recall that one edge has its extremes on (x_1,y_1) and (x_2,y_2);
        the same goes for the second edge, which extremes are (x_3,y_3) and (x_4,y_4).

        :param pt1: first point
        :param pt2: second point
        :param pt3: third point
        :param pt4: fourth point
        :return:
        """
        det_A = (pt4.x - pt3.x) * (pt1.y - pt2.y) - (pt4.y - pt3.y) * (pt1.x - pt2.x)
        if det_A == 0:
            return False

        # If it's not zero, then the 2x2 system has exactly one solution, which is:
        det_mu = (pt1.x - pt3.x) * (pt1.y - pt2.y) - (pt1.y - pt3.y) * (pt1.x - pt2.x)
        det_lambda = (pt4.x - pt3.x) * (pt1.y - pt3.y) - (pt4.y - pt3.y) * (pt1.x - pt3.x)

        mu = det_mu / det_A
        lambd = det_lambda / det_A

        if 0 < lambd < 1 and 0 < mu < 1:
            return True
        else:
            return False

    def __get_violated_edges(self):
        """
        This function gets a list of chosen edges (ys) and returns a list of violated edges
        :param ys: chosen edges
        :return: a list of violated edges in the (a,b,c,d)-tuple form
        """

        # Gets the values from the model
        ys = [self.__Edge(i+1,j+1)
              for i in range(self.__n_nodes)
              for j in range(self.__n_nodes)]
        ys_values = self.__model.solution.get_values([self.__ypos(i,j)
                                                      for i in range(self.__n_nodes)
                                                      for j in range(self.__n_nodes)])
        # extracts only the selected edges
        ones = [ys[i] for i,y in enumerate(ys_values) if y>0.5]

        # extracts the violated edges:
        # recall that we're interested in the (a,b,c,d) tuple,
        # rather than the two corresponding edges.

        violated = [(v.source, v.destination, u.source, u.destination)
                    for i,v in enumerate(ones)
                    for u in ones[i+1:]
                    if self.__are_crossing(self.__points[v.source-1],
                                           self.__points[v.destination-1],
                                           self.__points[u.source-1],
                                           self.__points[u.destination-1])]

        return violated

    def __add_violating_constraint(self, i,j,k):
        # Skipping the points that do not occur in the (a,b) and (b,a) edges
        a = self.__points[i]
        b = self.__points[j]
        c = self.__points[k]

        if self.cross_mode=='lazy':
            constraint_add = self.__model.linear_constraints.advanced.add_lazy_constraints
        else:
            constraint_add = self.__model.linear_constraints.add

        if not (c == a or c == b):
            current_couple = [self.__ypos(i,j), self.__ypos(j,i)]
            crossings = [self.__ypos(k,l)
                         for l,d in enumerate(self.__points[k+1:], start=k+1)
                         if self.__are_crossing(a,b,c,d)]

            if len(crossings) > 0:
                summation =  crossings + current_couple
                coefficients = [1] * len(summation)
                constraint_add(
                    lin_expr=[cplex.SparsePair(
                        ind=summation,
                        val=coefficients
                    )],
                    senses=["L"],
                    rhs=[1]
                )

    def build_model(self):
        """
        Builds the model with the right interface.
        Raise an error if the specified interface doesn't exist.
        """

        if self.__interface == 'cplex':
            self.__build_model_cplex()
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
        if self.cross_mode == 'lazy' or self.cross_mode == 'no':
            # This simply means that CPLEX has everything he needs to have inside his enviroment
            self.__model.solve()
        elif self.cross_mode == 'loop':
            xs = True       # "are there any crosses in this solution?"
            opt = False     # "has the optimum been reached?"
            starting_time = time.time()
            current_time = time.time()
            for i in range(3):
                self.__model.solve()
            while xs and not opt and current_time-starting_time < 600:
                self.__model.solve()

                crossed = self.__get_violated_edges()
                print(crossed)

                if len(crossed) > 0:
                    for xs in crossed:
                        self.__add_violating_constraint(xs[0], xs[1], xs[2])
                        self.__add_violating_constraint(xs[2], xs[3], xs[0])
                else:
                    # No crossings detected? Check.
                    xs = False

                if self.__model.solution.get_status() == self.__model.solution.status.MIP_optimal:
                    # Optimality reached? Check.
                    opt = True
                current_time = time.time()


        elif self.cross_mode == 'callback':
            # Something will happen here...
            raise ValueError("Callbacks not available, yet!")
        else:
            raise ValueError("Unrecognized cross-strategy; given: " + str(self.cross_mode))

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
        parser.add_argument('--interface', choices=['docplex', 'cplex'], help='Choose the interface ')
        parser.add_argument('--C', type=int, help='the maximum number of cables linked to a substation')
        parser.add_argument('--rins', type=int, help='the frequency with which the RINS Heuristic will be applied')
        parser.add_argument('--timeout', type=int, help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--polishtime', type=int, help='the time to wait before applying polishing')
        parser.add_argument('--outfolder', type=str, help='name of the folder to be created inside the /out' +
                                                          ' directory, which contains everything related to this run')
        parser.add_argument('--noslack', action="store_true",
                            help='type --noslack if you do not want the soft version of the problem')
        parser.add_argument('--crossings', choices=['no', 'lazy', 'loop', 'callback'],
                            help='Choose how you want to address the crossing problem')

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

        if args.noslack:
            self.__slack = False

        if args.crossings:
            self.cross_mode = args.crossings

        # Since the parameters have been updated, We must re-build the input files, names, etc.
        self.__build_input_files()
        self.__build_name()

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

    # Get and set methods, in the Pythonic way

    @property
    def cross_mode(self):
        return self.__cross_mode

    @cross_mode.setter
    def cross_mode(self, cm):
        if not (cm=='no' or cm=='lazy' or cm=='loop' or cm=='callback'):
            raise ValueError("Unrecognized crossing strategy; given: "+str(cm))
        self.__cross_mode = cm

    @property
    def cluster(self):
        return self.__cluster

    @cluster.setter
    def cluster(self, c):
        if not type(c) == bool:
            raise TypeError("Expecting 'cluster' to be a boolean, either set True or False; given:" + str(c))
        self.__cluster = c

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
        if d <= -1 or d >= 31:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [0-30]. Given: " + str(d))
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
