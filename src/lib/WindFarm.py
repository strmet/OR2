"""

Everything related to the instance of our problem is written in here.

"""

# Docplex 2.4.56 must be used on the cluster to match the CPLEX' version installed
# To install it type: pip3 install docplex=2.4.56 --user
from docplex.mp.model import Model
try:
    import plotly.offline as py
    from plotly.graph_objs import *
    import matplotlib.pyplot as plt
except:
    pass
import networkx as nx
import pprint as pp
import os.path
import warnings
import time
import argparse
import cplex
import math
from collections import namedtuple
import multiprocessing
from .callback import LazyCallback
import random
import sys


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
    def __init__(self):

        # The CPLEX/DOCPLEX model itself
        self.__model = None

        # Model variables/parameters
        self.__name = ''
        self.__y_start = 0
        self.__f_start = 0
        self.__x_start = 0
        self.__substation_slack_start = 0
        self.__flux_slack_start = 0
        self.__points = []
        self.__cables = []
        self.c = 7

        # Data counters
        self.__n_nodes = 0
        self.__n_cables = 0
        self.__n_turbines = 0
        self.__n_substations = 0
        self.__best_sol = []
        self.__best_incumbent = 10e12
        self.__true_gap = 1
        self.__starting_time = 0

        # "Named tuples", very useful for describing input data
        # without creating new classes
        self.__EdgeSol = namedtuple("EdgeSol", ["idx", "s", "d"])
        self.__CableSol = namedtuple("CableSol", ["idx", "s", "d", "capacity"])
        self.__FluxSol = namedtuple("FluxSol", ["idx", "s", "d"])
        self.__Point = namedtuple("Point", ["x", "y", "power"])
        self.__Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])

        # Operating parameters, commenting out whatever we don't use
        self.__project_path = ''
        self.__cluster = False
        self.time_limit = 60
        self.__rins = 7
        self.polishtime = int(1e12)
        self.__substation_slack = True
        self.__flux_slack = True
        self.cross_mode = 'no'
        self.__debug_mode = False
        self.verbosity = 0
        self.__interface = 'cplex'
        self.__metaheuristic = None
        self.__overall_wait_time = 1200

        # Dataset selection and consequent input files building, and output parameters
        self.data_select = 1

        # Building the input/output files, while parsing the command line.
        self.__build_input_files()
        self.out_dir_name = 'test'

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
                if int(words[2]) < 0.5:
                    self.__n_substations += 1

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
                words = line.split()
                cables.append(self.__Cable(int(words[0]), float(words[1]), int(words[2])))

        self.__n_cables = len(cables)
        self.__cables = cables

    def __fpos(self, i_off, j_off):

        """

        Get the position of the column inside the model.
        Must be used the number starting from 0.

        :param i_off: First turbine number
        :param j_off: Second turbine number

        :return: Column index

        """

        return self.__f_start + i_off*self.__n_nodes + j_off

    def __xpos(self, i_off, j_off, k_off):

        """

        Get the position of the column inside the model

        :param i_off: First turbine number
        :param j_off: Second turbine number
        :param k_off: Cable index

        :return: Column index

        """

        return self.__x_start + i_off*self.__n_nodes*self.__n_cables + j_off*self.__n_cables + k_off

    def __ypos(self, i_off, j_off):

        """

        Get the position of the column inside the model.
        Must be used the number starting from 0.

        :param i_off: First turbine number
        :param j_off: Second turbine number

        :return: Column index

        """

        return self.__y_start + i_off*self.__n_nodes + j_off

    def __substation_slackpos(self, slack_off):

        """

        Get the position of the column inside the model.

        :param slack_off: Offset w.r.t to the substation/turbine indexed by slack_off

        :return: Offset w.r.t slackstart

        """

        return self.__substation_slack_start + slack_off

    def __flux_slackpos(self, slack_off):

        """

        Get the position of the column inside the model.

        :param slack_off: Offset w.r.t to the substation indexed by slack_off

        :return: Offset w.r.t slackstart

        """

        return self.__flux_slack_start + slack_off

    def __build_model_cplex(self):

        """

        Build the model using classical cplex API

        :return:

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
                   for j in range(self.__n_nodes)],
            lb=[0] * (self.__n_nodes**2),
            ub=[max([cable.capacity for cable in self.__cables])] * (self.__n_nodes**2),
        )

        # Add x(i,j,k) variables
        self.__x_start = self.__model.variables.get_num()
        self.__model.variables.add(
            types=[self.__model.variables.type.binary]
                  * (self.__n_nodes**2)
                  * self.__n_cables,
            names=["x({0},{1},{2})".format(i+1, j+1, k+1)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)
                   for k in range(self.__n_cables)],
            obj=[cable.price * WindFarm.get_distance(v,u)
                 for v in self.__points
                 for u in self.__points
                 for cable in self.__cables]
        )

        # Slack variables on parameter C
        if self.__substation_slack:
            # Add s(h) (slack) variables
            self.__substation_slack_start = self.__model.variables.get_num()
            self.__model.variables.add(
                types=[self.__model.variables.type.continuous]
                      * self.__n_substations,
                names=["sc({0})".format(h+1)
                       for h, point in enumerate(self.__points)
                       if point.power < -0.5],
                obj=[1e9] * self.__n_substations,
                lb=[0] * self.__n_substations,
                ub=[self.__n_turbines] * self.__n_substations
            )
        else:
            # No variables should be added, then.
            self.__substation_slack_start = -1

        # Slack variables on flux
        if self.__flux_slack:
            # Add slack variables for flux
            self.__flux_slack_start = self.__model.variables.get_num()
            max_capacity = max([cable.capacity for cable in self.__cables])
            self.__model.variables.add(
                types=[self.__model.variables.type.continuous]
                      * self.__n_nodes,
                names=["s2({0})".format(h+1)
                       for h in range(self.__n_nodes)],
                obj=[1e9] * self.__n_nodes,
                ub=[max_capacity] * self.__n_nodes,
                lb=[0] * self.__n_nodes
            )
        else:
            # No variables should be added, then.
            self.__flux_slack_start = -1

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
            for k in range(self.__n_cables)
        ])

        if self.__flux_slack:
            # No slack variable on the substation
            self.__model.variables.set_upper_bounds([
                (self.__flux_slackpos(h), 0)
                for h,point in enumerate(self.__points)
                if point.power<-0.5
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
            senses=(["L"] if self.__flux_slack else ["E"]) * self.__n_turbines,
            rhs=[1] * self.__n_turbines
        )

        # Flow balancing constraint
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__fpos(h,j) for j in range(self.__n_nodes) if h!=j] +
                    [self.__fpos(j,h) for j in range(self.__n_nodes) if j!=h] +
                    ([self.__flux_slackpos(h)] if self.__flux_slack else []),
                val=[1] * (self.__n_nodes - 1) + [-1] * (self.__n_nodes - 1) +
                    ([1] if self.__flux_slack else [])
            )
                for h, point in enumerate(self.__points)
                if point.power > 0.5
            ],
            senses=["E"] * self.__n_turbines,
            rhs=[point.power for point in self.__points if point.power > 0.5]
        )

        # Maximum number of cables linked to a substation
        self.__model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[self.__ypos(i,h) for i in range(self.__n_nodes)] +
                    ([self.__substation_slackpos(h)] if self.__substation_slack else []),
                val=[1] * self.__n_nodes +
                    ([-1] if self.__substation_slack else [])
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
                ind=[self.__ypos(i,j)] + [self.__xpos(i,j,k) for k in range(self.__n_cables)],
                val=[1] + [-1] * self.__n_cables
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
                ind=[self.__fpos(i,j)] + [self.__xpos(i,j,k) for k in range(self.__n_cables)],
                val=[-1] + [cable.capacity for cable in self.__cables]
            )
                for i in range(self.__n_nodes)
                for j in range(self.__n_nodes)
            ],
            senses=["G"] * (self.__n_nodes**2),
            rhs=[0] * (self.__n_nodes**2)
        )

        # (in case) No-crossing lazy constraints
        if self.cross_mode == 'lazy' or self.cross_mode == 'normal':
            for i, a in enumerate(self.__points):
                # We want to lighten these constraints as much as possible; for this reason, j=i+1
                for j, b in enumerate(self.__points[i+1:], start=i+1):
                    # EdgeSol has this structure: ["idx", "s", "d"]
                    current_couple = [
                        self.__EdgeSol(self.__ypos(i,j), i, j),
                        self.__EdgeSol(self.__ypos(j,i), j, i)
                    ]
                    for k, c in enumerate(self.__points):
                        if not (c == a or c == b):
                            violating_cds = [self.__EdgeSol(self.__ypos(k,l), k, l)
                                             for l,d in enumerate(self.__points[k+1:], start=k+1)
                                             if WindFarm.are_crossing(a,b,c,d)]
                            if len(violating_cds) > 0:
                                self.__add_violating_constraint(current_couple + violating_cds)

        # Adding the parameters to the model
        self.__model.parameters.mip.strategy.rinsheur.set(self.__rins)
        if self.polishtime < self.time_limit:
            self.__model.parameters.mip.polishafter.time.set(self.polishtime)

        # The following asks CPLEX to measure its time with REAL CPU time.
        #self.__model.parameters.clocktype.set(1)

        self.__model.parameters.timelimit.set(self.time_limit)

        # Writing the model to a proper location
        self.__model.write(self.__project_path + "/out/" + self.out_dir_name + "/lpmodel.lp")

    def __build_model_docplex(self):

        """

        Build the model using docplex API

        :return: None

        """

        if not self.__interface == 'docplex':
            raise NameError("For some reason the docplex model has been called when " +
                            "the 'interface' variable has been set to: " + self.__interface)

        self.__model = Model(name=self.__name)

        # Time measurement within the solver
        #self.__model.set_TimeMode('CPUTime')
        self.__model.set_time_limit(self.time_limit)

        # Add y(i,j) variables
        self.__y_start = self.__model.get_statistics().number_of_variables
        self.__model.binary_var_list(
            ((i+1, j+1)
             for i in range(self.__n_nodes)
             for j in range(self.__n_nodes)),
            name="y%s"
        )

        # Add f(i,j) variables
        self.__f_start = self.__model.get_statistics().number_of_variables
        self.__model.continuous_var_list(
            ((i+1,j+1)
             for i in range(self.__n_nodes)
             for j in range(self.__n_nodes)),
            name="f%s",
            ub=max([cable.capacity for cable in self.__cables]),
            lb=0
        )

        # Add x(i,j,k) variables
        self.__x_start = self.__model.get_statistics().number_of_variables
        self.__model.binary_var_list(
            ((i+1,j+1,k+1)
             for i in range(self.__n_nodes)
             for j in range(self.__n_nodes)
             for k in range(self.__n_cables)),
            name="x%s"
        )

        # Slack variables on parameter C
        if self.__substation_slack:
            self.__substation_slack_start = self.__model.get_statistics().number_of_variables
            self.__model.continuous_var_list(
                (h+1
                 for h, point in enumerate(self.__points)
                 if point.power < -0.5),
                name="s1(%s)",
                ub=self.__n_turbines,
                lb=0
            )
        else:
            # No slack variables, then.
            self.__substation_slack_start = -1

        # Slack variables on flux
        if self.__flux_slack:
            # Add slack variables for flux
            self.__flux_slack_start = self.__model.get_statistics().number_of_variables
            self.__model.continuous_var_list(
                (h+1 for h in range(self.__n_turbines)),
                name="s2(%s)",
                ub=max([cable.capacity for cable in self.__cables]),
                lb=0
            )
        else:
            # No variables should be added, then.
            self.__flux_slack_start = -1

        # No self-loops constraints on y(i,i) variables (\forall i)
        self.__model.add_constraints(
            self.__model.get_var_by_index(self.__ypos(i,i)) == 0
            for i in range(self.__n_nodes)
        )

        # No self-loops constraints on f(i,i) variables (\forall i)
        self.__model.add_constraints(
            self.__model.get_var_by_index(self.__fpos(i,i)) == 0
            for i in range(self.__n_nodes)
        )

        # No self-loops constraints on x(i,i, k) variables (\forall i, \forall k))
        self.__model.add_constraints(
            self.__model.get_var_by_index(self.__xpos(i,i,k)) == 0
            for i in range(self.__n_nodes)
            for k in range(self.__n_cables)
        )

        # Out-degree constraints (substations)
        self.__model.add_constraints(
            self.__model.sum(
                self.__model.get_var_by_index(self.__ypos(h,j))
                for j in range(self.__n_nodes)
            )
            ==
            0
            for h, point in enumerate(self.__points)
            if point.power < -0.5
        )

        # Out-degree constraints (turbines)
        self.__model.add_constraints(
            self.__model.sum(
                self.__model.get_var_by_index(self.__ypos(h,j))
                for j in range(self.__n_nodes)
            )
            ==
            1
            for h, point in enumerate(self.__points)
            if point.power > 0.5
        )

        # Maximum number of cables linked to a substation
        self.__model.add_constraints(
            self.__model.sum(
                self.__model.get_var_by_index(self.__ypos(i,h))
                for i in range(self.__n_nodes)
            )
            <=
            self.c + (self.__model.get_var_by_index(self.__substation_slackpos(h))
                      if self.__substation_slack else 0)
            for h,point in enumerate(self.__points)
            if point.power < -0.5
        )

        # Flow balancing constraint
        self.__model.add_constraints(
            self.__model.sum(
                self.__model.get_var_by_index(self.__fpos(h,j))
                for j in range(self.__n_nodes)
            )
            ==
            self.__model.sum(
                self.__model.get_var_by_index(self.__fpos(j,h))
                for j in range(self.__n_nodes)
            ) + self.__points[h].power
            - (self.__model.get_var_by_index(self.__flux_slackpos(h))
               if self.__flux_slack else 0)
            for h,point in enumerate(self.__points)
            if point.power > 0.5
        )

        # Avoid double cable between two points
        self.__model.add_constraints(
            self.__model.get_var_by_index(self.__ypos(i,j))
            ==
            self.__model.sum(
                self.__model.get_var_by_index(self.__xpos(i,j,k))
                for k in range(self.__n_cables)
            )
            for i in range(self.__n_nodes)
            for j in range(self.__n_nodes)
        )

        # Guarantee that the cable is enough for the connection
        self.__model.add_constraints(
            self.__model.sum(
                cable.capacity * self.__model.get_var_by_index(self.__xpos(i,j,k))
                for k,cable in enumerate(self.__cables)
            )
            >=
            self.__model.get_var_by_index(self.__fpos(i,j))
            for i in range(self.__n_nodes)
            for j in range(self.__n_nodes)
        )

        # No-crossing lazy constraints don't work in docplex.
        if not self.cross_mode == 'no':
            raise ValueError("No crossings when using docplex.")

        # Objective function
        self.__model.minimize(
            self.__model.sum(
                cable.price * WindFarm.get_distance(u, v) * self.__model.get_var_by_index(self.__xpos(i, j, k))
                for k, cable in enumerate(self.__cables)
                for i, u in enumerate(self.__points)
                for j, v in enumerate(self.__points)
            )
            + (self.__model.sum(
                1e9 * self.__model.get_var_by_index(self.__substation_slackpos(h))
                for h,point in enumerate(self.__points)
                if point.power < 0.5
            ) if self.__substation_slack else 0)
            + (self.__model.sum(
                1e9 * self.__model.get_var_by_index(self.__flux_slackpos(h))
                for h in range(self.__n_turbines)
            ) if self.__flux_slack else 0)
        )

        # Adding the parameters to the model
        self.__model.parameters.mip.strategy.rinsheur.set(self.rins)
        if self.polishtime < self.time_limit:
            self.__model.parameters.mip.polishafter.time.set(self.polishtime)
        self.__model.parameters.timelimit.set(self.time_limit)

        # Writing the model to a proper location
        self.__model.export_as_lp(path=self.__project_path+"/out/" + self.out_dir_name + "/lpmodel.lp")

    def __get_solution(self, var='x'):

        """

        Reads the solution from CPLEX or DOCPLEX and stores it in three appropriate lists.
        (recall that a selected solution is a variable set to '1').

            - If var is set to 'x', a list of "CableSol" named tuples will be returned.
            - If var is set to 'y', a list of "EdgeSol" named tuples will be returned.
            - If var is set to 'f', a list of "FluxSol" named tuples will be returned.

        WARNING:
            If you call this function BEFORE CPLEX or DOCPLEX find an incumbent,
            or just after adding a constraint to the model (see loop method),
            an error will be raised.

        :param var: Default = 'x'. May be set to 'x', 'y' or 'f'.
        :return: The corresponding solution list.

        """

        if self.__interface == 'cplex':
            get_value = self.__model.solution.get_values
            xpos = self.__xpos
            ypos = self.__ypos
            fpos = self.__fpos
        elif self.__interface == 'docplex':
            get_value = self.__model.solution.get_value
            xpos = lambda i, j, k:  self.__model.get_var_by_index(self.__xpos(i, j, k))
            ypos = lambda i, j:  self.__model.get_var_by_index(self.__ypos(i, j))
            fpos = lambda i, j:  self.__model.get_var_by_index(self.__fpos(i, j))
        else:
            raise ValueError("Unknown interface. I've got: " + str(self.__interface))

        if var == 'x':
            sol = [self.__CableSol(self.__xpos(i, j, k), i, j, k)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)
                   for k in range(self.__n_cables)
                   if get_value(xpos(i, j, k)) > 0.5]
        elif var == 'y':
            sol = [self.__EdgeSol(self.__ypos(i, j), i, j)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)
                   if get_value(ypos(i, j)) > 0.5]
        elif var == 'f':
            sol = [self.__FluxSol(self.__fpos(i, j), i, j)
                   for i in range(self.__n_nodes)
                   for j in range(self.__n_nodes)
                   if get_value(fpos(i, j)) > 0.5]
        else:
            raise ValueError("Invalid solution request. 'x', 'y' or 'f' are possible values, given " + str(var))

        return sol

    def __build_input_files(self):

        """

        py:function:: __build_input_files(self)

        Sets the input file correctly, based on the dataset selection

        """

        if not type(self.data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(self.data_select))
        if self.data_select <= 0 or self.data_select >= 32:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-31]. Given: " + str(self.data_select))

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

    def __build_custom_parameters(self):

        """

        py:function:: __build_custom_parameters(self)

        Set the name and some constant parameters of the wind farm correctly,
        based on the dataset selection

        """

        if not type(self.data_select) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(self.data_select))
        if self.data_select <= 0 or self.data_select >= 32:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-31]. Given: " + str(self.data_select))

        # We assume that, in this context, we'll never have a WF >=10
        wf_number = 0
        if 0 <= self.data_select <= 6:
            wf_number = 1
            self.c = 7
        elif 7 <= self.data_select <= 15:
            wf_number = 2
            self.c = 100
        elif 16 <= self.data_select <= 19:
            wf_number = 3
            self.c = 4
        elif 20 <= self.data_select <= 21:
            wf_number = 4
            self.c = 10
        elif 26 <= self.data_select <= 29:
            wf_number = 5
            self.c = 10
        elif 30 <= self.data_select <= 31:
            wf_number = 6
            self.c = 12

        if wf_number == 0:
            raise ValueError("Something went wrong with the Wind Farm number;\n" +
                             "check the dataset selection parameter: " + str(self.data_select))

        self.__name = "Wind Farm 0" + str(wf_number)

    def __plot_high_quality(self, edges):

        """

        py:function:: plot_high_quality(inst, edges)

        Plot the solution using standard libraries

        :param export: whatever this means

        """

        G = nx.DiGraph()

        mapping = {}

        for i in range(self.__n_nodes):
            if self.__points[i].power < -0.5:
                mapping[i] = 'S{0}'.format(i + 1)
            else:
                mapping[i] = 'T{0}'.format(i + 1)

        for index, node in enumerate(self.__points):
            G.add_node(index)

        available_colours = ['black', 'red', 'gray', 'sienna', 'purple', 'navy', 'darkcyan', 'darkkhaki',
                             'coral', 'chocolate', 'salmon', 'olive', 'fuchsia', 'blue', 'seagreen', 'pink']
        for edge in edges:
            G.add_edge(edge.s, edge.d, color=available_colours[edge.capacity])

        pos = {i: (point.x, point.y) for i, point in enumerate(self.__points)}

        # Avoid re scaling of axes
        plt.gca().set_aspect('equal', adjustable='box')

        # draw graph
        net_edges = G.edges()
        colors = [G[u][v]['color'] for u, v in net_edges]

        hfont = {'fontname':'sans-serif'}
        plt.title(self.__name + " (" + str(self.__model.solution.get_objective_value()) + ")", **hfont, fontsize=16)

        nx.draw(G, pos, with_labels=True, node_size=1300, alpha=0.3, arrows=True, labels=mapping, node_color='g',
                linewidth=20, edge_color=colors)
        plt.show()

    def __get_violated_edges(self, selected_edges):

        """

        When called, this function returns a list of violations, which are a list of y_pos indexes,
        ready to be added to CPLEX or DOCPLEX.

        :param selected_edges:
        :return:    [list_1, ..., list2_m], where each list_i is a list:
                    [idx_1, ..., idx_m], where idx_i = y_pos(some_turb, some_other_turb)

        """

        constraints_to_be_added = []

        for ab in selected_edges:
            edges_violating_ab = [e2 for e2 in selected_edges
                                  # Filter out anything that goes to/comes from a and b.
                                  if not (e2.s == ab.s or e2.d == ab.s or e2.s == ab.d or e2.d == ab.d)
                                  # Extract the violated edges only.
                                  and WindFarm.are_crossing(self.__points[ab.s],
                                                            self.__points[ab.d],
                                                            self.__points[e2.s],
                                                            self.__points[e2.d])]

            if len(edges_violating_ab) > 0:
                for violating_edge in edges_violating_ab:
                    constraints_to_be_added.append([
                        violating_edge,
                        self.__EdgeSol(self.__ypos(ab.s, ab.d), ab.s, ab.d),
                        self.__EdgeSol(self.__ypos(ab.d, ab.s), ab.d, ab.s)
                    ])

        return constraints_to_be_added

    def __reduce_metaheuristic_iteration(self, edge_ab):

        """

        :param edge_ab: The edge selected by the metaheuristic algorithm
        :return: List of EdgeSol => all the violations through edge_ab

        """

        full_graph = [self.__EdgeSol(self.__ypos(i, j), i, j)
                      for i in range(self.__n_nodes)
                      for j in range(self.__n_nodes)]

        filtered_edges = [edge_cd for edge_cd in full_graph
                          if not (edge_cd.s == edge_ab.s or edge_cd.d == edge_ab.s or edge_cd.s == edge_ab.d or edge_cd.d == edge_ab.d)]

        violations = [edge_cd for edge_cd in filtered_edges
                      if WindFarm.are_crossing(self.__points[edge_ab.s],
                                               self.__points[edge_ab.d],
                                               self.__points[edge_cd.s],
                                               self.__points[edge_cd.d])]

        return violations

    def __add_violating_constraint(self, crossings):

        """

        Adds the violating constraint by index.
        Crossings contains the index of every edge variable to be added.

        :param crossings: a list [EdgeSol_1, ..., EdgeSol_n],
        where EdgeSol_i are the conflicting edges, containing their idx and names
        :return: None

        """

        if self.cross_mode == 'lazy':
            constraint_add = self.__model.linear_constraints.advanced.add_lazy_constraints
        else:
            constraint_add = self.__model.linear_constraints.add

        if len(crossings) > 0:
            crossings = [cross.idx for cross in crossings]
            coefficients = [1] * len(crossings)
            constraint_add(
                lin_expr=[cplex.SparsePair(
                    ind=crossings,
                    val=coefficients
                )],
                senses=["L"],
                rhs=[1]
            )

    # Public methods
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
        parser.add_argument('--interface', choices=['docplex', 'cplex'],
                            help='Choose the interface ')
        parser.add_argument('--rins', type=int,
                            help='the frequency with which the RINS Heuristic will be applied')
        parser.add_argument('--timeout', type=int,
                            help='timeout in which the optimizer will stop iterating')
        parser.add_argument('--polishtime', type=int,
                            help='the time to wait before applying polishing')
        parser.add_argument('--outfolder', type=str,
                            help='name of the folder to be created inside the /out' +
                                                          ' directory, which contains everything related to this run')
        parser.add_argument('--nosubstationslack', action="store_true",
                            help='type --noslack if you do not want the soft version of the problem')
        parser.add_argument('--nofluxslack', action="store_true",
                            help='type --noslack if you do not want the slack on the flux')
        parser.add_argument('--crossings', choices=['no', 'lazy', 'loop', 'callback', 'normal'],
                            help='Choose how you want to address the crossing problem')
        parser.add_argument('--metaheuristic', choices=['hard', 'soft'],
                            help='Choose one metaheuristic method to wrap the execution')
        parser.add_argument('--overall_wait_time', type=int,
                            help='Used in hard/soft fixing and loop method to stop the execution')

        args = parser.parse_args()

        if args.outfolder:
            self.out_dir_name = args.outfolder

        if args.dataset:
            self.data_select = args.dataset

        if args.cluster:
            self.__cluster = True

        if args.interface == 'docplex' or args.interface == 'cplex':
            self.__interface = args.interface
        else:
            warnings.warn("Invalid interface; '" + str(args.interface) + "' given. "
                          + "Using the default value: " + self.__interface,
                          ParseWarning)
            self.__interface = 'cplex'

        if args.rins:
            if args.rins <= -2:
                warnings.warn("RINS parameter given is not valid. Given: " + str(args.rins) +
                              ". Using the default value: " + str(self.__rins))
            else:
                self.__rins = args.rins

        if args.timeout:
            self.time_limit = args.timeout

        if args.polishtime:
            self.polishtime = args.polishtime

        if args.nosubstationslack:
            self.__substation_slack = False

        if args.nofluxslack:
            self.__flux_slack = False

        if args.crossings:
            self.cross_mode = args.crossings

        if args.metaheuristic:
            self.__metaheuristic = args.metaheuristic

        if args.overall_wait_time:
            self.__overall_wait_time = args.overall_wait_time
            if self.__overall_wait_time<self.time_limit:
                raise ValueError("Can't set the overall wait time under the single iteration time.")

        self.__build_input_files()
        self.__build_custom_parameters()

    def build_model(self):

        """

        Builds the model with the right interface.
        Raise an error if the specified interface doesn't exist.

        """

        if self.__interface == 'cplex':
            self.__build_model_cplex()

        elif self.__interface == 'docplex':
            self.__build_model_docplex()

        else:
            raise ValueError("The given interface isn't a valid option." +
                             "Either 'docplex' or 'cplex' are valid options; given: " +
                             str(self.__interface))

    def __exact_solve(self):

        """

        Simply solves the problem by invoking the .solve() method within the model selected.

        """

        if self.cross_mode == 'callback':
            # CPLEX disables parallelism, force it to the maximum number of cores available
            self.__model.parameters.threads.set(multiprocessing.cpu_count())

            # Install the callback
            lazycb = self.__model.register_callback(LazyCallback)

            # Give some arguments and functions to the callback
            lazycb.n_nodes = self.__n_nodes
            lazycb.get_violated_edges = self.__get_violated_edges
            lazycb.ypos = self.__ypos
            lazycb.EdgeSol = self.__EdgeSol
            lazycb.sum_time = 0
            lazycb.n_cables = self.__n_cables
            lazycb.model = self.__model

            self.__model.solve()

        elif self.cross_mode == 'lazy' or self.cross_mode == 'no' or self.cross_mode == 'normal':
            # This simply means that CPLEX has everything he needs to have inside his enviroment
            self.__model.solve()

        elif self.cross_mode == 'loop':
            self.__model.parameters.advance.set(1)
            xs = True       # "are there any crosses in this solution?"
            opt = False     # "has the optimum been reached?"

            for i in range(3):
                self.__model.solve()

            while (xs or not opt) and time.clock() - self.__starting_time < self.__overall_wait_time * self.__model.parameters.threads.get():
                self.__model.solve()
                self.plot_solution(edges=self.__get_solution(var='x'), high=False)

                violations = self.__get_violated_edges(self.__get_solution(var='y'))

                if len(violations) > 0:
                    xs = True
                    for violation in violations:
                        self.__add_violating_constraint(violation)
                else:
                    xs = False
                    self.__best_sol = self.__get_solution()
                    self.__best_incumbent = self.__model.solution.get_objective_value()

                if self.__model.solution.get_status() == self.__model.solution.status.MIP_optimal:
                    opt = True

            self.__model.parameters.advance.set(0)
        else:
            raise ValueError("Unrecognized cross-strategy; given: " + str(self.cross_mode))

    def __hard_fix(self):

        """

        Solve the problem using the hard fixing math-heuristic

        :return: None

        """

        optimum = False     # "has the optimum been reached?"
        probability = 0.9
        self.__model.parameters.advance.set(1)
        starting_gap = 1
        while starting_gap >= 0.70:
            self.__exact_solve()
            starting_gap = self.__model.solution.MIP.get_mip_relative_gap()

        starting_best_bound = starting_gap * self.__model.solution.get_objective_value()

        while not optimum and time.clock() - self.__starting_time < self.__overall_wait_time * self.__model.parameters.threads.get():

            sol = self.__get_solution(var='y')
            self.plot_solution(self.__get_solution(var='x'))

            for edge in sol:
                if random.random() > probability:

                    self.__model.variables.set_lower_bounds(
                        edge.idx,
                        1
                    )

                    violations = self.__reduce_metaheuristic_iteration(edge)

                    for violation in violations:
                        self.__model.variables.set_upper_bounds(
                            violation.idx,
                            0
                        )

            self.__exact_solve()

            if self.__model.solution.get_status() == self.__model.solution.status.MIP_optimal:
                optimum = True

            if self.__best_incumbent > self.__model.solution.get_objective_value():
                self.__best_sol = self.__get_solution()
                self.__best_incumbent = self.__model.solution.get_objective_value()

            probability -= 0.05
            probability = max(probability, 0.5)

        self.__true_gap = starting_best_bound / self.__best_incumbent
        print("True gap: ", self.__true_gap)

    def __soft_fix(self):

        """

        Solve the problem using the soft fixing math-heuristic

        :return: None

        """

        optimum = False     # "has the optimum been reached?"
        k = 0.90
        self.__model.parameters.advance.set(1)
        starting_gap = 1
        while starting_gap >= 0.70:  # We do this to AT LEAST have a feasible solution (with no slacks==1)
            self.__exact_solve()
            starting_gap = self.__model.solution.MIP.get_mip_relative_gap()

        starting_best_bound = starting_gap * self.__model.solution.get_objective_value()
        while not optimum and time.clock() - self.__starting_time < self.__overall_wait_time * self.__model.parameters.threads.get():

            selected_edges = self.__get_solution(var='y')
            self.plot_solution(self.__get_solution(var='x'))

            self.__model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[edge.idx for edge in selected_edges],
                    val=[1] * len(selected_edges)
                )],
                senses=["G"],
                rhs=[self.__n_nodes*(1-k)-1]
            )

            self.__exact_solve()

            if self.__model.solution.get_status() == self.__model.solution.status.MIP_optimal:
                optimum = True

            if self.__best_incumbent > self.__model.solution.get_objective_value():
                self.__best_sol = self.__get_solution()
                self.__best_incumbent = self.__model.solution.get_objective_value()

            k -= 0.10
            k = max(k, 0.1)

        self.__true_gap = starting_best_bound / self.__best_incumbent
        print("True gap: ", self.__true_gap)
        self.plot_solution(self.__best_sol)

    def solve(self):

        """

        Call the right method to solve the problem

        :return: None

        """

        print("Rins = ", self.__model.parameters.mip.strategy.rinsheur.get())
        self.__model.parameters.threads.set(multiprocessing.cpu_count())
        start_time = time.time()
        if self.__cluster:
            self.__model.parameters.randomseed = random.randint(0, sys.maxsize)
            self.__model.parameters.advance.set(0)
            print("Advanced model:", self.__model.parameters.advance.get())
            print("Dataset:", self.__data_select)

        self.__starting_time = time.clock()
        if self.__metaheuristic is None:
            self.__exact_solve()
        elif self.__metaheuristic == 'hard':
            self.__hard_fix()
        elif self.__metaheuristic == 'soft':
            self.__soft_fix()
        else:
            raise ValueError("Unrecognized heuristic technique; given: " + str(self.__metaheuristic))

        print("Elapsed time CPU: ", time.clock() - self.__starting_time)
        print("Elpased true time: ", time.time() - start_time)

    def release(self):

        """

        Release all the information about the model

        :return: None

        """

        self.__model.end()

    def write_results(self, file_name='results.csv'):

        """
        Write the file to be used for perfromance profiling script

        :param file_name: Output file name
        :return: None

        """

        with open(self.__project_path + "/out/" + self.out_dir_name + "/" + file_name, 'r') as fp:
            file_content = fp.readlines()

            num_columns = len(file_content[0].split(sep=','))

        with open(self.__project_path + "/out/" + self.out_dir_name + "/" + file_name, 'a') as fp:

            num_tokens_last_line = len(file_content[-1].split(sep=','))

            if num_tokens_last_line == num_columns:  # Write down the instance name
                fp.write("\ndata" + str(self.__data_select))

            fp.write("," + str(self.__model.solution.get_objective_value()))

    def write_solutions(self):

        """

        Writes the solutions obtained by first invoking the built-in function of the model,
        and then by returning our private __get_solution() method, which returns the list of the
        x(i,j,k) variables set to one.

        :return: the list of x(i,j,k) variables set to one from the solution

        """

        self.__model.solution.write(self.__project_path + "/out/" + self.out_dir_name + "/mysol.sol", 'r')
        return self.__get_solution()

    def read_input(self):

        """

        This function reads the input files by invoking the private methods which read
        both the turbines and the cables files.

        :return: None

        """

        self.__read_turbines_file()
        self.__read_cables_file()

    def plot_solution(self, edges=None, high=False):

        """

        py:function:: plot_solution(inst, edges)
        Plots the solution using the plot.ly library

        :param show: if =True, the exported plot will be shown right away.
        :param edges: list of edges to be plotted
        :param high: if =True, an high-quality img will be plotted, also

        :return: None

        """

        if edges is None:
            edges = self.__get_solution(var='x')

        G = nx.DiGraph()

        for index, node in enumerate(self.__points):
            G.add_node(index, pos=(node.x, node.y))

        for edge in edges:
            G.add_edge(edge.s, edge.d, weight=edge.capacity)

        edge_trace = Scatter(
            x=[],
            y=[],
            line=Line(width=0.5, color='#888'),
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
            self.__plot_high_quality(edges=self.__get_solution(var='x'))


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

    @staticmethod
    def are_crossing(pt1, pt2, pt3, pt4):

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

        if 1e-3 < lambd < 1-1e-3 and 1e-3 < mu < 1-1e-3:
            return True
        else:
            return False

    # Get and set methods, in the Pythonic way

    @property
    def cross_mode(self):
        return self.__cross_mode

    @cross_mode.setter
    def cross_mode(self, cm):
        if not (
                cm == 'no'
                or
                cm == 'lazy'
                or
                cm == 'loop'
                or
                cm == 'callback'
                or
                cm == 'normal'
        ):
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
    def data_select(self):
        return self.__data_select

    @data_select.setter
    def data_select(self, d):
        if not type(d) == int:
            raise TypeError("Expecting an integer value representing the dataset. Given: " + str(d))
        if d <= 0 or d >= 32:
            raise ValueError("The dataset you're trying to reach is out of range.\n" +
                             "Range: [1-31]. Given: " + str(d))
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
        return self.__n_cables

    @num_cables.setter
    def num_cables(self, nc):
        if not type(nc) == int:
            raise TypeError("The number of cables must be a positive integer number; given:" + str(nc))
        if not nc >= 0:
            raise AttributeError("The number of cables must be a positive integer number; given:" + str(nc))
        self.__n_cables = nc

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
