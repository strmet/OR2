from docplex.mp.model import Model
from collections import namedtuple
import math
import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx
import matplotlib.pyplot as plt
import cplex
import argparse

# Named tuples describing input data
Edge = namedtuple("Edge", ["source", "destination"])
Point = namedtuple("Point", ["x", "y", "power"])
Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])

# Named tuple to store result in the graph form
CableSol = namedtuple("CableSol", ["source", "destination", "capacity"])


def main():

    inst = Instance()

    inst.turb_file = 'wf01/wf01.turb'
    inst.cbl_file = 'wf01/wf01_cb01.cbl'
    inst.name = 'Wind Farm wf01'

    parse_command_line(inst)

    read_turbines_file(inst)
    read_cables_file(inst)
    print(inst.C)
    print("Solving...")
    if inst.interface == 'cplex':
        model = build_model_classical_cplex(inst)
    else:
        model = build_model_docplex(inst)
    model.solve()

    sol = get_solution(inst, model)

    #plot_solution(inst, sol)

    #plot_high_quality(inst, sol, export=True)


def get_solution(inst, model):
    sol = []
    edges = [Edge(i, j) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]
    if inst.interface == 'cplex':
        for edge in edges:
            for k in range(inst.num_cables):
                val = model.solution.get_values("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))
                if val > 0.5:
                    sol.append(CableSol(edge.source + 1, edge.destination + 1, k + 1))
    else:
        for edge in edges:
            for k in range(inst.num_cables):
                val = model.solution.get_value("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))
                if val > 0.5:
                    sol.append(CableSol(edge.source + 1, edge.destination + 1, k + 1))

    return sol


def parse_command_line(inst):

    parser = argparse.ArgumentParser(description='Process details about instance and interface.')

    parser.add_argument('--cf', type=str, nargs=1,
                        help='cable file path, starting from folder \'data\'')
    parser.add_argument('--tf', type=str, nargs=1,
                        help='substation and turbine file path, starting from folder \'data\'')
    parser.add_argument('--cluster', action="store_true", help='type --cluster if you want to use the cluster')
    parser.add_argument('--interface', choices=['docplex', 'cplex'],  help='Choose interface ')
    parser.add_argument('--C', type=int,  help='the maximum number of cables linked to a substation')

    args = parser.parse_args()

    if args.cf and args.tf:
        inst.cbl_file = args.cf[0]
        inst.turb_file = args.tf[0]
    elif (args.cf is None and args.tf is not None) or (args.cf is not None and args.tf is None):
        raise NameError("Both --cf and --tf must be set")

    if args.cluster:
        inst.cluster = True

    if args.interface == 'docplex':
        inst.interface = 'docplex'
    else:
        inst.interface = 'cplex'

    if args.C is not None:
        inst.C = args.C


def build_model_classical_cplex(inst):
    model = cplex.Cplex()

    model.set_problem_name(inst.name)
    model.objective.set_sense(model.objective.sense.minimize)

    edges = [Edge(i, j) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]

    # Add y(i,j) variables
    inst.y_start = model.variables.get_num()
    name_y_edges = ["y({0},{1})".format(i + 1, j + 1) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]

    for index, edge in enumerate(name_y_edges):
        model.variables.add(
            types=[model.variables.type.binary],
            names=[name_y_edges[index]],
                            )
        if inst.debug_mode:
            if ypos(index, inst) != model.variables.get_num() - 1:
                raise NameError('Number of variables and index do not match')

    # Add f(i,j) variables
    inst.f_start = model.variables.get_num()
    name_f_edges = ["f({0},{1})".format(i + 1, j + 1) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]

    for index, edge in enumerate(name_f_edges):
        model.variables.add(
            types=[model.variables.type.continuous],
            names=[name_f_edges[index]]
                            )
        if inst.debug_mode:
            if fpos(index, inst) != model.variables.get_num() - 1:
                raise NameError('Number of variables and index do not match')

    # Add x(i,j,k) variables
    inst.x_start = model.variables.get_num()

    for index, edge in enumerate(edges):
        for k in range(inst.num_cables):
            model.variables.add(
                types=[model.variables.type.binary],
                names=["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)],
                obj=[inst.cables[k].price * get_distance(inst.points[edge.source], inst.points[edge.destination])]
            )
            if inst.debug_mode:
                if xpos(index, k, inst) != model.variables.get_num() - 1:
                    raise NameError('Number of variables and index do not match')

    # No self-loop constraint
    for i in range(inst. n_nodes):
        model.variables.set_upper_bounds([("y({0},{1})".format(i + 1, i + 1), 0)])

    for i in range(inst. n_nodes):
        model.variables.set_upper_bounds([("f({0},{1})".format(i + 1, i + 1), 0)])


    # Out-degree constraints
    for h in range(len(inst.points)):
        if inst.points[h].power < -0.5:
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                                ind=["y({0},{1})".format(h + 1, j + 1) for j in range(inst.n_nodes)],
                                val=[1.0] * inst.n_nodes
                )],
                senses=["E"],
                rhs=[0]
            )
        else:
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                                ind=["y({0},{1})".format(h + 1, j + 1) for j in range(inst.n_nodes)],
                                val=[1.0] * inst.n_nodes
                )],
                senses=["E"],
                rhs=[1]
            )

    # Flow balancing constraint
    for h in range(len(inst.points)):
        if inst.points[h].power > - 0.5:
            sum = ["f({0},{1})".format(h + 1, j + 1) for j in range(inst.n_nodes)if h != j] \
                  + \
                  ["f({0},{1})".format(j + 1, h + 1) for j in range(inst.n_nodes) if h != j]
            coefficients = [1] * (inst.n_nodes - 1) + [-1] * (inst.n_nodes - 1)
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                            ind=sum,
                            val=coefficients,
                )],
                senses=["E"],
                rhs=[inst.points[h].power]
            )

    # Avoid double cable between two points
    for edge in edges:
        sum = ["y({0},{1})".format(edge.source + 1, edge.destination + 1)] \
               + \
              ["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1) for k in range(inst.num_cables)]
        coefficients = [1] + [-1] * (inst.num_cables)
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                        ind=sum,
                        val=coefficients,
            )],
            senses=["E"],
            rhs=[0]
        )

    # Guarantee that the cable is enough for the connection
    for edge in edges:
        sum = ["f({0},{1})".format(edge.source + 1, edge.destination + 1)] \
               + \
              ["x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1) for k in range(inst.num_cables)]
        coefficients = [-1] + [inst.cables[k].capacity for k in range(inst.num_cables)]
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                        ind=sum,
                        val=coefficients,
            )],
            senses=["G"],
            rhs=[0]
        )

    return model


def build_model_docplex(inst):
    edges = [Edge(i, j) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]

    model = Model(name=inst.name)
    model.set_time_limit(inst.time_limit)

    # Add y_i.j variables
    inst.y_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        model.binary_var(name="y({0},{1})".format(edge.source + 1, edge.destination + 1))
        if inst.debug_mode:
            if ypos(index, inst) != model.get_statistics().number_of_variables - 1:
                raise NameError('Number of variables and index do not match')

    # Add f_i.j variables
    inst.f_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        model.continuous_var(name="f({0},{1})".format(edge.source + 1, edge.destination + 1))
        if inst.debug_mode:
            if fpos(index, inst) != model.get_statistics().number_of_variables - 1:
                raise NameError('Number of variables and index do not match')

    # Add x_i.j.k variables
    inst.x_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        for k in range(inst.num_cables):
            model.binary_var(name="x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1))

        if inst.debug_mode:
            if xpos(index, k, inst) != model.get_statistics().number_of_variables - 1:
                raise NameError('Number of variables and index do not match')

    # No self-loops constraints on y_i.i variables
    for i in range(inst. n_nodes):
        var = model.get_var_by_name("y({0},{1})".format(i + 1, i + 1))
        model.add_constraint(var == 0)

    # No self-loops constraints on f_i.i variables
    for i in range(inst.n_nodes):
        var = model.get_var_by_name("f({0},{1})".format(i + 1, i + 1))
        model.add_constraint(var == 0)

    # Out-degree constraints
    for h in range(len(inst.points)):
        if inst.points[h].power < -0.5:
            model.add_constraint(
                model.sum(model.get_var_by_name("y({0},{1})".format(h + 1, j + 1)) for j in range(inst.n_nodes))
                ==
                0
            )
        else:
            model.add_constraint(
                model.sum(model.get_var_by_name("y({0},{1})".format(h + 1, j + 1)) for j in range(inst.n_nodes))
                ==
                1
            )

    # Flow balancing constraint
    for h in range(len(inst.points)):
        if inst.points[h].power > - 0.5:
            model.add_constraint(
                model.sum(model.get_var_by_name("f({0},{1})".format(h + 1, j + 1)) for j in range(inst.n_nodes))
                ==
                model.sum(model.get_var_by_name("f({0},{1})".format(j + 1, h + 1)) for j in range(inst.n_nodes))
                + inst.points[h].power
            )

    # Avoid double cable between two points
    for edge in edges:
        model.add_constraint(
            model.get_var_by_name("y({0},{1})".format(edge.source + 1, edge.destination + 1))
            ==
            model.sum(model.get_var_by_name("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)) for k in range(inst.num_cables))
        )

    # Guarantee that the cable is enough for the connection
    for edge in edges:
        model.add_constraint(
            model.sum(
                model.get_var_by_name("x({0},{1},{2})".format(edge.source + 1, edge.destination + 1, k + 1)) * inst.cables[k].capacity
                for k in range(inst.num_cables)
            )
            >=
            model.get_var_by_name("f({0},{1})".format(edge.source + 1, edge.destination + 1))
        )

    # Objective function
    model.minimize(
        model.sum(
            inst.cables[k].price * get_distance(inst.points[i], inst.points[j]) * model.get_var_by_name("x({0},{1},{2})".format(i + 1, j + 1, k + 1))
            for k in range(inst.num_cables) for i in range(inst.n_nodes) for j in range(inst.n_nodes)
        )
    )

    return model

def read_turbines_file(inst):

    """
    py:function:: read_turbines_file(inst)

    Read the turbines file

    :param class inst: Instance of the problem

    """

    file = open("../data/" + inst.turb_file, "r")
    points = []

    for index, line in enumerate(file):
        if index >= 20: break
        words = list(map(int, line.split()))
        points.append(
            Point(words[0], words[1], words[2])
        )

    file.close()
    inst.n_nodes = len(points)
    inst.points = points


def read_cables_file(inst):

    """
    py:function:: read_cables_file(inst)
    Read the cables file

    :param class inst: Instance of the problem

    """

    file = open("../data/" + inst.cbl_file, "r")

    cables = []
    for index, line in enumerate(file):
        if index >= 10: break
        words = line.split()
        cables.append(
            Cable(int(words[0]), float(words[1]), int(words[2]))
        )

    file.close()
    inst.num_cables = len(cables)
    inst.cables = cables


def get_distance(point1, point2):

    """
    py:function:: get_distance(point1, point2)
    Get the distance between two poins

    :param point1 int: First point
    :param point1 int: Second point

    """

    return math.sqrt(
        (point1.x - point2.x)**2
        +
        (point1.y - point2.y)**2
    )


def plot_solution(inst, edges):

    """
    py:function:: plot_solution(inst, edges)

    Plot the solution using the plot.ly library

    :param inst instance: Instance of the problem
    :param edges: List of edges given back by CPLEX
    :type edges: List of CableSol

    """

    G = nx.DiGraph()

    for index, node in enumerate(inst.points):
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
        text=["Substation #{0}".format(i + 1) if inst.points[i].power < -0.5 else "Turbine #{0}".format(i + 1) for i in range(inst.n_nodes)],
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
                        title='<br><b style="font-size:20px>'+inst.name+'</b>',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(scaleanchor="x", scaleratio=1,showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

    py.plot(fig, filename='wind_farm.html')


def plot_high_quality(inst, edges, export=False):

    """
    py:function:: plot_high_quality(inst, edges)

    Plot the solution using standard libraries

    :param inst: First point
    :param edges: List of edges given back by CPLEX
    :type edges: List of CableSol

    """

    G = nx.DiGraph()

    mapping = {}

    for i in range(inst.n_nodes):
        if (inst.points[i].power < -0.5):
            mapping[i] = 'S{0}'.format(i + 1)
        else:
            mapping[i] = 'T{0}'.format(i + 1)

    for index, node in enumerate(inst.points):
        G.add_node(index)

    for edge in edges:
        G.add_edge(edge.source - 1, edge.destination - 1)

    pos = {i: (point.x, point.y) for i, point in enumerate(inst.points)}

    # Avoid re scaling of axes
    plt.gca().set_aspect('equal', adjustable='box')

    # draw graph
    nx.draw(G, pos, with_labels=True, node_size=1300, alpha=0.3, arrows=True, labels=mapping, node_color='g', linewidth=10)

    if (export == True):
        plt.savefig('../imgs/foo.svg')


    # show graph
    plt.show()

def ypos(offset, inst):

    """
    py:function:: ypos(offset, inst)

    Plot the solution using standard libraries

    :param offset int: Offset w.r.t ystart and the edge indexed by (i, j)
    :param inst: Instance of the problem

    """

    return inst.y_start + offset


def fpos(offset, inst):

    """
    py:function:: fpos(offset, inst)

    Plot the solution using standard libraries

    :param offset int: Offset w.r.t fstart and the edge indexed by (i, j)
    :param inst: Instance of the problem

    """
    return inst.f_start + offset


def xpos(offset, k, inst):

    """
    py:function:: xpos(offset, k, inst)

    Plot the solution using standard libraries

    :param offset int: Offset w.r.t xstart and the edge indexed by (i, j)
    :param k int: index of the cable considered
    :param inst: Instance of the problem

    """
    return inst.x_start + offset * inst.num_cables + k


class Instance:

    """
    py:class:: instance()

    This class stores all the useful information about input data and parameters

    """
    ## Model
    name = ''
    y_start = 0
    f_start = 0
    x_start = 0

    ## Input data
    cbl_file = ''
    turb_file = ''
    n_nodes = 0
    num_cables = 0
    points = []
    cables = []
    C = 10

    ## Parameters
    #model_type
    #num_threads
    cluster = False
    time_limit = 3600
    debug_mode = False
    interface = 'cplex'
    # available_memory
    # max_nodes
    # cutoff
    # integer_costs


if __name__ == "__main__":
    main()
