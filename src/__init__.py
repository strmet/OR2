from docplex.mp.model import Model
from collections import namedtuple
import math
import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx
import matplotlib.pyplot as plt


Edge = namedtuple("Edge", ["source", "destination"])
Point = namedtuple("Point", ["x", "y", "power"])
Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])

CableSol = namedtuple("CableSol", ["source", "destination", "capacity"])

def main():

    inst = instance()

    inst.turb_file = 'wf01/wf01.turb'
    inst.cbl_file = 'wf01/wf01_cb01.cbl'
    inst.name = 'Wind Farm wf01'

    inst = read_turbines_file(inst)
    inst = read_cables_file(inst)

    edges = [Edge(i, j) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]

    model = Model(name=inst.name)
    model.set_time_limit(inst.time_limit)

    # Add y_i.j variables
    inst.y_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        model.binary_var(name="y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        if inst.debug_mode:
            if ypos(index, inst) != model.get_statistics().number_of_variables - 1:
                raise NameError('Number of variables and index do not match')

    # Add f_i.j variables
    inst.f_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        model.continuous_var(name="f_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        if inst.debug_mode:
            if fpos(index, inst) != model.get_statistics().number_of_variables - 1:
                raise NameError('Number of variables and index do not match')

    # Add x_i.j.k variables
    inst.x_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        for k in range(inst.num_cables):
            model.binary_var(name="x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1))

        if inst.debug_mode:
            if xpos(index, k, inst) != model.get_statistics().number_of_variables - 1:
                raise NameError('Number of variables and index do not match')

    # No self-loops constraints
    for i in range(inst. n_nodes):
        var = model.get_var_by_name("y_{0}.{1}".format(i + 1, i + 1))
        model.add_constraint(var == 0)

    for i in range(inst.n_nodes):
        var = model.get_var_by_name("f_{0}.{1}".format(i + 1, i + 1))
        model.add_constraint(var == 0)

    # Out- degree constraints
    for h in range(len(inst.points)):
        if inst.points[h].power < -0.5:
            model.add_constraint(
                model.sum(model.get_var_by_name("y_{0}.{1}".format(h + 1, j + 1)) for j in range(inst.n_nodes))
                ==
                0
            )
        else:
            model.add_constraint(
                model.sum(model.get_var_by_name("y_{0}.{1}".format(h + 1, j + 1)) for j in range(inst.n_nodes))
                ==
                1
            )

    # Flow balancing constraint
    for h in range(len(inst.points)):
        if inst.points[h].power > - 0.5:
            model.add_constraint(
                model.sum(model.get_var_by_name("f_{0}.{1}".format(h + 1, j + 1)) for j in range(inst.n_nodes))
                ==
                model.sum(model.get_var_by_name("f_{0}.{1}".format(j + 1, h + 1)) for j in range(inst.n_nodes))
                + inst.points[h].power
            )

    # Avoid double cable between two points
    for edge in edges:
        model.add_constraint(
            model.get_var_by_name("y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
            ==
            model.sum(model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) for k in range(inst.num_cables))
        )

    # Guarantee that the cable is enough for the connection
    for edge in edges:
        model.add_constraint(
            model.sum(
                model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) * inst.cables[k].capacity
                for k in range(inst.num_cables)
            )
            >=
            model.get_var_by_name("f_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        )

    # Objective function
    model.minimize(
        model.sum(
            inst.cables[k].price * get_distance(inst.points[i], inst.points[j]) * model.get_var_by_name("x_{0}.{1}.{2}".format(i + 1, j + 1, k + 1))
            for k in range(inst.num_cables) for i in range(inst.n_nodes) for j in range(inst.n_nodes)
        )
    )

    model.get_objective_expr()
    model.export_as_lp("model.lp")
    print("Solving...")
    model.solve()

    sol = []
    for edge in edges:
        for k in range(inst.num_cables):
            val = model.solution.get_value("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1))
            if val > 0.5:
                sol.append(CableSol(edge.source + 1, edge.destination + 1, k + 1))

    model.print_solution()

    plot_solution(inst, sol)

def read_turbines_file(inst):
    file = open("../data/" + inst.turb_file, "r")
    points = []

    for index, line in enumerate(file):
        if index > 20: break
        words = list(map(int, line.split()))
        points.append(
            Point(words[0], words[1], words[2])
        )

    file.close()
    inst.n_nodes = len(points)
    inst.points = points
    return inst


def read_cables_file(inst):
    file = open("../data/" + inst.cbl_file, "r")

    cables = []
    for index, line in enumerate(file):
        if index > 10: break
        words = line.split()
        cables.append(
            Cable(int(words[0]), float(words[1]), int(words[2]))
        )

    file.close()
    inst.num_cables = len(cables)
    inst.cables = cables
    return inst


def get_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x)**2
        +
        (point1.y - point2.y)**2
    )


def plot_solution(inst, edges):
    G = nx.Graph()

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
            showscale = False,
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
    fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br><b style="font-size:20px>'+inst.name+'</b>',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False))
             )

    py.plot(fig, filename='wind_farm.html')


def plot_easy(inst, edges):

    G = nx.Graph()

    for index, node in enumerate(inst.points):
        G.add_node(index, pos=(node.x, node.y))


    for edge in edges:
        G.add_edge(edge.source - 1, edge.destination - 1)

    # draw graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)

    # show graph
    plt.show()

def ypos(offset, inst):
    return inst.y_start + offset


def fpos(offset, inst):
    return inst.f_start + offset


def xpos(offset, k, inst):
    return inst.x_start + offset * inst.num_cables + k


class instance():

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

    ## Parameters
    #model_type
    #num_threads
    time_limit = 3600
    debug_mode = False
    # available_memory
    # max_nodes
    # cutoff
    # integer_costs


if __name__ == "__main__":
    main()
