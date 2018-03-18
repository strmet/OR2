from docplex.mp.model import Model, ModelStatistics
from collections import namedtuple
import math
import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx


Edge = namedtuple("Edge", ["source", "destination"])
Point = namedtuple("Point", ["x", "y", "power"])
Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])


def main():

    inst = instance()

    inst.turb_file = 'wf01/wf01.turb'
    inst.cbl_file = 'wf01/wf01_cb01.cbl'
    inst.name = 'Wind Farm wf01'

    points, inst = read_turbines_file(inst)
    cables, inst = read_cables_file(inst)

    n = inst.n_nodes
    num_cables = inst.num_cables

    edges = [Edge(i, j) for i in range(inst.n_nodes) for j in range(inst.n_nodes)]

    model = Model(name=inst.name)
    model.set_time_limit(inst.time_limit)

    # Add y_i.i variables
    inst.y_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        model.binary_var(name="y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        if ypos(index, inst) != model.get_statistics().number_of_variables - 1:
            raise NameError('Number of variables and index do not match')

    inst.f_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        model.continuous_var(name="f_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        if fpos(index, inst) != model.get_statistics().number_of_variables - 1:
            raise NameError('Number of variables and index do not match')

    inst.x_start = model.get_statistics().number_of_variables
    for index, edge in enumerate(edges):
        for k in range(inst.num_cables):
            model.binary_var(name="x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1))

        if xpos(index, k, inst) != model.get_statistics().number_of_variables - 1:
            raise NameError('Number of variables and index do not match')


    for i in range(inst.n_nodes):
        var = model.get_var_by_name("y_{0}.{1}".format(i + 1, i + 1))
        model.add_constraint(var == 0, ctname="y_{0}.{1}=0".format(i + 1, i + 1))

    for i in range(inst.n_nodes):
        var = model.get_var_by_name("f_{0}.{1}".format(i + 1, i + 1))
        model.add_constraint(var == 0, ctname="y_{0}.{1}=0".format(i + 1, i + 1))

    for h in range(len(points)):
        if points[h].power < -0.5:
            model.add_constraint(
                model.sum(model.get_var_by_name("y_{0}.{1}".format(h + 1, j + 1)) for j in range(n))
                ==
                0
            )
        else:
            model.add_constraint(
                model.sum(model.get_var_by_name("y_{0}.{1}".format(h + 1, j + 1)) for j in range(n))
                ==
                1
            )

    for h in range(len(points)):
        if points[h].power > 0.5:
            model.add_constraint(
                model.sum(model.get_var_by_name("f_{0}.{1}".format(h + 1, j + 1)) for j in range(n))
                ==
                model.sum(model.get_var_by_name("f_{0}.{1}".format(j + 1, h + 1)) for j in range(n))
                + points[h].power
            )

    for edge in edges:
        model.add_constraint(
            model.get_var_by_name("y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
            ==
            model.sum(model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) for k in range(num_cables))
        )

    for edge in edges:
        model.add_constraint(
            model.get_var_by_name("y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
            ==
            model.sum(model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) for k in range(num_cables))
        )

    for edge in edges:
        model.add_constraint(
            model.sum(
                model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) * cables[k].capacity
                for k in range(num_cables)
            )
            >=
            model.get_var_by_name("f_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        )

    # Objective function
    model.minimize(
        model.sum(
            cables[k].price * get_distance(points[i], points[j]) * model.get_var_by_name("x_{0}.{1}.{2}".format(i + 1, j + 1, k + 1))
            for k in range(0, num_cables) for i in range(0, n) for j in range(0, n)
        )
    )

    model.export_as_lp("mode.lp")
    #print("Solving...")
    #model.solve()

    #model.print_solution()
    #plot_solution(points, inst)

def read_turbines_file(inst):
    file = open("../data/" + inst.turb_file, "r")
    points = []

    for index, line in enumerate(file):
        #if index > 20: break
        words = list(map(int, line.split()))
        points.append(
            Point(words[0], words[1], words[2])
        )

    file.close()
    inst.n_nodes = len(points)
    return points, inst


def read_cables_file(inst):
    file = open("../data/" + inst.cbl_file, "r")

    cables = []
    for index, line in enumerate(file):
        if index > 3: break
        words = line.split()
        cables.append(
            Cable(int(words[0]), float(words[1]), int(words[0]))
        )

    file.close()
    inst.num_cables = len(cables)
    return cables, inst

def get_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x)**2
        +
        (point1.y - point2.y)**2
    )

def plot_solution(nodes, inst):
    G = nx.Graph()

    for index, node in enumerate(nodes):
        G.add_node(index, pos=(node.x, node.y))

    pos = nx.get_node_attributes(G, 'pos')

    dmin = 1
    ncenter = 0

    for n in pos:
        x, y = pos[n]
        d = (x-0.5)**2+(y-0.5)**2
        if d < dmin:
            ncenter=n
            dmin=d

    p = nx.single_source_shortest_path_length(G, ncenter)

    edge_trace = Scatter(
        x = [],
        y = [],
        line = Line(width=1,color='#888'),
        hoverinfo = 'none',
        mode = 'lines'
    )

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=["Substation"],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale = True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='Greens',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2))
        )

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'].append(len(adjacencies))
        node_info = '# of connections: '+str(len(adjacencies))
        node_trace['text'].append(node_info)

    fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>'+inst.name,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    showarrow = False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False))
             )

    py.plot(fig, filename='networkx.html')

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

    ## Parameters
    #model_type
    #num_threads
    time_limit = 3600
    # available_memory
    # max_nodes
    # cutoff
    # integer_costs

if __name__ == "__main__":
    main()
