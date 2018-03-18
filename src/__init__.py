from docplex.mp.model import Model
from collections import namedtuple
import math
import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx


Edge = namedtuple("Edge", ["source", "destination"])
Point = namedtuple("Point", ["x", "y", "power"])
Cable = namedtuple("Cable", ["capacity", "price", "max_usage"])


def main():

    points = read_turbines_file("wf05/wf05.turb")
    cables = read_cables_file("wf05/wf05_cb04.cbl")

    n = len(points)
    num_cables = len(cables)

    edges = [Edge(i, j) for i in range(0, n) for j in range(0, n)]

    model = Model(name='wind_farm')

    for e in edges:
        model.binary_var(name="y_{0}.{1}".format(e.source + 1, e.destination + 1))
        model.continuous_var(name="f_{0}.{1}".format(e.source + 1, e.destination + 1))

    for e in edges:
        for k in range(0, num_cables):
            model.binary_var(name="x_{0}.{1}.{2}".format(e.source + 1, e.destination + 1, k + 1))


    for i in range(n):
        var = model.get_var_by_name("y_{0}.{1}".format(i + 1, i + 1))
        model.add_constraint(var == 0, ctname="y_{0}.{1}=0".format(i + 1, i + 1))

    for i in range(n):
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
            model.sum(model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source +1, edge.destination + 1, k + 1)) for k in range(num_cables))
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

    print("Solving...")
    model.solve()

    #model.print_solution()
    plot_solution(points)

def read_turbines_file(name):
    file = open("../data/" + name, "r")
    points = []

    for index, line in enumerate(file):
        if index > 20: break
        words = list(map(int, line.split()))
        points.append(
            Point(words[0], words[1], words[2])
        )

    file.close()
    return points


def read_cables_file(name):
    file = open("../data/" + name, "r")
    cables = []
    for index, line in enumerate(file):
        if index > 3: break
        words = line.split()
        cables.append(
            Cable(int(words[0]), float(words[1]), int(words[0]))
        )

    file.close()
    return cables

def get_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x)**2
        +
        (point1.y - point2.y)**2
    )

def plot_solution(nodes):
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
            size=20,
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
                title='<br>Wind Farm',
                titlefont = dict(size=16),
                showlegend = False,
                hovermode = 'closest',
                margin = dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow = False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    py.plot(fig, filename='networkx.html')

def ypos(i, j):
    return

if __name__ == "__main__":
    main()
