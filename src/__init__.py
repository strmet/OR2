import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx
import matplotlib.pyplot as plt


# our own libraries
from lib.instance import *


def main():

    inst = Instance(trb_f='../data/wf01/wf01.turb', cbl_f='../data/wf01/wf01_cb01.cbl', name="Wind Farm wf01")

    inst.parse_command_line()

    inst.read_turbines_file()
    inst.read_cables_file()

    if inst.interface == 'cplex':
        model = inst.build_model_classical_cplex()
    else:
        model = inst.build_model_docplex()

    print("Solving...")
    model.solve()

    sol = inst.get_solution(model)

    #plot_solution(inst, sol)

    #plot_high_quality(inst, sol, export=True)



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


if __name__ == "__main__":
    main()
