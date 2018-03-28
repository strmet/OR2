import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx
import matplotlib.pyplot as plt


# our own libraries
from lib.instance import *


def main():

    inst = Instance(dataset_selection=5)

    inst.parse_command_line()

    inst.read_turbines_file()
    inst.read_cables_file()

    if inst.interface == 'cplex':
        model = inst.build_model_classical_cplex()
    else:
        model = inst.build_model_docplex()

    print("Solving...")

    model.solve()
    # Writing our solution inside a '.sol' file
    model.solution.write("../out/mysol.sol")

    # inst.plot_solution(model)

    # inst.plot_high_quality(model, export=True)




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
