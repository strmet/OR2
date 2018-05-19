# our own libraries
from lib.WindFarm import WindFarm
from lib.Heuristics import Heuristics
import networkx as nx
import math

def main():

    # Initializes our instance. The class "WindFarm" has everything we need.
    """
    The following will:
        - build the WindFarm object
        - call the command line parser
        - set its own parameters
        - build the input files
        - build the output folders
    """
    '''
    wf = WindFarm()
    # Reads the turbines and the cables file
    wf.parse_command_line()

    wf.read_input()

    # Builds the model with such
    wf.build_model()
    # Starts the CPLEX/DOCPLEX solver
    print("Solving...")
    wf.solve()

    if wf.cluster:
        wf.write_results()


    # Writing our solution inside a '.sol' file
    #inst.write_solutions()

    # Plotting our solution
    wf.plot_solution(high=True)

    wf.release()

    '''
    wf2 = Heuristics()

    wf2.parse_command_line()

    wf2.read_input()

    edges = wf2.MST_randomized_costs(delta_interval=0)

    prec, succ, graph = wf2.direct_mst(edges)

    #wf2.plot(graph)

    edges = wf2.grasp(num_edges=12)

    prec, succ, graph = wf2.direct_mst(edges)

    cost = wf2.cost_solution(prec, succ)
    print(cost)
    print(math.log(cost, 10))  # debugging
    wf2.plot(graph)

    tree, cost = wf2.genetic_algorithm()
    print(cost)
    print(math.log(cost, 10))  # debugging

    wf2.plot(tree)

    if tree.out_degree(0)>0:
        raise ValueError("LOLLONI")



if __name__ == "__main__":
    main()
