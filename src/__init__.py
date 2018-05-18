# our own libraries
from lib.WindFarm import WindFarm
from lib.Heuristics import Heuristics
import networkx as nx

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

    #edges = wf2.grasp(num_edges=12)

    #prec, succ, graph = wf2.direct_mst(edges)

    #cost = wf2.cost_solution(prec, succ)
    #print(cost)
    #wf2.plot(graph)

    wf2.genetic_algorithm()
    chromosome = wf2.encode(graph)
    print(chromosome)

    tree2 = wf2.decode(chromosome)

    wf2.plot(tree2)

if __name__ == "__main__":
    main()
