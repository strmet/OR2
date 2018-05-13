# our own libraries
from lib.WindFarm import WindFarm
from lib.Heuristics import Heuristics

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
    #wf.plot_solution(high=True)

    #wf.release()

    '''
    wf2 = Heuristics()

    wf2.parse_command_line()

    wf2.read_input()

    edges = wf2.MST_randomized_costs(delta_interval=0.1)

    prec, succ, graph = wf2.direct_mst(edges)

    wf2.plot(graph)

    edges = wf2.grasp(num_edges=5)

    prec, succ, graph = wf2.direct_mst(edges)

    wf2.plot(graph)


if __name__ == "__main__":
    main()
