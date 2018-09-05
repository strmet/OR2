# our own libraries
from lib.WindFarm import WindFarm
from lib.Heuristics import Heuristics
import networkx as nx
import math
import time
import pprint as pp

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

    '''wf = WindFarm()
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

    # wf.release()
    optimum = wf.get_solution('y')'''

    wf2 = Heuristics()

    wf2.parse_command_line()

    wf2.read_input()

    strutt_dati, cost = wf2.genetic_algorithm()

    print("Final genetic algorithm cost:")
    print(cost)
    print(math.log(cost, 10))  # debugging
    print("Final genetic algorithm solution:")
    print(strutt_dati)

    wf2.write_results(best_obj=cost)



if __name__ == "__main__":
    main()
