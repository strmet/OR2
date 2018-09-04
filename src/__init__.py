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

    '''# TODO: calcolare prec, calcolare il costo effettivo, plottare, portare il grafico sulla tesina
    wf2.plot(wf2.get_graph([0, 9, 10, 2, 12, 4, 14, 6, 16, 17, 18, 19, 11, 21, 13, 23, 24, 25, 26, 27, 28, 20, 30, 22, 32, 33, 25, 35, 36, 37, 38, 39, 31, 41, 42, 43, 44, 45, 37, 47, 48, 57, 49, 42, 51, 52, 53, 46, 56, 0, 0, 50, 59, 60, 62, 54, 55, 0, 0, 58, 67, 68, 61, 71, 56, 0, 65, 66, 75, 77, 69, 79, 80, 0, 0, 74, 73, 76, 77, 78, 79]), 10^7)

    wf2.plot(
        wf2.get_graph([0, 9, 10, 2, 3, 13, 5, 6, 7, 17, 18, 19, 11, 21, 15, 23, 24, 25, 26, 27, 36, 20, 21, 31, 32, 33, 25, 35, 27, 28, 29, 39, 40, 41, 42, 34, 44, 45, 46, 38, 48, 49, 50, 51, 43, 53, 45, 55, 56, 0, 57, 58, 59, 52, 61, 54, 55, 65, 0, 66, 67, 60, 70, 55, 72, 0, 0, 0, 76, 68, 69, 79, 80, 0, 73, 0, 75, 76, 77, 78, 79]),
        20195738.825338222
    )
    input()'''
    strutt_dati, cost = wf2.genetic_algorithm()

    print("Final genetic algorithm cost:")
    print(cost)
    print(math.log(cost, 10))  # debugging
    print("Final genetic algorithm solution:")
    print(strutt_dati)



if __name__ == "__main__":
    main()
