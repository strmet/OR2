# our own libraries
from lib.WindFarm import *


def main():

    # Initializes our instance. The class "WindFarm" has everything we need.
    inst = WindFarm(dataset_selection=1)

    # Correctly parses the command line
    inst.parse_command_line()

    # Reads the turbines and the cables file
    inst.read_input()

    inst.build_model()

    # Starts the CPLEX/DOCPLEX solver
    print("Solving...")
    inst.solve()

    # Writing our solution inside a '.sol' file
    #inst.write_solutions()

    # Plotting our solution
    inst.plot_solution(show=False, high=True, export=True)


if __name__ == "__main__":
    main()
