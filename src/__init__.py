# our own libraries
from lib.WindFarm import *


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

    inst = WindFarm(dataset_selection=1)
    # Reads the turbines and the cables file
    inst.parse_command_line()
    inst.read_input()

    # Builds the model with such
    inst.build_model()

    # Starts the CPLEX/DOCPLEX solver
    print("Solving...")
    inst.solve()

    # Writing our solution inside a '.sol' file
    #inst.write_solutions()

    # Plotting our solution
    #inst.plot_solution(show=False, high=True, export=True)


if __name__ == "__main__":
    main()
