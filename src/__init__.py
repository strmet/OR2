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

    wf = WindFarm(dataset_selection=1)
    # Reads the turbines and the cables file
    wf.parse_command_line()
    wf.read_input()

    # Builds the model with such
    wf.build_model()

    # Starts the CPLEX/DOCPLEX solver
    print("Solving...")
    wf.solve()

    # Writing our solution inside a '.sol' file
    #inst.write_solutions()

    # Plotting our solution
    wf.plot_solution(high=True)


if __name__ == "__main__":
    main()
