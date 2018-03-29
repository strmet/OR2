# our own libraries
from lib.instance import *


def main():

    inst = Instance(dataset_selection=1)

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
    print(inst.get_solution(model))
    inst.plot_solution(model)

    inst.plot_high_quality(model, export=True)


if __name__ == "__main__":
    main()
