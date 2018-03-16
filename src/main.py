from docplex.mp.model import Model
from collections import namedtuple

Edge = namedtuple("Edge", ["source", "destination"])


def main():

    substations, turbines = read_turbines_file("wf01/wf01.turb")
    cables = read_cables_file("wf01/wf01_cb01.cbl")

    n = len(turbines)  + len(substations)
    num_cables = len(cables)

    edges = [Edge(i, j) for i in range(1, n + 1) for j in range(1, n + 1)]

    print(edges)

    model = Model(agent='local', name='wind_farm')
    #model.edges = edges
    #print(model.edges)

    for e in edges:
        model.binary_var(name="y_{0}.{1}".format(e.source, e.destination))
        model.binary_var(name="f_{0}.{1}".format(e.source, e.destination))

    for e in edges:
        for k in range(1, num_cables + 1):
            model.binary_var(name="x_{0}.{1}.{2}".format(e.source, e.destination, k))
            #print(model.get_var_by_name(name="x_{0}.{1}.{2}".format(e.source,e.destination, k)))

    #for i in range(1, n + 1):
    #    var = model.get_var_by_name("y_{0}.{1}".format(i, i))
    #    model.add_equivalence(binary_var=var, linear_ct="eq", true_value=0)
    #print(model.number_of_constraints)
    '''
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            model.binary_var(name="y_{0}.{1}".format(i, j))
            #print(model.get_var_by_name(name="y_{0}.{1}".format(i, j)))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            model.binary_var(name="f_{0}.{1}".format(i, j))
            #print(model.get_var_by_name(name="f_{0}.{1}".format(i, j)))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, num_cables + 1):
                model.binary_var(name="x_{0}.{1}.{2}".format(i, j, k))
                print(model.get_var_by_name(name="x_{0}.{1}.{2}".format(i, j, k)))
    '''

    #for i in range(1, n + 1):

def read_turbines_file(name):
    file = open("../data/" + name, "r")
    substations = []
    turbines = []
    for line in file:
        if (int(line.split()[2]) < - 0.5):
            substations.append([int(num) for num in line.split()])
        else:
            turbines.append([int(num) for num in line.split()])

    file.close()
    return substations, turbines


def read_cables_file(name):
    file = open("../data/" + name, "r")
    cables = []
    for index, line in enumerate(file):
        cables.append([num for num in line.split()])
        cables[index][0], cables[index][2] = int(cables[index][0]), int(cables[index][2])
        cables[index][1] = float(cables[index][1])
    file.close()
    return cables


if __name__ == "__main__":
    main()
