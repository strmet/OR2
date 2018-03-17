from docplex.mp.model import Model

from .definitions import *
from .utils import *

def main():

    points = read_turbines_file("wf01/wf01.turb")
    cables = read_cables_file("wf01/wf01_cb01.cbl")
    print(points)
    n = len(points)
    num_cables = len(cables)

    edges = [Edge(i, j) for i in range(0, n) for j in range(0, n)]

    model = Model(name='wind_farm')
    #model.edges = edges
    #print(model.edges)

    for e in edges:
        model.binary_var(name="y_{0}.{1}".format(e.source + 1, e.destination + 1))
        model.continuous_var(name="f_{0}.{1}".format(e.source + 1, e.destination + 1))

    for e in edges:
        for k in range(0, num_cables):
            model.binary_var(name="x_{0}.{1}.{2}".format(e.source + 1, e.destination + 1, k + 1))
            #print(model.get_var_by_name(name="x_{0}.{1}.{2}".format(e.source,e.destination, k)))

    for i in range(n):
        var = model.get_var_by_name("y_{0}.{1}".format(i + 1, i + 1))
        model.add_constraint(var == 0, ctname="y_{0}.{1}=0".format(i + 1, i + 1))

    for i in range(n):
        var = model.get_var_by_name("f_{0}.{1}".format(i + 1, i + 1))
        con = model.add_constraint(var == 0, ctname="y_{0}.{1}=0".format(i + 1, i + 1))
        #print(con)

    #print(model.get_constraint_by_name("y_81.81=0"))

    for h in range(len(points)):
        if points[h].power < -0.5:
            con = model.add_constraint(
                model.sum(model.get_var_by_name("y_{0}.{1}".format(h + 1, j + 1)) for j in range(n))
                ==
                0
                )
            #print(con)
        else:
            con = model.add_constraint(
                model.sum(model.get_var_by_name("y_{0}.{1}".format(h + 1, j + 1)) for j in range(n))
                ==
                1
            )
            #print(con)

    for h in range(len(points)):
        if points[h].power > 0.5:
            con = model.add_constraint(
                model.sum(model.get_var_by_name("f_{0}.{1}".format(h + 1, j + 1)) for j in range(n))
                ==
                model.sum(model.get_var_by_name("f_{0}.{1}".format(j + 1, h + 1)) for j in range(n))
                + points[h].power
            )


    for edge in edges:
        con = model.add_constraint(
            model.get_var_by_name("y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
            ==
            model.sum(model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source +1, edge.destination + 1, k + 1)) for k in range(num_cables))
        )
        #print(con)

    for edge in edges:
        con = model.add_constraint(
            model.get_var_by_name("y_{0}.{1}".format(edge.source + 1, edge.destination + 1))
            ==
            model.sum(model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) for k in range(num_cables))
        )
        #print(con)

    for edge in edges:
        con = model.add_constraint(
            model.sum(
                model.get_var_by_name("x_{0}.{1}.{2}".format(edge.source + 1, edge.destination + 1, k + 1)) * cables[k].capacity
                for k in range(num_cables)
            )
            >=
            model.get_var_by_name("f_{0}.{1}".format(edge.source + 1, edge.destination + 1))
        )
        #print(con)



    # Objective function
    model.minimize(
        model.sum(
            cables[k].price * get_distance(points[i], points[j]) * model.get_var_by_name("x_{0}.{1}.{2}".format(i + 1, j + 1, k + 1))
            for k in range(0, num_cables) for i in range(0, n) for j in range(0, n)
        )
    )

    print("Solving...")
    model.solve(url=None, key=None)

    print(model.get_solve_status())

    print(model.print_solution())


if __name__ == "__main__":
    main()
