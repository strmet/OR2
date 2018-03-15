import cplex


def main():
    substations, turbines = read_turbines_file("wf01/wf01.turb")
    cables = read_cables_file("wf01/wf01_cb01.cbl")
    print(cables)

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
