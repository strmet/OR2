from lib.inout import *
from lib.core import *


def main():
    print(test())
    k = 4
    delta = 0.8
    G = g_input_read()
    patients = p_input_read()
    combinatorial_algorithm(G,delta,k,patients)


if __name__ == "__main__":
    main()