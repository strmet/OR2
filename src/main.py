from lib.inout import *
from lib.core import *


def main():
    print(test())

    G = input_read()
    combinatorial_algorithm(G,0.8,1)


if __name__ == "__main__":
    main()