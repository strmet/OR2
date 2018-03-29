from lib.inout import *
from lib.core import *


def main():
    k = 4
    delta = 0.8
    G = loadNetwork("../data/hint+hi2012_edge_file.txt")
    id_to_str, str_to_id = read_genes("../data/hint+hi2012_index_file.txt")
    patients = read_patients("../data/snvs.tsv", str_to_id)
    C = combinatorial_algorithm(G,delta,k,patients)
    print(C)

if __name__ == "__main__":
    main()