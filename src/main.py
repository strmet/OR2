from lib.inout import *
from lib.core import *
import time

def main():
    k = 4
    delta = 0.8
    G = loadNetwork("../data/hint+hi2012_edge_file.txt")
    id_to_str, str_to_id = read_genes("../data/hint+hi2012_index_file.txt")
    patients = read_patients("../data/snvs.tsv", str_to_id)
    t_start = time.time()
    C = combinatorial_algorithm(G,delta,k,patients)
    t_end = time.time()

    print("_________________")
    print("Final solution (ids): ", C)
    print("Final solution (genes' names): ", [id_to_str[el] for el in C])
    print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(t_end-t_start)))
    print("_________________")

if __name__ == "__main__":
    main()