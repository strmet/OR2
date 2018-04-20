from lib.inout import *
from lib.core import *
import time

def main():
    parameters = parse_command_line()
    k = parameters['k']
    delta = parameters['delta']
    G = load_network(parameters['genesin'])
    id_to_str, str_to_id = read_genes(parameters['proteinsin'])
    patients = read_patients(parameters['samplesin'], str_to_id)
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
