from lib.inout import *
from lib.core import *
#from lib.enumerationBDDE import *
import time


def main():
    parameters = parse_command_line()
    k = parameters['k']
    delta = parameters['delta']
    strategy = parameters['strategy']
    G = load_network(parameters['genesin'])
    id_to_str, str_to_id = read_genes(parameters['proteinsin'])
    patients = read_patients(parameters['samplesin'], str_to_id)
    t_start = time.time()

    if strategy == 'combinatorial':
        C, P_C = combinatorial_algorithm(G,k,patients, prob=parameters['prob'])
    elif strategy == 'enumerate':
        BDDE_instance = BDDE(G, patients, f_bound=cardinality_bound, k=k)
        BDDE_instance.enumeration_algorithm()
        C = BDDE_instance.best_subgraph
        P_C = BDDE_instance.best_score
    else:
        raise ValueError("Unkown strategy given. Input: " + str(strategy))
    t_end = time.time()

    print("_________________")
    print("Final solution (ids): ", C)
    print("Final solution (genes' names): ", [id_to_str[el] for el in C])
    if parameters['prob'] or strategy=='enumerate':
        print("Final solution cardinality: ", P_C)
    else:
        print("Final solution cardinality: ", len(P_C))
    print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(t_end-t_start)))
    print("_________________")


if __name__ == "__main__":
    main()

