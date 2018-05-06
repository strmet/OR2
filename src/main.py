from lib.inout import *
from lib.core import *


def main():
    parameters = parse_command_line()
    k = parameters['k']
    delta = parameters['delta']
    strategy = parameters['strategy']
    G = load_network(parameters['genesin'])
    id_to_str, str_to_id = read_genes(parameters['proteinsin'])
    if parameters['prob']:
        patients = read_patients_prob(parameters['samplesin'], str_to_id)
    else:
        patients = read_patients(parameters['samplesin'], str_to_id)

    if strategy == 'combinatorial':
        if parameters['prob']:
            C, P_C = prob_combinatorial_algorithm(G,k,patients)
        else:
            C, P_C = combinatorial_algorithm(G,k,patients)
    elif strategy == 'enumerate':
        BDDE_instance = BDDE(G, patients, k=k, prob=parameters['prob'])
        BDDE_instance.enumeration_algorithm()
        C = BDDE_instance.best_subgraph
        P_C = BDDE_instance.best_score
    else:
        raise ValueError("Unkown strategy given. Input: " + str(strategy))

    print("_________________")
    print("Final solution (ids): ", C)
    print("Final solution (genes' names): ", [id_to_str[el] for el in C])
    if parameters['prob'] or strategy=='enumerate':
        print("Final solution cardinality: ", P_C)
    else:
        print("Final solution cardinality: ", len(P_C))
    print("_________________")

if __name__ == "__main__":
    main()

