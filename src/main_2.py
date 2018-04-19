from lib import *

G = loadNetwork("../data/hint+hi2012_edge_file.txt")

id_to_str, str_to_id = read_genes("../data/hint+hi2012_index_file.txt")
patients = read_patients("../data/snvs.tsv", str_to_id)

parametri['k']=3
parametri['delta']=0.8
parametri['pazienti']=patients
parametri['score_func']=obj_func

bestSolution,bestScore=BDDE(G,parametri)   
