import pandas as pd
import argparse
import os
import time
import datetime
import networkx as nx

"""
    OLD/TESTING FUNCTIONS
"""

"""
def g_input_read():
    # generiamo un grafo completo un po' a caso
    G = nx.complete_graph(100)

    for (u,v) in G.edges():
        G[u][v]['weight'] = np.random.random() # ]0,1[

    return G

def p_input_read():
    p = {}
    genes = [i for i in range(100)]
    for i in range(100):
        id = "Patient#"+str(i+1)
        patient_genes = np.random.choice(genes, size=int(np.random.random()*100), replace=False)
        p[id] = set(patient_genes)

    return p
    
    
def testing(filename, genes_map):
    frame = pd.read_csv(filename,sep="\t")

    patients={}
    for index,row in frame.iterrows():
        patients.setdefault(row["sampleid"], set())
        if row["geneid"] in genes_map:
            patients[row["sampleid"]].add(genes_map[row["geneid"]])

    return patients
"""


def parse_command_line():
    """
    py:function:: parse_command_line()

    Parses the command line.

    :return: parameters, which is a dictionary. i.e. parameters['parameter'] = value_of_such_parameter.
    """
    # initializing 'parameters' with hard-coded defaults.
    current_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H:%M:%S')
    default_folder = "run__" + timestamp
    parameters = {
        'k': 2,
        'proteinsin': "../data/hint+hi2012_index_file.txt",
        'samplesin': "../data/matriceProb.csv",
        'genesin': "../data/hint+hi2012_edge_file.txt",
        'delta': 0.8,
        'timeout': 10e12,  # for now, this will be ignored
        'prob': False,
        'strategy': 'enumerate',
        'bestsol': 0,  #maximum for a single gene
        'bound': False,
        'outfolder': default_folder  # for now, this will be ignored
    }

    parser = argparse.ArgumentParser(description='Parser to receive input parameters.')

    parser.add_argument('--proteinsin', type=str, help='File path to the gene-to-id data set')
    parser.add_argument('--samplesin', type=str, help='File path to the samples data set')
    parser.add_argument('--genesin', type=str, help='File path to the genes (graph) data set')
    parser.add_argument('--delta', type=float, help='Delta parameter: edge weight tolerance')
    parser.add_argument('--k', type=int, help='Cardinality of the solution to be returned')
    parser.add_argument('--strategy', choices=['combinatorial', 'enumerate'], help='Choose the strategy')
    parser.add_argument('--prob', action="store_true", help='Type --prob if you want to use the' +
                                                            'probaiblistic version of the problem')
    parser.add_argument('--timeout', type=int, help='timeout (seconds) in which the optimizer will stop iterating')
    parser.add_argument('--outfolder', type=str, help='The name (only!) of the folder to be created inside' +
                                                      'the \'$project_path\'/out/ directory')
    parser.add_argument('--bestsol', type=float, help='The previous (k-1) best solution score')
    parser.add_argument('--bound', action="store_true", help='Type --bound if you want to use the bounds')

    args = parser.parse_args()

    if args.bound:
        parameters['bound'] = True

    if args.bestsol:
        parameters['bestsol'] = float(args.bestsol)

    if args.outfolder:
        parameters['outfolder'] = args.outfolde

    if args.strategy:
        parameters['strategy'] = args.strategy

    if args.prob:
        parameters['prob'] = True

    if args.timeout:
        if not (args.k>=1) or type(args.k)!=int:
            raise ValueError("The given 'k' is not a valid value (i.e. integer greater than 1). Given: " + str(args.k))
        parameters['timeout'] = args.timeout

    if args.k:
        if not (args.k>=1) or type(args.k)!=int:
            raise ValueError("The given 'k' is not a valid value (i.e. integer greater than 1). Given: " + str(args.k))
        parameters['k'] = args.k

    if args.delta:
        if not (0 < args.delta <= 1):
            raise ValueError("The given 'delta' is not a valid  value (i.e. in ]0,1]). Given: " + str(args.delta))
        parameters['delta'] = args.delta

    if args.genesin:
        if not os.path.isfile(args.genesin):
            raise FileNotFoundError("Can't find the 'genes' file; filename given: " + args.genesin)

        parameters['genesin'] = args.genesin

    if args.samplesin:
        if not os.path.isfile(args.samplesin):
            raise FileNotFoundError("Can't find the 'samples' file; filename given: " + args.samplesin)

        parameters['samplesin'] = args.samplesin

    if args.proteinsin:
        if not os.path.isfile(args.proteinsin):
            raise FileNotFoundError("Can't find the 'proteins' file; filename given: " + args.proteinsin)

        parameters['proteinsin'] = args.proteinsin

    return parameters


def read_patients_prob(filename, genes_map,filter=[], prob=True):

    frame = pd.read_csv(filename,sep="\t")
    patients={}
    for index,row in frame.iterrows():
        patients.setdefault(row["sampleid"], {})
        if row["geneid"] in genes_map and row["geneid"] not in filter :
            patients[row["sampleid"]][genes_map[row["geneid"]]]=row["prob"]

    patients = {p:patients[p] for p in patients if len(patients[p])>0}

    return patients

# filtro e' un set formato dai nomi dei geni come le chiavi di genes_map quindi TTN,ecc.
def read_patients(filename, genes_map,filter=[]):

    with open(filename, "r") as file:
        patients={}
        lines=file.readlines()
        for l in lines:
            s=l.split("\t")
            # Nota: se un gene risulta mutato in un paziente ma non compare nell'elenco dei geni dato in input,
            #       lo eliminiamo nell'elenco dei geni mutati di tale paziente, considerandolo (evidentemente)
            #       non rilevante.
            patients[s[0]]=set([genes_map[gene] for gene in s[1:] if gene in genes_map and gene not in filter])

    return patients

def read_filter(filename,genes_map):
    with open(filename, "r") as file:
        filter=set()
        lines=file.readlines()
        for l in lines:
            l=l.replace("\n","")#l.strip()
            filter.add(l)
    return filter

def read_genes(filename):

    id_to_str={}  # dato l'id del gene, d[id] = "stringa del gene"
    str_to_id={}  # data la stringa del gene, d[stringa] = "id del gene"

    with open(filename, "r") as file:
        lines=file.readlines()
        for l in lines:
            s=l.split(" ")
            gene_id = int(s[0])
            gene_str = s[1].split("\t")[0]
            id_to_str[gene_id] = gene_str
            str_to_id[gene_str] = gene_id

    return id_to_str, str_to_id


def load_network(filename):

    with open(filename, "r") as file:
        G=nx.Graph()
        lines=file.readlines()
        for l in lines:
            s=l.split(" ")
            s=[ int(ss) for ss in s]
            G.add_edge(s[0], s[1] ,weight=s[2])

    return G

