import networkx as nx
import numpy as np

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
"""


def read_patients(filename, genes_map):
    file  = open(filename, "r")
    patients={}
    lines=file.readlines()
    for l in lines:
        s=l.split("\t")
        # Nota: se un gene risulta mutato in un paziente ma non compare nell'elenco dei geni dato in input,
        #       lo eliminiamo nell'elenco dei geni mutati di tale paziente, considerandolo (evidentemente)
        #       non rilevante.
        patients[s[0]]=set([genes_map[gene] for gene in s[1:] if gene in genes_map])
    #print(patients)
    return patients


def read_genes(filename):
    file  = open(filename, "r")
    id_to_str={} # dato l'id del gene, d[id] = "stringa del gene"
    str_to_id={} # data la stringa del gene, d[stringa] = "id del gene"
    lines=file.readlines()
    for l in lines:
        s=l.split(" ")
        gene_id = int(s[0])
        gene_str = s[1].split("\t")[0]
        id_to_str[gene_id] = gene_str
        str_to_id[gene_str] = gene_id
    #print("id_to_str: ", id_to_str)
    #print("str_do_id: ", str_to_id)
    #input()
    #print (genes)
    return id_to_str, str_to_id


def loadNetwork(filename):
    file  = open(filename, "r")
    G=nx.Graph()
    lines=file.readlines()
    for l in lines:
        s=l.split(" ")
        s=[ int(ss) for ss in s]
        G.add_edge(s[0], s[1] ,weight=s[2])
    return G
