import networkx as nx
import time
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt 
from lib.core import *
from lib.inout import *
from lib.enumerationBDDE import *


G = loadNetwork("../data/hint+hi2012_edge_file.txt")
id_to_str, str_to_id = read_genes("../data/hint+hi2012_index_file.txt")
patients = read_patients("../data/snvs.tsv", str_to_id)
k=3
sum=0
max=0
for v in G:
    dv=len(list(G.neighbors(v)))
    sum+=dv
    if(dv>max):
        max=dv
avg=sum/len(G)
print("Grado medio: "+str(avg))
print("Grado massimo: "+str(max))

"""
t_start = time.time()
bestSolution,bestScore=enumerating_algorithm(G, k, patients, delta=0.8, prob=False)
t_end = time.time()
print("_________________")
print("Final solution (ids): ", bestSolution)
print("Final solution (genes' names): ", [id_to_str[el] for el in bestSolution])
print("Score final solution: "+str(bestScore))
print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(t_end-t_start)))
print("_________________")
"""
