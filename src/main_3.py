import networkx 
import time
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt 
from lib.core import *
from lib.inout import *
from lib.enumerationBDDE import *




def bruteForce(G,param=None):
    global parametri
    parametri=param
    ListaNodi=list(G.nodes)
    bestSolution=[] 
    bestScore=0
    print("size:"+str(len(ListaNodi)))
    cont_i=0
    for i in ListaNodi:
        cont_i+=1
        print("i:"+str(cont_i))
        #cont_j=0
        for j in ListaNodi:
            #cont_j+=1
            #print("j: "+str(cont_j))
            for k in ListaNodi:
                if(i!=j and j!=k and i!=k):#ho nodi diversi
                    S=G.subgraph([i,j,k])
                    #print( str(len(S))+str(S.edges))
                    if len(list(nx.connected_components(S)))==1:#allora deve essere un sottografo
                        #print("Find subgraph")
                        score=parametri['scoringFunction'](parametri['pazienti'],S)
                        if(score>bestScore ):
                            print("Best solution updated!")
                            print( str(S.nodes)+" -> "+str(score))
                            bestScore=score
                            bestSolution=S
    return (bestSolution,bestScore) 




def enumerateAll(G, k, patients, delta=0.8, prob=False):
    G = delta_removal(G, delta)

    if prob:
        obj_func = prob_cover
    else:
        obj_func = score_cover


    parametri={}
    parametri['k']=k
    parametri['soglia']=100 #per ora non usata
    parametri['pazienti']=patients
    parametri['funzioneLimite']=calcoloFunzioneLimiteSimple
    parametri['scoringFunction']=score_cover
    
    bestSolution,bestScore=bruteForce(G, parametri)
    print(bestSolution)
    print(bestScore)
    return (bestSolution,bestScore)  
    







G = loadNetwork("../data/hint+hi2012_edge_file.txt")
id_to_str, str_to_id = read_genes("../data/hint+hi2012_index_file.txt")
patients = read_patients("../data/snvs.tsv", str_to_id)
k=3
t_start = time.time()

bestSolution,bestScore=enumerateAll(G, k, patients, delta=0.8, prob=False)
t_end = time.time()
print("_________________")
print("Final solution (ids): ", bestSolution)
print("Final solution (genes' names): ", [id_to_str[el] for el in bestSolution])
print("Score final solution: "+str(bestScore))
print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(t_end-t_start)))
print("_________________")