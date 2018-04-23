import networkx as nx

def BDDE(G,param=None):
    global parametri
    parametri=param
    ListaNodi=list(G.nodes)
    bestSolution=[] 
    bestScore=0
    print("size:"+str(len(ListaNodi)))
    cont=0
    for v in ListaNodi:
        cont+=1
        print(str(cont)+ "  ID:"+str(v))
        radice=v
        parametri['radice']=radice
        B=nx.DiGraph()
        parametri['B']=B
        #print("Radice: "+v)
        DEPTH(G,[],v,[])
        G.remove_node(v)
        
        #Esamino albero binomiale ricavato da sopra
        if(len(B)!=0):
           
            leafs=[]
            radice=None
            for n in B:
                if(B.out_degree(n)==0):#allora � una foglia
                    leafs.append(n)
                if(B.in_degree(n)==0):#allora � la radice
                    radice=n
            print("nfoglie: "+str(len(leafs)))
            for n in leafs:
                #trovo ramo che parte dalla radice e va fino alla foglia
                #corrisponde al sottografo connesso che vogliamo esaminare
                Percorso=list(nx.shortest_path(B,radice,n))
                Percorso=[node.data for node in Percorso ]
                score=parametri['scoringFunction'](parametri['pazienti'],Percorso)
                #print( str(Percorso))
                #print(score)
                if(score>bestScore and len(Percorso)==parametri['k']):
                    print("Best solution updated!")
                    print( str(Percorso)+" -> "+str(score))
                    bestScore=score
                    bestSolution=Percorso
    return (bestSolution,bestScore)    
        
def BREADTH(G,S,n,U):
    #print("BREADTH")
    #print("S: "+str(S)+"    Vertice:"+str(n)+"    U:"+str(U))    
    vn=n.data
    if vn in U :
        return None
    
    S1=S+[vn]
    if(not boundFunction(S1)):
        return None
    
    B=parametri['B']   
    n1=Nodo(vn)
    for nxx in getNodesFromBranch(n):#nxx sarebbe n*
        n2=BREADTH(G,S1,nxx,U)
        if(n2!=None):
            B.add_edge(n1,n2)
    return n1
    
def DEPTH(G,S,v,beta):
    #print("DEPTH")
    #print("S: "+str(S)+"    Vertice:"+str(v)+"    beta:"+str(beta))
    S1=S+[v]
    if(not boundFunction(S1)):
        return None
    
    n=Nodo(v)
    beta1=[]
        
    #extensione set: xn=dnk-cnk
    xn=getxn(G,S,v)
    xn.sort(reverse=True)
    #print("Xn:"+str(xn))
    B=parametri['B']
    for i  in range(0,len(beta),1):
        n1=BREADTH(G,S1,beta[i],xn)
        if(n1!=None):
            B.add_edge(n,n1)
            beta1.append(n1)
    for v in xn:
        n1=DEPTH(G,S1,v,beta1)
        if(n1!=None):
            B.add_edge(n,n1)
            beta1.append(n1)
    return n
 
    
def getNodesFromBranch(n):
    B=parametri['B']
    W=[]
    neighbors=B.neighbors(n)
    for v in neighbors:
        if B.has_edge(n,v):
            W.append(v)
    return W

def boundFunction(S):
    
    return parametri['funzioneLimite'](parametri['pazienti'],S,parametri['k'],
                                parametri['soglia']
                                )
def getxn(G,S,v):
    neighbors=G.neighbors(v)
    L=nx.Graph()
    L.add_nodes_from(neighbors)
    
    if parametri['radice'] in L:
        L.remove_node(parametri['radice'])
    for n in S:
        if n!=v:#in teoria questo non dovrebbe servire
            L.remove_nodes_from(G.neighbors(n))
    return list(L.nodes())

def calcoloFunzioneLimiteSimple(pazienti,S,k,soglia):
    if(len(S)>k):
        return False
    return True
def parametriDefault():
    print("uso parametri di default")
    parametri={}
    parametri['k']=10
    parametri['soglia']=0.8
    parametri['matriceBinaria']=None
    parametri['pazienti']=None
    parametri['funzioneLimite']=calcoloFunzioneLimiteSimple
    return parametri
#Nota:i dati del nodo rappresenta l'etichetta del vertice
#tale etichetta pu� essere qualsiasi cosa inclusi stringhe e numeri
class Nodo(object):
    def __init__(self,data):
        self.data=data
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return str(self) 

